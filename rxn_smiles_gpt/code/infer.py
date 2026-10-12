"""
Inference on arbitrary product SMILES — the low-memory path: one prompt at
a time with a KV-cached beam search (beam x heads x length x head_dim cache
per block, tens of MB total, no training arrays loaded).

Usage:
    ../gpt-env/bin/python -I code/infer.py "CCOC(=O)C1CCCCC1O"
    ../gpt-env/bin/python -I code/infer.py --beam 10 SMILES1 SMILES2 ...
    ../gpt-env/bin/python -I code/infer.py          # interactive, one SMILES per line

Options: --beam 10 | --top 3 | --ckpt runs/retro_gpt_modal.pt | --device auto
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

DEFAULT_CKPTS = ["retro_gpt_modal.pt", "retro_gpt.pt"]  # in runs/


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    import torch
    from gpt_model import GPT
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = GPT(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def canon_product(sm):
    """Map-stripped canonical product (retro_pipeline.canon_side); '' if invalid."""
    from retro_pipeline import canon_side, strip_maps
    return canon_side(strip_maps(sm))


def top_reactants(model, product_smiles, device, beam=10, top_k=3):
    """Beam-decode reactants for one product. Returns (prod, [(sm, score, valid), ...]).
    Same scoring as gpu_core.beam_eval: length norm score / len^0.7, deduped
    by string, ranked best-first."""
    import torch
    import torch.nn.functional as F
    from tokenizer import SmilesTokenizer
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    prod = canon_product(product_smiles)
    if not prod:
        return None, []

    tok = SmilesTokenizer(os.path.join(HERE, "vocab.txt"))
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    prod_ids = tok.encode(prod)[1:-1].tolist()          # product tokens only
    prompt = [cls] + prod_ids + [sep]                   # the training-time format
    L = model.max_length
    if len(prompt) + 2 > L:
        raise ValueError(f"product tokenizes to {len(prod_ids)} tokens "
                         f"(max_len {L}) — try a smaller molecule")

    pt = torch.tensor([prompt], dtype=torch.int64, device=device)

    with torch.no_grad():
        # prefill once over the prompt (is_causal), then extend all beams
        logits, caches = model(pt)
        cand = F.log_softmax(logits[:, -1].float(), -1)  # (1, V)
        idx = cand.reshape(-1).topk(beam).indices        # first step, single path
        t_tok = idx.view(beam, 1)                        # token to feed each beam row
        scores = cand.reshape(-1)[idx]                   # (beam,)

        # replicate the (1, h, Lp, hd) cache for the beam rows
        caches = [{"k": c["k"].repeat(beam, 1, 1, 1),
                   "v": c["v"].repeat(beam, 1, 1, 1)} for c in caches]
        seq = pt.repeat(beam, 1)
        seq = torch.cat([seq, t_tok.view(beam, 1)], dim=1)

        fin = t_tok.view(-1) == sep
        fin_top = []                                      # (norm_score, string)
        for b in torch.nonzero(fin).flatten().tolist():
            gen = seq[b].tolist()[len(prompt):]
            fin_top.append((float(scores[b]) / len(gen) ** 0.7, tok.decode(gen)))

        b0_cache = torch.arange(beam, device=device)
        V = model.vocab_size
        for _ in range(L - len(prompt) - 1):
            if bool(fin.all()):
                break
            logits, caches = model(t_tok.view(-1, 1), cache=caches)
            logp = F.log_softmax(logits[:, 0].float(), -1)  # (beam, V)
            cand = scores.view(beam, 1) + logp
            cand[fin] = -1e9                                # finished beams can't grow
            idx = cand.reshape(-1).topk(beam).indices
            b_src, t_tok = idx // V, idx % V
            scores = cand.reshape(-1)[idx]
            caches = [{"k": c["k"][b_src], "v": c["v"][b_src]} for c in caches]
            seq = seq[b_src]
            seq = torch.cat([seq, t_tok.view(beam, 1)], dim=1)
            new_fin = (t_tok == sep) & ~fin
            fin = fin | new_fin
            for b in torch.nonzero(new_fin).flatten().tolist():
                gen = seq[b].tolist()[len(prompt):]
                fin_top.append((float(scores[b]) / len(gen) ** 0.7, tok.decode(gen)))

    # dedupe by string, rank best-first
    seen = {}
    for s, sm2 in fin_top:
        if sm2 not in seen or s > seen[sm2]:
            seen[sm2] = s
    pool = sorted(seen.items(), key=lambda t: t[1], reverse=True)[:top_k]

    out = []
    for sm2, s in pool:
        m = Chem.MolFromSmiles(sm2)
        out.append((sm2, s, m is not None,
                    Chem.MolToSmiles(m) if m is not None else ""))
    return prod, out


def run_one(model, args, sm):
    prod, picks = top_reactants(model, sm, args.device, beam=args.beam,
                                top_k=args.top)
    if prod is None:
        print(f"  invalid SMILES, cannot parse: {sm}")
        return
    print(f"product: {prod}")
    if not picks:
        print("  (no completed candidate within the length cap)")
        return
    for rank, (cand, score, valid, canon) in enumerate(picks[:args.top], 1):
        tag = "valid" if valid else "INVALID SMILES"
        print(f"  rank {rank}: {cand}    [{tag}]")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles", nargs="*", help="product SMILES (omit for stdin)")
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--ckpt", default=None,
                    help="path to a .pt checkpoint (default: newest of "
                         "runs/retro_gpt_modal.pt, runs/retro_gpt.pt)")
    ap.add_argument("--device", default=None, help="default: cuda>mps>cpu")
    args = ap.parse_args()

    ckpt = args.ckpt
    if ckpt is None:
        for name in DEFAULT_CKPTS:
            path = os.path.join(ROOT, "runs", name)
            if os.path.exists(path):
                ckpt = path
                break
        if ckpt is None:
            raise SystemExit(f"no checkpoint in {ROOT}/runs -- pass --ckpt")
    args.device = args.device or pick_device()

    print(f"loading {os.path.basename(ckpt)} on {args.device} (beam {args.beam})")
    model = load_model(ckpt, args.device)
    lines = args.smiles or [l.strip() for l in sys.stdin if l.strip()]
    for sm in lines:
        run_one(model, args, sm)


if __name__ == "__main__":
    main()