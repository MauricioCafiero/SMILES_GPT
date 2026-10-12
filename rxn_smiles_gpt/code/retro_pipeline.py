"""
USPTO-50k retrosynthesis with the ../gpt SmilesTokenizer + CafChemGPT GPT.

Stages: build -> train -> eval

Format: next-token sequences over unmapped SMILES. The atom-map labels
(":n" inside brackets) are stripped and each side is canonically
re-serialized by RDKit, then tokenized as

    [CLS] product [SEP] reactants [SEP]

Padding is the tokenizer's [pad] id (0); loss stays unmasked, matching
the repo's convention (the model learns the [SEP]->[PAD] shutdown).

The gpt repo is only read from (tokenizer vocab + model code) — no writes.
"""

import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

GPT_REPO = "/Users/cafierom/python_mac/gpt"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")

MAP_RE = re.compile(r"\[([^:\]]+):\d+\]")


def strip_maps(sm: str) -> str:
    return MAP_RE.sub(r"[\1]", sm)


def canon_side(side: str) -> str:
    """Canonicalize a dot-separated SMILES list, per molecule."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mols = []
    for frag in side.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            return ""
        mols.append(Chem.MolToSmiles(m))
    # order-independent: sort fragments so sets are comparable
    return ".".join(sorted(mols))


def load_tok():
    sys.path.insert(0, os.path.join(GPT_REPO, "code"))
    from smiles_tokenizer import SmilesTokenizer
    return SmilesTokenizer(os.path.join(GPT_REPO, "data", "vocab.txt"))


def cmd_build():
    """Strip maps, canonicalize, tokenize, save retro_{train,val}.npz."""
    import pandas as pd
    tok = load_tok()
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    unk_ids = list(tok.unk_token_ids)
    max_len = 256

    for name, fname in (("train", "train.parquet"), ("val", "val.parquet")):
        t0 = time.time()
        df = pd.read_parquet(os.path.join(DATA, fname))
        seqs, gts, n_bad = [], [], 0
        for i, rsmi in enumerate(df["rxn_smiles"]):
            parts = rsmi.split(">")
            reag = canon_side(strip_maps(parts[0]))
            prod = canon_side(strip_maps(parts[2]))
            if not reag or not prod:
                n_bad += 1
                continue
            seqs.append((prod, reag))
            gts.append(reag)
            if (i + 1) % 10000 == 0:
                print(f"  {name} canon {i+1}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"  {name}: kept {len(seqs)}/{len(df)} bad={n_bad} ({time.time()-t0:.0f}s)")

        # coverage + token-length stats
        lens, unk_rows = [], 0
        for prod, reag in seqs:
            ids = tok.encode(prod + reag)[1:-1]
            if np.isin(ids, unk_ids).any():
                unk_rows += 1
            lens.append(len(ids))
        lens = np.array(lens)
        print(f"  {name}: tokens med/p99/max = {int(np.median(lens))}/"
              f"{int(np.percentile(lens, 99))}/{lens.max()}, "
              f"rows w/ unk = {unk_rows}/{len(seqs)}")

        xs, ys, gts_kept, prod_kept, dropped = [], [], [], [], 0
        for (prod, reag), gt in zip(seqs, gts):
            ids = ([cls] + list(tok.encode(prod)[1:-1]) + [sep]
                   + list(tok.encode(reag)[1:-1]) + [sep])
            if len(ids) > max_len - 1:
                dropped += 1
                continue
            ids = ids + [pad] * (max_len - len(ids))
            xs.append(np.array(ids[:-1], np.int64))
            ys.append(np.array(ids[1:], np.int64))
            gts_kept.append(gt)
            prod_kept.append(prod)
        x, y = np.stack(xs), np.stack(ys)
        print(f"  {name}: tokenized kept={len(x)} dropped_long={dropped} (max_len {max_len})")

        np.savez(os.path.join(DATA, f"retro_{name}.npz"), x=x, y=y)
        with open(os.path.join(DATA, f"retro_{name}_gt.json"), "w") as f:
            json.dump({"products": prod_kept, "gts_react_canon": gts_kept,
                       "dropped_long": dropped}, f)
        print(f"  saved data/retro_{name}.npz  x{x.shape}")


def load_model(max_len=None):
    import torch
    sys.path.insert(0, os.path.join(GPT_REPO, "code"))
    from CafChemGPT import GPT
    ckpt = torch.load(os.path.join(RUNS, "retro_gpt.pt"),
                      map_location="cpu", weights_only=False)
    if max_len:
        ckpt["config"]["max_length"] = max_len
    model = GPT(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model


def cmd_train(epochs=10, batch_size=256, lr=1e-3):
    import torch
    sys.path.insert(0, os.path.join(GPT_REPO, "code"))
    from CafChemGPT import GPT, train_gpt

    d = np.load(os.path.join(DATA, "retro_train.npz"))
    x, y = d["x"], d["y"]
    print(f"train arrays: {x.shape}")

    tok = load_tok()
    model = GPT(num_blocks=4, max_length=x.shape[1] + 1, vocab_size=tok.vocab_size(),
                embed_dim=256, num_heads=4, key_dim=64, ff_dim=1024,
                dropout_rate=0.1)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
    train_gpt(model, x, y, epochs=epochs, batch_size=batch_size, lr=lr,
              warmup_steps=1 * (len(x) // batch_size), min_lr_frac=0.1)

    os.makedirs(RUNS, exist_ok=True)
    torch.save({"config": model.config(), "state_dict": model.state_dict()},
               os.path.join(RUNS, "retro_gpt.pt"))
    print("saved runs/retro_gpt.pt")


def cmd_eval(max_rows=200, batch_rows=32):
    """Greedy top-1 eval on the val split with batched decoding.

    Left-padded prompt batches are safe under causal attention: a row only
    attends to its own left context, so the pad prefix of one row cannot
    leak into another row's generation.
    """
    import torch
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    d = np.load(os.path.join(DATA, "retro_val.npz"))
    x = d["x"][:max_rows]
    with open(os.path.join(DATA, "retro_val_gt.json")) as f:
        gt = json.load(f)
    model = load_model(x.shape[1] + 1)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device).eval()
    print(f"device: {device}")

    tok = load_tok()
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    L = model.max_length

    def prod_ids_of(row):
        ids = []
        for j in row:
            if j == sep:
                break
            if j != pad:
                ids.append(int(j))
        return ids

    n = len(x)
    n_valid = n_strict = n_lenient = 0
    samples = []
    t0 = time.time()
    for b0 in range(0, n, batch_rows):
        batch_idx = list(range(b0, min(b0 + batch_rows, n)))
        prompts = [[cls] + prod_ids_of(x[i]) + [sep] for i in batch_idx]
        Lp = max(len(p) for p in prompts)
        mat = np.full((len(batch_idx), Lp), pad, np.int64)
        for r, p in enumerate(prompts):
            mat[r, Lp - len(p):] = p  # left pad
        mat = torch.from_numpy(mat).to(device)
        live = torch.ones(len(batch_idx), dtype=torch.bool, device=device)
        with torch.no_grad():
            for _ in range(L - Lp):
                logits, _ = model(mat[:, -L:])
                nxt = logits[:, -1].argmax(-1)
                nxt = torch.where(live, nxt, torch.tensor(sep, device=device,
                                                          dtype=nxt.dtype))
                live &= nxt != sep
                mat = torch.cat([mat, nxt[:, None]], dim=1)
                if not live.any():
                    break
        for r, i in enumerate(batch_idx):
            gen_ids = mat[r].tolist()[Lp:]  # the generated region only
            sm = tok.decode(gen_ids)
            m = Chem.MolFromSmiles(sm)
            if m is None:
                continue
            n_valid += 1
            pred = Counter(Chem.MolToSmiles(m).split("."))
            gt_c = Counter(gt["gts_react_canon"][i].split("."))
            strict = pred == gt_c
            if strict:
                n_strict += 1
                n_lenient += 1
            else:
                missing = gt_c - pred
                extra = pred - gt_c
                # reagent tolerance: every ground-truth molecule present,
                # prediction may carry up to 2 extra (reagent-like) molecules
                if not missing and sum(extra.values()) <= 2:
                    n_lenient += 1
            if len(samples) < 5:
                samples.append((gt["products"][i], gt["gts_react_canon"][i],
                                sm, strict))
        done = b0 + len(batch_idx)
        print(f"  {done}/{n}: strict {n_strict/done:.1%} "
              f"lenient {n_lenient/done:.1%} ({time.time()-t0:.0f}s)", flush=True)
    print(f"eval {n} rows in {time.time()-t0:.0f}s: valid {n_valid/n:.1%}, "
          f"top-1 strict {n_strict/n:.1%}, top-1 lenient {n_lenient/n:.1%}")
    for prod, gt_r, pred_r, strict in samples[:5]:
        print(f"example strict={strict}")
        print(f"  prod : {prod}")
        print(f"  gt   : {gt_r}")
        print(f"  pred : {pred_r}")


def cmd_evalbeam(beam=10, max_rows=200, batch_rows=8):
    """Batched beam search eval (no length norm; sum logprob)."""
    import torch
    import torch.nn.functional as F
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    d = np.load(os.path.join(DATA, "retro_val.npz"))
    x = d["x"][:max_rows]
    with open(os.path.join(DATA, "retro_val_gt.json")) as f:
        gt = json.load(f)
    model = load_model(x.shape[1] + 1)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device).eval()
    tok = load_tok()
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    L = model.max_length

    def prod_ids_of(row):
        ids = []
        for j in row:
            if j == sep:
                break
            if j != pad:
                ids.append(int(j))
        return ids

    n = len(x)
    n_valid = n_strict = n_lenient = n_copy = 0
    samples = []
    t0 = time.time()
    for b0 in range(0, n, batch_rows):
        batch_idx = list(range(b0, min(b0 + batch_rows, n)))
        B = len(batch_idx)
        prompts = [[cls] + prod_ids_of(x[i]) + [sep] for i in batch_idx]
        Lp = max(len(p) for p in prompts)
        mat = np.full((B, Lp), pad, np.int64)
        for r, p in enumerate(prompts):
            mat[r, Lp - len(p):] = p
        # (B*beam, pos) beam expansion; scores (B, beam)
        seq = torch.from_numpy(np.repeat(mat, beam, axis=0)).to(device)
        scores = torch.full((B, beam), -1e9, device=device)
        scores[:, 0] = 0.0  # only beam 0 active at the prompt
        fin_score = torch.full((B,), -1e9, device=device)   # best finished per row
        fin_str = [""] * B
        live = torch.ones(B, dtype=torch.bool, device=device)
        with torch.no_grad():
            for step in range(L - Lp):
                logits, _ = model(seq[:, -L:])
                logp = F.log_softmax(logits[:, -1], -1).view(B, beam, -1)
                cand = scores[:, :, None] + logp     # (B, beam, V)
                cand = cand.reshape(B, -1)
                top_pr = cand.topk(beam, dim=-1).indices
                b_src = top_pr // logp.size(-1)
                t_tok = top_pr % logp.size(-1)
                scores = cand.gather(1, top_pr)
                seq = seq.view(B, beam, -1)[torch.arange(B, device=device)[:, None], b_src]
                seq = seq.reshape(B * beam, -1)
                seq = torch.cat([seq, t_tok.reshape(B * beam, 1)], dim=1)
                # rows where the best-finished score beats any live beam can stop
                done_rows = (t_tok == sep).view(B, beam).any(1)
                newfin = done_rows & live
                if newfin.any():
                    best_len = (cand.gather(1, top_pr) /
                                max(step + 1, 1))            # crude length norm
                    for r in torch.nonzero(newfin).flatten().tolist():
                        with_fin = (t_tok[r] == sep).nonzero().flatten()
                        b = int(with_fin[best_len[r][with_fin].argmax()])
                        gen = seq.view(B, beam, -1)[r, b, Lp:].tolist()
                        fin_str[r] = tok.decode(gen)
                        fin_score[r] = scores[r, b]
                    live &= ~newfin
                if not live.any():
                    break
        for r, i in enumerate(batch_idx):
            sm = fin_str[r] or tok.decode(seq.view(B, beam, -1)[r, 0, Lp:].tolist())
            m = Chem.MolFromSmiles(sm)
            copy = (Chem.CanonSmiles(gt["products"][i]) == sm)
            n_copy += int(copy)
            if m is None:
                continue
            n_valid += 1
            pred = Counter(Chem.MolToSmiles(m).split("."))
            gt_c = Counter(gt["gts_react_canon"][i].split("."))
            strict = pred == gt_c
            if strict:
                n_strict += 1
                n_lenient += 1
            else:
                missing = gt_c - pred
                extra = pred - gt_c
                if not missing and sum(extra.values()) <= 2:
                    n_lenient += 1
            if len(samples) < 5:
                samples.append((gt["products"][i], gt["gts_react_canon"][i], sm, strict))
        done = b0 + B
        print(f"  {done}/{n}: valid {n_valid/done:.1%} strict {n_strict/done:.1%} "
              f"lenient {n_lenient/done:.1%} copy {n_copy/done:.1%} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"beam={beam} eval {n} rows in {time.time()-t0:.0f}s: valid {n_valid/n:.1%}, "
          f"top-1 strict {n_strict/n:.1%}, top-1 lenient {n_lenient/n:.1%}, "
          f"copy-rate {n_copy/n:.1%}")
    for prod, gt_r, pred_r, strict in samples[:5]:
        print(f"example strict={strict}")
        print(f"  prod : {prod}")
        print(f"  gt   : {gt_r}")
        print(f"  pred : {pred_r}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == "build":
        cmd_build()
    elif cmd == "train":
        # optional epochs as second arg
        cmd_train(epochs=arg or 10)
    elif cmd == "eval":
        cmd_eval(max_rows=arg or 200)
    elif cmd == "evalbeam":
        cmd_evalbeam(beam=arg or 10)
    else:
        print("usage: retro_pipeline.py [build|train|eval|evalbeam] [arg]")