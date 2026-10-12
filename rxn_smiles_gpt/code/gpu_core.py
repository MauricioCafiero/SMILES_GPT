"""
Train/eval core for the USPTO-50k retro GPT — designed to run in a Modal
container at /app/code with data on a Modal Volume at /vol/data.

Independent of the reference repo's conventions: the loss is masked on real
tokens (pad excluded), sequences are the same [CLS] prod [SEP] reactants
[SEP] format built by build_aug.py / retro_pipeline.py.
"""

import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

CODE = "/app/code" if os.path.isdir("/app/code") else os.path.dirname(os.path.abspath(__file__))
DATA = "/vol/data" if os.path.isdir("/vol/data") else os.path.join(os.path.dirname(CODE), "data")
# test the mount itself, not the subdirectory it will only contain later —
# os.path.isdir("/vol/runs") is False at import (dir exists not yet), which
# silently rerouted all checkpoint writes to container-local disk
RUNS = "/vol/runs" if os.path.isdir("/vol") else os.path.join(os.path.dirname(CODE), "runs")
CKPT = os.path.join(RUNS, "retro_gpt.pt")

# Set by the caller (modal_app) to persist the checkpoint — e.g. vol.commit().
# Without it, function-side writes to the mount are lost when the container
# exits (this is what erased the first run's weights).
KEEP_CKPT = None

MODEL_CFG = dict(num_blocks=8, embed_dim=384, num_heads=6, key_dim=64,
                 ff_dim=1536, dropout_rate=0.1)


def vocab_size():
    with open(os.path.join(CODE, "vocab.txt")) as f:
        return sum(1 for _ in f)


def make_model(max_length=256, device="cuda"):
    import torch
    from gpt_model import GPT
    model = GPT(max_length=max_length, vocab_size=vocab_size(), **MODEL_CFG)
    return model.to(device)


def save_ckpt(model):
    import torch
    os.makedirs(RUNS, exist_ok=True)
    torch.save({"config": model.config(), "state_dict": model.state_dict()}, CKPT)
    if KEEP_CKPT is not None:
        KEEP_CKPT()


def load_variant(idx):
    import torch
    d = np.load(os.path.join(DATA, f"aug{idx}.npz"))
    return (torch.from_numpy(d["x"].astype(np.int64)),
            torch.from_numpy(d["y"].astype(np.int64)))


def load_val():
    import torch
    d = np.load(os.path.join(DATA, "retro_val.npz"))
    return (torch.from_numpy(d["x"].astype(np.int64)),
            torch.from_numpy(d["y"].astype(np.int64)))


def val_loss(model, device, log):
    """Teacher-forced masked loss on the val split (cheap overfit guard)."""
    import torch
    import torch.nn.functional as F

    vx, vy = load_val()
    vx, vy = vx.to(device), vy.to(device)
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for b0 in range(0, len(vx), 2048):
            xb, yb = vx[b0:b0 + 2048], vy[b0:b0 + 2048]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(xb)
                per = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      yb.reshape(-1), reduction="none")
            mask = yb.reshape(-1) != 0
            tot += float((per * mask).sum()); cnt += int(mask.sum())
    log(f"  val loss {tot/cnt:.4f}")
    return tot / cnt


def train(model, epochs, n_variants=4, batch_size=512, lr=1e-3, device="cuda",
          log=print, subset=None, patience=6):
    """Masked-loss training, cycling through the augmentation variants.
    Monitors val loss each epoch; stops early after `patience` non-improving
    epochs (best checkpoint is re-saved before return)."""
    import time
    import torch
    import torch.nn.functional as F

    pad_id = 0
    n0 = len(load_variant(0)[0]) if subset is None else subset
    steps_per_epoch = math.ceil(n0 / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup = max(1, int(0.1 * total_steps))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / warmup if s < warmup
        else max(0.1, 0.1 + 0.5 * (1 - 0.1) * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))
    ))

    best_val, best_state, stale = float("inf"), None, 0
    import copy
    for ep in range(epochs):
        x, y = load_variant(ep % n_variants)
        if subset:
            x, y = x[:subset], y[:subset]
        x, y = x.to(device), y.to(device)
        perm = torch.randperm(len(x), device=device)
        x, y = x[perm], y[perm]
        t0 = time.time(); running = 0.0; nb = 0
        model.train()
        for b0 in range(0, len(x), batch_size):
            xb, yb = x[b0:b0 + batch_size], y[b0:b0 + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(xb)
                per = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      yb.reshape(-1), reduction="none")
            mask = yb.reshape(-1) != pad_id
            loss = (per * mask).sum() / mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += float(loss.item()); nb += 1
        log(f"epoch {ep+1}/{epochs} loss {running/nb:.4f} "
            f"({time.time()-t0:.0f}s) lr {sched.get_last_lr()[0]:.2e}")
        del x, y
        v = val_loss(model, device, log)
        if v < best_val - 1e-4:
            best_val, stale = v, 0
            best_state = copy.deepcopy({k: t.cpu() for k, t in model.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                log(f"early stop after {patience} epochs without val improvement")
                break
        model.eval()
        save_ckpt(model)
    if best_state is not None:
        model.load_state_dict({k: t.to(device) for k, t in best_state.items()})
        log(f"restored best epoch (val {best_val:.4f})")
        # overwrite the per-epoch checkpoint with the best weights, so the
        # file on the volume is the model that produced best val
        save_ckpt(model)
        with open(os.path.join(RUNS, "train_state.json"), "w") as f:
            json.dump({"best_val": best_val, "epochs_run": ep + 1}, f)
        if KEEP_CKPT is not None:
            KEEP_CKPT()
    return model


def beam_eval(model, beam=10, batch_rows=8, device="cuda", log=print, top_k=3,
              eval_rows=None, return_preds=False):
    """Batched beam search on the canonical val split; top-1/top-k metrics.
    Batches are length-sorted so prompts and generations inside a batch have
    similar lengths (fewer wasted decode steps). batch_rows=8 keeps the KV
    cache (which scales as batch_rows x beam) light enough for an 8 GB laptop;
    on a datacenter GPU the bigger batches bought little wall time anyway."""
    import torch
    import torch.nn.functional as F
    from tokenizer import SmilesTokenizer

    tok = SmilesTokenizer(os.path.join(CODE, "vocab.txt"))
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    L = model.max_length
    sys.path.insert(0, CODE)
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    d = np.load(os.path.join(DATA, "retro_val.npz"))
    with open(os.path.join(DATA, "retro_val_gt.json")) as f:
        gt = json.load(f)
    xs = d["x"]
    n = len(xs) if eval_rows is None else min(eval_rows, len(xs))

    def prod_ids_of(row):
        ids = []
        for j in row:
            if j == sep:
                break
            if j != pad:
                ids.append(int(j))
        return ids

    # length-sort: group rows so each batch shares similar prompt length
    prompt_lens = [len(prod_ids_of(xs[i])) for i in range(n)]
    order = sorted(range(n), key=lambda i: prompt_lens[i])
    xs = xs[order]
    gt = {**gt, "gts_react_canon": [gt["gts_react_canon"][i] for i in order],
          "products": [gt["products"][i] for i in order]}

    fin_top = [[] for _ in range(n)]  # per row: [(norm_score, string), ...]
    t0 = time.time()
    for b0 in range(0, n, batch_rows):
        batch_idx = list(range(b0, min(b0 + batch_rows, n)))
        B = len(batch_idx)
        prompts = [[cls] + prod_ids_of(xs[i]) + [sep] for i in batch_idx]
        Lp = max(len(p) for p in prompts)
        mat = np.full((B, Lp), pad, np.int64)
        for r, p in enumerate(prompts):
            mat[r, Lp - len(p):] = p
        mat_t = torch.from_numpy(mat).to(device)
        scores = torch.full((B, beam), -1e9, device=device)
        scores[:, 0] = 0.0
        rows_idx = torch.arange(B, device=device)[:, None]
        V = model.vocab_size

        # hypothesis bookkeeping per row: (norm_score, string); a hypothesis
        # is recorded the step its beam emits [SEP]; finished beams are frozen
        fin_mask = torch.zeros(B, beam, dtype=torch.bool, device=device)

        with torch.no_grad():
            # prefill over the full prompt (no cache yet)
            logits, caches = model(mat_t)
            logp = F.log_softmax(logits[:, -1].float(), -1)           # (B, V)
            cand = (scores[:, :, None] + logp[:, None, :]).reshape(B, -1)
            top_pr = cand.topk(beam, dim=-1).indices
            scores = cand.gather(1, top_pr)
            t_tok = top_pr % V                                       # (B, beam)
            # all beam slots descend from the same prompt: just replicate caches
            caches = [{"k": c["k"].unsqueeze(1).repeat(1, beam, 1, 1, 1)
                              .reshape(B * beam, c["k"].size(1), Lp, c["k"].size(3)),
                       "v": c["v"].unsqueeze(1).repeat(1, beam, 1, 1, 1)
                             .reshape(B * beam, c["v"].size(1), Lp, c["v"].size(3))}
                      for c in caches]
            fin_mask = t_tok == sep
            seq = torch.repeat_interleave(mat_t, beam, dim=0)
            seq = torch.cat([seq, t_tok.reshape(B * beam, 1)], dim=1)

            def record(seq_all, scores_all, new_fin):
                if not bool(new_fin.any()):
                    return
                sc = scores_all.detach().cpu().numpy()
                sq = seq_all.view(B, beam, -1).cpu().numpy()
                for r in range(B):
                    for b in np.nonzero(new_fin.cpu().numpy()[r])[0]:
                        gen = sq[r, b, Lp:]
                        s = sc[r, b] / len(gen) ** 0.7
                        fin_top[batch_idx[r]].append(
                            (float(s), tok.decode(list(map(int, gen)))))

            record(seq, scores, fin_mask)

            for step in range(L - Lp - 1):
                if bool(fin_mask.all()):
                    break
                logits, caches = model(t_tok.reshape(B * beam, 1), cache=caches)
                logp = F.log_softmax(logits[:, 0].float(), -1).view(B, beam, V)
                cand = scores[:, :, None] + logp
                cand[fin_mask] = -1e9   # finished beams cannot be extended
                top_pr = cand.reshape(B, -1).topk(beam, dim=-1).indices
                b_src = top_pr // V
                t_tok = (top_pr % V).view(B, beam)
                scores = cand.reshape(B, -1).gather(1, top_pr)
                # keep finished beams frozen (score/SEP, no candidate growth)
                t_tok = torch.where(fin_mask, torch.full_like(t_tok, sep), t_tok)
                scores = torch.where(fin_mask, torch.full_like(scores, -1e9),
                                     scores)
                # gather cache along the beam axis to the selected sources
                caches = [{"k": c["k"].view(B, beam, *c["k"].shape[1:])[rows_idx, b_src]
                              .reshape(B * beam, *c["k"].shape[1:]),
                           "v": c["v"].view(B, beam, *c["v"].shape[1:])[rows_idx, b_src]
                                .reshape(B * beam, *c["v"].shape[1:])}
                          for c in caches]
                seq = seq.view(B, beam, -1)[rows_idx, b_src].reshape(B * beam, -1)
                seq = torch.cat([seq, t_tok.reshape(B * beam, 1)], dim=1)
                new_fin = (t_tok == sep) & ~fin_mask
                fin_mask = fin_mask | new_fin
                record(seq, scores, new_fin)

        done = b0 + B
        if (b0 // batch_rows) % 10 == 0 or done == n:
            log(f"  beam eval {done}/{n} ({time.time()-t0:.0f}s)")

    # score and report on best finished hypothesis per row
    n_valid = n_strict = n_lenient = n_topk = 0
    pred1, gt_sorted, picks, pros = [], [], [], []
    for i in range(n):
        # dedupe hypotheses by string; keep best-scored instance of each
        seen = {}
        for s, sm2 in fin_top[i]:
            if sm2 not in seen or s > seen[sm2]:
                seen[sm2] = s
        pool = sorted(seen.items(), key=lambda t: t[1], reverse=True)
        pick = pool[:top_k] if pool else [("", -1.0)]
        sm = pick[0][0]
        pred1.append(sm)
        gt_sorted.append(gt["gts_react_canon"][i])
        picks.append(pick)
        pros.append(gt["products"][i])
        m = Chem.MolFromSmiles(sm)
        gt_c = Counter(gt["gts_react_canon"][i].split("."))
        any_k = False
        for s2, _ in pick:
            m2 = Chem.MolFromSmiles(s2)
            if m2 is None:
                continue
            pred2 = Counter(Chem.MolToSmiles(m2).split("."))
            if pred2 == gt_c or (not (gt_c - pred2) and
                                 sum((pred2 - gt_c).values()) <= 2):
                any_k = True
        if any_k:
            n_topk += 1
        if m is None:
            continue
        n_valid += 1
        pred = Counter(Chem.MolToSmiles(m).split("."))
        strict = pred == gt_c
        if strict:
            n_strict += 1
            n_lenient += 1
        elif not (gt_c - pred) and sum((pred - gt_c).values()) <= 2:
            n_lenient += 1
    log(f"BEAM RESULT (beam={beam}) over {n} rows in {time.time()-t0:.0f}s: "
        f"valid {n_valid/n:.1%} top-1 strict {n_strict/n:.1%} "
        f"top-1 lenient {n_lenient/n:.1%} top-{top_k} strict+lenient {n_topk/n:.1%}")
    res = {"n": n, "valid": n_valid / n,
           "top1_strict": n_strict / n, "top1_lenient": n_lenient / n,
           f"top{top_k}": n_topk / n}
    if return_preds:
        res["preds"] = pred1           # top-1 string per row
        res["gt"] = gt_sorted          # aligned ground truth
        res["picks"] = picks           # top-k (score, string) per row
        res["pros"] = pros             # the input product SMILES per row
    return res


def run(epochs, beam=10, subset=None, eval_rows=None):
    """Full job: train then beam-eval. Returns/prints a summary dict."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda s: print(s, flush=True)
    log(f"device {device} torch {torch.__version__}")
    model = make_model(device=device)
    log(f"model params: {sum(p.numel() for p in model.parameters()):,}")
    train(model, epochs, subset=subset, log=log)
    res = beam_eval(model, beam=beam, device=device, eval_rows=eval_rows)
    with open(os.path.join(RUNS, "results.json"), "w") as f:
        json.dump({"epochs": epochs, **res, "config": model.config()}, f, indent=2)
    return res