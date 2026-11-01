"""
Score a checkpoint on the canonical val split through the inference path —
`infer.top_reactants` (single unpadded prompt per row, KV-cached beam search).
This is the same decode Davros's tool uses, so the numbers here are the
model's *as-deployed* accuracy. Do not batch prompts for score reporting:
gpu_core.beam_eval's left-padded batching shifts absolute positions
(learned pos-emb) and scored ~10% where this scores ~32.5% top-1 (2026-10).

Usage:
    ../gpt-env/bin/python -I code/eval.py --rows 40        # smoke slice
    ../gpt-env/bin/python -I code/eval.py                  # beam 10, all 1001
    ../gpt-env/bin/python -I code/eval.py --out eval.json  # full dump
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

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


def main():
    import torch
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from infer import load_model, top_reactants

    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--rows", type=int, default=None,
                    help="first N val rows (default: all)")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--ckpt", default=None,
                    help="path to a .pt checkpoint (default: newest of "
                         "runs/retro_gpt_modal.pt, runs/retro_gpt.pt)")
    ap.add_argument("--device", default=None, help="default: cuda>mps>cpu")
    ap.add_argument("--out", default=None,
                    help="write the full prediction dump to this JSON path")
    parse = ap.parse_args()

    ckpt = parse.ckpt
    if ckpt is None:
        for name in DEFAULT_CKPTS:
            path = os.path.join(ROOT, "runs", name)
            if os.path.exists(path):
                ckpt = path
                break
        if ckpt is None:
            raise SystemExit(f"no checkpoint in {ROOT}/runs -- pass --ckpt")

    with open(os.path.join(ROOT, "data", "retro_val_gt.json")) as f:
        gt = json.load(f)
    products = gt["products"]
    gts = gt["gts_react_canon"]
    n = len(products) if parse.rows is None else min(parse.rows, len(products))

    device = parse.device or pick_device()
    print(f"loading {os.path.basename(ckpt)} on {device} (beam {parse.beam})",
          flush=True)
    model = load_model(ckpt, device)

    n_valid = n_strict = n_lenient = n_topk = 0
    preds = []
    t0 = time.time()
    for i in range(n):
        _, picks = top_reactants(model, products[i], device, beam=parse.beam,
                                 top_k=parse.top)
        pred1 = picks[0][0] if picks else ""
        preds.append({"product": products[i], "gt": gts[i],
                      "top": [(c, round(s, 3), v, c2) for c, s, v, c2 in picks]})
        gt_c = Counter(gts[i].split("."))
        any_hit = False
        for _, _, _, canon in picks:
            if canon:
                pred_c = Counter(canon.split("."))
                if pred_c == gt_c or (not (gt_c - pred_c)
                                      and sum((pred_c - gt_c).values()) <= 2):
                    any_hit = True
        if any_hit:
            n_topk += 1
        m = Chem.MolFromSmiles(pred1)
        if m is None:
            preds[-1]["strict"] = False
        else:
            n_valid += 1
            pred_c = Counter(Chem.MolToSmiles(m).split("."))
            strict = pred_c == gt_c
            preds[-1]["strict"] = bool(strict)
            if strict:
                n_strict += 1
                n_lenient += 1
            elif not (gt_c - pred_c) and sum((pred_c - gt_c).values()) <= 2:
                n_lenient += 1
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  {i+1}/{n}: valid {n_valid/(i+1):.1%} "
                  f"top-1 strict {n_strict/(i+1):.1%} "
                  f"top-1 lenient {n_lenient/(i+1):.1%} "
                  f"top-{parse.top} {n_topk/(i+1):.1%} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"RESULT (beam={parse.beam}) over {n} rows in {time.time()-t0:.0f}s: "
          f"valid {n_valid/n:.1%} top-1 strict {n_strict/n:.1%} "
          f"top-1 lenient {n_lenient/n:.1%} top-{parse.top} {n_topk/n:.1%}")
    if parse.out:
        payload = {"ckpt": os.path.basename(ckpt), "beam": parse.beam, "n": n,
                   "top1_strict": n_strict / n, "top1_lenient": n_lenient / n,
                   f"top{parse.top}": n_topk / n, "valid": n_valid / n,
                   "preds": preds}
        os.makedirs(os.path.dirname(os.path.abspath(parse.out)), exist_ok=True)
        with open(parse.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {parse.out}")


if __name__ == "__main__":
    main()