"""
Build K randomized-SMILES variants of the train split (products + reactants
randomized independently per row), tokenized with the same [CLS] prod [SEP]
reactiants [SEP] / max_len=256 scheme as retro_pipeline.build.

Reads: data/retro_train_gt.json (canonical products + gts_react_canon)
Writes: data/aug{k}.npz  (k = 0..K-1)   variant 0 is a re-tokenization too
        (each epoch draws one variant; canonical strings are kept for eval)
"""

import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CODE = HERE
sys.path.insert(0, HERE)

MAX_LEN = 256
N_VAR = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SEED = 1234


def rand_smiles(sm, rng) -> str:
    """Randomized SMILES of a dot-separated side, fragment order shuffled."""
    from rdkit import Chem
    frags = []
    for frag in sm.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            return ""
        frags.append(Chem.MolToSmiles(m, doRandom=True, canonical=False))
    rng.shuffle(frags)
    return ".".join(frags)


def one_row(args):
    i, prod, reag, k = args
    rng = random.Random(SEED + 7919 * k + i)
    p = rand_smiles(prod, rng)
    r = rand_smiles(reag, rng)
    if not p or not r:
        return None
    return p, r


def main():
    from tokenizer import SmilesTokenizer
    tok = SmilesTokenizer(os.path.join(CODE, "vocab.txt"))
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

    with open(os.path.join(DATA, "retro_train_gt.json")) as f:
        gt = json.load(f)
    rows = list(zip(gt["products"], gt["gts_react_canon"]))
    print(f"{len(rows)} rows, {N_VAR} variants", flush=True)

    t0 = time.time()
    with Pool(6) as pool:
        per_var = [pool.map(one_row, [(i, p, r, k) for i, (p, r) in enumerate(rows)])
                   for k in range(N_VAR)]
    print(f"randomized {N_VAR}x{len(rows)} rows in {time.time()-t0:.0f}s", flush=True)

    for k, var in enumerate(per_var):
        var = [v for v in var if v is not None]
        xs, ys, dropped = [], [], 0
        for prod, reag in var:
            ids = ([cls] + tok.encode(prod)[1:-1].tolist() + [sep]
                   + tok.encode(reag)[1:-1].tolist() + [sep])
            if len(ids) > MAX_LEN - 1:
                dropped += 1
                continue
            ids = ids + [pad] * (MAX_LEN - len(ids))
            # int32: vocab < 65536, halves the volume footprint
            xs.append(np.array(ids[:-1], np.int32))
            ys.append(np.array(ids[1:], np.int32))
        np.savez(os.path.join(DATA, f"aug{k}.npz"),
                 x=np.stack(xs), y=np.stack(ys))
        print(f"  aug{k}: {len(xs)} kept, {dropped} dropped", flush=True)


if __name__ == "__main__":
    main()