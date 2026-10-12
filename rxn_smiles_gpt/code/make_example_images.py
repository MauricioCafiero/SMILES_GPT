"""
Draw the README example reactions as a scheme grid: product | true reactants |
model top-1 | best lower-rank candidate, for one example of each class:
  SUCCESSFUL     top-1 chemically exact
  RANK-RECOVERED top-1 wrong, but a lower-ranked candidate is exact/close
  UNSUCCESSFUL   nothing in the top-3 is exact or close
Writes images/example_reactions.png + images/example_text.txt (README uses
both; they always come from the same beam pass).

Usage: ../gpt-env/bin/python -I code/make_example_images.py [n_rows] [beam]
"""

import os
import sys
from collections import Counter

from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")
IMAGES = os.path.join(ROOT, "images")
sys.path.insert(0, HERE)


def frag_counts(sm, canon):
    """Counter of canonical fragment strings in a dot-separated set."""
    return Counter(canon(f) for f in sm.split("."))


def gt_match(sm, gt_c):
    """How candidate SMILES sm relates to ground truth gt_c (Counter):
    'exact', 'close' (≤1 fragment differs), or None."""
    if not sm:
        return None
    m = Chem.MolFromSmiles(sm)
    if m is None:
        return None
    pc = Counter(Chem.MolToSmiles(m).split("."))
    if pc == gt_c:
        return "exact"
    if len(pc - gt_c) + len(gt_c - pc) <= 1:
        return "close"
    return None


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    import gpu_core
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    from gpt_model import GPT
    RDLogger.DisableLog("rdApp.*")

    ck = torch.load(os.path.join(RUNS, "retro_gpt_modal.pt"),
                    map_location=device, weights_only=True)
    model = GPT(max_length=256, vocab_size=gpu_core.vocab_size(),
                **gpu_core.MODEL_CFG)
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    res = gpu_core.beam_eval(model, beam=beam, batch_rows=8, device=device,
                             eval_rows=n_rows, return_preds=True)

    def canon(s):
        return Chem.MolToSmiles(Chem.MolFromSmiles(
            ".".join(sorted(s.split(".")))))

    def mol(s):
        return Chem.MolFromSmiles(s)

    good = recovered = bad = top3 = None
    for i, (g, picks) in enumerate(zip(res["gt"], res["picks"])):
        top1 = picks[0][0]
        gt_c = frag_counts(g, canon)
        m1 = gt_match(top1, gt_c)
        lower = None                      # best lower-rank relation
        for rk, (p, _s) in enumerate(picks[1:], 2):
            rel = gt_match(p, gt_c)
            if rel and lower is None:
                lower = (rk, p, rel)
        if m1 == "exact" and good is None:
            good = i
        elif m1 != "exact" and lower and recovered is None:
            recovered = i
        elif not m1 and not lower and bad is None:
            bad = i
        if lower and top3 is None:        # any product with a rank>=2 hit
            top3 = i

    cats = [("SUCCESSFUL", good), ("RANK-RECOVERED", recovered),
            ("UNSUCCESSFUL", bad)]
    missing = [a for a, i in cats if i is None]
    if missing:
        print(f"no example found for {missing}; rerun with more rows "
              f"or a different n")
        return

    mols, legends = [], []
    for label, idx in cats:
        picks = res["picks"][idx]
        gt_c = frag_counts(res["gt"][idx], canon)
        m1 = gt_match(picks[0][0], gt_c)
        mols += [mol(res["pros"][idx]), mol(res["gt"][idx]), mol(picks[0][0])]
        legends += [f"product: {res['pros'][idx]}",
                    f"true reactants: {res['gt'][idx]}",
                    f"model top-1 ({m1 if m1 else 'wrong'})"]
        lower = next(((rk, p) for rk, (p, _s) in enumerate(picks[1:], 2)
                      if gt_match(p, gt_c)), None)
        if lower:
            rk, p = lower
            mols.append(mol(p))
            legends.append(f"model rank {rk} "
                           f"({gt_match(p, gt_c)} match: {p})")
        else:
            alt = next((p for p, _s in picks[1:] if mol(p)), picks[0][0])
            mols.append(mol(alt))
            legends.append(f"no top-3 match (rank 2: {alt})")

    import contextlib
    import io as _io
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        for label, idx in cats:
            print(f"=== {label} INFERENCE ===")
            print(f"         product: {res['pros'][idx]}")
            print(f"  true reactants: {res['gt'][idx]}")
            gt_c = frag_counts(res["gt"][idx], canon)
            for rk, (p, _s) in enumerate(res["picks"][idx], 1):
                rel = gt_match(p, gt_c)
                tag = f"   <- {rel} match" if rel else ""
                print(f"    beam rank {rk}: {p}{tag}")
    txt = (f"beam-{beam} over {n_rows} rows (MPS): "
           f"valid {res['valid']:.1%} top-1 {res['top1_strict']:.1%} "
           f"top-3 {res['top3']:.1%}\n" + buf.getvalue())

    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(380, 240),
                               legends=legends)
    os.makedirs(IMAGES, exist_ok=True)
    img.save(os.path.join(IMAGES, "example_reactions.png"))
    out = os.path.join(IMAGES, "example_text.txt")
    with open(out, "w") as f:
        f.write(txt)
    print(f"wrote example_reactions.png + example_text.txt; "
          f"rows: {dict((a, i) for a, i in cats)}")


if __name__ == "__main__":
    main()