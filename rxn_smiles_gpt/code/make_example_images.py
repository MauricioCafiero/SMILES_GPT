"""
Draw the two README examples (successful + unsuccessful inference) as an SVG
reaction scheme: for each, product | true reactants | top-1 prediction.
Writes images/example_reactions.svg. Deterministic — reuse the same beam run
as inference_demo (60 rows, beam 10).

Usage: ../gpt-env/bin/python -I code/make_example_images.py [n_rows] [beam]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")
IMAGES = os.path.join(ROOT, "images")
sys.path.insert(0, HERE)


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 60
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

    good = bad = None
    for i, (g, picks) in enumerate(zip(res["gt"], res["picks"])):
        match = any(sm and Chem.MolFromSmiles(sm)
                    and canon(sm) == canon(g) for sm, _ in picks)
        if match and good is None:
            good = i
        elif not match and bad is None:
            bad = i

    def mol(s):
        return Chem.MolFromSmiles(s)

    rows = [("SUCCESSFUL", good), ("UNSUCCESSFUL", bad)]
    mols, legends = [], []
    for label, idx in rows:
        top1 = res["picks"][idx][0][0]   # picks entries are (string, score)
        m = mol(top1)
        hit = (m is not None
               and canon(top1) == canon(res["gt"][idx]))
        mols.append(mol(res["pros"][idx]))
        legends.append(f"product: {res['pros'][idx]}")
        mols.append(mol(res["gt"][idx]))
        legends.append(f"true reactants: {res['gt'][idx]}")
        mols.append(m)
        legends.append(f"model top-1 ({'exact match' if hit else 'miss'})")

    # emit the same examples as a text file, so the README block and the PNG
    # always come from one and the same beam pass (run-to-run tie-breaking
    # can slightly reorder beams on local floats)
    import contextlib
    import io as _io
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        for label, idx in rows:
            print(f"=== {label} INFERENCE ===")
            print(f"         product: {res['pros'][idx]}")
            print(f"  true reactants: {res['gt'][idx]}")
            for rk, (p, s) in enumerate(res["picks"][idx], 1):
                tag = ("   <- exact match"
                       if p and Chem.MolFromSmiles(p)
                       and canon(p) == canon(res["gt"][idx]) else "")
                print(f"    beam rank {rk}: {p}{tag}")
    txt = f"beam-{beam} over {n_rows} rows (MPS): valid {res['valid']:.1%} " \
          f"top-1 {res['top1_strict']:.1%} top-3 {res['top3']:.1%}\n" + buf.getvalue()

    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(380, 240),
                               legends=legends)
    os.makedirs(IMAGES, exist_ok=True)
    img.save(os.path.join(IMAGES, "example_reactions.png"))
    out = os.path.join(IMAGES, "example_text.txt")
    with open(out, "w") as f:
        f.write(txt)
    print(f"wrote example_reactions.png + example_text.txt; rows: {rows}")


if __name__ == "__main__":
    main()