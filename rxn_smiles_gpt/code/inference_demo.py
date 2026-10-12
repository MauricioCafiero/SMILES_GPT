"""
Inference demo: load a saved checkpoint and beam-decode val rows with the
trained model, then print one exact-match (successful) and one miss
(unsuccessful) reaction for the README. Runs locally (MPS) — no Modal needed.

Usage: .venv/bin/python code/inference_demo.py [n_rows] [beam]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")
sys.path.insert(0, HERE)


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    import torch
    device = ("mps" if torch.backends.mps.is_available() else "cpu")

    import gpu_core
    from gpt_model import GPT
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    ck = torch.load(os.path.join(RUNS, "retro_gpt_modal.pt"),
                    map_location=device, weights_only=True)
    model = GPT(max_length=256, vocab_size=gpu_core.vocab_size(),
                **gpu_core.MODEL_CFG)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    print(f"loaded {sum(p.numel() for p in model.parameters()):,} params on {device}")

    res = gpu_core.beam_eval(model, beam=beam, batch_rows=8, device=device,
                             eval_rows=n_rows, return_preds=True)
    print(f"beam-{beam} over {res['n']} rows: valid {res['valid']:.1%} "
          f"top-1 {res['top1_strict']:.1%} top-3 {res['top3']:.1%}")

    def canon(s):
        return Chem.MolToSmiles(Chem.MolFromSmiles(
            ".".join(sorted(s.split(".")))))

    good = bad = None
    for i, (pro, g, picks) in enumerate(zip(res["pros"], res["gt"],
                                            res["picks"])):
        match = any(sm2 and Chem.MolFromSmiles(sm2)
                    and canon(sm2) == canon(g) for sm2, _ in picks)
        if match and good is None:
            good = i
        elif not match and bad is None:
            bad = i

    for label, idx in (("SUCCESSFUL", good), ("UNSUCCESSFUL", bad)):
        print(f"\n=== {label} INFERENCE ===")
        print(f"         product: {res['pros'][idx]}")
        print(f"  true reactants: {res['gt'][idx]}")
        for rk, (p, s) in enumerate(res["picks"][idx], 1):
            tag = ""
            if p and Chem.MolFromSmiles(p) and canon(p) == canon(res["gt"][idx]):
                tag = "   <- exact match"
            print(f"    beam rank {rk}: {p}{tag}")


if __name__ == "__main__":
    main()