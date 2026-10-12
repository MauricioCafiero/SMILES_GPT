"""
Train the USPTO-50k retrosynthesis GPT locally on any torch device
(cuda, MPS, or CPU) with the production recipe: masked loss, randomized-
SMILES augmentation variants, warmup+cosine LR, early stopping, per-epoch
checkpoints to runs/retro_gpt_local.pt.

Usage:
    ../gpt-env/bin/python -I code/train.py --epochs 50
    ../gpt-env/bin/python -I code/train.py --epochs 2 --subset 2000   # quick shape test
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--subset", type=int, default=None,
                    help="train on only the first N rows per variant")
    ap.add_argument("--batch", type=int, default=None,
                    help="default: 512 on cuda, 128 elsewhere")
    ap.add_argument("--device", default=None, help="default: cuda>mps>cpu")
    ap.add_argument("--ckpt", default="retro_gpt_local.pt")
    args = ap.parse_args()

    import torch
    import gpu_core

    device = args.device or pick_device()
    batch = args.batch or (512 if device == "cuda" else 128)
    log = print

    # data must exist as tokenized arrays; build variants if missing
    data_dir = gpu_core.DATA
    if not os.path.exists(os.path.join(data_dir, "aug0.npz")):
        raise SystemExit(f"no aug0.npz in {data_dir} — run: "
                         f"$PY code/build_aug.py 4 (needs the parquets and "
                         f"data/retro_train_gt.json)")

    gpu_core.CKPT_OVERRIDE = os.path.join(gpu_core.RUNS, args.ckpt)
    model = gpu_core.make_model(device=device)
    log(f"device {device}, batch {batch}, saving to "
        f"{gpu_core.CKPT_OVERRIDE}")
    gpu_core.train(model, args.epochs, batch_size=batch, device=device,
                   log=log, subset=args.subset)
    log("wrote best checkpoint to " + str(gpu_core.CKPT_OVERRIDE))


if __name__ == "__main__":
    main()