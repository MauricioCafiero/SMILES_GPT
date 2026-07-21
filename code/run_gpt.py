"""
CafChem GPT - PyTorch workflow driver.

Runs the end-to-end pipeline described in CLAUDE.md / GPT_CafChem.ipynb:

    Stage 1 - Build & save a foundation GPT from data/ZN305K_smiles.csv
    Stage 2 - Fine-tune on Tyrosinase1239_IC50.csv (transfer learning)
    Stage 3 - Inference: generate molecules and save generated_molecules.png

Stages are toggled with the environment variables below so you can re-run
just the part you need. Epoch counts are configurable; defaults are kept
modest so the whole pipeline runs on a laptop - raise them for quality.

    STAGE_FOUNDATION   = 1   # build + save foundation model
    STAGE_FINETUNE     = 1   # fine-tune on Tyrosinase
    STAGE_INFER        = 1   # generate molecules

    FOUNDATION_EPOCHS  = 10  # original used 50; 10 is a reasonable laptop default
    FINETUNE_EPOCHS    = 50  # frozen-then-unfrozen fine-tuning
    INFER_PROMPTS      = 12  # number of prompts for generation

Run from the repo root so that relative paths (data/..., code/) resolve:

    cd <repo root>
    python3 code/run_gpt.py
"""

import os
import sys

import numpy as np

# Make sibling modules importable when run as `python3 code/run_gpt.py`.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Run from repo root so data/ paths resolve.
ROOT = os.path.dirname(HERE)
if os.getcwd() != ROOT:
    os.chdir(ROOT)

from CafChemGPT import (
    get_device, make_datasets, make_gpt, train_gpt, save_gpt,
    test_vocab, trim_vocab, make_finetune_gpt, unfreeze_gpt,
    load_foundation, make_prompts, gen_mols,
)

# ---- config from env -------------------------------------------------------
STAGE_FOUNDATION = os.environ.get("STAGE_FOUNDATION", "1") == "1"
STAGE_FINETUNE   = os.environ.get("STAGE_FINETUNE",   "1") == "1"
STAGE_INFER      = os.environ.get("STAGE_INFER",      "1") == "1"

FOUNDATION_EPOCHS = int(os.environ.get("FOUNDATION_EPOCHS", "10"))
FOUNDATION_BATCH  = int(os.environ.get("FOUNDATION_BATCH",  "512"))
FOUNDATION_FILE   = "data/GPT_ZN305_pytorch"            # save_gpt appends .pt
# When 1, continue training from the saved foundation checkpoint instead of
# starting from a freshly initialized model. Useful for adding epochs to an
# undertrained foundation without discarding prior progress.
RESUME_FOUNDATION = os.environ.get("RESUME_FOUNDATION", "0") == "1"

FINETUNE_EPOCHS   = int(os.environ.get("FINETUNE_EPOCHS", "50"))
FINETUNE_BATCH    = int(os.environ.get("FINETUNE_BATCH",  "512"))
FINETUNE_FILE     = "data/GPT_Tyrosinase_finetuned"

INFER_PROMPTS     = int(os.environ.get("INFER_PROMPTS", "100"))
INFER_TEMP        = float(os.environ.get("INFER_TEMP", "1.5"))
# use_ramp controls the generation ramp (see gen_mols). Set INFER_RAMP=0 to
# disable it — useful for deterministic/low-temperature sampling.
INFER_RAMP        = os.environ.get("INFER_RAMP", "1") == "1"


def build_foundation():
    """Stage 1: train the foundation GPT on ZN305K and save it."""
    print("\n" + "=" * 70)
    print("STAGE 1 - Build foundation model on ZN305K_smiles.csv")
    print("=" * 70)
    print("Device:", get_device(), flush=True)

    cache = "data/ZN305K_tokenized.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        fx, fy = z["fx"], z["fy"]
        VOCAB_SIZE, max_length = int(z["VOCAB_SIZE"]), int(z["max_length"])
        from smiles_tokenizer import SmilesTokenizer
        tokenizer = SmilesTokenizer(vocab_file="data/vocab_305K.txt")
        print(f"Loaded tokenized cache {cache}: fx {fx.shape}, fy {fy.shape}", flush=True)
    else:
        fx, fy, VOCAB_SIZE, tokenizer, max_length = make_datasets(
            "data/ZN305K_smiles.csv", "SMILES"
        )
        np.savez(cache, fx=fx, fy=fy, VOCAB_SIZE=VOCAB_SIZE, max_length=max_length)
        print(f"Saved tokenized cache {cache}", flush=True)
    print(f"VOCAB_SIZE: {VOCAB_SIZE} | max_length: {max_length} | fx: {fx.shape} | fy: {fy.shape}", flush=True)

    ckpt_path = FOUNDATION_FILE + ".pt"
    if RESUME_FOUNDATION and os.path.exists(ckpt_path):
        print(f"RESUME_FOUNDATION=1: loading {ckpt_path} to continue training.", flush=True)
        from CafChemGPT import load_gpt
        # total_layers=2 matches the original foundation (see make_gpt call below).
        gpt = load_gpt(FOUNDATION_FILE, 2, max_length, VOCAB_SIZE)
    else:
        if RESUME_FOUNDATION:
            print(f"RESUME_FOUNDATION=1 but {ckpt_path} not found; starting fresh.", flush=True)
        gpt = make_gpt(2, max_length, VOCAB_SIZE)
    train_gpt(gpt, fx, fy, epochs=FOUNDATION_EPOCHS, batch_size=FOUNDATION_BATCH)
    save_gpt(gpt, FOUNDATION_FILE)
    print(f"Foundation model saved -> {FOUNDATION_FILE}.pt", flush=True)


def finetune():
    """Stage 2: transfer-learn on Tyrosinase (freeze then unfreeze)."""
    print("\n" + "=" * 70)
    print("STAGE 2 - Fine-tune on Tyrosinase1239_IC50.csv")
    print("=" * 70)
    print("Device:", get_device())

    data_path = "Tyrosinase1239_IC50.csv"

    # Check vocabulary compatibility, then trim the dataset.
    novel_tokens = test_vocab(data_path, "SMILES")
    trim_vocab(data_path, novel_tokens)

    fx, fy, VOCAB_SIZE, tokenizer, max_length = make_datasets(
        "Tyrosinase1239_IC50_trimmed.csv", "SMILES"
    )
    print(f"VOCAB_SIZE: {VOCAB_SIZE} | max_length: {max_length} | fx: {fx.shape} | fy: {fy.shape}")

    # Build fine-tuning model: foundation weights + 2 new blocks, old frozen.
    gpt_ft = make_finetune_gpt(2, freeze_old_layers=True)
    train_gpt(gpt_ft, fx, fy, epochs=FINETUNE_EPOCHS // 2, batch_size=FINETUNE_BATCH)

    # Save the frozen-only model BEFORE attempting the unfrozen phase. The
    # unfrozen phase trains all 2.7M params and can stall under MPS memory
    # pressure on the 8GB Mac; this checkpoint preserves the frozen work so a
    # stall never loses it. Usable as-is (loss ~0.17).
    save_gpt(gpt_ft, FINETUNE_FILE + "_frozen")
    print(f"Frozen-only model saved -> {FINETUNE_FILE}_frozen.pt")

    # Unfreeze everything and train a bit more (as in the notebook workflow).
    gpt_ft = unfreeze_gpt(gpt_ft)
    train_gpt(gpt_ft, fx, fy, epochs=FINETUNE_EPOCHS - FINETUNE_EPOCHS // 2,
              batch_size=FINETUNE_BATCH)

    save_gpt(gpt_ft, FINETUNE_FILE)
    print(f"Fine-tuned model saved -> {FINETUNE_FILE}.pt")
    return VOCAB_SIZE, tokenizer, max_length


def infer(VOCAB_SIZE=None, tokenizer=None, use_finetuned=False):
    """Stage 3: generate molecules and save an image."""
    print("\n" + "=" * 70)
    print("STAGE 3 - Inference / molecule generation")
    print("=" * 70)
    print("Device:", get_device(), flush=True)

    # Prefer the fine-tuned model when its file exists (and caller didn't
    # force the foundation model), so re-running inference uses the latest.
    want = os.environ.get("INFER_MODEL", "auto").lower()
    finetuned_exists = os.path.exists(FINETUNE_FILE + ".pt")
    use_ft = want == "finetuned" or (want == "auto" and finetuned_exists) or use_finetuned

    if use_ft:
        from CafChemGPT import load_gpt
        print(f"Loading fine-tuned model: {FINETUNE_FILE}.pt", flush=True)
        model = load_gpt(FINETUNE_FILE, 4, 166, VOCAB_SIZE)
    else:
        print("Loading foundation model.", flush=True)
        model = load_foundation()

    if tokenizer is None:
        from smiles_tokenizer import SmilesTokenizer
        tokenizer = SmilesTokenizer(vocab_file="data/vocab_305K.txt")
    if VOCAB_SIZE is None:
        VOCAB_SIZE = tokenizer.vocab_size()

    prompts = make_prompts(INFER_PROMPTS, 2)
    pic, novel_smiles = gen_mols(prompts, INFER_RAMP, model, tokenizer,
                                 INFER_TEMP, VOCAB_SIZE)
    pic.save("generated_molecules.png")
    print(f"Saved generated_molecules.png with {len(novel_smiles)} unique molecules.")
    print("Sample SMILES:", novel_smiles[:5])


if __name__ == "__main__":
    if STAGE_FOUNDATION:
        build_foundation()
    ft_vocab, ft_tok, ft_len = None, None, None
    if STAGE_FINETUNE:
        ft_vocab, ft_tok, ft_len = finetune()
    if STAGE_INFER:
        # Use the fine-tuned model if we just trained it; otherwise foundation.
        infer(ft_vocab, ft_tok, use_finetuned=STAGE_FINETUNE)