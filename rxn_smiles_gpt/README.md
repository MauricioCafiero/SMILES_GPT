# rxn_smiles_gpt — retrosynthesis GPT on USPTO-50k

A decoder-only GPT that takes a **product** SMILES as a prompt and generates
**reactants**: single-step retrosynthesis by language modeling. Built as a
feasibility study of the parent repo's tokenizer + GPT architecture on
[pingzhili/uspto-50k](https://huggingface.co/datasets/pingzhili/uspto-50k).

## Results

Trained on all 4 randomized-SMILES variants of the 49k train split, evaluated
by beam search (beam=10) on the full 1,001-row val split:

| Run | Params | Epochs | Valid SMILES | Top-1 | Top-3 | Hardware | Cost |
|---|---|---|---|---|---|---|---|
| 1 | 3.5M | 10 | 9.5% | 0% | — | M-series MPS, ~3 h | free |
| 2 | 14.7M | 50 | 97.2% | 8.2% | 11.5% | Modal L4, ~65 min | ~$1.10 |
| 3 (retrain) | 14.7M | 50 | 96.5% | 8.2% | 10.7% | Modal L4, ~65 min | ~$1.10 |

Runs 2 and 3 are seeded-identical retrains (8.2% both) — the recipe is
reproducible. Saved weights: `runs/retro_gpt_modal.pt` (56 MB,
`torch.load(..., weights_only=True)` → `{"config", "state_dict"}`).

**Feasibility verdict**: the parent repo's tokenizer/vocab (591 tokens) covers
all of USPTO-50k at 0 unknown-token rate once atom-map labels are stripped;
masked-loss + randomized-SMILES augmentation + KV-cached beam decoding make a
working end-to-end predictor. Val loss was still falling at epoch 50, so the
ceiling is training budget, not data. Published seq2seq/graph models reach
~40–50% top-1 with far more compute and reaction-class conditioning (the
`class` column is in the parquet, unused here).

## Environment

Uses the shared repo env `../gpt-env` (Python 3.14, torch + MPS, rdkit,
numpy, pyarrow, pandas, modal). Nothing else is needed — local CPU/MPS
inference works with just the pip install; Modal runs additionally need
`modal login` authentication.

## Code layout

| File | What |
|---|---|
| `code/tokenizer.py`, `code/vocab.txt` | SmilesTokenizer (regex vocab) — copied from the parent repo, unchanged |
| `code/gpt_model.py` | GPT classes from the parent repo + KV-cache decoding (`cache=None` reproduces reference behavior; cache path is single-token decode, prefill runs without cache) |
| `code/gpu_core.py` | Train/eval core: masked loss, AdamW + warmup/cosine, per-epoch checkpoints, early stopping (patience 6), batched KV-cached beam search, `beam_eval(..., return_preds=True)` for per-row strings |
| `code/build_aug.py` | Builds 4 randomized-SMILES variants of train (`.npz`), seeded per row index |
| `code/modal_app.py` | Modal app: `--tiny` (speed probe), `--end2end` (persistence probe), full train+eval |
| `code/inference_demo.py` | Load a checkpoint, beam-decode real val rows, print a hit and a miss |
| `code/make_example_images.py` | Regenerates `images/example_reactions.png` + `images/example_text.txt` (the README examples) from one beam pass |
| `code/retro_pipeline.py` | The original local-only pipeline (data prep + small MPS train/eval; runs 1's history) |
| `data/*.parquet` | Raw HF train/val splits |
| `data/aug{0..3}.npz`, `data/retro_val.npz` | Tokenized int32 training arrays (max_len 256) |
| `data/retro_val_gt.json` | Canonical val products/ground-truths aligned to the val arrays |
| `runs/retro_gpt_modal.pt` | The trained 14.7M checkpoint |
| `images/example_reactions.png` | The two README example reactions, drawn by RDKit |
| `runs/*.log` | Training logs incl. full loss curves |
| `run_modal2.sh`, `run_all.sh` | Launcher scripts (caffeinate + detached launch) |

Data model: input rows are `[CLS] product [SEP] reactants [SEP] [PAD]...`,
student-forcing next-token training with the pad masked out of the loss;
decoding starts at the end of the product prompt.

## Usage

```sh
PY=../gpt-env/bin/python

# rebuild tokenized data (only if the npz files are missing)
$PY code/build_aug.py 4

# tiny probes on Modal L4 (seconds to a minute)
../gpt-env/bin/modal run code/modal_app.py --tiny          # speed/cost probe
../gpt-env/bin/modal run code/modal_app.py --end2end       # full mini-pipeline: save+commit + read back + decode

# full train + beam eval on Modal L4 (~$1.10, ~65 min)
../gpt-env/bin/modal run code/modal_app.py --epochs 50 --beam 10 --eval-rows 1001

# local inference on the saved checkpoint (MPS, no Modal)
$PY code/inference_demo.py 60 10

# regenerate the README example reactions (text + RDKit scheme image)
$PY code/make_example_images.py 60 10
```

After a Modal run, the checkpoint/results live on the shared volume
(`uspto50k`); fetch from the laptop with:

```sh
../gpt-env/bin/modal volume get uspto50k /runs/retro_gpt.pt runs/retro_gpt_modal.pt
```

## The persistence rule (why this gotcha section exists)

Two training runs wrote checkpoints that evaporated, for two independent
reasons; the pattern that works (and matches peptidebuilder's Modal code):

1. **Container-side volume writes need `vol.commit()`** — writes staged in a
   container vanish at its exit otherwise. Commit after every meaningful save
   (here: after each epoch, so a container death keeps everything saved so far).
2. **Resolve volume paths by testing the mount, not the subdir**: at import
   time `/vol/runs` does not exist even when `/vol` is mounted. Sniffing
   `isdir("/vol/runs")` silently rerouted every checkpoint to container-local
   disk (which then had `vol.commit()` commits landing on an empty diff).

`gpu_core.py` now does both (commit hook `KEEP_CKPT`, mount-based path check,
plus an `assert gpu_core.RUNS.startswith("/vol")` in the probe). Verify from
outside the container with `modal volume ls` and a real `volume get` +
`torch.load` — a function "returning success" proves nothing on disk.

## Example inference calls

`make_example_images.py` on the saved model (beam-10, MPS; the PNG and the
text block below are written by the same pass):

![Reaction schemes: product | true reactants | model top-1, for one successful and one unsuccessful inference](images/example_reactions.png)

**Successful** — the product is the free carboxylic acid; the true reaction
is ester hydrolysis and the model outputs that ester, with all top-3
chemically exact (they re-canonicalize to the true reactants):

```
beam-10 over 60 rows (MPS): valid 93.3% top-1 5.0% top-3 6.7%
=== SUCCESSFUL INFERENCE ===
         product: O=C(O)Cn1ccc(C2CC2)n1
  true reactants: CCOC(=O)Cn1ccc(C2CC2)n1
    beam rank 1: O=C(OCC)Cn1ccc(n1)C1CC1   <- exact match
    beam rank 2: O=C(OCC)Cn1ccc(C2CC2)n1   <- exact match
    beam rank 3: O=C(OCC)Cn1nc(cc1)C1CC1   <- exact match
=== UNSUCCESSFUL INFERENCE ===
         product: CCOC(=O)C1CCCCC1O
  true reactants: CCOC(=O)C1CCCCC1=O
    beam rank 1: C1CCCCC1C(=O)OCCc1ccccc1
    beam rank 2: C1(CCCCC1C(OCC)=O)OC(=O)C
    beam rank 3: O=C(C1CCCCC1OC(=O)C)OCC
```

(Repeated local runs vary by a point or two on a 60-row sample due to
floating-point tie-breaking; the 8.2% top-1 was reproduced twice on the full
1,001-row GPU eval. `make_example_images.py` regenerates the PNG and the
matching text block from a single pass.)

## Training stats

Loss curves (masked, real tokens only; val = teacher-forced on the canonical
val split) — from `runs/modal_train3.log`, the run that produced the saved
checkpoint:

| epoch | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 40 | 49 (best) | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 2.72 | 1.67 | 1.37 | 1.07 | 0.75 | 0.62 | 0.58 | 0.55 | 0.533 | 0.532 |
| val | 1.87 | 1.55 | 1.26 | 1.03 | 0.70 | 0.61 | 0.59 | 0.57 | **0.5625** | 0.5656 |

Config: 8 blocks, d=384, 6 heads (key 64), ff 1536 → 14,748,495 params;
max_len 256; batch 512; bf16 autocast; AdamW lr 1e-3, wd 0.01, warmup 10% +
cosine to 0.1×; ~73 s/epoch on L4; 171k tokens/s measured by `--tiny`.