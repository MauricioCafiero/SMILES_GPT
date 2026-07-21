# CafChem SMILES GPT (PyTorch)

A small decoder-only GPT that generates small-molecule **SMILES** strings. Build a
**foundation** model on a large SMILES corpus, then **fine-tune** it on a
target dataset to bias generation toward a chemistry of interest.

This is a PyTorch refactor of an earlier TensorFlow/Keras SMILES GPT. The model
architecture, loss, and training recipe are matched to the original TF model so
that generation quality is preserved.

## Models in this repo

| File | What | Params | Trained on |
|------|------|-------|------------|
| `data/GPT_ZN305_pytorch.pt` | Foundation model (2 transformer blocks) | 1.41M | `data/ZN305K_smiles.csv` (305K SMILES) |
| `data/GPT_ZN305_mini.pt` | Mini foundation (1 transformer block) | ~0.7M | `data/ZN305K_smiles.csv` — base for the laptop-friendly workbench fine-tune |
| `data/GPT_Tyrosinase_finetuned.pt` | Fine-tuned model (4 blocks) | 2.73M | `Tyrosinase1239_IC50.csv` (tyrosinase inhibitors) |
| `data/GPT_Tyrosinase_finetuned_frozen.pt` | Frozen-phase checkpoint (intermediate) | 2.73M | Used by the Colab notebook to skip re-running the frozen phase |

Foundation: unmasked next-token loss **0.158**, generates **7/7 valid** drug-like
molecules. Fine-tune: **12/12 valid**, output enriched in tyrosinase-inhibitor
motifs (phenols/catechols, thiosemicarbazones, flavonoids).

## Architecture

Decoder-only transformer, post-norm residuals, matching the original TF model:

- `EMBEDDING_DIM = 256`, `N_HEADS = 2`, `KEY_DIM = 256`, `FEED_FORWARD_DIM = 256`, `DROPOUT = 0.1`
- **Per-head key dimension** attention: q/k/v project `embed_dim -> num_heads * key_dim`
  (`256 -> 512`), as in Keras `MultiHeadAttention(num_heads, key_dim)`. This is **not**
  the standard `embed_dim // num_heads` split — it's a larger attention and is what
  matches the original.
- Causal self-attention via `torch.nn.functional.scaled_dot_product_attention` (fast on Apple Silicon).
- LayerNorm `eps=1e-6`, ReLU FFN, GPT-2-style embedding init `N(0, 0.02)`.
- **Unmasked** cross-entropy loss (no `ignore_index` on padding). ~75% of targets are
  `[PAD]`; training on them teaches the `[SEP] -> [PAD]` shut-down transition, which is
  what makes generation stop cleanly instead of producing run-on chains.

## Device

Apple Silicon (MPS) is the default; CUDA is used when available; CPU is the final
fallback. See `get_device()` in `code/CafChemGPT.py`.

## Quick start

### Local (Mac) — train / fine-tune / infer

```bash
python -m venv gpt-env && source gpt-env/bin/activate
pip install torch pandas numpy scikit-learn matplotlib rdkit
python code/run_gpt.py
```

Stages are controlled with environment variables (defaults shown):

| Variable | Default | Meaning |
|----------|---------|---------|
| `STAGE_FOUNDATION` | 1 | Build/train the foundation on ZN305K |
| `STAGE_FINETUNE`   | 1 | Fine-tune on Tyrosinase (frozen then unfrozen) |
| `STAGE_INFER`      | 1 | Generate molecules |
| `FOUNDATION_EPOCHS`| 10 | Foundation epochs (the shipped model used ~170) |
| `RESUME_FOUNDATION`| 0 | Continue from `data/GPT_ZN305_pytorch.pt` instead of fresh |
| `FINETUNE_EPOCHS`  | 50 | Total fine-tune epochs (split frozen/unfrozen) |
| `FINETUNE_BATCH`   | 512 | Fine-tune batch size |
| `INFER_MODEL`      | auto | `foundation` / `finetuned` / `auto` (prefer finetuned) |
| `INFER_PROMPTS`    | 100 | Number of molecules to generate |
| `INFER_TEMP`       | 1.5 | Sampling temperature (use **0.7** for these models) |
| `INFER_RAMP`       | 1 | Temperature ramp (use **0** for these models) |

For the shipped models, inference works best at `INFER_TEMP=0.7 INFER_RAMP=0`:

```bash
STAGE_FOUNDATION=0 STAGE_FINETUNE=0 STAGE_INFER=1 \
INFER_MODEL=finetuned INFER_TEMP=0.7 INFER_RAMP=0 INFER_PROMPTS=100 \
python code/run_gpt.py
```

> On an 8 GB Mac the **unfrozen** fine-tune phase is slow (~130 s/epoch under MPS
> memory pressure, all 2.7M params trainable). The frozen phase is fast (~4 s/epoch).
> For the full fine-tune, use the Colab notebook below.

### Colab (GPU) — train the foundation or fine-tune

- **`notebooks/Colab_Foundation_Train.ipynb`** — trains the foundation from scratch on a GPU
  (bf16 autocast on CUDA), saves `data/GPT_ZN305_pytorch.pt`, downloads it back.
- **`notebooks/Colab_Finetune_Tyrosinase.ipynb`** — fine-tunes on Tyrosinase. Defaults to
  `SKIP_FROZEN=True`: loads the committed frozen checkpoint
  (`data/GPT_Tyrosinase_finetuned_frozen.pt`) and runs **only the unfrozen phase** on GPU,
  then downloads `data/GPT_Tyrosinase_finetuned.pt`. Set `SKIP_FROZEN=False` to re-run the
  frozen phase from the foundation instead.
- **`notebooks/Colab_Mini_Foundation_Train.ipynb`** — trains the **1-block mini foundation**
  (`data/GPT_ZN305_mini.pt`) from scratch on a GPU, same recipe as the 2-block one but a
  single transformer block (~0.7M params). Download it back and commit it for the workbench
  fine-tune below.

Both notebooks clone this repo (public), so no uploads are needed — the foundation
`.pt`, the Tyrosinase CSV, and the vocab are all tracked here.

### Notebook (interactive workflow)

`notebooks/GPT_CafChem.ipynb` walks through the full workflow end-to-end: tokenize, build
foundation, fine-tune with transfer learning, save/load, and generate molecules.

## Workbench mini fine-tune (laptop-friendly)

`workbench/` is a PyTorch rewrite of the legacy agentic med-chem prototype
(`legacy_code/`) that fine-tunes the **mini** 1-block foundation on a ChEMBL
target's bioactives and generates novel molecules. It reuses the
`code/CafChemGPT.py` toolkit and is deliberately small so it runs on an 8 GB Mac:

1-block foundation + 1 new block = a 2-block fine-tune (~half the 4-block Tyrosinase fine-tune).

- `workbench/gpt_node.py` — `gpt_node(chembl_id)`: reads `{chembl_id}_bioactives.csv`
  (fetched by the workbench's `getbioactives_node`, as in the legacy prototype), calls
  `finetune_gpt`, and returns `(smiles_list, gpt_string, [img])`.
- `workbench/finetune_gpt.py` — `finetune_gpt(df, chembl_id)`: caps the dataset to 2000,
  tokenizes, fine-tunes the mini foundation (frozen then unfrozen, 25 + 25 epochs by
  default), saves `data/GPT_{chembl_id}_mini_finetuned.pt`, generates 50 molecules, and
  caches them to `gen_smiles_{chembl_id}.csv`. `make_mini_finetune_gpt` loads the mini
  foundation and appends one fresh block (mirrors `CafChemGPT.make_finetune_gpt`).

Requires the committed `data/GPT_ZN305_mini.pt` (produce it with
`notebooks/Colab_Mini_Foundation_Train.ipynb`). `legacy_code/` is kept as reference
and is not used at runtime.

## Repository layout

```
code/
  CafChemGPT.py        model, training (train_gpt), inference, dataset utils
  smiles_tokenizer.py  SMILES tokenizer (DeepChem-style regex, no DeepChem dep)
  run_gpt.py           CLI driver for foundation / finetune / infer stages
data/
  ZN305K_smiles.csv              foundation training corpus
  vocab_305K.txt                 100-token vocabulary (used for training + inference)
  vocab.txt                      larger 591-token vocabulary
  GPT_ZN305_pytorch.pt           foundation model
  GPT_ZN305_mini.pt               mini foundation (1 block, for workbench fine-tune)
  GPT_Tyrosinase_finetuned.pt    fine-tuned model
  GPT_Tyrosinase_finetuned_frozen.pt  frozen-phase checkpoint (for Colab skip-frozen)
notebooks/
  Colab_Foundation_Train.ipynb     train foundation on Colab GPU
  Colab_Mini_Foundation_Train.ipynb train the 1-block mini foundation on Colab GPU
  Colab_Finetune_Tyrosinase.ipynb  fine-tune on Colab GPU
  GPT_CafChem.ipynb                interactive workflow notebook
workbench/
  gpt_node.py        workbench entry point (called by the agentic med-chem workbench)
  finetune_gpt.py    mini-foundation fine-tune + generation, reusing CafChemGPT
legacy_code/          old TF prototype (old_prot.py, finetune_gpt.py) — reference only
Tyrosinase1239_IC50.csv          fine-tune dataset
```

## Tokenizer

`code/smiles_tokenizer.py` is a standalone DeepChem-style regex tokenizer (no
DeepChem dependency, which doesn't run on Python > 3.11). Special tokens:
`[PAD]=0`, `[unused]=1`, `[CLS]=2`, `[SEP]=3`. The regex deliberately does **not**
use `[A-Z][a-z]?` — that pattern merges an aliphatic atom with a following
aromatic atom (`Cc`, `Nc`, `Oc`) into a token absent from the vocab, corrupting
~2.6% of tokens. Halogens use `Cl?/Br?` and every other atom is a single letter.

## Dependencies

`torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `rdkit`.

## License

See `LICENSE`.