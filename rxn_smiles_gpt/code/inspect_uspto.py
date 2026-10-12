import sys
sys.path.insert(0, "/Users/cafierom/python_mac/gpt/code")
import pandas as pd
import numpy as np
from smiles_tokenizer import SmilesTokenizer

df = pd.read_parquet("/Users/cafierom/python_mac/temp_repo/uspto50k/train.parquet")
print("shape:", df.shape)
print("cols:", list(df.columns))
print("keep counts:", df["keep"].value_counts().to_dict())

# split rxn_smiles "reactants>agents>products"
parts = df["rxn_smiles"].str.split(">", expand=True)
print("num parts:", parts.shape[1])
r, a, p = parts[0], parts[1], parts[2]
print("agents empty:", (a.str.strip() == "").sum(), "/", len(df))
print("sample rows:")
for i in range(5):
    print("  class", df["class"].iloc[i], "| rxn:", df["rxn_smiles"].iloc[i][:160])

tok = SmilesTokenizer("/Users/cafierom/python_mac/gpt/data/vocab.txt")
print("\nvocab size:", tok.vocab_size())

def stats(smiles_col, label):
    lens = smiles_col.str.len()
    print(f"{label}: char len min/med/max = {lens.min()}/{int(lens.median())}/{lens.max()}")
    # tokenizer coverage
    unk_frac = []
    token_lens = []
    fail = 0
    for s in smiles_col:
        ids = tok.encode(s)
        ids = ids[1:-1]  # drop CLS/SEP
        if len(ids) == 0:
            fail += 1
            continue
        token_lens.append(len(ids))
        unk = np.isin(ids, list(tok.unk_token_ids)).mean()
        unk_frac.append(unk)
    token_lens = np.array(token_lens)
    unk_frac = np.array(unk_frac)
    print(f"{label}: token len min/med/p95/max = {token_lens.min()}/{int(np.median(token_lens))}/{int(np.percentile(token_lens,95))}/{token_lens.max()}")
    print(f"{label}: mean unk frac = {unk_frac.mean():.4%}, rows with any unk = {(unk_frac>0).mean():.1%}, empty = {fail}")

stats(r, "reactants")
stats(p, "products")
stats(a, "agents")

# full reaction string
full = df["rxn_smiles"]
stats(full, "full rxn_smiles")

# dedup / sanity
print("\nunique prod_smiles:", df["prod_smiles"].nunique(), "/", len(df))
print("prod_smiles == last part:", (p == df["prod_smiles"]).mean())

# chars in rxn_smiles not covered by tokenizer regex patterns quickly -> check unk tokens encountered
alltok = set()
for s in full.sample(3000, random_state=0):
    toks = tok._tokenize(s)
    alltok.update(t for t in toks if t.startswith("?") is None)
unknown = [t for t in alltok if t not in tok.vocab]
print("unknown vocab tokens seen in 3k sample:", unknown[:20], "count uniq:", len(unknown))