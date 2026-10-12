import sys
sys.path.insert(0, "/Users/cafierom/python_mac/gpt/code")
import pandas as pd
import numpy as np
import re
from collections import Counter
from smiles_tokenizer import SmilesTokenizer

df = pd.read_parquet("/Users/cafierom/python_mac/temp_repo/uspto50k/train.parquet")
parts = df["rxn_smiles"].str.split(">", expand=True)
r, a, p = parts[0], parts[1], parts[2]

tok = SmilesTokenizer("/Users/cafierom/python_mac/gpt/data/vocab.txt")

def stats(smiles_col, label):
    unk_frac, token_lens = [], []
    for s in smiles_col:
        if not s: continue
        ids = tok.encode(s)[1:-1]
        if len(ids) == 0: continue
        token_lens.append(len(ids))
        unk_frac.append(np.isin(ids, list(tok.unk_token_ids)).mean())
    tl, uf = np.array(token_lens), np.array(unk_frac)
    print(f"{label}: token len med/p95/max = {int(np.median(tl))}/{int(np.percentile(tl,95))}/{tl.max()}, "
          f"mean unk = {uf.mean():.1%}, rows w/ unk = {(uf>0).mean():.1%}")

print("== prod_smiles column (separate, likely unmapped) ==")
print(df["prod_smiles"].head(3).tolist())
stats(df["prod_smiles"], "prod_smiles")
print("prod_smiles vs split p:", (df["prod_smiles"] == p).mean(), "  match unmapped:", (df["prod_smiles"].str.replace(r'\[...?:\d+\]', '', regex=True) == p.str.replace(r'\[...?:\d+\]', '', regex=True)).mean())

print("\n== mapped slices of rxn_smiles ==")
stats(r, "reactants(mapped)")
stats(p, "products(mapped)")
stats(df["rxn_smiles"], "full(mapped)")

# what fraction of chars are the mapping labels vs real atoms
def mapped_stats(s_col, label):
    brack = 0; total = 0
    for s in s_col:
        total += len(s)
        brack += sum(m.end()-m.start() for m in re.finditer(r'\[[^\]]*\]', s))
    print(f"{label}: {brack/total:.1%} of chars inside brackets")

mapped_stats(df["rxn_smiles"], "full")
mapped_stats(df["prod_smiles"], "prod_smiles")

# distinct bracket-atom forms (what a vocab must cover)
c = Counter()
for s in df["rxn_smiles"].sample(5000, random_state=0):
    c.update(re.findall(r'\[[^\]]*\]', s))
print("\ntop bracket tokens:", c.most_common(15))
print("distinct bracket-atom forms in 5k rows:", len(c))

# class distribution
print("\nclass distribution:", df["class"].value_counts().sort_index().to_dict())

# is prod_smiles unmapped? check for ':' inside brackets
print("prod_smiles rows with ':' (atom-map) in prod_smiles:", df["prod_smiles"].str.contains(r'\[.*?:.*?\]', regex=True).mean())