import sys
sys.path.insert(0, "/Users/cafierom/python_mac/gpt/code")
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

df = pd.read_parquet("/Users/cafierom/python_mac/temp_repo/uspto50k/train.parquet")
p = df["rxn_smiles"].str.split(">").str[2]

def canon(s, strip_map=True):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    if strip_map:
        for a in m.GetAtoms(): a.SetAtomMapNum(0)
        Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m)

# match prod_smiles after stripping maps + canonicalizing
match = 0; none = 0
for ps, s in zip(df["prod_smiles"][::50], p[::50]):
    cm = canon(s)
    if cm is None: none += 1
    elif canon(ps) == cm: match += 1
print(f"prod_smiles == canonicalized unmapped rxn product: {match}/{match+none} (None: {none})")

# validity of prod_smiles
bad = sum(Chem.MolFromSmiles(s) is None for s in df["prod_smiles"])
print(f"prod_smiles RDKit-valid: {1-bad/len(df):.2%}")

# validity of mapped slices
badp = sum(Chem.MolFromSmiles(s) is None for s in p[::50])
badr = sum(Chem.MolFromSmiles(s) is None for s in df["rxn_smiles"].str.split(">").str[0][::50])
print(f"mapped product parse: {1-badp/len(p[::50]):.2%}   mapped reactants parse: {1-badr/len(p[::50]):.2%}")

# a concrete example
i = 0
print("\nrxn product :", p[i])
print("prod_smiles :", df["prod_smiles"][i])

# prod_smiles_pop
print("\nprod_smiles_pop range:", df["prod_smiles_pop"].min(), df["prod_smiles_pop"].max())

# length distribution of full mapped rxn in tokens (needs map-aware vocab anyway -> chars)
full = df["rxn_smiles"]
print("full rxn char len p50/p95/p99:", *(int(full.str.len().quantile(q)) for q in (0.5,0.95,0.99)))