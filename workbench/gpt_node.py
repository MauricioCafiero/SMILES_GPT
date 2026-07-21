"""
Workbench GPT node (PyTorch) — rewrite of legacy_code/old_prot.py::gpt_node.

Called by a small agentic med-chem workbench. Reads the ChEMBL bioactives CSV
produced by the workbench's getbioactives_node, fine-tunes the mini GPT on them
(via finetune_gpt), and returns the generated molecules, a summary string, and
a grid image.

getbioactives_node is provided by the workbench runtime — it is not defined in
this repo (same arrangement as legacy_code/old_prot.py). It is referenced as a
bare name, so this module imports cleanly on its own and only fails if
gpt_node is actually called without the workbench having provided
getbioactives_node in scope.
"""

import os

import pandas as pd

from finetune_gpt import finetune_gpt


def gpt_node(chembl_id):
    """Fine-tune the mini GPT on a ChEMBL target's bioactives and generate
    novel molecules.

        Args:
            chembl_id: the ChEMBL ID to query
        Returns:
            smiles_list: list of generated SMILES strings
            gpt_string: human-readable summary of the run
            img: a list containing the grid image of the generated molecules
    """
    print("GPT node")
    print("=" * 51)

    chembl_id = chembl_id.upper()
    # Fetch the bioactives CSV if the workbench hasn't already cached it.
    if not os.path.exists(f"{chembl_id}_bioactives.csv"):
        getbioactives_node([chembl_id])  # workbench-provided

    try:
        df = pd.read_csv(f"{chembl_id}_bioactives.csv")
        smiles_list, gpt_string, img = finetune_gpt(df, chembl_id)
    except Exception as exc:  # pragma: no cover - workbench-facing safety net
        print(f"gpt_node failed: {exc}")
        return [], "", [None]

    return smiles_list, gpt_string, [img]