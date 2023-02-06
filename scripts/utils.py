from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from typing import List, Optional, Tuple, Union

def clean_smiles(sd_file):
    smiles = []
    with Chem.SDMolSupplier(sd_file) as suppl:
        for mol in suppl:
            if mol is None: continue
            elif Chem.rdMolDescriptors.CalcExactMolWt(mol) > 500:
                continue
            else: 
                smiles.append(Chem.MolToSmiles(mol)+"\n")

    with open("../datasets/processed/clean_smiles.txt","w") as f:
        for i in range(len(smiles)):
            f.writelines(smiles[i])
    

def scale_model_input(sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
    """
    Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
    current timestep.

    Args:
        sample (`torch.FloatTensor`): input sample
        timestep (`int`, optional): current timestep

    Returns:
        `torch.FloatTensor`: scaled input sample
    """
    return sample
