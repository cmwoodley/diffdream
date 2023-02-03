from rdkit import Chem
from rdkit.Chem import AllChem

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
    