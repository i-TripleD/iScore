import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
#from rdkit.Chem import AllChem

def extract_ligands_from_pdbs(pdb_ids):
    """Extracts SMILES strings of ligands from SDF files in subdirectories named by PDB IDs.

    Args:
        pdb_ids (list): A list of PDB IDs.

    Returns:
        list: A list of SMILES strings representing the largest ligand in each PDB file's SDF. 
              If no ligand was found or the SDF file is invalid, None is returned for that position.
    """
    smiles_list = []

    for pdb_id in pdb_ids:
        pdb_id = os.path.splitext(pdb_id)[0]
        sdf_file_path = f"Training/Data/PDBs/{pdb_id}/{pdb_id}_ligand.sdf"
        try:
            suppl = Chem.SDMolSupplier(sdf_file_path, sanitize=False, removeHs=True)
            if suppl:
                for mol in suppl:
                    if mol is not None:
                        smiles_list.append(Chem.MolToSmiles(mol))
                        break  # Only take the first molecule (assumes one ligand per SDF)
                    else:
                        smiles_list.append(None)
            else:
                smiles_list.append(None)
        except FileNotFoundError:
            smiles_list.append(None)

    return smiles_list
    
def calculate_descriptors(smiles_list):
    """Calculates RDKit descriptors for a list of SMILES strings.

    Args:
        smiles_list (list): A list of SMILES strings.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated descriptors.
    """

    descriptor_names = [
        'MolLogP', 'MolMR', 'ExactMolWt', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 
        'NumHeteroatoms', 'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings',
        'RingCount', 'TPSA', 'LabuteASA', 'Kappa1', 'Kappa2', 'Kappa3', 
        'Chi0', 'Chi1', 'Chi0n', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n',
        'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
        'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 
        'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 
        'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 
        'SMR_VSA1', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6',
        'SMR_VSA7', 'SMR_VSA9', 'SMR_VSA10',
        'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 
        'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA10', 'SlogP_VSA11', 
        'SlogP_VSA12',
        'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
        'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10',
        'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
        'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'VSA_EState10'
    ]

    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc_values = [getattr(Descriptors, desc)(mol) for desc in descriptor_names]
            descriptors.append(desc_values)
        else:
            descriptors.append([None] * len(descriptor_names))  # For invalid SMILES

    return pd.DataFrame(descriptors, columns=descriptor_names)

# Read input data
pocket_data = pd.read_csv("Training/Data/dpout_explicitp.txt", delim_whitespace=True)
pocket_data['pdb'] = pocket_data['pdb'].astype(str).str.replace('.pdb', '', regex=False) 
pkd_data = pd.read_csv("Training/Data/pKd.csv", header=None, names=["pdb", "pKd"])  

# Extract SMILES from SDF files
ligands = extract_ligands_from_pdbs(pocket_data["pdb"])

# Calculate descriptors
df_descriptors = calculate_descriptors(ligands)

columns_to_drop_set1 = [
    "pdb", "lig", "overlap", "PP-crit", "PP-dst", "crit4", 
    "crit5", "crit6", "crit6_continue", "nb_AS_norm", "apol_as_prop_norm", 
    "mean_loc_hyd_dens_norm", "polarity_score_norm", "as_density_norm", 
    "as_max_dst_norm", "drug_score"
]

columns_to_drop_set2 = {
    "pock_vol","nb_AS","mean_as_ray","mean_as_solv_acc","apol_as_prop","mean_loc_hyd_dens","hydrophobicity_score","volume_score","polarity_score","charge_score","flex","prop_polar_atm","as_density","as_max_dst",
    "convex_hull_volume","surf_pol_vdw14","surf_pol_vdw22","surf_apol_vdw14","surf_apol_vdw22","n_abpa","ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","pKd"
}

merged_data = pd.concat([df_descriptors, pocket_data], axis=1)
merged_data = merged_data.merge(pkd_data, on="pdb", how="left")

iScoreData = merged_data.drop(columns=columns_to_drop_set1)
UFSData = iScoreData.drop(columns=columns_to_drop_set2, errors='ignore')

iScoreData.to_csv("Training/Data/iScore_Training-set.csv", index=False)
UFSData.to_csv("Training/Data/UFS_Training-set.csv", index=False)
