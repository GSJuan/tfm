from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import pandas as pd
from scipy.stats import ttest_ind
from io import StringIO
import sys
from utils import clean_error_message
from rdkit.rdBase import WrapLogs, BlockLogs
#WrapLogs() #https://www.rdkit.org/docs/source/rdkit.rdBase.html
BlockLogs()

def is_valid_smiles(smiles):
    """Check if a SMILES string is valid.
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None
    """
    
    validity = {
        "syntactically_valid": False,
        "semantically_valid": False,
        "error": ""
    }
    
        
    #Syntatically valid yet semantically invalid smiles="N=N=N"
    #Perfectly valid smiles = "CC(=O)O"

    sio = sys.stderr = StringIO()
    m = Chem.MolFromSmiles(smiles,sanitize=False) #https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html

    if m is None:
        validity["error"] = clean_error_message(sio.getvalue())
        return validity

    validity["syntactically_valid"] = True

    try:
        Chem.SanitizeMol(m, catchErrors=False)  #https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.SanitizeMol
    except Exception as e:
        validity["error"] = clean_error_message(e)
        return validity

    validity["semantically_valid"] = True
    return validity
    

def calculate_properties(smiles):
    """Calculate chemical properties of a molecule given its SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    properties = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logP': Descriptors.MolLogP(mol),
        'num_h_bond_donors': Descriptors.NumHDonors(mol),
        'num_h_bond_acceptors': Descriptors.NumHAcceptors(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'topological_polar_surface_area': Descriptors.TPSA(mol),
        'qed': QED.qed(mol) #https://www.rdkit.org/new_docs/source/rdkit.Chem.QED.html#
    }
    return properties


def calculate_validity(smiles_list):
    """Calculate the validity percentage of a list of SMILES strings."""

    results = {
        "total": len(smiles_list),
        "syntactically_valid": 0,
        "semantically_valid": 0,
        "valid_smiles": [],
    }

    for smiles in smiles_list:
        validity = is_valid_smiles(smiles)
        if validity["syntactically_valid"]:
            results["syntactically_valid"] += 1
        if validity["semantically_valid"]:
            results["semantically_valid"] += 1
            results["valid_smiles"].append(smiles)

    results["syntactically_valid_percentage"] = results["syntactically_valid"] / results["total"] * 100
    results["semantically_valid_percentage"] = results["semantically_valid"] / results["total"] * 100

    return results

        

def calculate_novelty(generated_smiles, reference_smiles):
    """Calculate the novelty percentage of generated SMILES strings."""
    reference_set = set(reference_smiles)
    novel_smiles = [s for s in generated_smiles if s not in reference_set]
    novelty = len(novel_smiles) / len(generated_smiles)
    return novelty

def calculate_uniqueness(smiles_list):
    """Calculate the uniqueness percentage of a list of SMILES strings."""
    unique_smiles = set(smiles_list)
    uniqueness = len(unique_smiles) / len(smiles_list)
    return uniqueness

def calculate_drug_likeness(smiles_list):
    """Calculate the average drug-likeness (QED) of a list of SMILES strings."""
    qed_scores = [calculate_properties(s)['qed'] for s in smiles_list if calculate_properties(s) is not None]
    average_qed = sum(qed_scores) / len(qed_scores) if qed_scores else 0
    return average_qed

def compare_properties(dataset_properties, generated_properties):
    """Compare the properties of dataset and generated molecules."""
    comparison_results = {}
    for property in dataset_properties.columns:
        t_stat, p_value = ttest_ind(dataset_properties[property], generated_properties[property], nan_policy='omit')
        comparison_results[property] = {'t_stat': t_stat, 'p_value': p_value}
    return comparison_results

def evaluate(generated_smiles_list, smiles_list = None):
    """Calculate metrics."""
    
    # Validate molecules
    validity_metrics = calculate_validity(generated_smiles_list)
    valid_generated_smiles = validity_metrics["valid_smiles"]

    uniqueness = None
    drug_likeness = None
    novelty= None

    # Calculate metrics
    if len(valid_generated_smiles) > 0:
        uniqueness = calculate_uniqueness(valid_generated_smiles)
        drug_likeness = calculate_drug_likeness(valid_generated_smiles)
        #check if baseline has been given
        if smiles_list != None:
            novelty = calculate_novelty(valid_generated_smiles, smiles_list)
    #else: print("Not a single valid molecule generated!!")

    return validity_metrics, novelty, uniqueness, drug_likeness

def main():
    # Example data
    dataset_smiles = ['CCO', 'CCN', 'CCC']  # Replace with actual dataset SMILES
    generated_smiles = ['CCO', 'CCS', 'CCCl']  # Replace with actual generated SMILES

    # Calculate validity
    validity, valid_generated_smiles = calculate_validity(generated_smiles)
    print(f"Validity: {validity:.2f}")

    # Calculate novelty
    novelty = calculate_novelty(valid_generated_smiles, dataset_smiles)
    print(f"Novelty: {novelty:.2f}")

    # Calculate uniqueness
    uniqueness = calculate_uniqueness(valid_generated_smiles)
    print(f"Uniqueness: {uniqueness:.2f}")

    # Calculate drug-likeness
    drug_likeness = calculate_drug_likeness(valid_generated_smiles)
    print(f"Average Drug-likeness (QED): {drug_likeness:.2f}")

    # Calculate properties for both sets
    dataset_properties = pd.DataFrame([calculate_properties(s) for s in dataset_smiles if calculate_properties(s) is not None])
    generated_properties = pd.DataFrame([calculate_properties(s) for s in valid_generated_smiles if calculate_properties(s) is not None])

    # Compare properties
    comparison_results = compare_properties(dataset_properties, generated_properties)
    for property, results in comparison_results.items():
        print(f"Property: {property}, t_stat: {results['t_stat']:.2f}, p_value: {results['p_value']:.2e}")

if __name__ == "__main__":
    main()
