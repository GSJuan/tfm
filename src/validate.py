from rdkit.Chem import MolFromSmiles, SanitizeMol

def validate_single_smiles(smiles):
    m = MolFromSmiles(smiles,sanitize=False)
    if m is None:
      print('invalid SMILES')
    else:
      try:
        SanitizeMol(m)
        print('Valid SMILES, both sintactically and chemically')
      except:
        print('invalid chemistry')