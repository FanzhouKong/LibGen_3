from rdkit import Chem
def sanitize_smiles(smiles):
    """
    Sanitize a SMILES string to ensure it is in a valid format.
    This function takes a SMILES string and attempts to sanitize it using RDKit's
    Chem.SanitizeMol function. If the sanitization fails, it returns None.

    Args:
        smiles (str): The SMILES string to be sanitized.
    Returns:
        str or None: The sanitized SMILES string if successful, otherwise None.
    """
    if isinstance(smiles, str):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            return None
    elif isinstance(smiles, Chem.Mol):
        mol = smiles

    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return (mol)
    except:
        return None