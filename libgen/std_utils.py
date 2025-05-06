import re
from tqdm import tqdm
from .identifier_utils import *
from .chem_utils import *
from rdkit import Chem
def standaridize_name(std_list_raw):
    for index, row in std_list_raw.iterrows():
        if row['inchikey'] != row['inchikey']:
            std_list_raw.at[index, 'name'] = remove_parentheses(row['name']).strip('-')
    return std_list_raw
            # continue
def remove_parentheses(text):
    # Use regex to remove text inside parentheses and any surrounding spaces
    result = re.sub(r"\s*\(.*?\)\s*", "", text)
    return result
def clean_std_list(std_list, adducts):
    smiles_cleaned = []
    if 'smiles' in std_list.columns.str.lower():
        # std_list.drop(columns = ['smiles'], inplace = True)
        for index, row in std_list.iterrows():
            if is_smiles(row['smiles']):
                smiles_cleaned.append(desalter(Chem.MolFromSmiles(row['smiles'])))
            else:
                smiles_cleaned.append(np.nan)
    for i in range(len(smiles_cleaned)):
        if smiles_cleaned[i] != smiles_cleaned[i]:
            smiles_from_name = name_to_smiles(std_list.iloc[i]['name'])
            if is_smiles(smiles_from_name):
                smiles_cleaned[i] = desalter(Chem.MolFromSmiles(smiles_from_name))
    std_list['smiles'] = smiles_cleaned
    for index, row in std_list.iterrows():
        std_list.at[index, 'mono_mass'] = get_mono_mass(row['smiles'])
        for a in adducts:
            if get_formal_charge(row['smiles']) == 0:
                if a not in ['[M]+', '[M]-']:
                    std_list.at[index, a] = calculate_precursormz(a, row['smiles'])
                else:
                    std_list.at[index, a] = 0.0
            elif get_formal_charge(row['smiles']) == 1:
                if a =='[M]+':
                    std_list.at[index, a] = calculate_precursormz(a, row['smiles'])
                else:
                    std_list.at[index, a] = 0.0
            elif get_formal_charge(row['smiles']) == -1:
                if a == '[M]-':
                    std_list.at[index, a] = calculate_precursormz(a, row['smiles'])
                else:
                    std_list.at[index, a] = 0.0
    return std_list
# def standardize_sheet(std_list, adducts = ['[M+H]+', '[M+Na]+', '[M+NH4]+', '[M]+']):
#     std_list = standaridize_name(std_list)
#     smiles = []

#     for index, row in tqdm(std_list.iterrows(), total = len(std_list)):
#         smiles_row= everything_to_smiles(row['inchikey'])
#         if smiles_row != smiles_row:
#             smiles_row = everything_to_smiles(row['name'])
#         smiles.append(smiles_row)
#     std_list['smiles_fetched'] = smiles
#     std_list.dropna(subset = ['smiles_fetched'], inplace = True)
#     desalted_smiles =[]
#     for index, row in tqdm(std_list.iterrows(), total = len(std_list)):
#         desalted_smiles.append(desalter(row['smiles_fetched']))
#     std_list['smiles']=desalted_smiles

#     for index, row in std_list.iterrows():
#         for a in adducts:
#             std_list.at[index, a] = calculate_precursormz(a, row['smiles'])
#     std_list.drop(columns = ['smiles_fetched'], inplace = True)
#     return std_list