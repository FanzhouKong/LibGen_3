from .Tylos import *
from .search_utils import *
from tqdm import tqdm
from .spectral_denoising import spectral_denoising_batch
def libgen_pipeline(std_list, mzml_dir, adducts, mass_error = 0.005):
    std_list = complete_inchikey(std_list)
    print('making all matches...')
    all_matches = make_all_matches(std_list, mzml_dir, adducts, mass_error=mass_error)
    all_matches = complete_inchikey(all_matches, smiles_col='reference_smiles')
    if len(all_matches)==0:
        print('no matches found')
        return()
    print(f'there are {len(all_matches)} features found to matches with precursor mz at {mass_error} error')
    # print('returning only all matches')
    # return all_matches
    print('denoising...')
    # return all_matches
    msms_denoised = spectral_denoising_batch(all_matches['peaks'], 
                                                all_matches['reference_smiles'],
                                                all_matches['reference_adduct']
                                                )
    
    all_matches['peaks_denoised'] = msms_denoised
    for index, row in all_matches.iterrows():
        all_matches.at[index, 'ei'] = get_ei(row['peaks'], row['peaks_denoised'], row['reference_mz'])
    lower_quantile = all_matches['ei'].mean()-2*all_matches['ei'].std()
    return all_matches[all_matches['ei']>lower_quantile]
def complete_inchikey(std_list, smiles_col='smiles'):
    inchikey_all = []
    if 'inchikey' not in std_list.columns.str.lower():
        for index, row in std_list.iterrows():
        
            inchikey_all.append(smiles_to_inchikey(row[smiles_col], full = True))
        std_list['inchikey'] = inchikey_all
    else:
        for index, row in std_list.iterrows():
            if row['inchikey'] != row['inchikey']:
                inchikey_all.append(smiles_to_inchikey(row[smiles_col], full = True))
            else:
                inchikey_all.append(row['inchikey']) 
        # if row['inchikey'] != row['inchikey']:
        #     std_list.at[index, 'inchikey'] = smiles_to_inchikey(row[smiles_col], full = True)
    
    return std_list
from .identifier_utils import smiles_to_inchikey
def mzrt_pipeline(std_list,mzml_dir,adduct):

    std_list = complete_inchikey(std_list)
    rt_all = pd.DataFrame()
    for mix in tqdm(std_list['mix'].unique()):
        std_list_mix = string_search(std_list, 'mix', mix)
        ms1, ms2 = read_mzml(os.path.join(mzml_dir, mix))

        rt_mix = get_rt_mix(std_list_mix, adduct, ms1, ms2, 0.005, unique_identifier='inchikey')
        rt_all = pd.concat([rt_all, rt_mix], ignore_index=True)
    rt_all = standardize_col(rt_all)
    rt_all = rt_all[['name', 'inchikey', 'adduct','smiles','mz', 'rt','mix','ms1_intensity']]
    rt_all.rename(columns = {'mz':'precursor_mz'}, inplace = True)
    return rt_all
def get_rt_mix(std_list_mix, adduct, ms1, ms2, mass_error = 0.005, unique_identifier = 'inchikey'):
    if unique_identifier not in std_list_mix.columns:
        unique_identifier = 'inchikey'
        std_list_mix.insert(0, unique_identifier, [smiles_to_inchikey(x) for x in std_list_mix['smiles']])
    elif std_list_mix[unique_identifier].isna().any():
        std_list_mix[unique_identifier] = [smiles_to_inchikey(x) for x in std_list_mix['smiles']]
    
    matches = make_all_matches_mix(std_list_mix, adduct, ms1, ms2, mass_error, unique_identifier)
    matches_unique = pd.DataFrame()
    for i in std_list_mix[unique_identifier].unique():
        matches_i = string_search(matches, unique_identifier, i)
        matches_i.sort_values(by = 'ms1_intensity', ascending = False, inplace = True)
        matches_unique = pd.concat([matches_unique, matches_i.iloc[0:1]], ignore_index=True)
    return matches_unique
def get_ei(peak, peak_denoised, pmz):
    
    peak = so.remove_precursor(peak, pmz)
    peak_denoised = so.remove_precursor(peak_denoised, pmz)
    if isinstance(peak, float) or isinstance(peak_denoised, float):
        return np.nan
    return np.sum(peak_denoised.T[1])/np.sum(peak.T[1])

def make_all_matches(std_list, mzml_dir, adducts, unique_identifier = None,mass_error = 0.005):
    all_matches = pd.DataFrame()
    for mix in tqdm(std_list['mix'].unique()):
        std_list_mix = string_search(std_list, 'mix', mix)
        try:
            ms1, ms2 = read_mzml(os.path.join(mzml_dir, mix))
        except:
            continue
        all_matches_mix = make_all_matches_mix(std_list_mix, adducts,ms1, ms2,mass_error,unique_identifier)
        all_matches = pd.concat([all_matches, all_matches_mix], ignore_index=True)
    return all_matches
def make_all_matches_mix(std_list_mix,adducts, ms1, ms2,mass_error,unique_identifier=None):
    all_masses = make_all_flat_masses(std_list_mix, adducts)
    mass_sorted, intensity_sorted, index_sorted, rt_list = build_index(ms1)
    all_features = find_feature_targeted(all_masses, mass_sorted, intensity_sorted, index_sorted, rt_list, ms1, ms2 )
    all_matches = feature_matching(all_features, std_list_mix, adducts, mass_error,unique_identifier=unique_identifier, return_raw=False)
    return(all_matches)
def make_all_flat_masses(std_list_mix, adducts):
    flattened_array = list(itertools.chain(*std_list_mix[adducts].values))
    all_masses = [x for x in flattened_array if x == x]
    return(all_masses)
def feature_matching(feature_targeted, std_list_mix, adducts, mass_error = 0.005,unique_identifier = None, return_raw = False):
    feature_targeted.sort_values(by = 'precursor_mz', inplace=True, ascending=True)
    mix_matched = pd.DataFrame()
    for index, row in std_list_mix.iterrows():
        compound_matched = pd.DataFrame()

        for a in adducts:
            adduct_matched = quick_search_sorted(feature_targeted, 'precursor_mz', row[a]-mass_error, row[a]+mass_error)
            if len(adduct_matched)>0:
                # adduct_matched.insert(0, 'reference_mz', row[a])
                adduct_matched.insert(1, 'reference_name', row['name'])
                adduct_matched.insert(1, 'reference_mz', row[a])
                if unique_identifier is not None:
                    adduct_matched.insert(1,unique_identifier, row[unique_identifier])
                adduct_matched.insert(2, 'reference_adduct', a)

                # adduct_matched.insert(3, 'reference_rt', row['reference_rt'])
                adduct_matched.insert(4, 'reference_smiles', row['smiles'])
                adduct_matched.insert(6, 'reference_mix', row['mix'])
                # adduct_matched.insert(7, 'reference_rt', row['rt_a'])
                compound_matched  = pd.concat([compound_matched, adduct_matched], ignore_index=True)
    
        if len(compound_matched)>0:
            if return_raw == False:
                compound_matched = dereplicate(compound_matched)
            mix_matched = pd.concat([mix_matched, compound_matched],ignore_index=True)
            # compound_matched = pd.concat([compound_matched, adduct_matched], ignore_index=True)
    return(mix_matched)
def dereplicate(compound_matched):
    if len(compound_matched)==0:
        return(pd.DataFrame())
    df_return = pd.DataFrame()
    guessed_rt = compound_matched.iloc[np.argmax(compound_matched['ms1_intensity'])]['rt_apex']
    if quick_search_values(compound_matched, 'rt_apex', guessed_rt-10/60, guessed_rt+10/60)['peaks'].isna().all():
        return(pd.DataFrame())# must be at least 1 msms around targeted rt
    compound_matched.dropna(subset = 'peaks', inplace = True)
    if len(compound_matched)==0:
        return(compound_matched)
    comment = []
    for ma in compound_matched['reference_adduct'].unique():
        # comment = []
        current_adduct = string_search(compound_matched, 'reference_adduct', ma)
        rt_matched = quick_search_values(current_adduct, 'rt_apex', guessed_rt-10/60, guessed_rt+10/60)
        if len(rt_matched)>0:
            current_adduct_left_over = current_adduct.drop(rt_matched.index.tolist())
            major = rt_matched[rt_matched['ms1_intensity']==rt_matched['ms1_intensity'].max()]
            minor = rt_matched[rt_matched['ms1_intensity']<rt_matched['ms1_intensity'].max()]
            if len(major)>1:
                major.sort_values(by = 'rt_offset', ascending=True, inplace = True)
                major = major[0:1]
            df_return = pd.concat([df_return, major], ignore_index=True)
            comment.append('Major')
            df_return = pd.concat([df_return, minor], ignore_index=True)
            comment.extend(['Minor']*len(minor))
            seed_msms = major.iloc[0]['peaks']
            for i, j in current_adduct_left_over.iterrows():
                entropy_temp =so.entropy_similairty(seed_msms
                                                    , j['peaks'], pmz = major.iloc[0]['precursor_mz'])
                if entropy_temp>0.75 and j['ms1_intensity']>0.1*major.iloc[0]['ms1_intensity']:
                    df_return= pd.concat([df_return, pd.DataFrame([j])], ignore_index=True)
                    comment.append('isomer')
    if len(df_return)==0:
        return (df_return)
    df_return.insert(4, 'comment', comment)
    return df_return
def find_feature_compound(compound_row, adducts,ms1, ms2):
    masses = compound_row[adducts]
    masses = [m for m in masses if m ==m]
    all_features = find_feature_targeted(masses, ms1, ms2)
    seed_rt = all_features.iloc[np.argmax(all_features['ms1_intensity'])]['rt_apex']
    # for index, row in all_features.iterrows():
    #     if row['rt']-seed_rt
    return masses

def check_missing(std_list, to_check, col ='inchikey'):
    missing_key = pd.DataFrame()
    for index, row in std_list.iterrows():
        if row[col] not in to_check[col].unique():
            missing_key= pd.concat([missing_key, pd.DataFrame([row])], axis =0)
    return missing_key