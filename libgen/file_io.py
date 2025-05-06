import pandas as pd 
import re
import os
from . import spectral_operations as so
import re
import shutil
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz
from .search_utils import *
import pymzml as pymzml
import json


def read_mzml(mzml_path, parent_dir =  None, rt_max = None,if_mix = False):

    if if_mix == True and parent_dir != None:
        mix = mzml_path
        mzml_base_name = mix
        mzml_path = os.path.join(parent_dir, mzml_base_name)
    if if_mix == False and parent_dir is not None:
        mzml_path = os.path.join(parent_dir,mzml_path)
    if mzml_path[-5:]!='.mzML' and mzml_path[-5:]!='.mzml':
        mzml_path = mzml_path+'.mzML'
    ms1_2 = _load_mzml_data(mzml_path, rt_max = rt_max)
    
    ms2 = string_search(ms1_2, 'ms_level', 2)
    ms1 = string_search(ms1_2, 'ms_level', 1)
    ms2_modify = ms2.copy()
    for index, row in ms2.iterrows():
        ms1_row= string_search(ms1, 'cycle', row['cycle']) 
        actual_precursor_mz, actual_precursor_intensity, peak_purity = get_precursor_statistics(ms1_row.iloc[0]['peaks'], row['precursor_mz'], 
                                                                                                row['isolation_window'][0], row['isolation_window'][1])
        ms2_modify.loc[index, 'ms1_precursor_mz'] = actual_precursor_mz
        ms2_modify.loc[index, 'ms1_precursor_intensity'] = actual_precursor_intensity
        ms2_modify.loc[index, 'peak_purity'] = peak_purity
        if actual_precursor_mz != np.nan:
            ms2_modify.loc[index, 'mz_offset'] = abs(row['precursor_mz']-actual_precursor_mz)
    
    # ms2.sort_values(by = ['ms1_pmz', 'ms1_precursor_intensity'], inplace = True)
    # ms2 =ms2[ms2['peak_purity'] !=0]
    # ms2 = ms2[ms2['mz_offset']<0.5]
    ms2_modify.reset_index(inplace=True, drop=True)
    return(ms1, ms2_modify)
def _load_mzml_data(file: str, n_most_abundant=400, rt_max = None) -> tuple:
    """Load data from an mzml file as a dictionary.

    Args:
        filename (str): The name of a .mzml file.
        n_most_abundant (int): The maximum number of peaks to retain per MS2 spectrum.
        callback (callable): A function that accepts a float between 0 and 1 as progress. Defaults to None.

    Returns:
        a pandas dataframe with all the raw data

    """

    id = 1
    all_specs = pymzml.run.Reader(file, obo_version="4.1.33")
    total_spec_count = all_specs.info['spectrum_count']
    # ms_list = np.zeros(total_spec_count,dtype=int)
    ms_list = [None]*total_spec_count
    # mono_mzs_list = np.zeros(total_spec_count)
    mono_mzs_list = [None]*total_spec_count
    mono_mzs_intensity_list = np.zeros((total_spec_count))
    # pmz_intensity_list = np.zeros(total_spec_count)
    pmz_intensity_list = [None]*total_spec_count
    polarity_list = [None]*total_spec_count
    select_windows_list = [None]*total_spec_count
    # cycles = np.zeros(total_spec_count,dtype=int)
    cycles = [None]*total_spec_count
    cycle = -1
    # scan_idx = np.zeros(total_spec_count,dtype=int)
    scan_idx = [None]*total_spec_count
    cur_idx = 0
    peak_list = [None]*total_spec_count
    # rt_list = np.zeros(total_spec_count)
    rt_list = [None]*total_spec_count
    for spec in pymzml.run.Reader(file, obo_version="4.1.33", build_index_from_scratch=True):
        try:
            rt, masses, intensities, ms_order,mono_mz,mono_mz_intensity,polarity, (prec_windows_lower, prec_windows_upper)=_extract_mzml_info(spec)
            if rt_max!= None and rt>rt_max:
                break

            to_keep = intensities > 0
            # return_mass = masses
            try:
                masses = masses[to_keep]
                intensities = intensities[to_keep]
            except:
                continue

            if ms_order == 1:
                cycle = cycle+1
            cycles[cur_idx]=cycle
            if not np.all(masses[:-1] <= masses[1:]):
                order = np.argsort(masses)
                masses = masses[order]
                intensities = intensities[order]

            # Only keep top n_most_abundant peaks
            # if ms_order == 2 and len(masses) > n_most_abundant:
            #     sortindex = np.argsort(intensities)[::-1][:n_most_abundant]
            #     sortindex.sort()
            #     masses, intensities = masses[sortindex], intensities[sortindex]
            peak = np.array([masses, intensities], dtype=np.float64).T
            # peak = pack_spectra(masses, intensities)
            peak_list[cur_idx]=peak
            ms_list[cur_idx]=ms_order
            mono_mzs_list[cur_idx]=mono_mz
            mono_mzs_intensity_list[cur_idx]=mono_mz_intensity
            # pmz_intensity_list[cur_idx]=precursor_intensity
            polarity_list[cur_idx]=polarity
            select_windows_list[cur_idx]=(prec_windows_lower, prec_windows_upper)
            scan_idx[cur_idx]=cur_idx
            rt_list[cur_idx]=rt
            cur_idx = cur_idx+1
        except:
            pass
    return_df = pd.DataFrame({
            'scan_idx':scan_idx,
            'cycle':cycles,
            'ms_level':ms_list,
            'precursor_mz':mono_mzs_list,
            'precursor_intensity':mono_mzs_intensity_list,
            # 'precursor_intensity':pmz_intensity_list,
            'polarity':polarity_list,
            'rt':rt_list,
            'peaks':peak_list,
            'isolation_window':select_windows_list
        })
    return(return_df)






def _extract_mzml_info(input_dict: dict) -> tuple:
    """Extract basic MS coordinate arrays from a dictionary.

    Args:
        input_dict (dict): A dictionary obtained by iterating over a Pyteomics mzml.read function.

    Returns:
        tuple: The rt, masses, intensities, ms_order, prec_mass, mono_mz, charge arrays retrieved from the input_dict.
            If the `ms level` in the input dict does not equal 2, the charge, mono_mz and prec_mass will be equal to 0.

    """
    rt = input_dict.scan_time_in_minutes()
    masses = input_dict.mz
    intensities = input_dict.i
    ms_order = input_dict.ms_level
    mono_mz = 0
    mono_mz_intensity = 0
    polarity = np.nan
    # prec_mass = mono_mz = charge = 0
    # if ms_order == 2 and len(input_dict.selected_precursors) > 0:
    if ms_order == 1 and input_dict['isolation window target m/z'] is None:
        polarity = '+'
        if input_dict['negative scan'] is not None:
            polarity = '-'
        elif input_dict['positive scan'] is None:
            raise Exception("Can't determine polarity")
    if ms_order == 2 and input_dict['isolation window target m/z'] is not None:
        polarity = '+'
        if input_dict['negative scan'] is not None:
            polarity = '-'
        elif input_dict['positive scan'] is None:
            raise Exception("Can't determine polarity")
        mono_mz = input_dict['isolation window target m/z']
        if 'i' in input_dict.selected_precursors[0].keys():
            mono_mz_intensity = input_dict.selected_precursors[0]['i']
        else:
            mono_mz_intensity = np.nan
        # precursor_intensity =input_dict.selected_precursors[0]["i"]
        # prec_mass = _calculate_mass(mono_mz, charge)

        prec_windows_center = mono_mz
        try:
            prec_windows_lower = prec_windows_center-(input_dict["isolation window lower offset"])
            prec_windows_upper = prec_windows_center+(input_dict["isolation window upper offset"])
        except:
            prec_windows_lower = prec_windows_center-0.5
            prec_windows_upper = prec_windows_center+0.5
    else:
        prec_windows_lower, prec_windows_upper = 0., 0.

    return (rt, masses, intensities, ms_order,
            # prec_mass,
            mono_mz,mono_mz_intensity, polarity, (prec_windows_lower, prec_windows_upper))
def get_precursor_statistics(ms1, pmz, isolation_left, isolation_right):
    isolation_winow = so.search_spectrum(ms1, isolation_left, isolation_right)
    if isinstance(isolation_winow, float):
        return np.nan, 0, 0
    isolation_window_total_intensity = so.search_spectrum(ms1, isolation_left, isolation_right).T[1].sum()
    actual_precursor_mz, actual_precursor_intensity = get_exact_precursor(ms1, pmz)
    return actual_precursor_mz, actual_precursor_intensity, actual_precursor_intensity/isolation_window_total_intensity*100

    pass
def get_exact_precursor(ms1, pmz, search_window = 0.01):
    rough_region = so.search_spectrum(ms1, pmz-search_window, pmz+search_window)
    if isinstance(rough_region, float):
        return np.nan, 0
    intensity_idx = np.argmax(rough_region.T[1])
    return rough_region.T[0][intensity_idx], rough_region.T[1][intensity_idx]



def read_msp(file_path):
    
    """
    Reads the MSP files into the pandas dataframe, and sort/remove zero intensity ions in MS/MS spectra.

    Args:
        file_path (str): target path path for the MSP file.
    Returns:
        pd.DataFrame: DataFrame containing the MS/MS spectra information
    """
    
    spectra = []
    spectrum = {}
    if os.path.exists(file_path)== False:
        raise FileNotFoundError(f"File not found: {file_path}")
        return ()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            # Handle metadata
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'name':
                    # Save current spectrum and start a new one
                    if spectrum:
                        spectra.append(spectrum)
                    spectrum = {'name': value, 'peaks': []}
                else:
                    spectrum[key] = value
            
            # Handle peak data (assumed to start with a number)
            elif line[0].isdigit():
                
                peaks = line.split()
                m_z = float(peaks[0])
                intensity = float(peaks[1])
                spectrum['peaks'].append((([m_z, intensity])))
        # Save the last spectrum
        if spectrum:
            spectra.append(spectrum)
    df = pd.DataFrame(spectra)
    df['peaks'] = [so.sort_spectrum(so.remove_zero_ions(np.array(peak))) for peak in df['peaks']]
    for column in df.columns:
        if column != 'peaks':  # Skip 'peaks' column
            try:
                df[column] = pd.to_numeric(df[column], errors='raise')
            except:
                pass
    df = standardize_col(df) 
    return df
def write_to_msp_libgen(df, file_path, msms_col = 'peaks_denoised', normalize = False):
    
    """
    Pair function of read_msp.
    Exports a pandas DataFrame to an MSP file.

    Args:
        df (pd.DataFrame): DataFrame containing spectrum information. Should have columns for 'name', 'peaks', and other metadata.
        file_path (str): Destination path for the MSP file.
    Returns:
        None
    """
    from .identifier_utils import everything_to_formula
    if normalize == True:
        df[msms_col] = [so.normalize_spectrum(peak) for peak in df[msms_col]]
    df = standardize_col(df)
    df['formula']=[everything_to_formula(x) for x in df['smiles']]
    
    using_columns = ['name', 'precursor_mz', 'inchikey','adduct', 'comment', 'smiles', 'formula','rt',msms_col]
    with open(file_path, 'w') as f:
        for _, row in df.iterrows():
            # Write the name of the spectrum
            if isinstance(row[msms_col], float):
                continue
            if 'name' in df.columns:
                f.write(f"Name: {row['name']}\n")
            
            # Write other metadata if available
            for col in using_columns:
                if col not in ['name', msms_col] and 'peak' not in col:
                    f.write(f"{col.capitalize()}: {row[col]}\n")
            
            # Write the peaks (assuming each peak is a tuple of (m/z, intensity))
            f.write(f"Num Peaks: {len(row[msms_col])}\n")
            for mz, intensity in row[msms_col]:
                f.write(f"{mz} {intensity}\n")
            
            # Separate spectra by an empty line
            f.write("\n")
def write_to_msp(df, file_path, msms_col = 'peaks', normalize = False):
    
    """
    Pair function of read_msp.
    Exports a pandas DataFrame to an MSP file.

    Args:
        df (pd.DataFrame): DataFrame containing spectrum information. Should have columns for 'name', 'peaks', and other metadata.
        file_path (str): Destination path for the MSP file.
    Returns:
        None
    """
    if normalize == True:
        df[msms_col] = [so.normalize_spectrum(peak) for peak in df[msms_col]]
    with open(file_path, 'w') as f:
        for _, row in df.iterrows():
            # Write the name of the spectrum
            if isinstance(row[msms_col], float):
                continue
            if 'name' in df.columns:
                f.write(f"Name: {row['name']}\n")
            
            # Write other metadata if available
            for col in df.columns:
                if col not in ['name', msms_col] and 'peak' not in col:
                    f.write(f"{col.capitalize()}: {row[col]}\n")
            
            # Write the peaks (assuming each peak is a tuple of (m/z, intensity))
            f.write(f"Num Peaks: {len(row[msms_col])}\n")
            for mz, intensity in row[msms_col]:
                f.write(f"{mz} {intensity}\n")
            
            # Separate spectra by an empty line
            f.write("\n")
def save_df(df, save_path):
    """
    Pair function of save_df.

    Save a DataFrame contaning MS/MS spectra to a CSV file, converting any columns containing 2D numpy arrays to string format.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        save_path (str): The file path where the DataFrame should be saved. If the path does not end with '.csv', it will be appended automatically.
    Returns:
        None
    Notes:
        - This function identifies columns in the DataFrame that contain 2D numpy arrays with a second dimension of size 2.
        - These identified columns are converted to string format before saving to the CSV file.
        - The function uses tqdm to display a progress bar while processing the rows of the DataFrame.
    """

    data = df.copy()
    cols = []
    for c in df.columns:
        if isinstance(df.iloc[0][c], np.ndarray):
            if np.shape(df.iloc[0][c])[1]==2:
                cols.append(c)
    print(cols)
    if save_path.endswith('.csv') == False:
        save_path = save_path+'.csv'
    for col in cols:
        specs = []
        for index, row in tqdm(data.iterrows(), total = len(data)):
            specs.append(so.arr_to_str(row[col]))
        data[col]=specs
    data.to_csv(save_path, index = False)
def read_df(path, keep_ms1_only = False):
    """
    Pair function of write_df.
    Reads a CSV file into a DataFrame, processes specific columns based on a pattern check, 
    and MS/MS in string format to 2-D numpy array (string is used to avoid storage issue in csv files).
    
    Args:
        path (str): The file path to the CSV file.
    Returns:
        pandas.DataFrame: The processed DataFrame with specific columns converted.
    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file contains parsing errors.
    Notes:
        - The function assumes that the first row of the CSV file contains the column headers.
        - The `check_pattern` function is used to determine which columns to process.
        - The `so.str_to_arr` function is used to convert the values in the selected columns.
    """
    df = pd.read_csv(path)
    print('done read in df...')
    for col in df.columns:
        if check_pattern(df[col].iloc[0]):
            df[col] = [so.str_to_arr(y[col]) for x,y in df.iterrows()]
    df =  standardize_col(df)

    if ':' in df.iloc[0]['peaks']:
        df['peaks']=[so.msdial_to_array(row['peaks']) for index, row in df.iterrows()]

    if keep_ms1_only == False:
        df.dropna(subset=['peaks'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return(df)

from .constant import standard_mapping
def standardize_col(df):
    """
    Standardizes column names in the given DataFrame based on a provided mapping. Help to read in and processing files with MS Dial generated msp files.
    
    Args:
    df (pd.DataFrame): The DataFrame whose column names need to be standardized.
    
    standard_mapping (dict): A dictionary where keys are common variations of the name, 
                              and values are the standard name.
    
    Returns:
    pd.DataFrame: DataFrame with standardized column names.
    """
    
    # Create a mapping for case-insensitive column names
    new_columns = []
    for col in df.columns:
        # Convert the column name to lowercase
        col_lower = col.lower()
        col_lower = col_lower.replace('reference_', '')
        # Map the column name to the standard one if found in the standard mapping
        standardized_col = standard_mapping.get(col_lower)
        if standardized_col is not None:
            new_columns.append(standardized_col)
        else:
            new_columns.append(col_lower)
    
    # Assign the new standardized columns back to the DataFrame
    df.columns = new_columns
    return df

def check_pattern(input_string):
    """
    Helper function for read_df.
    Regular expression to match pairs of floats in standard or scientific notation separated by a tab, with each pair on a new line

    Args:
        input_string (str): input string to check for the pattern
    Returns:
        bool: True if the pattern is found, False otherwise
    """
    
    if isinstance(input_string, str):
        if '\t' in input_string:
            return True
    return False
def export_denoising_searches(results, save_dir, top_n = 10):
    """
    Pair function of import_denoising_searches.
    Exports the results of a denoising search to a JSON file.

    Args:
        results (list): The list of results from a denoising search.
        save_path (str): The file path where the results should be saved. If the path does not end with '.json', it will be appended automatically.
    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(results)):
        if results[i].empty or len(results[i])==0:
            continue
        else:
            temp = results[i].head(top_n)
            pmz_temp = temp.iloc[0]['precursor_mz']
            write_to_msp(temp, os.path.join(save_dir, f"denoising_search_{i}_{pmz_temp:0.4f}.msp"), msms_col='query_peaks_denoised')
