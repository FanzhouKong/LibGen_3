from itertools import chain

import pandas as pd
import numpy as np
import re
import itertools
from . import spectral_operations as so
from .search_utils import *
from .file_io import *
from tqdm import tqdm
from scipy.signal import find_peaks
import pybaselines
import seaborn as sns
def map_ms2(features_df, ms2, mass_tolerance = 0.01, use_msms = 'max_intensity'):

    ms2['adjusted_intensity']=[x*y/100 if x == x and y== y else 0 for x, y in zip(ms2['precursor_intensity'], ms2['peak_purity'])]
    ms2_working = ms2.copy()
    peaks = []
    features_df.sort_values(by = 'ms1_intensity', ascending=False, inplace=True)
    ms2_working.sort_values(by = 'precursor_mz', ascending=True, inplace=True)
    for index, row in features_df.iterrows():
        if len(ms2_working)==0:
            return features_df

        pmz_matched = quick_search_sorted(ms2_working, 'precursor_mz', row['precursor_mz']-mass_tolerance, row['precursor_mz']+mass_tolerance)
        ms2_under_peak = quick_search_values(pmz_matched, 'rt', row['rt_start'], row['rt_end'])

        if len(ms2_under_peak)>0:
            if use_msms == 'max_intensity':
                max_idx = np.argmax(ms2_under_peak['ms1_precursor_intensity'])
            elif use_msms == 'max_adjusted_intensity':
                max_idx = np.argmax(ms2_under_peak['adjusted_intensity'])

            # features_df.loc[index, 'peak']=ms2_under_peak.iloc[max_idx]['peaks']
            peaks.append(ms2_under_peak.iloc[max_idx]['peaks'])
            features_df.loc[index, 'ms2_scan_idx']=int(ms2_under_peak.iloc[max_idx]['scan_idx'])
            features_df.loc[index, 'ms2_precursor_mz']=ms2_under_peak.iloc[max_idx]['precursor_mz']
            features_df.loc[index, 'peak_purity']=ms2_under_peak.iloc[max_idx]['peak_purity']
            features_df.loc[index, 'rt_offset']=abs(ms2_under_peak.iloc[max_idx]['rt']-row['rt_apex'])  
            ms2_working.drop(ms2_under_peak.index.tolist(), inplace=True)
        else:
            peaks.append(np.nan)
    features_df['peaks']=peaks
    return(features_df)



def just_eic(ms1, mass_center, baseline = None):
    mass_sorted, intensity_sorted, index_sorted, rt_list = build_index(ms1)
    ion_trace = flash_eic(mass_center, mass_sorted, intensity_sorted, index_sorted)
    ax= sns.lineplot(x = rt_list, y = ion_trace, label = 'EIC', color = 'blue')
    if baseline is not None:
        ax = sns.lineplot(x = rt_list, y = baseline, color = 'orange', label = 'baseline', linestyle='--')
    return rt_list, ion_trace

def get_current_peak(peaks_all, all_apex_intensity,intensity_list_smoothed):
    current_peak_idx = np.argmax(all_apex_intensity)
    apex_peak = np.array(peaks_all[current_peak_idx])
    apex_peak_range = [current_peak_idx,current_peak_idx]
    l = 1
    r = 1
    while current_peak_idx-l>0:
        left_peak = peaks_all[current_peak_idx-l]
        if apex_peak[0]-left_peak[2]<=2 and intensity_list_smoothed[apex_peak[0]]>min(intensity_list_smoothed[apex_peak[1]], intensity_list_smoothed[left_peak[1]])/1.3:
            apex_peak[0]=left_peak[0]
            apex_peak_range[0]=current_peak_idx-l
            l = l+1
        else:
            break
    while current_peak_idx+r<len(peaks_all):
        right_peak = peaks_all[current_peak_idx+r]
        if right_peak[0]-apex_peak[2]<=2 and intensity_list_smoothed[apex_peak[2]]>min(intensity_list_smoothed[apex_peak[1]], intensity_list_smoothed[right_peak[1]])/1.3:
            apex_peak[2]=right_peak[2]
            apex_peak_range[1]=current_peak_idx+r
            r = r+1
        else:
            break
    peaks_all = peaks_all[0:apex_peak_range[0]]+peaks_all[apex_peak_range[1]+1:]
    all_apex_intensity = np.array([intensity_list_smoothed[x[1]]for x in peaks_all])
    return apex_peak, peaks_all, all_apex_intensity
def find_feature_targeted( masses, mass_sorted, intensity_sorted, index_sorted, rt_list, ms1, ms2, mass_error = 0.005, if_QE = True):

    mass_tolerance = mass_error
    if if_QE ==True:
        n = 2
        intensity_threshold = 30000
    else:
        n = 1
        intensity_threshold = 1000
    # ms1, ms2 = read_mzml(mix, mzml_dir)
    pmz = []
    apex_intensity = []
    rt_apex = []
    rt_start = []
    rt_end = []
    peak_range = []
    n_scans = []
    reci_snrs = []

    for mass in (masses):
        ion_trace = flash_eic(mass, mass_sorted, intensity_sorted, index_sorted)

        idx_left, idx_right = mass_sorted.searchsorted([mass-mass_tolerance, mass+mass_tolerance])
        peaks_all, raw_apex_idx_all,reci_snrs_all = detect_all_peaks(ion_trace, n_neighbor=n, intensity_threshold=intensity_threshold)

        if len(peaks_all)>0:
            for p, r, a in zip(peaks_all, reci_snrs_all, raw_apex_idx_all):
                pmz_statistics = guess_pmz(mass, mass_sorted,
                                               intensity_sorted, index_sorted, idx_left, idx_right, int(a), mass_error= mass_error)
                if pmz_statistics[0]==pmz_statistics[0]:
                    pmz.append(pmz_statistics[0])
                    apex_intensity.append(pmz_statistics[1])
                    rt_apex.append(gaussian_estimator(tuple([int(a)-1,int(a), int(a)+1]),rt_list, ion_trace))
                    rt_start.append(rt_list[p[0]])
                    rt_end.append(rt_list[p[2]])
                    peak_range.append(p)
                    n_scans.append(p[2]-p[0])
                    reci_snrs.append(r)
    feature_targeted = pd.DataFrame(zip(pmz,
                                        rt_apex, rt_start, rt_end,
                                        apex_intensity,
                                        n_scans,peak_range,reci_snrs),
                                    columns=['precursor_mz',
                                             'rt_apex', 'rt_start', 'rt_end',
                                             'ms1_intensity',
                                             'n_scnas', 'ms1_scan_range', 'reci_snr'])
    feature_targeted_ms1 = find_isotope_snapshot(feature_targeted,ms1)
    feature_targeted_ms1_ms2 = map_ms2(feature_targeted_ms1, ms2)
    
    return(feature_targeted_ms1_ms2)
def find_isotope_snapshot(feature_targeted, ms1):
    ms1_snaps = []
    for index, row in feature_targeted.iterrows():
        ms1_scan = ms1.iloc[row['ms1_scan_range'][1]]
        ms1_snaps.append(so.search_spectrum(ms1_scan['peaks'], row['precursor_mz']-0.01,row['precursor_mz']+3 ))
    feature_targeted['ms1_snapshot']=ms1_snaps
    return feature_targeted
def guess_pmz(target_mass,mass_sorted, intensity_sorted,index_sorted,idx_start, idx_end, peak_apex, mass_error,guess_step = 1):
    # idx_start, idx_end = mass_sorted.searchsorted([seed_mass-mass_tolerance,seed_mass+mass_tolerance ])
    mass_range = mass_sorted[idx_start:idx_end]
    index_range = index_sorted[idx_start:idx_end]
    intensity_range = intensity_sorted[idx_start:idx_end]
    if np.max(intensity_range)==0:
        return(np.nan, np.nan)
    pmz_candidates = np.zeros(2*guess_step+1)
    intensity_candidates = np.zeros(2*guess_step+1)
    pmz_idx = 0
    anchor_pmz = np.nan
    for i in range(peak_apex-guess_step, peak_apex+guess_step+1):
        if len(intensity_range[index_range==i])>0:
            difference_array = np.absolute(mass_range[index_range==i]-target_mass)

            mass_anchor = difference_array.argmin()
            pmz_candidates[pmz_idx]=mass_range[index_range==i][mass_anchor]
            if i == peak_apex:
                anchor_pmz = mass_range[index_range==i][mass_anchor]
            intensity_candidates[pmz_idx]=intensity_range[index_range==i][mass_anchor]
            pmz_idx = pmz_idx+1
        # if i == peak_apex:
        #     apex_intensity = intensity_candidates[pmz_idx]
    if anchor_pmz==np.nan:
        return(np.nan, np.nan,np.nan)
    # if pmz_idx<(2*guess_step+1):
    #     return(np.nan, np.nan,np.nan)
    pmz_candidates=pmz_candidates[0:pmz_idx]
    intensity_candidates=intensity_candidates[0:pmz_idx]
    weighted_pmz = np.sum([x*y/np.sum(intensity_candidates) for x, y in zip(pmz_candidates, intensity_candidates)])
    apex_intensity = flash_eic(weighted_pmz, mass_sorted, intensity_sorted, index_sorted, mass_error = mass_error)[peak_apex]
    return (weighted_pmz, apex_intensity, anchor_pmz)
def build_index(ms1, use_binned = False):
    # ms1.reset_index(inplace = True, drop = True)
    if use_binned == False:
        col = 'peaks'
    else:
        col = 'peaks_binned'
    mass_nested = [None]*len(ms1)
    intensity_nested = [None]*len(ms1)
    index_nested = [None]*len(ms1)
    rt_list = np.zeros(len(ms1))
    for index, row in (ms1.iterrows()):
        mass_temp, intensity_temp = row[col].T
        mass_nested[index]=(mass_temp)
        intensity_nested[index]=(intensity_temp)
        index_nested[index]=([index]*len(mass_temp))
        rt_list[index]=(row['rt'])
    mass_flatten = np.array(list(itertools.chain.from_iterable(mass_nested)))
    intensity_flatten = np.array(list(itertools.chain.from_iterable(intensity_nested)))
    index_flatten = np.array(list(itertools.chain.from_iterable(index_nested)))
    order = np.argsort(mass_flatten)
    mass_sorted = mass_flatten[order]
    intensity_sorted = intensity_flatten[order]
    index_sorted = index_flatten[order]
    # loc_sorted = np.arange(len(mass_sorted))
    return(mass_sorted, intensity_sorted, index_sorted, rt_list)
def flash_eic(pmz, mass_sorted, intensity_sorted, index_sorted, mass_error=0.005, gap_fill = True):
    index_start, index_end = mass_sorted.searchsorted([pmz-mass_error, pmz+mass_error+1E-9])
    index_range = index_sorted[index_start:index_end]

    intensity_range = intensity_sorted[index_start:index_end]

    intensity_list = np.zeros(np.max(index_sorted)+1)
    for idx in range(0,len(index_range)):
        intensity_list[index_range[idx]]= intensity_list[index_range[idx]]+intensity_range[idx]
    if gap_fill == True:
        intensity_list = gap_filling(intensity_list, max_gap=2)
    return(intensity_list)
def gap_filling(intensity_list, max_gap = 2):
    zero_ranges = zero_runs(intensity_list)
    intensity_list_twisted = intensity_list.copy()
    for zr in zero_ranges:
        if zr[1]-zr[0]<=max_gap and zr[0]!=0 and zr[1]!=len(intensity_list):
            gradient = (intensity_list[zr[1]]-intensity_list[zr[0]-1])/(zr[1]-zr[0]+1)
            for j in range(0, zr[1]-zr[0]):

                intensity_list_twisted[j+zr[0]]=intensity_list_twisted[j+zr[0]]+ intensity_list[zr[0]-1]+gradient*(j+1)
    return(intensity_list_twisted)
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges
def moving_average( x, n=2):
    window = np.ones(2*n + 1) / (2*n + 1)

    # Use 'same' mode in convolve to ensure the output array has the same size as the input array
    # 'valid' mode would reduce the size of the array to avoid edge effects
    # 'full' mode would increase the size to include all possible overlaps
    smoothed_arr = np.convolve(x, window, mode='same')

    return smoothed_arr
def find_peak_width(peak, intensity_list_smoothed, ratio = 0.5):
    peak = peak[1]
    peak_height = intensity_list_smoothed[peak]
    half_max = peak_height *ratio
    left_idx = np.where(intensity_list_smoothed[:peak] <= half_max)[0]
    if len(left_idx) > 0:
        left_idx = left_idx[-1]
    else:
        left_idx = peak
    right_idx = np.where(intensity_list_smoothed[peak:] <= half_max)[0]
    if len(right_idx) > 0:
        right_idx = peak + right_idx[0]
    else:
        right_idx = peak
    fwhm_width = right_idx - left_idx
    return(fwhm_width)
def get_peaks(intensity_list: np.ndarray) -> list:
    """Detects peaks in an array.

    Args:
        int_array (np.ndarray): An array with intensity values.

    Returns:
        list: A regular Python list with all peaks.
            A peak is a triplet of the form (start, center, end)

    """
    apex, _ = find_peaks(intensity_list)
    peak_list = []
    for cur_apex_idx in apex:
        peak_list.append(get_edges(intensity_list, cur_apex_idx))
    return(peak_list)
def get_edges(intensity_list, cur_apex_idx):
    # gradient = np.diff(intensity_list)
    left_edge_idx = cur_apex_idx-1
    right_edge_idx = cur_apex_idx+1
    while left_edge_idx>0:
        if intensity_list[left_edge_idx-1]<=intensity_list[left_edge_idx] and intensity_list[left_edge_idx]>0:
            left_edge_idx = left_edge_idx-1
        else:
            break
    while right_edge_idx <len(intensity_list)-1:
        if intensity_list[right_edge_idx+1]<=intensity_list[right_edge_idx] and intensity_list[right_edge_idx]>0:
            right_edge_idx = right_edge_idx+1
        else:
            break

    return([left_edge_idx, cur_apex_idx, right_edge_idx])


#%%
def gaussian_estimator(
        peak: tuple,
        mz_array: np.ndarray,
        int_array: np.ndarray
) -> float:
    """Three-point gaussian estimator.

    Args:
        peak (tuple): A triplet of the form (start, center, end)
        mz_array (np.ndarray): An array with mz values.
        int_array (np.ndarray): An array with intensity values.

    Returns:
        float: The gaussian estimate of the center.
    """
    start, center, end = peak

    m1, m2, m3 = mz_array[center - 1], mz_array[center], mz_array[center + 1]
    i1, i2, i3 = int_array[center - 1], int_array[center], int_array[center + 1]
    # print(m1,m2,m3)
    if i1 == 0:  # Case of sharp flanks
        m = (m2 * i2 + m3 * i3) / (i2 + i3)
    elif i3 == 0:
        m = (m1 * i1 + m2 * i2) / (i1 + i2)
    else:
        l1, l2, l3 = np.log(i1), np.log(i2), np.log(i3)
        m = (
                ((l2 - l3) * (m1 ** 2) + (l3 - l1) * (m2 ** 2) + (l1 - l2) * (m3 ** 2))
                / ((l2 - l3) * (m1) + (l3 - l1) * (m2) + (l1 - l2) * (m3))
                * 1
                / 2
        )

    return m
def detect_all_peaks(intensity_list, intensity_threshold = 30000, n_neighbor=2, return_all = False, return_baseline = False):
    intensity_list_smoothed = np.array(moving_average(intensity_list, n= n_neighbor))
    peaks_all =get_peaks(intensity_list_smoothed)
    if len(peaks_all)==0:
        return([], np.nan, np.nan)
    peaks_return = [None]*len(peaks_all)
    reci_snrs = np.zeros(len(peaks_all))
    raw_apex_idx = np.zeros(len(peaks_all), dtype=int)
    # smoothed_apex_intensity =np.zeros(len(peaks_all))
    idx = 0
    all_apex_intensity = np.array([intensity_list_smoothed[x[1]]for x in peaks_all])
    if np.max(all_apex_intensity)<=intensity_threshold:
        return ([], np.nan, np.nan)
    while np.max(all_apex_intensity)>intensity_threshold:
        apex_peak, peaks_all, all_apex_intensity= get_current_peak(peaks_all,all_apex_intensity, intensity_list_smoothed )
        if idx == 0:
            fwhm = find_peak_width(apex_peak, intensity_list_smoothed, ratio = 1/1.5)
            fwhm = np.ceil(fwhm*1.32)
            # print(fwhm)
        peaks_return[idx]=(apex_peak)
        # smoothed_apex_intensity[idx]=intensity_list_smoothed[apex_peak[1]]
        raw_apex_idx[idx]=(apex_peak[0]+np.argmax(intensity_list[apex_peak[0]:apex_peak[2]+1]))
        idx = idx+1
        if len(peaks_all)==0:
            break
    baseline = pybaselines.smooth.snip(intensity_list,  decreasing = True, max_half_windowint = fwhm, smooth_half_window = int(fwhm))[0]
    # print(fwhm)
    # baseline = pybaselines.classification.dietrich(intensity_list_smoothed, smooth_half_window =0,poly_order=2,interp_half_window=fwhm)[0]
    # baseline = pybaselines.classification.fastchrom(intensity_list_smoothed, half_window = fwhm)[0]

    baseline[baseline<0]=0
    for i in range(0, idx):
        reci_snrs[i]=np.median(baseline[peaks_return[i][0]:peaks_return[i][2]+1])/intensity_list_smoothed[peaks_return[i][1]]
    raw_apex_idx = raw_apex_idx[0:idx]

    peaks_return = peaks_return[0:idx]
    reci_snrs=reci_snrs[0:idx]
    if return_all==True:
        toggle = raw_apex_idx>-1
    else:
        toggle = reci_snrs<1/3
    if return_baseline==False:
        return np.array(peaks_return)[toggle], raw_apex_idx[toggle], reci_snrs[toggle]
    else:
        return np.array(peaks_return)[toggle], raw_apex_idx[toggle], reci_snrs[toggle], baseline
# def libgen_function(std_list, mzml_dir, if_QE = True):
#     if if_QE ==True:
#         mass_error  = 0.002
#     else:
#         mass_error = 0.005
#     adducts = find_adducts(std_list.columns)
#     from itertools import  chain
#     print('extracting features, and mapping by mzrt')
#     matched_all = pd.DataFrame()
#     for mix in tqdm(std_list['mix'].unique()):
#         std_list_mix = string_search(std_list, 'mix', mix)
#         masses = std_list_mix[adducts].values
#         masses = np.array(list(chain.from_iterable(masses)))
#         keep = masses>50
#         masses = masses[keep]
#         feature_targeted = find_feature_targeted(masses=masses, mix=mix, mzml_dir=mzml_dir,mass_error = 0.005, if_QE= if_QE)

#         matched = feature_matching(feature_targeted, std_list_mix, adducts, mass_error = mass_error)
#         matched_all = pd.concat([matched_all, matched], ignore_index=True)
#     msms_denoised = []
#     eis = []
#     msms_raw = []
#     print('denoising....')
#     matched_all_predenoising = matched_all.copy()
#     for index, row in tqdm(matched_all.iterrows(), total = len(matched_all)):

#         entropy_temp = so.entropy_identity(row['msms'], row['msms_mf'], pmz = row['precursor_mz'])
#         if entropy_temp<0.99:
#             msms_d1 = drf.spectral_denoising(row['msms'], row['reference_smiles'], row['reference_adduct'], max_allowed_deviation=0.02)
#             ei1 = drf.get_ei(msms_d1, row['msms'], row['precursor_mz'])
#             msms_d2 = drf.spectral_denoising(row['msms_mf'], row['reference_smiles'], row['reference_adduct'])
#             ei2 = drf.get_ei(msms_d2, row['msms_mf'], row['precursor_mz'])
#             if ei1>ei2:
#                 msms_raw.append(row['msms'])
#                 msms_denoised.append(msms_d1)
#                 eis.append(ei1)
#             else:
#                 msms_raw.append(row['msms_mf'])
#                 msms_denoised.append(msms_d2)
#                 eis.append(ei2)
#         else:
#             msms_d1 = drf.spectral_denoising(row['msms'], row['reference_smiles'], row['reference_adduct'])
#             ei1 = drf.get_ei(msms_d1, row['msms'], row['precursor_mz'])
#             msms_denoised.append(msms_d1)
#             msms_raw.append(row['msms'])
#             eis.append(ei1)
#     matched_all.drop(columns=['msms', 'msms_pmz',
#                               'msms_pmz_intensity', 'msms_idx', 'rt_offset', 'msms_mf', 'msms_mf_pmz',
#                               'msms_mf_pmz_intensity', 'rt_offset_mf'], inplace=True)
#     matched_all['msms_raw']=msms_raw
#     matched_all['msms_denoised']=msms_denoised
#     matched_all['eis']=eis
#     return matched_all,matched_all_predenoising






