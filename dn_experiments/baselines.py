"""
module used to compute different types of baselines for neural signals and to add the DF/F signals to the twop_df.
Author: jonas.braun@epfl.ch
"""
import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from twoppp import utils, rois

import params, stimulation

def get_resting_baselines(neurons, rest, fs=params.fs_2p,
                          min_rest_s=params.rest_base_min_rest_s,
                          max_search_t_s=params.rest_base_max_search_t_s,
                          frac_rest=params.rest_base_frac_rest):
    """
    Compute the baseline for neurons based on the resting behavior.
    compute resting baseline as follows:
    1. find all instances where fly starts resting and is resting for at least frac_rest * 100 percent during the next min_rest_s seconds.
        - ignore all repeated starts of resting if they are within max_search_t_s[1] // 2 seconds after the previous start
    2. compute the median across all repetitions
    3. compute the minimum of each neuron from  max_search_t_s[0] s before to max_search_t_s[1] s after start of resting

    Parameters:
        neurons (numpy.ndarray): Neuronal data to calculate the baseline for.
        rest (numpy.ndarray): Resting behavior data.
        fs (float, optional): Sampling frequency. Defaults to params.fs_2p.
        min_rest_s (float, optional): Minimum duration of resting in seconds. Defaults to params.rest_base_min_rest_s.
        max_search_t_s (tuple, optional): Search time range in seconds, specified as (before, after) resting onset.
                                          Defaults to params.rest_base_max_search_t_s.
        frac_rest (float, optional): Fraction of time within the search range that must be spent resting. Defaults to params.rest_base_frac_rest.

    Returns:
        numpy.ndarray: baselines for each neuron.
    """
    # 1. find onsets of resting
    rest_start = np.diff(rest.astype(int)) == 1
    i_rest_start = np.where(rest_start)[0]
    
    n_s_pre = int(np.abs(fs*max_search_t_s[0]))
    assert min_rest_s <= max_search_t_s[1]
    n_s_post = int(fs*max_search_t_s[1])
    n_s_total = n_s_pre + n_s_post
                   
    # aligned_neurons = np.zeros((n_s_total, neurons.shape[1], len(i_rest_start)))  # old version only usable for ROI signals
    aligned_neurons = np.zeros((len(i_rest_start), n_s_total) + neurons.shape[1:])  # this makes it possible to be used for stacks and ROI signals
    true_rest = 0
    for i, i_rs in enumerate(i_rest_start):
        if i and i_rs < i_rest_start[i-1] + n_s_post // 2:
            continue  # if already considered recent resting start
        if i_rs < n_s_pre + 1:
            continue  # if resting start is at the very beginning of the trial
        if i_rs >= len(neurons) - n_s_post - 1:
            continue  # if resting start is at the very end of the trial
        if np.mean(rest[i_rs:i_rs+int(min_rest_s*fs)]) >= frac_rest:
            # if fraction of resting in selected time frame is sufficient
            try:
                # extract signals around onsets of resting
                aligned_neurons[true_rest,:,:,] = neurons[i_rs-n_s_pre:i_rs+n_s_post, :]
                true_rest += 1
            except:
                print(f"Error while computing resting baseline: i_rs: {i_rs}, n_s_pre {n_s_pre}, ns_post {n_s_pre} len(neurons) {len(neurons)}")
            
    aligned_neurons = aligned_neurons[:true_rest]  # only keep as many as were not rejected in the loop
    
    # compute the median across all repetitions of starting to rest
    # then take the temporal minimum
    resting_baselines = np.min(np.median(aligned_neurons, axis=0), axis=0)

    return resting_baselines

def get_trough_baselines(neurons, n_peaks=params.trough_baseline_n_peaks,
                         min_height=params.trough_baseline_min_height,
                         min_distance=params.trough_baseline_min_distance,
                         n_samples_around=params.trough_baseline_n_samples_around):
    """
    Compute baselines for neurons based on population activity troughs.
    find the baseline of a neuron by considering the time points when the entire population is at its minimum
    1. standardise 'neurons' and average across neurons
    2. find 'n_peaks' most prominent troughs
    3. extract signals of each neuron aroun these time points (considering 'n_samples_around' before and after)
    4. compute the median across repetitions for each neuron
    5. compute the minimum across time for each neuron

    Parameters:
        neurons (numpy.ndarray): Neuronal data to calculate the baseline for.
        n_peaks (int, optional): Number of most prominent trough peaks to consider. Defaults to params.trough_baseline_n_peaks.
        min_height (float, optional): Minimum peak height to be considered a trough. Defaults to params.trough_baseline_min_height.
        min_distance (int, optional): Minimum distance (in samples) between troughs. Defaults to params.trough_baseline_min_distance.
        n_samples_around (int, optional): Number of samples to consider around each trough. Defaults to params.trough_baseline_n_samples_around.

    Returns:
        numpy.ndarray: baselines for each neuron.
    """

    neurons_std_mean = -1 * np.mean((neurons - neurons.mean(axis=0)) / neurons.std(axis=0), axis=-1)
    peaks, peak_properties = find_peaks(neurons_std_mean, height=min_height, distance=min_distance)
    sorted_peaks = peaks[np.flip(np.argsort(peak_properties["peak_heights"]))]
    sorted_peaks = sorted_peaks[sorted_peaks >= n_samples_around]
    sorted_peaks = sorted_peaks[sorted_peaks < len(neurons_std_mean) - n_samples_around]
    selected_peaks = sorted_peaks[:np.minimum(len(sorted_peaks), n_peaks)]
    
    neuron_mins = np.zeros((2*n_samples_around, neurons.shape[1], len(selected_peaks)))
    for i_peak, selected_peak in enumerate(selected_peaks):
        neuron_mins[:,:,i_peak] = neurons[selected_peak-n_samples_around:selected_peak+n_samples_around]
    through_baselines = np.min(np.median(neuron_mins, axis=-1), axis=0)
    
    return through_baselines

def get_me_trough_baselines(neurons, me, n_peaks=params.trough_baseline_n_peaks,
                         min_height=params.me_trough_baseline_min_height,
                         min_distance=params.trough_baseline_min_distance,
                         n_samples_around=params.trough_baseline_n_samples_around,
                         sigma_me=params.me_trough_baseline_sigma_me):
    """
    Compute baselines for neurons based on motion energy troughs.
    find the baseline of a neuron by considering the time points when the motion energy is at its minimum
    1. low pass filter and quantile normalise motion energy
    2. find 'n_peaks' most prominent troughs
    3. extract signals of each neuron aroun these time points (considering 'n_samples_around' before and after)
    4. compute the median across repetitions for each neuron
    5. compute the minimum across time for each neuron

    Parameters:
        neurons (numpy.ndarray): Neuronal data to calculate the baseline for.
        me (numpy.ndarray): Motion energy data.
        n_peaks (int, optional): Number of trough peaks to consider. Defaults to params.trough_baseline_n_peaks.
        min_height (float, optional): Minimum peak height to be considered a trough. Defaults to params.me_trough_baseline_min_height.
        min_distance (int, optional): Minimum distance (in samples) between troughs. Defaults to params.trough_baseline_min_distance.
        n_samples_around (int, optional): Number of samples to consider around each trough. Defaults to params.trough_baseline_n_samples_around.
        sigma_me (float, optional): Sigma value for Gaussian filtering of motion energy. Defaults to params.me_trough_baseline_sigma_me.

    Returns:
        numpy.ndarray: baselines for each neuron.
    """
    me_q = utils.normalise_quantile(-1 * gaussian_filter1d(me, sigma=sigma_me))
    peaks, peak_properties = find_peaks(me_q, height=min_height, distance=min_distance)
    sorted_peaks = peaks[np.flip(np.argsort(peak_properties["peak_heights"]))]
    sorted_peaks = sorted_peaks[sorted_peaks >= n_samples_around]
    sorted_peaks = sorted_peaks[sorted_peaks < len(me_q) - n_samples_around]
    selected_peaks = sorted_peaks[:np.minimum(len(sorted_peaks), n_peaks)]
    
    neuron_mins = np.zeros((2*n_samples_around, neurons.shape[1], len(selected_peaks)))
    for i_peak, selected_peak in enumerate(selected_peaks):
        neuron_mins[:,:,i_peak] = neurons[selected_peak-n_samples_around:selected_peak+n_samples_around]
    through_baselines = np.min(np.median(neuron_mins, axis=-1), axis=0)
    
    return through_baselines

def add_baseline_to_df(twop_df, baseline_sub, baseline_div, fstring="dff", neurons_regex=params.baseline_neurons_regex):
    """
    Add baseline-subtracted and baseline-divided neuronal signals, i.e. DF/F to the twop_df DataFrame.

    Parameters:
        twop_df (pandas.DataFrame): DataFrame containing two-photon imaging data.
        baseline_sub (numpy.ndarray): Baseline to subtract from the neuronal signals.
        baseline_div (numpy.ndarray): Baseline to divide the neuronal signals by.
        fstring (str, optional): Prefix for the new columns. Defaults to "dff".
        neurons_regex (str, optional): Regular expression to match neuronal columns in the DataFrame. Defaults to params.baseline_neurons_regex.

    Returns:
        pandas.DataFrame: Updated DataFrame with baseline-subtracted and baseline-divided signals.
    """
    neurons_filt = twop_df.filter(regex=neurons_regex).values
    dff = (neurons_filt - baseline_sub) / baseline_div
    for i_roi in range(neurons_filt.shape[1]):
        twop_df[f"{fstring}_{i_roi}"] = dff[:, i_roi]
    return twop_df

def get_baseline_in_twop_df(twop_df, baseline_exclude_stim_range=params.baseline_exclude_stim_range,
                            exclude_stim_low_high=params.baseline_exclude_stim_low_high,
                            normalisation_type=params.all_normalisation_types, return_baselines=False, qmax=params.baseline_qmax):
    """
    Calculate baseline values and add them to the twop_df DataFrame.

    Parameters:
        twop_df (pandas.DataFrame): DataFrame containing two-photon imaging data.
        baseline_exclude_stim_range (tuple, optional): Time range to exclude for baseline calculation. Defaults to params.baseline_exclude_stim_range.
        exclude_stim_low_high (tuple, optional): Tuple of length 2 indicating whether to exclude stimulation periods for lower and upper limit computation. Defaults to params.baseline_exclude_stim_low_high.
        normalisation_type (str or list, optional): Type of normalisation to perform. Defaults to params.all_normalisation_types.
        return_baselines (bool, optional): Whether to return computed baseline values. Defaults to False.
        qmax (float, optional): Quantile value for computing the qmax baseline. Defaults to params.baseline_qmax.

    Returns:
        pandas.DataFrame or tuple: Updated DataFrame with baseline-subtracted and baseline-divided signals.
                                    If return_baselines is True, also returns a dictionary of computed baseline values.
    """
    # print("prepare neural data outside of stimulation period for baseline computation")
    neurons_for_baseline = twop_df.filter(regex=params.baseline_neurons_regex).values
    
    if baseline_exclude_stim_range is not None:
        stim_starts, stim_stops = stimulation.get_all_stim_starts(twop_df, return_stops=True)
        stim_starts += baseline_exclude_stim_range[0]
        stim_stops += baseline_exclude_stim_range[1]
        stim = np.zeros((len(neurons_for_baseline)), dtype=bool)
        for start, stop in zip(stim_starts, stim_stops):
            stim[start:stop] = True
        no_stim = np.logical_not(stim)
    else:
        no_stim = np.ones((len(neurons_for_baseline)), dtype=bool)
    
    # print("compute baseline per neuron")
    if not isinstance(normalisation_type, list):
        normalisation_type = [normalisation_type]
    all_baselines = {}
    if "dff" in normalisation_type:
        _, f0 = rois.get_dff_from_traces(signals=neurons_for_baseline[no_stim,:], return_f0=True)
        twop_df = add_baseline_to_df(twop_df, baseline_sub=f0, baseline_div=f0, fstring="dff_filt")
        all_baselines["dff"] = f0
    if "zscore" in normalisation_type or "std" in normalisation_type:
        mu = np.mean(neurons_for_baseline[no_stim,:], axis=0)
        std = np.std(neurons_for_baseline[no_stim,:], axis=0)
        twop_df = add_baseline_to_df(twop_df, baseline_sub=mu, baseline_div=std, fstring="neuron_std")
        all_baselines["mu"] = mu
        all_baselines["std"] = std
    if "silent" in normalisation_type:
        silent = np.convolve(twop_df["silent"].values, np.array([True, True, True]), mode="same")  # also take the next two frames if a fly was silent
        silent_no_stim = np.logical_and(no_stim, silent)
        _, f0_silent = rois.get_dff_from_traces(signals=neurons_for_baseline[silent_no_stim,:], return_f0=True)
        twop_df = add_baseline_to_df(twop_df, baseline_sub=f0_silent, baseline_div=f0_silent, fstring="dff_silent_filt")
        all_baselines["silent"] = f0_silent
    if "fastsilent" in normalisation_type:
        silent = np.convolve(twop_df["silent"].values, np.array([True, True, True]), mode="same")  # also take the next two frames if a fly was silent
        fast = np.convolve(twop_df["fast"].values, np.array([True, True, True]), mode="same")  # also take the next two frames if a fly was silent
        silent_no_stim = np.logical_and(no_stim, silent) if exclude_stim_low_high[0] else silent
        fast_no_stim = np.logical_and(no_stim, fast) if exclude_stim_low_high[1] else fast
        print("# samples for fast and slow calculation:", np.sum(fast_no_stim), np.sum(silent_no_stim))
        _, f0_silent = rois.get_dff_from_traces(signals=neurons_for_baseline[silent_no_stim,:], return_f0=True)
        _, f0_fast = rois.get_dff_from_traces(signals=-1*neurons_for_baseline[fast_no_stim,:], return_f0=True)
        f0_fast = f0_fast * -1  # compute maximum value by first multiplying by 1, then computing F0 and multiplying by -1 again
        twop_df = add_baseline_to_df(twop_df, baseline_sub=f0_silent, baseline_div=f0_fast - f0_silent, fstring="dff_fastsilent_filt")
        all_baselines["silent"] = f0_silent
        all_baselines["fast"] = f0_fast
    if "rest_qmax" in normalisation_type:
        rest = twop_df.rest.values
        if exclude_stim_low_high[0]:
            f0_rest = get_resting_baselines(neurons_for_baseline[no_stim,:], rest[no_stim])
        else:
            f0_rest = get_resting_baselines(neurons_for_baseline, rest)
        if exclude_stim_low_high[1]:
            f0_qmax = np.quantile(neurons_for_baseline[no_stim,:], q=qmax, axis=0)
        else:
            f0_qmax = np.quantile(neurons_for_baseline, q=qmax, axis=0)
        twop_df = add_baseline_to_df(twop_df, baseline_sub=f0_rest, baseline_div=f0_qmax - f0_rest, fstring="dff_rest_qmax_filt")
        all_baselines["rest_baseline"] = f0_rest
        all_baselines["qmax"] = f0_qmax
    if "trough_qmax" in normalisation_type:
        if exclude_stim_low_high[0]:
            f0_trough = get_trough_baselines(neurons_for_baseline[no_stim,:])
        else:
            f0_trough = get_trough_baselines(neurons_for_baseline)
        if exclude_stim_low_high[1]:
            f0_qmax = np.quantile(neurons_for_baseline[no_stim,:], q=qmax, axis=0)
        else:
            f0_qmax = np.quantile(neurons_for_baseline, q=qmax, axis=0)
        twop_df = add_baseline_to_df(twop_df, baseline_sub=f0_trough, baseline_div=f0_qmax - f0_trough, fstring="dff_trough_qmax_filt")
        all_baselines["trough_baseline"] = f0_trough
        all_baselines["qmax"] = f0_qmax
    if "me_trough_qmax" in normalisation_type:
        me = twop_df.me_all.values
        if exclude_stim_low_high[0]:
            f0_me_trough = get_me_trough_baselines(neurons_for_baseline[no_stim,:], me[no_stim])
        else:
            f0_me_trough = get_me_trough_baselines(neurons_for_baseline, me)
        if exclude_stim_low_high[1]:
            f0_qmax = np.quantile(neurons_for_baseline[no_stim,:], q=qmax, axis=0)
        else:
            f0_qmax = np.quantile(neurons_for_baseline, q=qmax, axis=0)
        twop_df = add_baseline_to_df(twop_df, baseline_sub=f0_me_trough, baseline_div=f0_qmax - f0_me_trough, fstring="dff_me_trough_qmax_filt")
        all_baselines["me_trough_baseline"] = f0_me_trough
        all_baselines["qmax"] = f0_qmax
    if return_baselines:
        return twop_df, all_baselines
    else:
        return twop_df