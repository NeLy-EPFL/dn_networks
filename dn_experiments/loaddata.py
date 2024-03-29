"""
loading data from files
Author: jonas.braun@epfl.ch
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter

from twoppp import load, rois, utils
from twoppp.behaviour import synchronisation, fictrac, sleap

import params, behaviour, baselines, stimulation, motionenergy  # , sleap


def get_background_image(fly_dir, background_crop=params.background_crop, background_key=params.background_key,
                         background_sigma=params.background_sigma, background_med=params.background_med, force_new_background=False):
    """
    Load and process the background image for a fly's data directory.
    For data downloaded from Harvard dataverse, this will load the "background_image.tif" file instead of newly computing it.
    It can be forced to calculate a new background.

    Args:
        fly_dir (str): The directory containing the fly's data.
        background_crop (tuple): Cropping parameters for the background image.
        background_key (str): Key for the background image in summary statistics.
        background_sigma (float): Sigma for Gaussian smoothing of the background image.
        background_med (int): Size of the median filter for the background image.
        force_new_background (bool): Force recomputation of the background image instead of loading saved background image.

    Returns:
        background (numpy.ndarray): Processed background image.
    """
    if os.path.isfile(os.path.join(fly_dir, load.PROCESSED_FOLDER, "background_image.tif")) and not force_new_background:
        background = utils.get_stack(os.path.join(fly_dir, load.PROCESSED_FOLDER, "background_image.tif"))
        print("loading pre-computed background image. Params are not applied.")
        return background
    try:
        with open(os.path.join(fly_dir, load.PROCESSED_FOLDER, "compare_trials.pkl"), "rb") as f:
            summary_stats = pickle.load(f)
        background = np.array(summary_stats[background_key])
        if len(background.shape) == 3:
            background = np.mean(background, axis=0)
    except FileNotFoundError:
        print(f"Warning: could not find summary image for fly {fly_dir}")
        background = np.zeros((320,736))

    background = background[background_crop[0]:background.shape[0]-background_crop[1],
                            background_crop[2]:background.shape[1]-background_crop[3]]
    background = gaussian_filter(median_filter(background, size=background_med), sigma=background_sigma)
    
    return background

def get_roi_centers(fly_dir):
    """
    Load ROI centers from a file in a fly's data directory.

    Args:
        fly_dir (str): The directory containing the fly's data.

    Returns:
        roi_centers (list): List of ROI centers.
    """
    try:
        roi_centers = rois.read_roi_center_file(os.path.join(fly_dir, load.PROCESSED_FOLDER, "ROI_centers.txt"))
    except FileNotFoundError:
        print(f"Did not find ROI centers for {fly_dir}. Will continue without")
        roi_centers = []
    return roi_centers 

def get_filtered_twop_df(fly_dir, all_trial_dirs, neurons_med=params.neurons_med, neurons_sigma=params.neurons_sigma, neurons_filt_regex=params.neurons_filt_regex):
    """
    Load and filter Two-Photon (2P) data for a fly's data directory.

    Args:
        fly_dir (str): The directory containing the fly's data.
        all_trial_dirs (list): List of trial directories.
        neurons_med (int): Size of the median filter for neuronal signal data.
        neurons_sigma (float): Sigma for Gaussian smoothing of neuronal signal data.
        neurons_filt_regex (str): Regular expression for filtering neuron data. Will be used as keys in the twop_df.

    Returns:
        twop_df (pandas.DataFrame): Filtered 2P data DataFrame.
    """
    twop_dfs = []
    for i_trial, trial_dir in enumerate(all_trial_dirs):
        if not os.path.isdir(trial_dir):
            trial_dir = load.get_trials_from_fly(fly_dir, startswith=trial_dir)[0][0]
        twop_dfs.append(pd.read_pickle(os.path.join(trial_dir, load.PROCESSED_FOLDER, "twop_df.pkl")))
        neurons = twop_dfs[i_trial].filter(regex="neuron").filter(regex="^(?!.*?denoised)").filter(regex="^(?!.*?filt)").values
        neurons_filt = gaussian_filter1d(median_filter(neurons, size=(neurons_med, 1)), sigma=neurons_sigma, axis=0)
        for i_roi in range(neurons.shape[1]):
            twop_dfs[i_trial][f"{neurons_filt_regex}_{i_roi}"] = neurons_filt[:, i_roi]
    twop_df = pd.concat(twop_dfs).fillna(0)
    # fix velocities for wheel trials and when something went wrong with fictrac
    if "v_forw" not in twop_df.keys() and "v" in twop_df.keys():  # wheel trials
        v = twop_df.v.values
        v_forw = v.copy()
        v = np.abs(v_forw)
        twop_df["v_forw"] = v_forw
        twop_df["v"] = v
    elif "v_forw" not in twop_df.keys() and "v" not in twop_df.keys():
        print(f"Warning: missing 'v' and 'v_forw' in fly {fly_dir} with trials {all_trial_dirs}")
    return twop_df

def get_beh_df_with_me(fly_dir, all_trial_dirs, q_me=params.q_me, add_sleap=True, add_me=False):
    """
    Load behavior data with optional motion energy (ME) computation and SLEAP predictions.

    Args:
        fly_dir (str): The directory containing the fly's data.
        all_trial_dirs (list): List of trial directories.
        q_me (float): Quantile for normalizing motion energy.
        add_sleap (bool): Add SLEAP predictions to behavior data.
        add_me (bool): Add motion energy to behavior data.

    Returns:
        beh_df (pandas.DataFrame): Behavior data DataFrame.
    """
    beh_dfs = []
    for i_trial, trial_dir in enumerate(all_trial_dirs):
        if not os.path.isdir(trial_dir):
            trial_dir = load.get_trials_from_fly(fly_dir, startswith=trial_dir)[0][0]
        beh_df_file = os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl")
        beh_df = pd.read_pickle(beh_df_file)
        if "me_front" not in beh_df.keys() and add_me:
            print(f"Motion energy not yet computed for trial {trial_dir}. Will compute now.")
            beh_df = motionenergy.compute_and_add_me_to_df(trial_dir, beh_df_file, camera_name=params.me_cam)
        if ("mef_tita" not in beh_df.keys()) and add_sleap:  #  or "frtita_neck_dist" not in beh_df.keys()
            beh_df = sleap.add_sleap_to_beh_df(trial_dir, beh_df)
            beh_df.to_pickle(beh_df_file)
        beh_dfs.append(beh_df)
    beh_df = pd.concat(beh_dfs).fillna(0)

    # fix velocities for wheel trials and when something went wrong with fictrac
    no_tracking = False
    if "v_forw" not in beh_df.keys() and "delta_rot_lab_forward" in beh_df.keys():
        v_forw = fictrac.filter_fictrac(beh_df["delta_rot_lab_forward"], 5, 10)
        beh_df["v_forw"] = v_forw
    elif "v_forw" not in beh_df.keys() and hasattr(beh_df, 'v'):  # wheel trials
        v = beh_df.v.values
        v_forw = v.copy()
        v = np.abs(v_forw)
        beh_df["v_forw"] = v_forw
        beh_df["v"] = v

    elif "v_forw" not in beh_df.keys() and not hasattr(beh_df, 'v'):  # no ball trials
        beh_df["v_forw"] = np.zeros(beh_df.shape[0])
        beh_df["v"] = np.zeros(beh_df.shape[0])
        print(f"Warning: missing 'v' and 'v_forw' in fly {fly_dir} with trials {all_trial_dirs}")
        no_tracking = True

    # integrate v_forw to get the displacement 'd_forw'
    beh_df["d_forw"] = np.cumsum(beh_df["v_forw"].values) / params.fs_beh

    if add_me and not no_tracking:
        beh_df["me_front_q"] = utils.normalise_quantile(beh_df["me_front"].values, q=q_me)
        beh_df["me_back_q"] = utils.normalise_quantile(beh_df["me_back"].values, q=q_me)
        beh_df["me_mid_q"] = utils.normalise_quantile(beh_df["me_mid"].values, q=q_me)
        beh_df["me_all_q"] = utils.normalise_quantile(beh_df["me_all"].values, q=q_me)
        beh_df["silent"] = np.logical_and.reduce((beh_df["me_front_q"] < params.thres_silent_me_front,
                                                beh_df["me_all_q"] < params.thres_silent_me_all,
                                                beh_df["v"] < params.thres_silent_v))
        beh_df["fast"] = beh_df["me_all_q"] > params.thres_fast_me_all

    return beh_df


def load_process_data(fly_dir, all_trial_dirs, twop_df_save_name=params.twop_df_save_name, beh_df_save_name=params.beh_df_save_name, load_twop=True,
                      add_sleap=True, add_me=True, vnccut_mode=False):
    """
    Load and process data for a fly's data directory, including 2P data and behavior data.
    Low level interface only to be used internally.

    Args:
        fly_dir (str): The directory containing the fly's data.
        all_trial_dirs (list): List of trial directories.
        twop_df_save_name (str): Filename for saving the filtered 2P data.
        beh_df_save_name (str): Filename for saving the behavior data.
        load_twop (bool): Whether to load 2P data of beh_df only.
        add_sleap (bool): Whether to add sleap data to beh_df.
        add_me (bool): Whether to add motion energy data to beh_df
        vnccut_mode (bool): Whether to treat data as coming from vnccut experiments.

    Returns:
        twop_df (pandas.DataFrame): Filtered 2P data DataFrame.
        beh_df (pandas.DataFrame): Behavior data DataFrame.
    """
    if load_twop:
        beh_df = get_beh_df_with_me(fly_dir=fly_dir, all_trial_dirs=all_trial_dirs, add_sleap=add_sleap, add_me=add_me)
        twop_df = get_filtered_twop_df(fly_dir=fly_dir, all_trial_dirs=all_trial_dirs)
        twop_df = stimulation.fix_stim_power_signal(twop_df)
        if not vnccut_mode:
            twop_df = behaviour.get_beh_info_in_twop_df(beh_df, twop_df)
    else:
        beh_df = get_beh_df_with_me(fly_dir=fly_dir, all_trial_dirs=all_trial_dirs, add_sleap=True)
        twop_df = None
    beh_df = stimulation.fix_stim_power_signal(beh_df)
    if not vnccut_mode:
        twop_df, beh_df = behaviour.add_beh_class_to_dfs(twop_df, beh_df)
        twop_df, beh_df = behaviour.get_beh_trigger_into_dfs(twop_df, beh_df, beh="back")
        twop_df, beh_df = behaviour.get_beh_trigger_into_dfs(twop_df, beh_df, beh="walk")
        twop_df, beh_df = behaviour.get_beh_trigger_into_dfs(twop_df, beh_df, beh="rest")
        twop_df, beh_df = behaviour.get_beh_trigger_into_dfs(twop_df, beh_df, beh="groom")
    
    if load_twop and not vnccut_mode:
        twop_df = baselines.get_baseline_in_twop_df(twop_df)
    elif load_twop:
        twop_df = baselines.get_baseline_in_twop_df(twop_df, normalisation_type=["qmin_qmax"], exclude_stim_low_high=[False,False])

    # TODO: get backwards start in twop_df
    if load_twop and twop_df_save_name is not None:
        pass
    if beh_df_save_name is not None:
        pass
    if load_twop:
        return twop_df, beh_df
    else:
        return beh_df

def load_data(fly_dir, all_trial_dirs, overwrite=False, add_sleap=True, add_me=True, vnccut_mode=False):
    """
    Load 2P and behavior data for a fly's data directory.
    High level interface also for external use.

    Args:
        fly_dir (str): The directory containing the fly's data.
        all_trial_dirs (list): List of trial directories.
        overwrite (bool): Whether to overwrite existing data files.
        add_sleap (bool): Whether to add sleap data to beh_df.
        add_me (bool): Whether to add motion energy data to beh_df
        vnccut_mode (bool): Whether to treat data as coming from vnccut experiments.

    Returns:
        twop_df (pandas.DataFrame): Filtered 2P data DataFrame.
        beh_df (pandas.DataFrame): Behavior data DataFrame.
    """
    FILE_PRESENT = False
    if FILE_PRESENT and not overwrite:
        twop_df = None
        beh_df = None
    else:
        return load_process_data(fly_dir, all_trial_dirs, load_twop=True, add_sleap=add_sleap, add_me=add_me, vnccut_mode=vnccut_mode)

def load_beh_data_only(fly_dir, all_trial_dirs, overwrite=False):
    """
    Load behavior data (without 2P data) for a fly's data directory.
    High level interface to also be used externally.

    Args:
        fly_dir (str): The directory containing the fly's data.
        all_trial_dirs (list): List of trial directories.
        overwrite (bool): Whether to overwrite existing data files.

    Returns:
        beh_df (pandas.DataFrame): Behavior data DataFrame.
    """
    FILE_PRESENT = False
    if FILE_PRESENT and not overwrite:
        beh_df = None
    else:
        return load_process_data(fly_dir, all_trial_dirs, load_twop=False)