import os.path
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d, median_filter

import params

# TODO: remove this file. This processing is now part of twoppp

def fill_nans_with_previous(array):
    print(f"found {np.sum(np.isnan(array))} nans. will replace them with previous value")
    array = array.copy()
    if np.isnan(array[0]):
        array[0] = 0
    
    while any(np.isnan(array)):
        mask = np.isnan(array)
        indices = np.where(mask)[0]
        array[mask] = array[indices-1]
    return array

def read_sleap_output(trial_dir, med_filt=params.sleap_med_filt, sigma_gauss=params.sleap_sigma_gauss):
    sleap_output_file = os.path.join(trial_dir, "behData", "images", "sleap_output.h5")

    with h5py.File(sleap_output_file, "r") as f:
        # dset_names = list(f.keys())
        locations = np.squeeze(f["tracks"][:].T)  # returns (N_samples, N_keypoints, 2)
        node_names = [n.decode() for n in f["node_names"][:]]

    n_samples, n_keypoints, n_dim = locations.shape
    assert len(node_names) == n_keypoints

    for i_k in range(n_keypoints):
        for i_d in range(n_dim):
            locations[:,i_k, i_d] = fill_nans_with_previous(locations[:,i_k, i_d])
            locations[:,i_k, i_d] = gaussian_filter1d(median_filter(locations[:,i_k, i_d], size=med_filt), sigma=sigma_gauss)

    return locations, node_names

def get_angle(a,b,c):
    ba = a - b
    bc = c - b

    if len(ba.shape) == 2 and len(bc.shape) == 2:
        cosine_angle = np.sum(ba*bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    elif len(ba.shape) == 2 and len(bc.shape) == 1:
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    elif len(ba.shape) == 1 and len(bc.shape) == 2:
        cosine_angle = np.dot(ba, bc.T) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    elif len(ba.shape) == 1 and len(bc.shape) == 1:
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def joint_motionenergy(x1,y1, moving_average=params.joint_motion_energy_moving_average):
    x2 = np.ones_like(x1)*x1[0]
    x2[1:] = x1[:-1]
    y2 = np.ones_like(y1)*y1[0]
    y2[1:] = y1[:-1]
    me = np.sqrt(np.sum([np.square(x1-x2), np.square(y1-y2)], axis=0))
    return np.convolve(me, np.ones(moving_average), 'same') / moving_average

def add_sleap_to_beh_df(trial_dir, beh_df):
    # beh_df = pd.read_pickle(beh_df) if os.path.isfile(beh_df) else beh_df
    assert isinstance(beh_df, pd.DataFrame)

    locations, node_names = read_sleap_output(trial_dir)
    n_samples, n_keypoints, n_dim = locations.shape
    assert n_samples == len(beh_df)
    assert n_dim == 2

    i_neck = node_names.index("neck")
    neck_fix = np.median(locations[:,i_neck,:], axis=0)

    for i_k, keypoint in enumerate(node_names):
        for i_d, (d, neck_d) in enumerate(zip(["x","y"], neck_fix)):
            beh_df[f"{keypoint}_{d}"] = locations[:,i_k, i_d]
            beh_df[f"{keypoint}_{d}_rel_neck"] = locations[:,i_k, i_d] - neck_d

    i_coxa = node_names.index("frcofe")
    coxa_fix = np.median(locations[:,i_coxa,:], axis=0)
    i_frtita = node_names.index("frtita")
    beh_df["frleg_height"] = locations[:,i_frtita,1] - coxa_fix[1]

    i_frfeti = node_names.index("frfeti")
    beh_df["frtita_neck_dist"] = np.sqrt((locations[:,i_frtita,0] - neck_fix[0])**2 + (locations[:,i_frtita,1] - neck_fix[1])**2)
    beh_df["frfeti_neck_dist"] = np.sqrt((locations[:,i_frfeti,0] - neck_fix[0])**2 + (locations[:,i_frfeti,1] - neck_fix[1])**2)

    i_anus = node_names.index("anus")
    i_ovum = node_names.index("ovum")
    i_stripe = node_names.index("stripe4")
    beh_df["anus_dist"] = np.sqrt(np.sum(np.square(locations[:,i_anus]-locations[:,i_stripe]), axis=-1))
    beh_df["ovum_dist"] = np.sqrt(np.sum(np.square(locations[:,i_ovum]-locations[:,i_stripe]), axis=-1))

    # compute angles of frontleg
    # frfeti, coxa, neck -> femur angle
    beh_df["ang_frfemur"] = get_angle(locations[:,i_frfeti], coxa_fix, neck_fix)
    # coxa, frfeti, frtita -> tibia angle
    beh_df["ang_frtibia"] = get_angle(coxa_fix, locations[:,i_frfeti], locations[:,i_frtita])
    # neck, coxa, frfeti, frtita -> tibia/neck angle
    beh_df["ang_frtibia_neck"] = get_angle(locations[:,i_frtita], coxa_fix, neck_fix)
    # compute butt angle
    beh_df["ang_abd"] = get_angle(locations[:,i_anus], locations[:,i_stripe], coxa_fix)

    # compute leg motion energy
    i_mrtita = node_names.index("mrtita")
    i_hrtita = node_names.index("hrtita")
    beh_df["mef_tita"] = joint_motionenergy(locations[:,i_frtita,0], locations[:,i_frtita,1])
    beh_df["mem_tita"] = joint_motionenergy(locations[:,i_mrtita,0], locations[:,i_mrtita,1])
    beh_df["meh_tita"] = joint_motionenergy(locations[:,i_hrtita,0], locations[:,i_hrtita,1])

    return beh_df

    
