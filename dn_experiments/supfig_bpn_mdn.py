"""
Module to generate supplementary figure analysing stimulation at different locations
Author: jonas.braun@epfl.ch
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import center_of_mass
import pickle
import cv2

from twoppp import load, utils
from twoppp.plot import videos
import twoppp.plot as myplt

import loaddata, params, plotpanels

def make_stim_loc_plots(driver_lines=["MDN", "BPN"], fly_dirs=None, figures_path=None):
    """
    Generate and save plots of behavioral responses to optogenetic stimulation at different locations.

    Parameters:
        driver_lines (list of str, optional): List of driver lines to consider.
        fly_dirs (list of str, optional): List of fly directories to process.
        figures_path (str, optional): Directory path to save generated figures.

    Returns:
        None
    """
    if fly_dirs is None:
        fly_dirs = [# "/mnt/nas2/JB/220816_MDN3xCsChrimson/Fly5", # "/mnt/nas2/JB/230125_BPNxCsChrimson/Fly1",
                    "/mnt/nas2/JB/230914_MDN3xCsChrimson/Fly1", "/mnt/nas2/JB/230914_MDN3xCsChrimson/Fly2",
                    "/mnt/nas2/JB/230914_BPNxCsChrimson/Fly3", "/mnt/nas2/JB/230914_BPNxCsChrimson/Fly4",
                    "/mnt/nas2/JB/230915_MDN3xCsChrimson/Fly11",
                    "/mnt/nas2/JB/230921_BPNxCsChrimson/Fly5", "/mnt/nas2/JB/230921_BPNxCsChrimson/Fly6"]
    if figures_path is None:
        figures_path = params.plot_base_dir

    STIM_PS = [1,5,10,20]
    P_COLOURS = [myplt.DARKGREEN, myplt.DARKYELLOW, myplt.DARKORANGE, myplt.DARKRED]
    STIM_LOCS = ["thorax", "cc", "head"]  # "t2", "t1", 

    fs = params.fs_beh
    N_samples_stim = params.n_s_beh_5s
    i_t_stim = [int(-N_samples_stim),int(2*N_samples_stim)]
    t_stim = np.arange(i_t_stim[0], i_t_stim[1]) / fs 

    for driver_line in driver_lines:
        genotype_fly_dirs = []
        for fly_dir in fly_dirs:
            if driver_line in fly_dir:
                genotype_fly_dirs.append(fly_dir)
        N_flies = len(genotype_fly_dirs)

        all_stim_responses = np.zeros((len(STIM_PS), len(STIM_LOCS), len(t_stim), 0))
        
        fig, axs = plt.subplots(N_flies+1,len(STIM_LOCS),figsize=(2.5*len(STIM_LOCS),3*(N_flies+1)), sharex=True, squeeze=False, sharey=True)

        for i_fly, fly_dir in enumerate(genotype_fly_dirs):
            trial_dirs = load.get_trials_from_fly(fly_dir, contains="plevels")
            trial_dirs = sum(trial_dirs, [])
            beh_df_dirs = [os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl") for trial_dir in trial_dirs]
            
            n_stim_responses = np.zeros((len(STIM_PS), len(STIM_LOCS)), dtype=int)
            stim_responses = np.zeros((len(STIM_PS), len(STIM_LOCS), len(t_stim), 10))

            for i_t, (trial_dir, beh_df) in enumerate(zip(trial_dirs, beh_df_dirs)):
                trial_name = trial_dir.split(os.sep)[-1]
                leg_position_file = os.path.join(trial_dir, load.PROCESSED_FOLDER, "leg_positions_pixels.pkl")
                i_loc = -1
                for i_, stim_loc in enumerate(STIM_LOCS):
                    if stim_loc in trial_name:
                        i_loc = i_
                        break
                if i_loc == -1:
                    continue
                if not isinstance(beh_df, pd.DataFrame):
                    beh_df = loaddata.get_beh_df_with_me(fly_dir, all_trial_dirs=[trial_name], add_sleap=False, add_me=False)
                stim_starts = np.argwhere(np.diff(beh_df["laser_stim"].to_numpy().astype(int))==1).flatten()

                for i_stim_start in stim_starts:
                    stim_p = int(beh_df.laser_power_uW.values[i_stim_start+2*100])
                    if stim_p in STIM_PS:
                        i_p = STIM_PS.index(stim_p)
                        stim_responses[i_p, i_loc, :, n_stim_responses[i_p, i_loc]] = beh_df.v_forw.values[i_stim_start+i_t_stim[0]:i_stim_start+i_t_stim[1]]
                        n_stim_responses[i_p, i_loc] += 1

            if len(np.unique(n_stim_responses)) == 2:
                assert 0 in n_stim_responses
            elif len(np.unique(n_stim_responses)) > 1:
                raise NotImplementedError

            stim_responses = stim_responses[:,:,:,:n_stim_responses[0,0]]
            if np.min(n_stim_responses):
                all_stim_responses = np.concatenate((all_stim_responses, stim_responses[:,:,:,:n_stim_responses[0,0]]), axis=-1)
        
            for i_loc, ax in enumerate(axs[i_fly]):
                for i_p, (p, p_color) in enumerate(zip(STIM_PS, P_COLOURS)):
                    if np.sum(np.abs(stim_responses[i_p, i_loc,:,:])) > 0:
                        myplt.plot_mu_sem(mu=np.mean(stim_responses[i_p, i_loc,:,:], axis=-1),
                                        err=utils.conf_int(stim_responses[i_p, i_loc,:,:], axis=-1),
                                        x=t_stim,
                                        label=f"{p} uW",
                                        ax=ax,
                                        color=p_color,
                                        linewidth=2
                                        )
                    else:
                        a = 0
                if i_fly == 0:
                    ax.set_title(f"{STIM_LOCS[i_loc]}".upper(), fontsize=16)
                if i_loc == 0:
                    ax.set_ylabel(f"Fly{i_fly+1}\n"+r"$v_{||}$ (mm/s)", fontsize=16)
                
                ax.set_xlim([-5,10])
                ax.set_xlabel("t (s)", fontsize=16)
                ax.set_xticks([0,5])
                plotpanels.make_nice_spines(ax)
                myplt.shade_categorical(catvar=np.concatenate((np.zeros((N_samples_stim)), np.ones((N_samples_stim)), np.zeros((N_samples_stim)))),
                                    x=t_stim, ax=ax, colors=[myplt.WHITE, myplt.BLACK])

        for i_loc, ax in enumerate(axs[-1]):
            for i_p, (p, p_color) in enumerate(zip(STIM_PS, P_COLOURS)):
                myplt.plot_mu_sem(mu=np.mean(all_stim_responses[i_p, i_loc,:,:], axis=-1),
                                err=utils.conf_int(all_stim_responses[i_p, i_loc,:,:], axis=-1),
                                x=t_stim,
                                label=f"{p} uW",
                                ax=ax,
                                color=p_color,
                                linewidth=2
                                )
            ax.set_xlim([-5,10])
            ax.set_xlabel("t (s)", fontsize=16)
            ax.set_xticks([0,5])
            plotpanels.make_nice_spines(ax)
            myplt.shade_categorical(catvar=np.concatenate((np.zeros((N_samples_stim)), np.ones((N_samples_stim)), np.zeros((N_samples_stim)))),
                                    x=t_stim, ax=ax, colors=[myplt.WHITE, myplt.BLACK])

        if driver_line == "BPN":
            ax.set_ylim([-5,20])
        else:
            ax.set_ylim([-3,1])
        
        fig.suptitle(f"{driver_line} x CsChrimson", fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_path, f"{driver_line}xCsChrimson_beh_responses.pdf"), dpi=300)

if __name__ == "__main__":
    make_stim_loc_plots()
