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

overwrite_save = False

driver_lines = ["MDN", "BPN"]  # , "AN6"]
base_dirs = ["/mnt/nas2/JB/220816_MDN3xCsChrimson", "/mnt/nas2/JB/230125_BPNxCsChrimson"]  # , "/mnt/nas2/JB/230125_AN6xCsChrimson"]

STIM_PS = [1,5,10,20]
P_COLOURS = [myplt.DARKBLUE, myplt.DARKGREEN, myplt.DARKYELLOW, myplt.DARKRED]
STIM_LOCS = ["thorax", "t2", "t1", "cc", "head"]

fs = 100
N_samples_stim = 500
i_t_stim = [int(-N_samples_stim),int(2*N_samples_stim)]
t_stim = np.arange(i_t_stim[0], i_t_stim[1]) / fs 

for driver_line, base_dir in zip(driver_lines, base_dirs):
    fly_dirs = load.get_flies_from_datedir(base_dir)
    N_flies = len(fly_dirs)

    
    fig, axs = plt.subplots(N_flies,len(STIM_LOCS),figsize=(2.5*len(STIM_LOCS),3*N_flies), sharex=True, squeeze=False)  # , sharey=True)

    for i_fly, fly_dir in enumerate(fly_dirs):
        trial_dirs = load.get_trials_from_fly(fly_dir, contains="plevels")  # fullp low
        trial_dirs = sum(trial_dirs, [])
        video_dirs = [os.path.join(trial_dir, "behData", "images", "camera_5.mp4") for trial_dir in trial_dirs]
        beh_df_dirs = [os.path.join(trial_dir, load.PROCESSED_FOLDER, "beh_df.pkl") for trial_dir in trial_dirs]
        
        n_stim_responses = np.zeros((len(STIM_PS), len(STIM_LOCS)), dtype=int)
        stim_responses = np.zeros((len(STIM_PS), len(STIM_LOCS), len(t_stim), 10))

        for i_t, (trial_dir, beh_df, video_dir) in enumerate(zip(trial_dirs, beh_df_dirs, video_dirs)):
            trial_name = trial_dir.split(os.sep)[-1]
            leg_position_file = os.path.join(trial_dir, load.PROCESSED_FOLDER, "leg_positions_pixels.pkl")
            for i_, stim_loc in enumerate(STIM_LOCS):
                if stim_loc in trial_name:
                    i_loc = i_
                    break

            if not isinstance(beh_df, pd.DataFrame):
                if driver_line == "AN6":
                    beh_df = loaddata.get_beh_df_with_me(fly_dir, all_trial_dirs=[trial_name], add_sleap=True, add_me=False)
                else:
                    beh_df = loaddata.get_beh_df_with_me(fly_dir, all_trial_dirs=[trial_name], add_sleap=False, add_me=False)
                # beh_df = pd.read_pickle(beh_df)
            stim_starts = np.argwhere(np.diff(beh_df["laser_stim"].to_numpy().astype(int))==1).flatten()

            if driver_line == "AN6":
                pass
                # TODO

            for i_stim_start in stim_starts:
                stim_p = int(beh_df.laser_power_uW.values[i_stim_start+2*100])
                i_p = STIM_PS.index(stim_p)
                if driver_line == "AN6":
                    stim_responses[i_p, i_loc, :, n_stim_responses[i_p, i_loc]] = -1* beh_df.hrtita_y_rel_neck.values[i_stim_start+i_t_stim[0]:i_stim_start+i_t_stim[1]]
                else:
                    stim_responses[i_p, i_loc, :, n_stim_responses[i_p, i_loc]] = beh_df.v_forw.values[i_stim_start+i_t_stim[0]:i_stim_start+i_t_stim[1]]
                
                n_stim_responses[i_p, i_loc] += 1

        if len(np.unique(n_stim_responses)) == 2:
            assert 0 in n_stim_responses
        elif len(np.unique(n_stim_responses)) > 2:
            raise NotImplementedError
        # continue # TODO
    # continue # TODO
    
        for i_loc, ax in enumerate(axs[i_fly]):
            for i_p, (p, p_color) in enumerate(zip(STIM_PS, P_COLOURS)):
                if np.sum(np.abs(stim_responses[i_p, i_loc,:,:])) > 0:
                    # ""
                    myplt.plot_mu_sem(mu=gaussian_filter1d(np.mean(stim_responses[i_p, i_loc,:,:], axis=1), sigma=1),
                                    err=gaussian_filter1d(utils.conf_int(stim_responses[i_p, i_loc,:,:], axis=1), sigma=1),
                                    x=t_stim,
                                    label=f"{p} uW",
                                    ax=ax,
                                    color=p_color,
                                    linewidth=2
                                    )
                    # ""
                    # ax.plot(t_stim, gaussian_filter1d(stim_responses[i_p, i_loc,:,:], axis=1, sigma=1), color=p_color, linewidth=2, alpha=0.5)
            # ax.legend(frameon=False, bbox_to_anchor=(-0.9,0.9), fontsize=12)
            # ax.axhline(y=0, color='k', linewidth=2)
            if i_loc == 0:
                if driver_line == "BPN":
                    ax.set_ylabel(f"Fly{i_fly+1}\n"+r"$v_{||}$ (mm/s)", fontsize=16)
                elif driver_line == "AN6":
                    ax.set_ylabel(f"Fly{i_fly+1}\n y hind leg (px)", fontsize=16)
            if i_fly == 0:
                ax.set_title(f"{STIM_LOCS[i_loc]}".upper(), fontsize=16)

            if driver_line == "BPN":
                ax.set_ylim([-5,10])
            elif driver_line == "AN6":
                ax.set_ylim([-200,0])
            else:
                ax.set_ylim([-4,2.5])
            ax.set_xlim([-5,10])
            ax.set_xlabel("t (s)", fontsize=16)
            ax.set_xticks([0,5])
            plotpanels.make_nice_spines(ax)
            myplt.shade_categorical(catvar=np.concatenate((np.zeros((N_samples_stim)), np.ones((N_samples_stim)), np.zeros((N_samples_stim)))),
                                x=t_stim, ax=ax, colors=[myplt.WHITE, myplt.BLACK])

    fig.suptitle(f"{driver_line} x CsChrimson", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(params.plot_base_dir, f"{driver_line}xCsChrimson_beh_responses.pdf"), dpi=300)
    a = 0
    """
    start_frames = []
    for i_t, beh_df in enumerate(beh_df_dirs):
        if not isinstance(beh_df, pd.DataFrame):
            beh_df = pd.read_pickle(beh_df)
        stim_start = np.argwhere(np.diff(beh_df["laser_stim"].to_numpy().astype(int))==1).flatten()
        stim_start = [s-(5*100) for s in stim_start]  # if beh_df["olfac_cond"].to_numpy()[s+2] == "H2O"
        start_frames.append(stim_start)

    videos.make_behaviour_grid_video(
        video_dirs=video_dirs,
        start_frames=start_frames,
        N_frames=15*100,
        stim_range=[5*100,10*100],
        out_dir=base_dir,
        video_name=f"{driver_line}xCsChr_stim_aligned.mp4",
        frame_rate=None,
        size=(120,-1))
    """



