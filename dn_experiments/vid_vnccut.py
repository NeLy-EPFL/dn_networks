"""
Module used to create videos from functional imaging data
Author: jonas.braun@epfl.ch
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_dilation
import params, summarydf, loaddata, stimulation, behaviour, plotpanels, baselines

from twoppp.plot import videos
from twoppp import load, utils

out_dir = os.path.join(params.video_base_dir, "revision")

def make_vnc_cut_example_stim_video(trial_dirs, overwrite=False, subtract_pre=True, clim=[0,1],
                                       show_centers=False, show_mask=False, show_beh=False,
                                       make_fig=False, make_vid=True, trigger="laser_start"):
    """
    Generate functional example stimulation video.
    Can also be used to generate natural behaviour videos.
    Also used to generate plots in Figure 2b

    Args:
        pre_stim (str): Pre-stimulation condition ("walk", "rest", "not_walk", None).
        overwrite (bool): Whether to overwrite existing pre-computed data.
        subtract_pre (bool): Whether to subtract the pre-stimulus baseline.
        clim (list): Color limit range for the video.
        show_centers (bool): Whether to show ROI centers.
        show_mask (bool): Whether to show ROI masks.
        show_beh (bool): Whether to show behavior.
        make_fig (bool): Whether to create a figure of the maximum response for this fly (Shown in Figure 2b).
        make_vid (bool): Whether to create a video.
        select_group_of_flies (str): Selected group of flies.
        trigger (str): Stimulation trigger ("laser_start", "olfac_start", etc.).

    Returns:
        None
    """
    y_crop_raw = [80,480-80]  # [80+40,480-80-40]
    y_crop_denoised = [0,320]  # [40,320-40]
    x_crop = [0,736]  # [40,736-40]

    fly_dir, _ = os.path.split(trial_dirs[0])
    fly_name = fly_dir[13:].replace("/", "_")

    stack_save = os.path.join(params.plotdata_base_dir, f"{fly_name}_stacks_filt.tif")
    baseline_save = os.path.join(params.plotdata_base_dir, f"{fly_name}_baseline.tif")
    qmax_save = os.path.join(params.plotdata_base_dir, f"{fly_name}_qmax.tif")
    dff_save = os.path.join(params.plotdata_base_dir, f"{fly_name}_dff.tif")

    # twop_df, beh_df = loaddata.load_data(fly_dir, all_trial_dirs=trial_dirs)
    beh_df = loaddata.get_beh_df_with_me(fly_dir=fly_dir, all_trial_dirs=trial_dirs, add_sleap=False, add_me=False)
    twop_df = loaddata.get_filtered_twop_df(fly_dir=fly_dir, all_trial_dirs=trial_dirs)
    twop_df = stimulation.fix_stim_power_signal(twop_df)
    # twop_df = behaviour.get_beh_info_in_twop_df(beh_df, twop_df)

    stim_starts_beh_sel = stimulation.get_all_stim_starts(beh_df, return_stops=False)
    stim_starts_twop_sel = stimulation.get_all_stim_starts(twop_df, return_stops=False)
    
    if dff_save is not None and os.path.isfile(dff_save) and not overwrite:
        dff = utils.get_stack(dff_save)
    else:
        if stack_save is not None and os.path.isfile(stack_save) and not overwrite:
            stacks = utils.get_stack(stack_save)
        else:
            # load 2p data
            stacks = []
            for trial_dir in trial_dirs:
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "loading stack")
                stack = utils.get_stack(os.path.join(trial_dir, load.PROCESSED_FOLDER, "green_com_warped.tif"))[30:-30,y_crop_raw[0]:y_crop_raw[1],x_crop[0]:x_crop[1]]
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "median filtering stack")
                stack = median_filter(stack, size=(params.neurons_med, params.neurons_med_xy, params.neurons_med_xy))
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "gaussian filtering stack")
                stack = gaussian_filter(stack, sigma=(params.neurons_sigma, params.neurons_sigma_xy, params.neurons_sigma_xy))
                stacks.append(stack)
                del stack
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "concatenating stacks")
            stacks = np.concatenate(stacks, axis=0)
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "saving stack")
            if stack_save is not None:
                utils.save_stack(stack_save, stacks)
        
        # check whether samples should be excluded for baseline computation
        baseline_exclude_stim_range = params.baseline_exclude_stim_range
        exclude_stim_low_high = params.baseline_exclude_stim_low_high
        if baseline_exclude_stim_range is not None:
            stim_starts, stim_stops = stimulation.get_all_stim_starts(twop_df, return_stops=True)
            stim_starts += baseline_exclude_stim_range[0]
            stim_stops += baseline_exclude_stim_range[1]
            stim = np.zeros((len(stacks)), dtype=bool)
            for start, stop in zip(stim_starts, stim_stops):
                stim[start:stop] = True
            no_stim = np.logical_not(stim)
        else:
            no_stim = np.ones((len(stacks)), dtype=bool)
        
        # compute quantile baseline instead of resting baseline
        if baseline_save is not None and os.path.isfile(baseline_save) and not overwrite:
            f0_rest = utils.get_stack(baseline_save)
        else:
            if exclude_stim_low_high[0]:
                # f0_rest = baselines.get_resting_baselines(stacks[no_stim,:], twop_df.rest.values[no_stim])
                f0_rest = np.quantile(stacks[no_stim,:], q=1-params.baseline_qmax, axis=0)
            else:
                # f0_rest = baselines.get_resting_baselines(stacks, twop_df.rest.values)
                f0_rest = np.quantile(stacks, q=1-params.baseline_qmax, axis=0)
            if baseline_save is not None:
                utils.save_stack(baseline_save, f0_rest)

        # compute quantile maxiumum
        if qmax_save is not None and os.path.isfile(qmax_save) and not overwrite:
            f0_qmax = utils.get_stack(qmax_save)
        else:
            if exclude_stim_low_high[1]:
                f0_qmax = np.quantile(stacks[no_stim,:], q=params.baseline_qmax, axis=0)
            else:
                f0_qmax = np.quantile(stacks, q=params.baseline_qmax, axis=0)
            if baseline_save is not None:
                utils.save_stack(qmax_save, f0_qmax)
        
        if stacks.shape[0] < 50000:
            dff = (stacks - f0_rest) / (f0_qmax - f0_rest)
            if dff_save is not None:
                utils.save_stack(dff_save, dff)
        else:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "den = f0_qmax - f0_rest")
            den = f0_qmax.astype(np.float16) - f0_rest.astype(np.float16)
            del f0_qmax
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "num = stacks - f0_rest")
            num = stacks.astype(np.float16) - f0_rest.astype(np.float16)
            del stacks, f0_rest
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "dff = num / den")
            dff = np.divide(num, den, dtype=np.float16)
            del num, den
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "save dff")
        
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "dff handling complete")

    # get stim trig average
    N_samples_pre = params.n_s_2p_1s
    N_samples_stim = params.n_s_2p_5s
    N_samples_post = 1  # params.n_s_2p_5s
    N_samples = N_samples_pre + N_samples_stim + N_samples_post
    stim_trig_stack = np.zeros((N_samples,) + dff.shape[1:], dtype=dff.dtype)

    for stim_start in stim_starts_twop_sel:
        stim_trig_stack += dff[stim_start-N_samples_pre:stim_start+N_samples_stim+N_samples_post,:,:]
    del dff
    stim_trig_stack = stim_trig_stack.astype(np.float32)
    stim_trig_stack /= len(stim_starts_twop_sel)

    # subtract baseline
    if subtract_pre:
        N_baseline_pre = params.response_n_baseline
        stim_trig_stack -= np.mean(stim_trig_stack[N_samples_pre-N_baseline_pre: N_samples_pre,:,:], axis=0)

    if show_centers:
        for roi_center in roi_centers:
            x = roi_center[1]-x_crop[0]
            y = roi_center[0]-y_crop_denoised[0]
            stim_trig_stack[:,y-1:y+1,x-1:x+1] = np.nan
    if show_mask:
        stim_trig_stack[np.repeat(roi_mask[np.newaxis,:,:], repeats=N_samples, axis=0)] = np.nan

    if make_fig:
        fig = make_stim_resp_figure(stim_trig_stack=stim_trig_stack, start_stim=N_samples_pre)
        fig.savefig(os.path.join(out_dir, f"{fly_name}_neural_resp_avg_{trigger}.pdf"), dpi=300)


    if make_vid:
        generator_2p = videos.generator_dff(stack=stim_trig_stack,
                                vmin=-0.8, vmax=0.8,  # clim[0], vmax=clim[1],
                                text="",
                                show_colorbar=True,
                                colorbarlabel=r"$\frac{\Delta F}{F}$")
        if trigger == "laser_start":
            color = (255,0,0)
            generator_2p = videos.stimulus_dot_generator(generator_2p, start_stim=[N_samples_pre], stop_stim=[N_samples_pre+N_samples_stim], color=color)  # red dot
        elif trigger == "olfac_start":
            color = (255,255,255)
            generator_2p = videos.stimulus_dot_generator(generator_2p, start_stim=[N_samples_pre], stop_stim=[N_samples_pre+N_samples_stim], color=color)  # blue dot
        else:
            color = (255,255,255)
            generator_2p = videos.stimulus_dot_generator(generator_2p, start_stim=[N_samples_pre], stop_stim=[N_samples_pre+params.n_s_2p_1s], color=color)  # only show green dot for 1s for natural behaviour
        if not show_beh:
            videos.make_video(os.path.join(out_dir, f"{fly_name}_neural_resp_example_{trigger}.mp4"), generator_2p, fps=params.fs_2p, n_frames=N_samples)
        else:
            N_beh_repeats = 4
            N_samples_beh_pre = params.n_s_beh_1s
            N_samples_beh_stim = params.n_s_beh_5s
            N_samples_beh_post = 10  # params.n_s_beh_5s
            N_samples_beh = N_samples_beh_pre + N_samples_beh_stim + N_samples_beh_post

            # sel_trials
            # i_trial = sel_trials[0]
            beh_start_trials = beh_df.iloc[stim_starts_beh_sel].index.get_level_values("Trial")
            for i_trial in sel_trials:
                if np.sum(beh_start_trials == i_trial) >= N_beh_repeats:
                    break


            # if i_trial != unique_trials[0]:
            #     raise NotImplementedError
            beh_trial_dir = trial_dir
            beh_video_dir = os.path.join(beh_trial_dir, "behData", "images", "camera_5.mp4")
            if trigger == "laser_start" or trigger == "olfac_start":
                stim_range = [N_samples_beh_pre, N_samples_beh_pre+N_samples_beh_stim]
            else:
                stim_range = [N_samples_beh_pre, N_samples_beh_pre+params.n_s_beh_1s]  # only show dot for 1s for natural behaviour
            beh_start_df = beh_df.iloc[stim_starts_beh_sel]
            beh_start_df = beh_start_df[beh_start_df.index.get_level_values("Trial") == i_trial]
            beh_start_frames = beh_start_df.index.get_level_values("Frame").values
            # beh_start_frames = beh_df.index.get_level_values("Frame")[stim_starts_beh_sel].values
            beh_generators = videos.make_behaviour_grid_video(
                video_dirs=[beh_video_dir],
                start_frames=[beh_start_frames[:N_beh_repeats]-N_samples_beh_pre],
                N_frames=N_samples_beh,
                stim_range=stim_range,
                out_dir=None,
                video_name=None,
                frame_rate=params.fs_beh,
                asgenerator=True,
                size=(100,-1),
                color=color,
                brighter=2.5 if "wheel" in beh_trial_dir else None)


            t_beh = (np.arange(N_samples_beh) - N_samples_beh_pre) / params.fs_beh
            t_2p = (np.arange(N_samples) - N_samples_pre) / params.fs_2p
            beh_samples = []
            for this_t_2p in t_2p:
                closest_beh_sample = np.argmin(np.abs(t_beh-this_t_2p))
                beh_samples.append(closest_beh_sample)
            beh_generators = [videos.utils_video.generators.resample(beh_generator, beh_samples) for beh_generator in beh_generators]

            rows = [generator_2p]
            for i in range(N_beh_repeats//2):
                i_start = i * 2
                rows.append(videos.utils_video.generators.stack(beh_generators[i_start:i_start+2], axis=1))
            generator = videos.utils_video.generators.stack(rows, axis=0)
            videos.make_video(os.path.join(out_dir, f"{fly_name}_neural_beh_resp_example_{trigger}.mp4"), generator, fps=params.fs_2p, n_frames=N_samples)


def assemble_vnc_cut_videos(fly_names, genotypes=["DNp09", "control"], video_name="vnc_cut_summary.mp4", fps=params.fs_2p):
    def crop_generator(generator, size):
        for i, item in enumerate(generator):
            yield item[:size[0], :size[1], :]

    def blacken_generator(generator, size):
        for i, item in enumerate(generator):
            item[:, size[1]:, :] = 0
            yield item

    all_generators = []
    for i_gen, (genotype, genotype_fly_names) in enumerate(zip(genotypes, fly_names)):
        genotype_generators = []
        for i_fly, fly_name in enumerate(genotype_fly_names):
            video_dir = os.path.join(out_dir, fly_name+"_neural_resp_example_laser_start.mp4")
            gen = videos.generator_video(video_dir)
            gen = videos.utils_video.generators.add_text(gen, text=f"Fly {i_fly+1}", pos=(10,300), color=(0,0,0))
            if i_fly == 0:
                gen = videos.utils_video.generators.add_text(gen, text=genotype, pos=(320,40), color=(0,0,0))
            if i_gen == 0:
                gen = crop_generator(gen, size=(320,746))
            elif i_fly < len(genotype_fly_names) - 1:
                gen = blacken_generator(gen, size=(320,746))
                # gen = videos.utils_video.generators.static_image(np.zeros((320,746,3), dtype=np.uint8), n_frames=98)
                # crop_generator(gen, size=(320,746))
                # gen_black = videos.utils_video.generators.static_image(np.zeros((320,116,3), dtype=np.uint8), n_frames=98)
                # gen = videos.utils_video.generators.stack((gen,gen_black), axis=1)
            
            genotype_generators.append(gen)
            genotype_generators.append(videos.utils_video.generators.static_image(np.zeros((10,960,3), dtype=np.uint8), n_frames=98))
        all_generators.append(videos.utils_video.generators.stack(genotype_generators, axis=0))

    generator = videos.utils_video.generators.stack(all_generators, axis=1)
    videos.make_video(os.path.join(out_dir, video_name), generator, fps=fps, n_frames=-1)


if __name__ == "__main__":
    """
    make_vnc_cut_example_stim_video(
        trial_dirs = ["/mnt/nas2/JB/231025_DfdxGCaMP6s_DNp09xCsChrimson/Fly1/002_xz_cc_p10_vnccut",
                      ]
    )
    make_vnc_cut_example_stim_video(
        trial_dirs = ["/mnt/nas2/JB/231025_DfdxGCaMP6s_DNp09xCsChrimson/Fly2/001_xz_cc_p10_vnccut",
                      ]
    )
    
    make_vnc_cut_example_stim_video(
        trial_dirs = ["/mnt/nas2/JB/231114_DfdxGCaMP6s_DNp09xCsChrimson/Fly1/001_xz_cc_p10_vnccut",
                      ]
    )
    
    make_vnc_cut_example_stim_video(
        trial_dirs = ["/mnt/nas2/JB/240109_DfdxGCaMP6s_PRxCsChrimson/Fly1/001_xz_cc_p10_vnccut",
                      ]
    )
    make_vnc_cut_example_stim_video(
        trial_dirs = ["/mnt/nas2/JB/240109_DfdxGCaMP6s_PRxCsChrimson/Fly2/001_xz_cc_p10_vnccut",
                      ]
    )
    make_vnc_cut_example_stim_video(
        trial_dirs = ["/mnt/nas2/JB/240109_DfdxGCaMP6s_PRxCsChrimson/Fly3/001_xz_cc_p10_vnccut",
                      ]
    )
    """
    fly_names = [
        ["231025_DfdxGCaMP6s_DNp09xCsChrimson_Fly1", "231025_DfdxGCaMP6s_DNp09xCsChrimson_Fly2", "231114_DfdxGCaMP6s_DNp09xCsChrimson_Fly1"],
        ["240109_DfdxGCaMP6s_PRxCsChrimson_Fly1", "240109_DfdxGCaMP6s_PRxCsChrimson_Fly2", "240109_DfdxGCaMP6s_PRxCsChrimson_Fly3"]
    ]
    assemble_vnc_cut_videos(fly_names=fly_names, genotypes=["DNp09", "control"])