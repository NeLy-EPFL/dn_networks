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


def make_functional_example_stim_video(pre_stim="walk", overwrite=False, subtract_pre=False, clim=[0,1],
                                       show_centers=False, show_mask=False, show_beh=False,
                                       make_fig=False, make_vid=True, select_group_of_flies="presentation", trigger="laser_start"):
    out_dir = os.path.join(params.video_base_dir, "presentation")

    presentation_flies = {
        "DNp09": 46,
        "aDN2": 32,
        "MDN": 50,
        "PR": 8,
    }
    presentation_natbeh_flies_MDN = {
        "MDN": 72,
    }
    presentation_natbeh_flies_aDN2 = {
        "aDN2": 32,
    }
    presentation_natbeh_flies_DNp09 = {
        "DNp09": 15,
    }
    if select_group_of_flies == "presentation":
        selected_flies = presentation_flies
    elif select_group_of_flies == "natbeh_DNp09":
        selected_flies = presentation_natbeh_flies_DNp09
    elif select_group_of_flies == "natbeh_aDN2":
        selected_flies = presentation_natbeh_flies_aDN2
    elif select_group_of_flies == "natbeh_MDN":
        selected_flies = presentation_natbeh_flies_MDN
    else:
        raise NotImplementedError

    stim_ps = {
        "DNp09": [10,20],
        "aDN2": [20],
        "MDN": [10,20],
        "PR": [10,20],
    }
    y_crop_raw = [80,480-80]  # [80+40,480-80-40]
    y_crop_denoised = [0,320]  # [40,320-40]
    x_crop = [0,736]  # [40,736-40]

    df = summarydf.load_data_summary()
    df = summarydf.filter_data_summary(df, no_co2=False)
    df = summarydf.get_selected_df(df, [{"walkon": "ball"}])

    for GAL4, fly_id in selected_flies.items():
        stim_p = stim_ps[GAL4]
        fly_df = df.loc[df.fly_id == fly_id]
        fly_dir = np.unique(fly_df.fly_dir)[0]
        print(fly_dir)
        stack_save = os.path.join(params.plotdata_base_dir, f"{GAL4}_{fly_id}_stacks_filt.tif")
        baseline_save = os.path.join(params.plotdata_base_dir, f"{GAL4}_{fly_id}_baseline.tif")
        qmax_save = os.path.join(params.plotdata_base_dir, f"{GAL4}_{fly_id}_qmax.tif")
        dff_save = os.path.join(params.plotdata_base_dir, f"{GAL4}_{fly_id}_dff.tif")

        roi_centers = loaddata.get_roi_centers(fly_dir)
        roi_mask = utils.get_stack(os.path.join(fly_dir, load.PROCESSED_FOLDER, "ROI_mask.tif"))[y_crop_denoised[0]:y_crop_denoised[1], x_crop[0]:x_crop[1]].astype(bool)
        roi_mask = np.logical_and(binary_dilation(roi_mask), np.logical_not(roi_mask))
        twop_df, beh_df = loaddata.load_data(fly_dir, all_trial_dirs=fly_df.trial_name.values)

        walk_pre, rest_pre, stim_starts_beh = behaviour.get_pre_stim_beh(beh_df, trigger=trigger,
                                                        stim_p=stim_p,
                                                        n_pre_stim=params.pre_stim_n_samples_beh,
                                                        trials=fly_df.trial_name.values,
                                                        return_starts=True)

        # select responses:
        if pre_stim == "walk":
            select_pre_beh = walk_pre
        elif pre_stim == "rest":
            select_pre_beh = rest_pre
        elif pre_stim is None:
            select_pre_beh = np.ones_like(rest_pre)
        elif pre_stim == "not_walk":
            select_pre_beh = np.logical_not(walk_pre)
        else:
            raise NotImplementedError
        stim_starts_beh_sel = np.array(stim_starts_beh)[select_pre_beh]
        stim_starts_twop_sel = beh_df.twop_index.values[stim_starts_beh_sel]
        unique_trials = np.unique(beh_df.index.get_level_values("Trial"))
        sel_trials = np.unique(beh_df.iloc[stim_starts_beh_sel].index.get_level_values("Trial"))
        if any(sel_trials != unique_trials[0]): # using stim starts from later trials -> will have to increase their twop index
            for i_trial in unique_trials[1:]:  # TODO: test this!
                trial_start_twop = np.where(np.diff(twop_df.index.get_level_values("Trial") == i_trial))[0][0] + 1
                stim_starts_twop_sel[beh_df.iloc[stim_starts_beh_sel].index.get_level_values("Trial") == i_trial] += trial_start_twop
        
        if dff_save is not None and os.path.isfile(dff_save) and not overwrite:
            dff = utils.get_stack(dff_save)
        else:
            if stack_save is not None and os.path.isfile(stack_save) and not overwrite:
                stacks = utils.get_stack(stack_save)
            else:
                # load 2p data
                stacks = []
                for trial_dir in tqdm(fly_df.trial_dir.values):
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
            
            # compute resting baseline
            if baseline_save is not None and os.path.isfile(baseline_save) and not overwrite:
                f0_rest = utils.get_stack(baseline_save)
            else:
                if exclude_stim_low_high[0]:
                    f0_rest = baselines.get_resting_baselines(stacks[no_stim,:], twop_df.rest.values[no_stim])
                else:
                    f0_rest = baselines.get_resting_baselines(stacks, twop_df.rest.values)
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

            dff = (stacks - f0_rest) / (f0_qmax - f0_rest)
            if dff_save is not None:
                utils.save_stack(dff_save, dff)

        # get stim trig average
        N_samples_pre = params.n_s_2p_1s
        N_samples_stim = params.n_s_2p_5s
        N_samples_post = params.n_s_2p_5s
        N_samples = N_samples_pre + N_samples_stim + N_samples_post
        stim_trig_stack = np.zeros((N_samples,) + dff.shape[1:], dtype=dff.dtype)

        for stim_start in stim_starts_twop_sel:
            stim_trig_stack += dff[stim_start-N_samples_pre:stim_start+N_samples_stim+N_samples_post,:,:]
        del dff
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
            fig.savefig(os.path.join(out_dir, f"{GAL4}_{fly_id}_neural_resp_avg.pdf"), dpi=300)


        if make_vid:
            generator_2p = videos.generator_dff(stack=stim_trig_stack,
                                    vmin=-1, vmax=1,  # clim[0], vmax=clim[1],
                                    text="",
                                    show_colorbar=True,
                                    colorbarlabel=r"$\frac{\Delta F}{F}$")
            generator_2p = videos.stimulus_dot_generator(generator_2p, start_stim=[N_samples_pre], stop_stim=[N_samples_pre+N_samples_stim])
            if not show_beh:
                videos.make_video(os.path.join(out_dir, f"{GAL4}_{fly_id}_neural_resp_example.mp4"), generator_2p, fps=params.fs_2p, n_frames=N_samples)
            else:
                N_beh_repeats = 4
                N_samples_beh_pre = params.n_s_beh_1s
                N_samples_beh_stim = params.n_s_beh_5s
                N_samples_beh_post = params.n_s_beh_5s
                N_samples_beh = N_samples_beh_pre + N_samples_beh_stim + N_samples_beh_post

                # sel_trials
                i_trial = sel_trials[0]
                if i_trial != unique_trials[0]:
                    raise NotImplementedError
                beh_trial_dir = os.path.join(fly_dir, fly_df.trial_name.values[0])
                beh_video_dir = os.path.join(beh_trial_dir, "behData", "images", "camera_5.mp4")

                beh_generators = videos.make_behaviour_grid_video(
                    video_dirs=[beh_video_dir],
                    start_frames=[stim_starts_beh_sel[:N_beh_repeats]-N_samples_beh_pre],
                    N_frames=N_samples_beh,
                    stim_range=[N_samples_beh_pre, N_samples_beh_pre+N_samples_beh_stim],
                    out_dir=None,
                    video_name=None,
                    frame_rate=params.fs_beh,
                    asgenerator=True,
                    size=(100,-1))

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
                videos.make_video(os.path.join(out_dir, f"{GAL4}_{fly_id}_neural_beh_resp_example_{trigger}.mp4"), generator, fps=params.fs_2p, n_frames=N_samples)



        
def make_stim_resp_figure(stim_trig_stack, start_stim, clim=[-0.8,0.8], n_avg=params.response_n_avg, n_latest_max=params.response_n_latest_max):
    # n_avg=params.response_n_latest_max, clim=[0,1]):
    
    # img = np.mean(stim_trig_stack[start_stim:start_stim+n_avg], axis=0)
    n_avg = n_avg // 2
    i_stim_response_max = np.argmax(np.abs(stim_trig_stack[start_stim:start_stim+n_latest_max]), axis=0) + start_stim
    img = np.zeros_like(stim_trig_stack[0])
    for i_y in range(stim_trig_stack.shape[1]):
        for i_x in range(stim_trig_stack.shape[2]):
            img[i_y, i_x] = np.mean(stim_trig_stack[i_stim_response_max[i_y, i_x]-n_avg:i_stim_response_max[i_y, i_x]+n_avg, i_y, i_x])


    fig = plt.figure(figsize=(4,2))
    mosaic = """
    AAAAAAAA.B
    """
    axd = fig.subplot_mosaic(mosaic)
    
    norm = plt.Normalize(clim[0], clim[1])
    cmap = plt.cm.jet  # params.cmap_ci  # plt.cm.jet
    cmap.set_bad(color="black")

    axd["A"].imshow(img, cmap=cmap, norm=norm)
    axd["A"].axis("off")
    cbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axd["B"],
                         ticks=[clim[0], 0, clim[1]], label=r"%$\frac{\Delta F}{F}$")
    cbar1.set_ticklabels([f"{clim[0]:.1f}", 0, f"{clim[1]:.1f}"])
    cbar1.outline.set_visible(False)
    cbar1.outline.set_linewidth(2)
    cbar1.ax.tick_params(width=2.5)
    cbar1.ax.tick_params(length=5)
    cbar1.ax.tick_params(labelsize=16)

    return fig



if __name__ == "__main__":
    # make_functional_example_stim_video(pre_stim="walk", overwrite=False,
    #     subtract_pre=True, show_centers=False, show_mask=False, show_beh=True,
    #     make_fig=True, make_vid=False, select_group_of_flies="presentation", trigger="laser_start")

    make_functional_example_stim_video(pre_stim="not_walk", overwrite=False,
        subtract_pre=True, show_centers=False, show_mask=False, show_beh=True,
        make_fig=True, make_vid=False, select_group_of_flies="natbeh_DNp09", trigger="laser_start")
    make_functional_example_stim_video(pre_stim="not_walk", overwrite=False,
        subtract_pre=True, show_centers=False, show_mask=False, show_beh=True,
        make_fig=True, make_vid=False, select_group_of_flies="natbeh_DNp09", trigger="walk_trig_start")

    make_functional_example_stim_video(pre_stim=None, overwrite=False,
        subtract_pre=True, show_centers=False, show_mask=False, show_beh=True,
        make_fig=True, make_vid=False, select_group_of_flies="natbeh_aDN2", trigger="laser_start")
    make_functional_example_stim_video(pre_stim=None, overwrite=False,
        subtract_pre=True, show_centers=False, show_mask=False, show_beh=True,
        make_fig=True, make_vid=False, select_group_of_flies="natbeh_aDN2", trigger="groom_trig_start")

    make_functional_example_stim_video(pre_stim=None, overwrite=False,
        subtract_pre=True, show_centers=False, show_mask=False, show_beh=True,
        make_fig=True, make_vid=False, select_group_of_flies="natbeh_MDN", trigger="laser_start")
    make_functional_example_stim_video(pre_stim=None, overwrite=False,
        subtract_pre=True, show_centers=False, show_mask=False, show_beh=True,
        make_fig=True, make_vid=False, select_group_of_flies="natbeh_MDN", trigger="back_trig_start")

    

