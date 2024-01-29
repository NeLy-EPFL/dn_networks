"""
Module to generate figures related to functional imaging during optogenetic stimulation.
Author: jonas.braun@epfl.ch
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import mannwhitneyu
import pickle
from tqdm import tqdm

import params, summarydf, loaddata, stimulation, behaviour, plotpanels

from twoppp import plot as myplt
presentation_flies = {
    "MDN": 50,
    "DNp09": 46,
    "aDN2": 32,
    "PR": 8,
}
presentation_natbeh_flies = {
    "MDN": 72,
    "DNp09": 15,
    "aDN2": 32,
}

mosaic_stim_resp_panel = """
..................................................
VVVVVVVVVVVVVVVV.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
VVVVVVVVVVVVVVVV.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
VVVVVVVVVVVVVVVV.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
BBBBBBBBBBBBBBBB.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
BBBBBBBBBBBBBBBB.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
BBBBBBBBBBBBBBBB.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
"""

mosaic_vnccut_stim_resp_panel = """
..................................................
IIIIIIIIIIIIIIII.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
IIIIIIIIIIIIIIII.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
IIIIIIIIIIIIIIII.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
IIIIIIIIIIIIIIII.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
JJJJJJJJJJJJJJJJ.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
JJJJJJJJJJJJJJJJ.NNNNNNNNL....SSSSSSSSSSSSSSSSSSSS
"""

mosaic_stim_resp_panel_presentation = """
VVVVVVVVVVVVVV.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
VVVVVVVVVVVVVV.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
VVVVVVVVVVVVVV.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
...............NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
BBBBBBBBBBBBBB.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
BBBBBBBBBBBBBB.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
BBBBBBBBBBBBBB.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
...............................................
"""

mosaic_vnccut_stim_resp_panel_presentation = """
IIIIIIIIIIIIII.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
IIIIIIIIIIIIII.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
IIIIIIIIIIIIII.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
IIIIIIIIIIIIII.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
IIIIIIIIIIIIII.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
IIIIIIIIIIIIII.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
IIIIIIIIIIIIII.NNNNNNNNL...SSSSSSSSSSSSSSSSSSSS
...............................................
"""

mosaic_stim_resp_panel_presentationsummary = """
....VVVVVVVVVVVVVVVVVVVVVV.....
....VVVVVVVVVVVVVVVVVVVVVV.....
....VVVVVVVVVVVVVVVVVVVVVV.....
....VVVVVVVVVVVVVVVVVVVVVV.....
...............................
....BBBBBBBBBBBBBBBBBBBBBB.....
....BBBBBBBBBBBBBBBBBBBBBB.....
....BBBBBBBBBBBBBBBBBBBBBB.....
....BBBBBBBBBBBBBBBBBBBBBB.....
...............................
...............................
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
"""

mosaic_vnccut_stim_resp_panel_presentationsummary = """
....IIIIIIIIIIIIIIIIIIIIII.....
....IIIIIIIIIIIIIIIIIIIIII.....
....IIIIIIIIIIIIIIIIIIIIII.....
....IIIIIIIIIIIIIIIIIIIIII.....
....IIIIIIIIIIIIIIIIIIIIII.....
....IIIIIIIIIIIIIIIIIIIIII.....
....IIIIIIIIIIIIIIIIIIIIII.....
....IIIIIIIIIIIIIIIIIIIIII.....
....IIIIIIIIIIIIIIIIIIIIII.....
...............................
...............................
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
SSSSSSSSSSSSSSSSSSSSSSSSSSL....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
DDDDDDDDDDDDDDDDDDDDDDDDDD.....
"""

mosaic_nat_resp_panel = """
VVVVVVVVVVVVVVVV.NNNNNNNNMMMMMMMML...SSSSSSSSSSSSSSSSSSSSSSSS
VVVVVVVVVVVVVVVV.NNNNNNNNMMMMMMMML...SSSSSSSSSSSSSSSSSSSSSSSS
VVVVVVVVVVVVVVVV.NNNNNNNNMMMMMMMML...SSSSSSSSSSSSSSSSSSSSSSSS
BBBBBBBBBBBBBBBB.NNNNNNNNMMMMMMMML...SSSSSSSSSSSSSSSSSSSSSSSS
BBBBBBBBBBBBBBBB.NNNNNNNNMMMMMMMML...SSSSSSSSSSSSSSSSSSSSSSSS
BBBBBBBBBBBBBBBB.NNNNNNNNMMMMMMMML...SSSSSSSSSSSSSSSSSSSSSSSS
.............................................................

"""

def align_roi_centers(roi_centers, x_target=[50,736-50], y_target=[25,320-25]):
    """
    Aligns the ROI centers to a target coordinate system defined by `x_target` and `y_target`.
    Considers the two most dorsal/ventral neurons for y axis alignment and the two most lateral neurons for x-axis alignment.

    Parameters:
        roi_centers (numpy.ndarray): A 2D numpy array containing ROI centers as (y, x) coordinates.
        x_target (list): Target x-coordinate range for alignment. Defaults to [50, 686].
        y_target (list): Target y-coordinate range for alignment. Defaults to [25, 295].

    Returns:
        numpy.ndarray: Aligned ROI centers.
    """
    roi_centers = np.array(roi_centers)
    if not roi_centers.size: # empty array
        return roi_centers
    y_min = np.min(roi_centers[:,0])
    y_max = np.max(roi_centers[:,0])
    x_min = np.min(roi_centers[:,1])
    x_max = np.max(roi_centers[:,1])
    y_c = np.mean([y_min, y_max]).astype(int)
    x_c = np.mean([x_min, x_max]).astype(int)
    y_s = np.diff(y_target) / (y_max - y_min)
    x_s = np.diff(x_target) / (x_max - x_min)
    y_c_target = np.mean(y_target).astype(int)
    x_c_target = np.mean(x_target).astype(int)
    
    roi_centers_corr = np.zeros_like(roi_centers)
    roi_centers_corr[:,0] = (roi_centers[:,0] - y_c) * y_s + y_c_target
    roi_centers_corr[:,1] = (roi_centers[:,1] - x_c) * x_s + x_c_target
    
    return roi_centers_corr

def get_one_fly_stim_resp_panel(fig, axd, fly_data, figure_params):
    """
    Generates a panel of plots related to stimulation response for a single fly.

    Parameters:
        fig (matplotlib.figure.Figure): The matplotlib figure to which the plots will be added.
        axd (dict): A dictionary of axes for different subplots.
        fly_data (dict): Data for the fly.
        figure_params (dict): Parameters for configuring the figure.

    Returns:
        None
    """
    fly_name = f"{fly_data['fly_df'].date.values[0]} Fly {fly_data['fly_df'].fly_number.values[0]}| " + \
               f"ID {fly_data['fly_df'].fly_id.values[0]}| {fly_data['fly_df'].CsChrimson.values[0]}"

    clim = fly_data["clim"] if figure_params["response_clim"] is None else figure_params["response_clim"]

    if figure_params["mode"] == "presentation":
        response_name = ""
        summary_title = ""
    else:
        response_name = f"{fly_data['n_sel_responses']} responses after {figure_params['pre_stim'].replace('_', ' ') if figure_params['pre_stim'] is not None else None}ing" #  +\
        #             f" ({fly_data['n_responses']} tot. {fly_data['n_other_responses']} other)"
        summary_title = None
    
    # V: volocity response
    if "V" in axd.keys():
        plotpanels.plot_ax_behavioural_response(fly_data["beh_responses"], ax=axd["V"],
                response_name=response_name, response_ylabel=figure_params["beh_response_ylabel"],
                ylim=figure_params["response_beh_lim"])
    
    # B: behavioural class
    if "B" in axd.keys():
        plotpanels.plot_ax_behprob(fly_data["beh_class_responses"], ax=axd["B"])

    try:
        # I: image of VNC cut:
        if "I" in axd.keys():
            plotpanels.plot_vnccut(fly_data, ax=axd["I"], mean="z", show_title=False)
        # J: image of VNC cut (coronal view):
        if "J" in axd.keys():
            plotpanels.plot_vnccut(fly_data, ax=axd["J"], mean="x", show_title=False)
    except FileNotFoundError:
        if not figure_params["force_vnc_images"]:
            print(f"Warning: not showing VNC images for fly {fly_name} because data not present.")
        else:
            raise FileNotFoundError(f"Error: Could not find VNC images for fly {fly_name}.")
    
    # N: all neurons matrix with confidence interval
    plotpanels.plot_ax_allneurons_confidence(fly_data["stim_responses"], ax=axd["N"],
        clim=fly_data["clim"], sort_ind=fly_data["sort_ind"])

    # L: legend colour bar
    plotpanels.plot_ax_cbar(fig=fig, ax=axd["L"], clim=clim, clabel=figure_params["clabel"])

    # S: summary of neurons over std image
    plotpanels.plot_ax_response_summary(background_image=fly_data["background_image"],
        roi_centers=fly_data["roi_centers"],
        ax=axd["S"], response_values=fly_data["response_values"],
        response_name="",  # response_name,
        fly_name=fly_name,
        q_max=figure_params["response_q_max"], clim=clim,
        title=summary_title)

def get_one_fly_nat_resp_panel(fig, axd, fly_data, figure_params):
    """
    Generates a panel of plots to compare stimulation respones and natural behavior response for a single fly.

    Parameters:
        fig (matplotlib.figure.Figure): The matplotlib figure to which the plots will be added.
        axd (dict): A dictionary of axes for different subplots.
        fly_data (dict): Data for the fly.
        figure_params (dict): Parameters for configuring the figure.

    Returns:
        None
    """
    clim = fly_data["clim"] if figure_params["response_clim"] is None else figure_params["response_clim"]

    beh_name = figure_params['trigger_2'].split('_')[0]
    GAL4 = fly_data['fly_df'].CsChrimson.values[0]
    if GAL4 == "DNP9":
        GAL4 = "DNp09"

    if figure_params["mode"] == "presentation":
        response_name = ""
        summary_title = ""
        fly_name = ""
    else:
        response_name = f"{fly_data['n_responses']} stim (walk pre: {fly_data['n_other_responses']}, rest pre: {fly_data['n_sel_responses']}) \n" + \
                    f"{fly_data['nat_n_responses']} nat beh (walk pre: {fly_data['nat_n_other_responses']}, rest pre {fly_data['nat_n_sel_responses']})"
        summary_title = f"left {figure_params['suptitle']} stim,\n right {beh_name} | fly ID {fly_data['fly_df'].fly_id.values[0]}"
        fly_name = f"{fly_data['fly_df'].date.values[0]} Fly {fly_data['fly_df'].fly_number.values[0]}"

    # V: volocity response
    if beh_name == "walk":
        beh_color = myplt.DARKGREEN
        beh_title = "walking"
    elif beh_name == "back":
        beh_color = myplt.DARKCYAN
        beh_title = "backward"
    elif beh_name == "groom" or beh_name == "olfac":
        beh_color = myplt.DARKRED
        beh_title = "grooming"
    else:
        raise NotImplementedError
    plotpanels.plot_ax_behavioural_response(fly_data["beh_responses"], ax=axd["V"],
            response_name=response_name, response_ylabel=figure_params["beh_response_ylabel"],
            beh_responses_2=fly_data["nat_beh_responses"], beh_response_2_color=beh_color)

    # B: behavioural class
    plotpanels.plot_ax_behprob(fly_data["beh_class_responses"], ax=axd["B"],
            labels_2=fly_data["nat_beh_class_responses"], beh_2=beh_name, ylabel=f"{beh_title}\nprobability")
    
    # N: all neurons matrix with confidence interval
    plotpanels.plot_ax_allneurons_confidence(fly_data["stim_responses"], ax=axd["N"],
        clim=clim, sort_ind=fly_data["sort_ind"], title=f"{GAL4} stim")

    # M: all neurons matrix with confidence interval for natural behaviour
    plotpanels.plot_ax_allneurons_confidence(fly_data["nat_responses"], ax=axd["M"],
        clim=clim, sort_ind=fly_data["sort_ind"], title=f"natural {beh_title}", ylabel="")

    # L: legend colour bar
    plotpanels.plot_ax_cbar(fig=fig, ax=axd["L"], clim=clim, clabel=figure_params["clabel"])


    # S: summary of neurons over std image
    plotpanels.plot_ax_response_summary(background_image=fly_data["background_image"],
        roi_centers=fly_data["roi_centers"],
        ax=axd["S"], response_values=None,
        response_values_left=fly_data["response_values"], response_values_right=fly_data["nat_response_values"],
        response_name=summary_title, title=summary_title,
        fly_name=fly_name,
        q_max=figure_params["response_q_max"], clim=clim)

def summarise_stim_resp(exp_df, figure_params, stim_resp_save=None, overwrite=False):
    """
    Summarize neural responses and behavioral data for multiple flies during stimulus-triggered averaging.

    Parameters:
        exp_df (pandas.DataFrame): Dataframe containing information about the experiments and flies.
        figure_params (dict): A dictionary containing various figure parameters and settings.
        stim_resp_save (str, optional): Path to save/load cached data. Default is None.
        overwrite (bool, optional): Whether to overwrite cached data if it exists. Default is False.

    Returns:
        matplotlib.figure.Figure: The generated summary figure.
    """
    if stim_resp_save is None:
        tmpdata_path = None
    else:
        tmpdata_path = os.path.dirname(stim_resp_save)

    if stim_resp_save is not None and os.path.isfile(stim_resp_save) and not overwrite:
        with open(stim_resp_save, "rb") as f:
            all_fly_data = pickle.load(f)
    else:
        # load data for all flies
        base_fly_data = {
            "fly_df": None,
            "fly_id": None,
            "fly_dir": None,
            "trial_names": None,
            "backgound_image": None,
            "roi_centers": None,
            "stim_responses": None,
            "response_values": None,
            "sort_ind": None,
            "clim": None,
            "beh_responses": None,
            "beh_class_responses": None,
            "n_responses": None,
            "n_sel_responses": None,
            "n_other_responses": None,
            "vnccut": False,
        }

        all_fly_data = []

        for i_fly, (fly_id, fly_df) in enumerate(exp_df.groupby("fly_id")):
            # stim_resp_fly_save = stim_resp_save[:-1] + str(fly_id) + stim_resp_save[-4:]
            # if stim_resp_fly_save is not None and os.path.isfile(stim_resp_fly_save) and not overwrite>=2:
            #     with open(stim_resp_fly_save, "rb") as f:
            #         fly_data = pickle.load(f)
            if "selected_fly_ids" in list(figure_params.keys()):
                if not fly_id in figure_params["selected_fly_ids"]:
                    continue
            fly_data = base_fly_data.copy()
            fly_data["fly_df"] = fly_df
            fly_data["fly_dir"] = np.unique(fly_df.fly_dir)[0]
            fly_data["trial_names"] = fly_df.trial_name.values
            # TODO: work out difference between all trials and selected trials
            fly_data["background_image"] = loaddata.get_background_image(fly_data["fly_dir"])
            fly_data["roi_centers"] = loaddata.get_roi_centers(fly_data["fly_dir"])
            if "vnccut" in list(figure_params.keys()):
                if figure_params["vnccut"]:
                    fly_data["vnccut"] = True
            if fly_data["vnccut"]:
                twop_df, beh_df = loaddata.load_data(fly_data["fly_dir"], all_trial_dirs=fly_data["trial_names"],
                                                     add_sleap=False, add_me=False, vnccut_mode=True)
            else:
                twop_df, beh_df = loaddata.load_data(fly_data["fly_dir"], all_trial_dirs=fly_data["trial_names"]) 
            twop_df, beh_df = loaddata.load_data(fly_data["fly_dir"], all_trial_dirs=fly_data["trial_names"]) 
                twop_df, beh_df = loaddata.load_data(fly_data["fly_dir"], all_trial_dirs=fly_data["trial_names"]) 

            all_stim_responses, all_beh_responses = stimulation.get_neural_responses(twop_df, figure_params["trigger"],
                                                                        trials=fly_data["trial_names"],
                                                                        stim_p=figure_params["stim_p"],
                                                                        return_var=figure_params["return_var"],
                                                                        neural_regex=figure_params["normalisation_type"])
            fly_data["n_responses"] = all_stim_responses.shape[-1]
            if fly_data["vnccut"]:
                all_beh_class_responses = None
                if figure_params["pre_stim"] is not None:
                    raise NotImplementedError("For vnccut experiments, 'pre_stim' must be None.")
            else:
                all_beh_class_responses = stimulation.get_beh_class_responses(beh_df, figure_params["trigger"],
                                                                            trials=fly_data["trial_names"],
                                                                            stim_p=figure_params["stim_p"])
            
                walk_pre, rest_pre = behaviour.get_pre_stim_beh(beh_df, trigger=figure_params["trigger"],
                                                                stim_p=figure_params["stim_p"],
                                                                n_pre_stim=params.pre_stim_n_samples_beh,
                                                                trials=fly_data["trial_names"])  # selected_trials
            del twop_df, beh_df

            # select responses:
            if figure_params["pre_stim"] == "walk":
                select_pre = walk_pre
                fly_data["n_other_responses"] = np.sum(rest_pre)
            elif figure_params["pre_stim"] == "rest":
                select_pre = rest_pre
                fly_data["n_other_responses"] = np.sum(walk_pre)
            elif figure_params["pre_stim"] == "not_walk":
                select_pre = np.logical_not(walk_pre)
                fly_data["n_other_responses"] = np.sum(walk_pre)
            elif figure_params["pre_stim"] is None:
                select_pre = np.ones((fly_data["n_responses"]), dtype=bool)
                fly_data["n_other_responses"] = 0
            else:
                raise NotImplementedError

            fly_data["n_sel_responses"] = np.sum(select_pre)
            if fly_data["n_sel_responses"] < figure_params["min_resp"]:
                continue
            
            fly_data["stim_responses"] = all_stim_responses[:,:,select_pre]
            fly_data["response_values"] = stimulation.summarise_responses(fly_data["stim_responses"])
            fly_data["sort_ind"] = np.argsort(fly_data["response_values"])
            fly_data["beh_responses"] = all_beh_responses[:,select_pre] if all_beh_responses is not None else None
            fly_data["beh_class_responses"] = all_beh_class_responses[:,select_pre] if all_beh_class_responses is not None else None
            if figure_params["response_clim"] is None:
                try:
                    fly_data["clim"] = np.quantile(np.abs(fly_data["response_values"]), q=figure_params["response_q_max"])
                except IndexError:
                    fly_data["clim"] = 1
            else:
                fly_data["clim"] = figure_params["response_clim"]
            if fly_data["clim"] < 0.5:
                fly_data["clim"] = 0.5

            all_fly_data.append(fly_data.copy())
            del fly_data

        if stim_resp_save is not None:
            with open(stim_resp_save, "wb") as f:
                pickle.dump(all_fly_data, f)

    collect_data_stat_comparison_activation(all_fly_data, pre_stim=figure_params["pre_stim"], overwrite=overwrite, tmpdata_path=tmpdata_path)
    
    if figure_params["mode"] == "presentationsummary":
        fig = plt.figure(figsize=(figure_params["panel_size"][0],figure_params["panel_size"][1]))  # layout="constrained"
        subfigs = fig.subfigures(nrows=1, ncols=1, squeeze=False)[:,0]
    else:
        if figure_params["mode"] == "presentation":
            nrows = 3
        else:
            nrows = len(all_fly_data) + 2
        fig = plt.figure(figsize=(figure_params["panel_size"][0],figure_params["panel_size"][1]*nrows))  # layout="constrained"
        subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=False)[:,0]

    _ = [subfig.patch.set_alpha(0) for subfig in subfigs]
    mosaic = figure_params["mosaic"]
    axds = [subfig.subplot_mosaic(mosaic) for subfig in subfigs]

    i_plot = -1
    for i_fly, fly_data in enumerate(tqdm(all_fly_data)):
        if figure_params["mode"] == "presentation":
            if fly_data["fly_df"].fly_id.values[0] != figure_params["pres_fly"]:
                continue
        elif figure_params["mode"] == "presentationsummary":
            continue
        i_plot += 1
        axd = axds[i_plot]
        get_one_fly_stim_resp_panel(fig, axd, fly_data, figure_params)

    if figure_params["mode"] == "presentationsummary":
        axd_summary = axds[0]
        ax_density = axd_summary["D"]
        summary_title = ""
    else:
        axd_summary = axds[-2]
        axd_summary["N"].axis("off")
        ax_density = axds[-1]["S"]
        if "B" in axds[-1].keys():
            axds[-1]["B"].axis("off")
        if "V" in axds[-1].keys():
            axds[-1]["V"].axis("off")
        if "I" in axds[-1].keys():
            axds[-1]["I"].axis("off")
        if "J" in axds[-1].keys():
            axds[-1]["J"].axis("off")
        axds[-1]["N"].axis("off")
        axds[-1]["L"].axis("off")
        summary_title = f"N = {len(all_fly_data)} flies"

    if not any([fly_data["vnccut"] for fly_data in all_fly_data]):
        # V: all fly volocity response
        beh_responses = np.concatenate([fly_data["beh_responses"] for fly_data in all_fly_data], axis=1)
        plotpanels.plot_ax_behavioural_response(beh_responses, ax=axd_summary["V"],
                response_name=f"N = {len(all_fly_data)} flies", response_ylabel=figure_params["beh_response_ylabel"],
                ylim=figure_params["response_beh_lim"])
        
        # B: all fly behavioural class
        beh_class_responses = np.concatenate([fly_data["beh_class_responses"] for fly_data in all_fly_data], axis=1)
        plotpanels.plot_ax_behprob(beh_class_responses, ax=axd_summary["B"])

    
    # L: all fly legend colour bar
    plotpanels.plot_ax_cbar(fig=fig, ax=axd_summary["L"], clim=0.8, clabel=figure_params["clabel"])

    # S: all fly summary of neurons over std image
    all_roi_centers = np.concatenate([align_roi_centers(fly_data["roi_centers"]) for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)
    all_response_values = np.concatenate([fly_data["response_values"] for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)
    all_n_response = np.array([fly_data["n_sel_responses"] for fly_data in all_fly_data if len(fly_data["roi_centers"])])
    plotpanels.plot_ax_response_summary(background_image=np.zeros_like(all_fly_data[0]["background_image"]),
        roi_centers=all_roi_centers,
        ax=axd_summary["S"], response_values=all_response_values,
        response_name="",  # response_name,
        title=summary_title,
        q_max=None, clim=0.8,
        min_dot_size=50, max_dot_size=50, min_dot_alpha=0.25,max_dot_alpha=0.5, crop_x=0)

    plotpanels.plot_ax_multi_fly_response_density(
        roi_centers=all_roi_centers,
        response_values=all_response_values,
        background_image=np.zeros_like(all_fly_data[0]["background_image"]),
        ax=ax_density,
        n_flies=len(all_n_response),
        clim=0.8,)

    fig.suptitle(figure_params["suptitle"], fontsize=30, y=1-0.01*5/(len(subfigs)))
    return fig


def summarise_natbeh_resp(exp_df, figure_params, stim_resp_save=None, overwrite=False):
    """
    Summarize neural responses and behavioral data for multiple flies during optogenetic stimulation with natural behaviors.

    Parameters:
        exp_df (pandas.DataFrame): Dataframe containing information about the experiments and flies.
        figure_params (dict): A dictionary containing various figure parameters and settings.
        stim_resp_save (str, optional): Path to save/load cached data. Default is None.
        overwrite (bool, optional): Whether to overwrite cached data if it exists. Default is False.

    Returns:
        matplotlib.figure.Figure: The generated summary figure.
    """
    if stim_resp_save is not None and os.path.isfile(stim_resp_save) and not overwrite:
        with open(stim_resp_save, "rb") as f:
            all_fly_data = pickle.load(f)
    else:
        # load data for all flies
        base_fly_data = {
            "fly_df": None,
            "fly_id": None,
            "fly_dir": None,
            "trial_names": None,
            "backgound_image": None,
            "roi_centers": None,
            "stim_responses": None,
            "nat_responses": None,
            "response_values": None,
            "nat_response_values": None,
            "sort_ind": None,
            "clim": None,
            "beh_responses": None,
            "nat_beh_responses": None,
            "beh_class_responses": None,
            "nat_beh_class_responses": None,
            "n_responses": None,
            "nat_n_responses": None,
            "n_sel_responses": None,
            "nat_n_sel_responses": None,
            "n_other_responses": None,
            "nat_n_other_responses": None,
        }

        all_fly_data = []

        for i_fly, (fly_id, fly_df) in enumerate(exp_df.groupby("fly_id")):
            # stim_resp_fly_save = stim_resp_save[:-1] + str(fly_id) + stim_resp_save[-4:]
            # if stim_resp_fly_save is not None and os.path.isfile(stim_resp_fly_save) and not overwrite>=2:
            #     with open(stim_resp_fly_save, "rb") as f:
            #         fly_data = pickle.load(f)
            # else:
            fly_data = base_fly_data.copy()
            fly_data["fly_df"] = fly_df
            fly_data["fly_dir"] = np.unique(fly_df.fly_dir)[0]
            fly_data["trial_names"] = fly_df.trial_name.values
            # TODO: work out difference between all trials and selected trials
            fly_data["background_image"] = loaddata.get_background_image(fly_data["fly_dir"])
            fly_data["roi_centers"] = loaddata.get_roi_centers(fly_data["fly_dir"])
            twop_df, beh_df = loaddata.load_data(fly_data["fly_dir"], all_trial_dirs=fly_data["trial_names"]) 

            all_stim_responses, all_beh_responses = stimulation.get_neural_responses(twop_df, figure_params["trigger"],
                                                                        trials=fly_data["trial_names"],
                                                                        stim_p=figure_params["stim_p"],
                                                                        return_var=figure_params["return_var"])
            all_beh_class_responses = stimulation.get_beh_class_responses(beh_df, figure_params["trigger"],
                                                                        trials=fly_data["trial_names"],
                                                                        stim_p=figure_params["stim_p"])
            fly_data["n_responses"] = all_stim_responses.shape[-1]
            walk_pre, rest_pre = behaviour.get_pre_stim_beh(beh_df, trigger=figure_params["trigger"],
                                                            stim_p=figure_params["stim_p"],
                                                            n_pre_stim=params.pre_stim_n_samples_beh,
                                                            trials=fly_data["trial_names"])  # selected_trials

            all_nat_responses, all_nat_beh_responses = stimulation.get_neural_responses(twop_df, figure_params["trigger_2"],
                                                                trials=fly_data["trial_names"],
                                                                stim_p=figure_params["stim_p_2"],
                                                                return_var=figure_params["return_var"])
            all_nat_beh_class_responses = stimulation.get_beh_class_responses(beh_df, figure_params["trigger_2"],
                                                                trials=fly_data["trial_names"],
                                                                stim_p=figure_params["stim_p_2"])
            fly_data["nat_n_responses"] = all_nat_responses.shape[-1]
            nat_walk_pre, nat_rest_pre = behaviour.get_pre_stim_beh(beh_df, trigger=figure_params["trigger_2"],
                                                            stim_p=figure_params["stim_p_2"],
                                                            n_pre_stim=params.pre_stim_n_samples_beh,
                                                            trials=fly_data["trial_names"])
            del twop_df, beh_df

            if figure_params["pre_stim"] == "walk":
                select_pre = walk_pre
                fly_data["n_sel_responses"] = np.sum(walk_pre)
                fly_data["n_other_responses"] = np.sum(rest_pre)
                nat_select_pre = nat_walk_pre
                fly_data["nat_n_sel_responses"] = np.sum(nat_walk_pre)
                fly_data["nat_n_other_responses"] = np.sum(nat_rest_pre)

            elif figure_params["pre_stim"] == "rest":
                select_pre = rest_pre
                fly_data["n_sel_responses"] = np.sum(rest_pre)
                fly_data["n_other_responses"] = np.sum(walk_pre)
                nat_select_pre = nat_rest_pre
                fly_data["nat_n_sel_responses"] = np.sum(nat_rest_pre)
                fly_data["nat_n_other_responses"] = np.sum(nat_walk_pre)

            elif figure_params["pre_stim"] == "not_walk":
                select_pre = np.logical_not(walk_pre)
                fly_data["n_sel_responses"] = np.sum(select_pre)
                fly_data["n_other_responses"] = np.sum(walk_pre)
                nat_select_pre = np.logical_not(nat_walk_pre)
                fly_data["nat_n_sel_responses"] = np.sum(nat_select_pre)
                fly_data["nat_n_other_responses"] = np.sum(nat_walk_pre)

            elif figure_params["pre_stim"] is None:
                select_pre = np.ones_like(rest_pre)
                fly_data["n_sel_responses"] = np.sum(select_pre)
                fly_data["n_other_responses"] = 0
                nat_select_pre = np.ones_like(nat_rest_pre)
                fly_data["nat_n_sel_responses"] = np.sum(nat_select_pre)
                fly_data["nat_n_other_responses"] = 0
            else:
                raise NotImplementedError

            if fly_data["n_sel_responses"] < figure_params["min_resp_2"] or fly_data["nat_n_sel_responses"] < figure_params["min_resp_2"]:
                print(fly_data["fly_dir"], fly_data["n_sel_responses"], fly_data["n_responses"], fly_data["nat_n_sel_responses"], fly_data["nat_n_responses"])
                continue
            
            fly_data["stim_responses"] = all_stim_responses[:,:,select_pre]
            fly_data["response_values"] = stimulation.summarise_responses(fly_data["stim_responses"])
            fly_data["sort_ind"] = np.argsort(fly_data["response_values"])
            fly_data["beh_responses"] = all_beh_responses[:,select_pre]
            fly_data["beh_class_responses"] = all_beh_class_responses[:,select_pre]

            fly_data["nat_responses"] = all_nat_responses[:,:,nat_select_pre]
            fly_data["nat_response_values"] = stimulation.summarise_responses(fly_data["nat_responses"])
            fly_data["nat_beh_responses"] = all_nat_beh_responses[:,nat_select_pre]
            fly_data["nat_beh_class_responses"] = all_nat_beh_class_responses[:,nat_select_pre]


            if figure_params["response_clim"] is None:
                try:
                    fly_data["clim"] = np.quantile(np.abs(fly_data["response_values"]), q=figure_params["response_q_max"])
                except IndexError:
                    fly_data["clim"] = 1
            else:
                fly_data["clim"] = figure_params["response_clim"]
            if fly_data["clim"] < 0.5:
                fly_data["clim"] = 0.5

            all_fly_data.append(fly_data.copy())
            del fly_data

        if stim_resp_save is not None:
            with open(stim_resp_save, "wb") as f:
                pickle.dump(all_fly_data, f)

    all_fly_data = [fly_data for fly_data in all_fly_data if fly_data["nat_n_sel_responses"] > 15 and fly_data["n_sel_responses"] > 15]

    if figure_params["mode"] == "presentation":
        nrows = 1
    elif figure_params["mode"] == "presentationsummary":
        return None
    else:
        nrows = len(all_fly_data)
    if not nrows:
        return None
    fig = plt.figure(figsize=(figure_params["panel_size"][0],figure_params["panel_size"][1]*nrows))  # layout="constrained"
    subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=False)[:,0]
    mosaic = figure_params["mosaic"]
    axds = [subfig.subplot_mosaic(mosaic) for subfig in subfigs]

    i_plt = -1
    for i_fly, fly_data in enumerate(tqdm(all_fly_data)):
        if figure_params["mode"] == "presentation":
            if fly_data["fly_df"].fly_id.values[0] != figure_params["pres_fly"]:
                continue
        i_plt += 1
        axd = axds[i_plt]
        get_one_fly_nat_resp_panel(fig, axd, fly_data, figure_params)
    return fig


def summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="pdf", natbeh=False, figures_path=None, tmpdata_path=None, compute_only=False):
    """
    Summarize all neuronal and behavioural responses for flies from multiple genotypes.

    Parameters:
        pre_stim (str, optional): The type of pre-stimulus behavior to consider. Default is "walk".
        overwrite (bool, optional): Whether to overwrite cached data if it exists. Default is False.
        mode (str, optional): The mode for generating summary figures (e.g., "pdf"). Default is "pdf".
        natbeh (bool, optional): Whether to include natural behavior data. Default is False.
        figures_path (str, optional): Path to save generated figures. Default is None.
        tmpdata_path (str, optional): Path to save/load cached data. Default is None.
        compute_only (bool, optional): Whether to compute summary data only without generating figures. Default is False.

    Returns:
        None
    """
    if figures_path is None:
        figures_path = params.plot_base_dir
    if tmpdata_path is None:
        tmpdata_path = params.plotdata_base_dir

    df = summarydf.load_data_summary()
    df = summarydf.filter_data_summary(df, no_co2=False)

    figure_params = {
        "trigger": "laser_start",
        "trigger_2": None,
        "stim_p": [10,20],
        "stim_p_2": None,
        "response_name": None,
        "response_name_2": None,
        "walkrest": True,
        "normalisation_type": params.default_response_regex[4:],
        "return_var": "v_forw",
        "beh_response_ylabel": "Forward vel. \n(mm/s)",
        "neural_response_ylabel": r"$\Delta$F/F",
        "clabel": r"$\Delta$F/F",
        "response_clim": 0.8,
        "response_beh_lim": None,
        "response_q_max": 0.95,
        "suptitle": "",
        "pre_stim": pre_stim,
        "panel_size": (15,4),
        "mosaic": mosaic_stim_resp_panel,
        "min_resp": 10,
        "min_resp_2": 5,
        "mode": mode,
        "pres_fly": None,
    }

    if mode == "presentation":
        figure_params["mosaic"] = mosaic_stim_resp_panel_presentation
    elif mode == "presentationsummary":
        figure_params["mosaic"] = mosaic_stim_resp_panel_presentationsummary
        figure_params["panel_size"] = (4,8)
    
    response_beh_lims = {
        "MDN": [-5,11],  # [-5,5],
        "DNp09": [-5,11],  # [0,15],
        "aDN2": [-5,11],  # [-5,5],
        "PR": [-5,11],  # [-5,5],
    }
    
    # MDN
    df_Dfd_MDN = summarydf.get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "MDN3", "walkon": "ball"}])
    figure_params_MDN = figure_params.copy()
    figure_params_MDN["suptitle"] = f"Backward walking DN (MDN) response after {pre_stim.replace('_', ' ') if pre_stim is not None else None}ing"
    figure_params_MDN["pres_fly"] = presentation_flies["MDN"]
    figure_params_MDN["response_beh_lim"] = response_beh_lims["MDN"] if "presentation" in mode else None
    figure_params_MDN["trigger_2"] = "back_trig_start"
    if not natbeh:
        fig_MDN = summarise_stim_resp(df_Dfd_MDN, figure_params_MDN, stim_resp_save=os.path.join(tmpdata_path, f"stim_resp_{pre_stim}_MDN3.pkl"),
                                        overwrite=overwrite)
    df_Dfd_MDN = summarydf.get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "MDN3", "walkon": "wheel"}])
    figure_params_MDN["panel_size"] = (18,4)
    figure_params_MDN["mosaic"] = mosaic_nat_resp_panel
    figure_params_MDN["pres_fly"] = presentation_natbeh_flies["MDN"]
    figure_params_MDN["pre_stim"] = None
    # figure_params_MDN["pre_stim"] = None
    if natbeh:
        fig_MDN_natbeh = summarise_natbeh_resp(df_Dfd_MDN, figure_params_MDN, stim_resp_save=os.path.join(tmpdata_path, f"natbeh_None_resp_MDN3.pkl"),
                                        overwrite=overwrite)

    # DNp09
    df_Dfd_DNp09 = summarydf.get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "DNp09", "walkon": "ball"},
                                      {"GCaMP": "Dfd", "CsChrimson": "DNP9" , "walkon": "ball"}])
    figure_params_DNp09 = figure_params.copy()
    figure_params_DNp09["suptitle"] = f"Forward walking DN (DNp09) response after {pre_stim.replace('_', ' ') if pre_stim is not None else None}ing"
    figure_params_DNp09["pres_fly"] = presentation_flies["DNp09"]
    figure_params_DNp09["response_beh_lim"] = response_beh_lims["DNp09"] if "presentation" in mode else None
    figure_params_DNp09["trigger_2"] = "walk_trig_start"
    if not natbeh:
        fig_DNp09 = summarise_stim_resp(df_Dfd_DNp09, figure_params_DNp09, stim_resp_save=os.path.join(tmpdata_path, f"stim_resp_{pre_stim}_DNp09.pkl"),
                                    overwrite=overwrite)
    figure_params_DNp09["panel_size"] = (18,4)
    figure_params_DNp09["mosaic"] = mosaic_nat_resp_panel
    figure_params_DNp09["pres_fly"] = presentation_natbeh_flies["DNp09"]
    figure_params_DNp09["pre_stim"] = "not_walk"
    if natbeh:
        fig_DNp09_natbeh = summarise_natbeh_resp(df_Dfd_DNp09, figure_params_DNp09, stim_resp_save=os.path.join(tmpdata_path, f"natbeh_not_walk_resp_DNp09.pkl"),
                                        overwrite=overwrite)

    # aDN2
    df_Dfd_aDN2 = summarydf.get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "aDN2", "walkon": "ball"}])
    figure_params_aDN2 = figure_params.copy()
    figure_params_aDN2["suptitle"] = f"Grooming DN (aDN2) response after {pre_stim.replace('_', ' ') if pre_stim is not None else None}ing"
    figure_params_aDN2["pres_fly"] = presentation_flies["aDN2"]
    figure_params_aDN2["response_beh_lim"] = response_beh_lims["aDN2"] if "presentation" in mode else None
    figure_params_aDN2["trigger_2"] = "olfac_start"  # "groom_trig_start"
    figure_params_aDN2["stim_p"] = [20]
    if not natbeh:
        fig_aDN2 = summarise_stim_resp(df_Dfd_aDN2, figure_params_aDN2, stim_resp_save=os.path.join(tmpdata_path, f"stim_resp_{pre_stim}_aDN2.pkl"),
                                        overwrite=overwrite)
    figure_params_aDN2["panel_size"] = (18,4)
    figure_params_aDN2["mosaic"] = mosaic_nat_resp_panel
    figure_params_aDN2["pres_fly"] = presentation_natbeh_flies["aDN2"]
    figure_params_aDN2["pre_stim"] = None
    if natbeh:
        fig_aDN2_olfac = summarise_natbeh_resp(df_Dfd_aDN2, figure_params_aDN2, stim_resp_save=os.path.join(tmpdata_path, f"natbeh_None_resp_aDN2.pkl"),
                                        overwrite=overwrite)

    # PR
    df_Dfd_PR = summarydf.get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "PR", "walkon": "ball"}])
    figure_params_PR = figure_params.copy()
    figure_params_PR["suptitle"] = f"control (no GAL4) response after {pre_stim.replace('_', ' ') if pre_stim is not None else None}ing"
    figure_params_PR["pres_fly"] = presentation_flies["PR"]
    figure_params_PR["response_beh_lim"] = response_beh_lims["PR"] if "presentation" in mode else None
    if not natbeh:
        fig_PR = summarise_stim_resp(df_Dfd_PR, figure_params_PR, stim_resp_save=os.path.join(tmpdata_path, f"stim_resp_{pre_stim}_PR.pkl"),
                                        overwrite=overwrite)

    if not natbeh and not compute_only:
        figs = [fig_DNp09, fig_aDN2, fig_MDN, fig_PR]
        with PdfPages(os.path.join(figures_path, f"fig_func_summary_{pre_stim}_to_stim_{mode}.pdf")) as pdf:
            _ = [pdf.savefig(fig, transparent=True) for fig in figs if fig is not None]
        _ = [plt.close(fig) for fig in figs if fig is not None]
    elif not compute_only:
        figs_natbeh = [fig_DNp09_natbeh, fig_aDN2_olfac, fig_MDN_natbeh]  # fig_aDN2_natbeh, 
        with PdfPages(os.path.join(figures_path, f"fig_func_summary_{pre_stim}_natbeh_{mode}.pdf")) as pdf:
            _ = [pdf.savefig(fig, transparent=True) for fig in figs_natbeh if fig is not None]
        _ = [plt.close(fig) for fig in figs_natbeh if fig is not None]
    

def collect_data_stat_comparison_activation(all_fly_data, pre_stim="walk", overwrite=True, tmpdata_path=None):
    """
    Collect and summarize data for statistical comparison of neural activation.

    Parameters:
        all_fly_data (list): A list of dictionaries, each containing data for one fly.
        pre_stim (str, optional): The type of pre-stimulus behavior to consider. Default is "walk".
        overwrite (bool, optional): Whether to overwrite cached data if it exists. Default is True.
        tmpdata_path (str, optional): Path to store temporary data. Default is None.

    Returns:
        None
    """
    if tmpdata_path is None:
        tmpdata_path = params.plotdata_base_dir
    n_activated_file = os.path.join(tmpdata_path, f"n_activated_stats_{pre_stim}_to_stim.csv")
    if not os.path.isfile(n_activated_file):
        active_df = pd.DataFrame(columns=["fly_id", "CsChrimson", "n_neurons", "n_active", "frac_active", "e_n_active", "e_f_active", 
                                          "n_deactive", "frac_deactive", "e_n_deactive", "e_f_deactive"])
        active_df.to_csv(n_activated_file, index=False)
    else:
        active_df = pd.read_csv(n_activated_file)

    for fly_data in all_fly_data:
        fly_id = fly_data["fly_df"]["fly_id"].values[0]
        if overwrite and fly_id in active_df.fly_id.values:
            active_df = active_df.drop(index=active_df.index[active_df.fly_id==fly_id])
        if not fly_id in active_df.fly_id.values:
            CsChrimson = fly_data["fly_df"]["CsChrimson"].values[0]
            if CsChrimson == "DNP9":
                CsChrimson = "DNp09"
            elif CsChrimson == "MDN":
                CsChrimson = "MDN3"
            response_values = fly_data["response_values"]
            n_neurons = len(response_values)
            n_active = np.sum(response_values > 0)
            n_deactive = np.sum(response_values < 0)
            frac_active = n_active / n_neurons
            frac_deactive = n_deactive / n_neurons
            e_n_active = np.sum(response_values[response_values > 0])
            e_n_deactive = np.sum(response_values[response_values < 0])
            e_f_active = np.sum(response_values[response_values > 0]) / n_neurons
            e_f_deactive = np.sum(response_values[response_values < 0]) / n_neurons
            active_df = active_df.append({"fly_id": fly_id, "CsChrimson": CsChrimson,
                                        "n_neurons": n_neurons, "n_active": n_active, "frac_active": frac_active,
                                        "n_deactive": n_deactive, "frac_deactive": frac_deactive,
                                        "e_n_active": e_n_active, "e_f_active": e_f_active,
                                        "e_n_deactive": e_n_deactive, "e_f_deactive": e_f_deactive}, ignore_index=True)
    
    active_df.to_csv(n_activated_file, index=False)

def plot_stat_comparison_activation(pre_stim="walk", figures_path=None, tmpdata_path=None):
    """
    Generate statistical comparison plots for neural activation during pre-stimulus behavior.

    Parameters:
        pre_stim (str, optional): The type of pre-stimulus behavior to consider. Default is "walk".
        figures_path (str, optional): Path to save generated figures. Default is None.
        tmpdata_path (str, optional): Path to store temporary data. Default is None.

    Returns:
        None
    """
    if figures_path is None:
        figures_path = params.plot_base_dir
    if tmpdata_path is None:
        tmpdata_path = params.plotdata_base_dir
    n_activated_file = os.path.join(tmpdata_path, f"n_activated_stats_{pre_stim}_to_stim.csv")
    active_df = pd.read_csv(n_activated_file)

    fig, axs = plt.subplots(1,3, figsize=(13,4))

    # ["n_active", "e_n_active", "frac_active", "e_f_active", "n_deactive", "e_n_deactive", "frac_deactive", "e_f_deactive"]
    for ax, var in zip(axs.flatten(), ["n_active", "e_n_active", "frac_active"]):

        sns.barplot(ax=ax, data=active_df, x="CsChrimson", y=var, order=["DNp09", "aDN2", "MDN3", "PR"], hue="CsChrimson",
                    palette=[myplt.DARKCYAN, myplt.DARKGREEN, myplt.DARKRED, myplt.DARKGRAY], saturation=1, dodge=False, width=0.5, alpha=1)
        sns.stripplot(ax=ax, data=active_df, x="CsChrimson", y=var, order=["DNp09", "aDN2", "MDN3", "PR"], color=myplt.BLACK, size=8)
        plotpanels.make_nice_spines(ax)
        ax.legend([], frameon=False)
    fig.tight_layout()

    fig.savefig(os.path.join(figures_path, f"fig_n_activated_stats_{pre_stim}_to_stim.pdf"))

    print(f"results of Mann-Whittney-U testing of each line agains control for {pre_stim} to stim")
    print("number of activated neurons:")
    print("DNp09 vs. control", mannwhitneyu(active_df["n_active"][active_df["CsChrimson"] == "DNp09"], active_df["n_active"][active_df["CsChrimson"] == "PR"]))
    print("aDN2 vs. control", mannwhitneyu(active_df["n_active"][active_df["CsChrimson"] == "aDN2"], active_df["n_active"][active_df["CsChrimson"] == "PR"]))
    print("MDN3 vs. control", mannwhitneyu(active_df["n_active"][active_df["CsChrimson"] == "MDN3"], active_df["n_active"][active_df["CsChrimson"] == "PR"]))
    print("fraction of activated neurons:")
    print("DNp09 vs. control", mannwhitneyu(active_df["frac_active"][active_df["CsChrimson"] == "DNp09"], active_df["frac_active"][active_df["CsChrimson"] == "PR"]))
    print("aDN2 vs. control", mannwhitneyu(active_df["frac_active"][active_df["CsChrimson"] == "aDN2"], active_df["frac_active"][active_df["CsChrimson"] == "PR"]))
    print("MDN3 vs. control", mannwhitneyu(active_df["frac_active"][active_df["CsChrimson"] == "MDN3"], active_df["frac_active"][active_df["CsChrimson"] == "PR"]))
    print("DFF sum of activated neurons:")
    print("DNp09 vs. control", mannwhitneyu(active_df["e_n_active"][active_df["CsChrimson"] == "DNp09"], active_df["e_n_active"][active_df["CsChrimson"] == "PR"]))
    print("aDN2 vs. control", mannwhitneyu(active_df["e_n_active"][active_df["CsChrimson"] == "aDN2"], active_df["e_n_active"][active_df["CsChrimson"] == "PR"]))
    print("MDN3 vs. control", mannwhitneyu(active_df["e_n_active"][active_df["CsChrimson"] == "MDN3"], active_df["e_n_active"][active_df["CsChrimson"] == "PR"]))

if __name__ == "__main__":
    summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="presentation", natbeh=False)
    summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="presentationsummary", natbeh=False)
    plot_stat_comparison_activation(pre_stim="walk")
    
    # Supp. File 1: Individual fly neural and behavioral responses to opto-genetic stimulation
    summarise_all_stim_resp(pre_stim="walk", overwrite=False, mode="pdf", natbeh=False)
    summarise_all_stim_resp(pre_stim="rest", overwrite=False, mode="pdf", natbeh=False)

    # Supp. Figure 2: Comparison of GNG-DN population neural activity during optogenetic stimulation versus corresponding natural behaviors.
    summarise_all_stim_resp(pre_stim=None, overwrite=False, mode="presentation", natbeh=True)
    summarise_all_stim_resp(pre_stim=None, overwrite=False, mode="pdf", natbeh=False, compute_only=True)
    summarise_all_stim_resp(pre_stim="not_walk", overwrite=False, mode="pdf", natbeh=False, compute_only=True)