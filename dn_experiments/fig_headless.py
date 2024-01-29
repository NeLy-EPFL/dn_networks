"""
Module to analyse headless experiments.
Author: jonas.braun@epfl.ch
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from tqdm import tqdm
from scipy.stats import mannwhitneyu, wilcoxon, ranksums

import params, summarydf, loaddata, stimulation, behaviour, plotpanels

from twoppp import plot as myplt


mosaic_headless_panel = """
V.W.X..B.C.D
"""

mosaic_headless_summary_panel = """
.XXX
.XXX
.XXX
.DDD
.DDD
.DDD
....
"""

def get_one_fly_headless_panel(fig, axd, fly_data, figure_params, set_baseline_zero=True):

    """
    Generate a panel for a single fly's behavioral response and add it to the figure.

    Args:
        fig (matplotlib.figure.Figure): The figure to which the panel should be added.
        axd (dict): A dictionary of subplot axes where the panel should be placed.
        fly_data (dict): Data for a single fly.
        figure_params (dict): Parameters for configuring the panel's appearance.
        set_baseline_zero (bool, optional): set stimulation onset to zero

    Returns:
        None
    """

    if (not isinstance(fly_data["beh_class_responses_pre"], np.ndarray) or (
        np.isnan(fly_data["beh_class_responses_pre"].all()))):
        return
    
    if figure_params["allflies_only"]:
        if fly_data['fly_df'].CsChrimson.values[0] == "PR":
            response_name = f"__ > CsChrimson"
        else:
            response_name = f"{fly_data['fly_df'].CsChrimson.values[0]} > CsChrimson"
    else:
        response_name = f"{fly_data['fly_df'].date.values[0]} {fly_data['fly_df'].CsChrimson.values[0]} Fly {fly_data['fly_df'].fly_number.values[0]}"
    # X: volocity response comparison
    plotpanels.plot_ax_behavioural_response(
        fly_data["beh_responses_pre"],
        ax=axd["X"],
        x="beh",
        response_name=response_name,
        response_ylabel=figure_params["beh_response_ylabel"] if figure_params["allflies_only"] else None,
        beh_responses_2=fly_data["beh_responses_post"],
        beh_response_2_color=behaviour.get_beh_color(figure_params["beh_name"]),
        period=figure_params["stats_period"],
        ylim=figure_params["ylim"],)

    if not figure_params["allflies_only"]:
        # V: volocity response pre head cut
        plotpanels.plot_ax_behavioural_response(
            fly_data["beh_responses_pre"],
            ax=axd["V"],
            x="beh",
            response_name="intact",
            response_ylabel=figure_params["beh_response_ylabel"],
            period=figure_params["stats_period"],
            ylim=figure_params["ylim"])

        # W: volocity response post head cut
        plotpanels.plot_ax_behavioural_response(
            fly_data["beh_responses_post"],
            ax=axd["W"],
            x="beh",
            response_name="amputated",
            response_ylabel=None,
            period=figure_params["stats_period"],
            ylim=figure_params["ylim"])
    
    
        if figure_params["ylim"] is None:
            ylim = axd["X"].get_ylim()
            axd["V"].set_ylim(ylim)
            axd["W"].set_ylim(ylim)

        # B: behavioural class
        plotpanels.plot_ax_behprob(fly_data["beh_class_responses_pre"], ax=axd["B"], ylabel="behaviour\nprobability")

        # C: behavioural class
        plotpanels.plot_ax_behprob(fly_data["beh_class_responses_post"], ax=axd["C"], ylabel=None)
    
    # D: behavioural class
    plotpanels.plot_ax_behprob(fly_data["beh_class_responses_pre"], ax=axd["D"],
            labels_2=fly_data["beh_class_responses_post"], beh_2=figure_params["beh_name"],
            ylabel=f"{figure_params['beh_name']}\nprobability" if figure_params["allflies_only"] else None)


def summarise_headless(exp_df, figure_params, headless_save=None, overwrite=False):
    """
    Summarize and visualize behavioral responses of multiple flies and add them to a figure.

    Args:
        exp_df (pandas.DataFrame): DataFrame containing experimental data for multiple flies.
        figure_params (dict): Parameters for configuring the appearance of the summary figure.
        headless_save (str): Path to save the temporary data as a pickle file.
        overwrite (bool): If True, overwrite the existing temporary data file.

    Returns:
        matplotlib.figure.Figure: The generated figure containing behavioral response panels.
    """

    base_fly_data = {
        "fly_df": None,
        "fly_id": None,
        "fly_dir": None,
        "trial_names": None,
        "beh_responses_pre": None,
        "beh_responses_post": None,
        "beh_class_responses_pre": None,
        "beh_class_responses_post": None,
        }
    if headless_save is not None and os.path.isfile(headless_save) and not overwrite:
        with open(headless_save, "rb") as f:
            all_fly_data = pickle.load(f)
    else:
        # load data for all flies

        all_fly_data = []
        summary_fly_data = base_fly_data.copy()

        for i_fly, (fly_id, fly_df) in enumerate(exp_df.groupby("fly_id")):
                    
            fly_data = base_fly_data.copy()
            fly_data["fly_df"] = fly_df
            fly_data["fly_dir"] = np.unique(fly_df.fly_dir)[0]
            fly_data["trial_names"] = fly_df.trial_name.values
            for index, trial_df in fly_df.iterrows():
                if not trial_df.walkon == "ball":
                    continue  # TODO: make no ball analysis
                else:
                    if trial_df["head"]:
                        beh_key = "beh_responses_pre"
                        beh_class_key = "beh_class_responses_pre"
                    else:
                        beh_key = "beh_responses_post"
                        beh_class_key = "beh_class_responses_post"
                beh_df = loaddata.load_beh_data_only(fly_data["fly_dir"], all_trial_dirs=[trial_df.trial_name]) 
            

                beh_responses = stimulation.get_beh_responses(beh_df, trigger=figure_params["trigger"],
                                                                            trials=[trial_df.trial_name],
                                                                            stim_p=figure_params["stim_p"],
                                                                            beh_var=figure_params["return_var"])
                beh_class_responses = stimulation.get_beh_class_responses(beh_df, trigger=figure_params["trigger"],
                                                                            trials=[trial_df.trial_name],
                                                                            stim_p=figure_params["stim_p"])

                if figure_params["return_var_flip"]:
                    beh_responses = -1 * beh_responses  # make the presentation more intuitive, e.g. front leg height

                if figure_params["return_var_multiply"] is not None:
                    beh_responses = figure_params["return_var_multiply"] * beh_responses

                if figure_params["return_var_change"] is not None:
                    beh_responses = beh_responses - np.mean(beh_responses[figure_params["return_var_change"][0]:figure_params["return_var_change"][1],:], axis=0)

                fly_data[beh_key] = beh_responses
                fly_data[beh_class_key] = beh_class_responses
                del beh_df

            # select responses
            # not necessary here.

            all_fly_data.append(fly_data.copy())
            del fly_data

        if headless_save is not None:
            with open(headless_save, "wb") as f:
                pickle.dump(all_fly_data, f)

    nrows = len(all_fly_data) + 1 if not figure_params["allflies_only"] else 1
    fig = plt.figure(figsize=(figure_params["panel_size"][0],figure_params["panel_size"][1]*nrows))  # layout="constrained"
    subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=False)[:,0]
    mosaic = figure_params["mosaic"]
    axds = [subfig.subplot_mosaic(mosaic) for subfig in subfigs]

    if not figure_params["allflies_only"]:
        for fly_data, axd in tqdm(zip(all_fly_data, axds)):
            get_one_fly_headless_panel(fig, axd, fly_data, figure_params)

    # summarise data
    summary_fly_data = base_fly_data.copy()
    summary_fly_data["fly_df"] = all_fly_data[0]["fly_df"]
    summary_fly_data["fly_df"].date = ""
    summary_fly_data["fly_df"].fly_number = "all"
    summary_fly_data["beh_responses_pre"] = np.concatenate([fly_data["beh_responses_pre"] for fly_data in all_fly_data], axis=-1)
    summary_fly_data["beh_responses_post"] = np.concatenate([fly_data["beh_responses_post"] for fly_data in all_fly_data], axis=-1)
    summary_fly_data["beh_class_responses_pre"] = np.concatenate([fly_data["beh_class_responses_pre"] for fly_data in all_fly_data], axis=-1)
    summary_fly_data["beh_class_responses_post"] = np.concatenate([fly_data["beh_class_responses_post"] for fly_data in all_fly_data], axis=-1)
    
    get_one_fly_headless_panel(fig, axds[-1], summary_fly_data, figure_params)

    fig.suptitle(figure_params["suptitle"])
    return fig


def summarise_all_headless(overwrite=False, allflies_only=True, tmpdata_path=None, figures_path=None):
    """
    Summarize and visualize behavioral responses of multiple fly genotypes and save them as a PDF.

    Args:
        overwrite (bool): If True, overwrite existing temporary data files.
        allflies_only (bool): If True, generate a summary for all flies.
        tmpdata_path (str): Path to the temporary data directory.
        figures_path (str): Path to the directory where PDF figures will be saved.

    Returns:
        None
    """
    if tmpdata_path is None:
        tmpdata_path = params.plotdata_base_dir
    if figures_path is None:
        figures_path = params.plot_base_dir
    df = summarydf.get_headless_df()

    figure_params = {
        "trigger": "laser_start",
        "stim_p": [10,20],
        "response_name": None,
        "beh_name": None,
        "return_var": "v_forw",
        "return_var_flip": False,
        "return_var_change": None,
        "return_var_multiply": None,
        "beh_response_ylabel": r"$v_{||}$ (mm/s)",
        "suptitle": "",
        "panel_size": (15,1),
        "mosaic": mosaic_headless_panel,
        "allflies_only": False,
        "ylim": None,
    }
    if allflies_only:
        figure_params["mosaic"] = mosaic_headless_summary_panel
        figure_params["allflies_only"] = True
        figure_params["panel_size"] = (3,4)
        add_str = "_allflies_only"
    else:
        add_str = ""

    # MDN
    df_MDN = summarydf.get_selected_df(df, [{"CsChrimson": "MDN3"}])
    figure_params_MDN = figure_params.copy()
    figure_params_MDN["suptitle"] = "MDN3 > CsChrimson"
    figure_params_MDN["beh_name"] = "back"
    figure_params_MDN["ylim"] = [-3,7]
    fig_MDN = summarise_headless(df_MDN, figure_params_MDN, headless_save=os.path.join(tmpdata_path, f"headless_MDN3.pkl"),
                                    overwrite=overwrite)
    
    # DNp09
    df_DNp09 = summarydf.get_selected_df(df, [{"CsChrimson": "DNp09"}])
    figure_params_DNp09 = figure_params.copy()
    figure_params_DNp09["suptitle"] = "DNp09 > CsChrimson"
    figure_params_DNp09["beh_name"] = "walk"
    figure_params_DNp09["ylim"] = [-3,7]
    fig_DNp09 = summarise_headless(df_DNp09, figure_params_DNp09, headless_save=os.path.join(tmpdata_path, f"headless_DNp09.pkl"),
                                    overwrite=overwrite)
    figure_params_DNp09["return_var"] = "anus_dist"
    figure_params_DNp09["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_DNp09["beh_response_ylabel"] = "anal plate (um)"
    figure_params_DNp09["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_DNp09["ylim"] = [-75,25]
    fig_DNp09_1 = summarise_headless(df_DNp09, figure_params_DNp09, headless_save=os.path.join(tmpdata_path, f"headless_DNp09_anus.pkl"),
                                    overwrite=overwrite)
    
    # aDN2
    df_aDN2 = summarydf.get_selected_df(df, [{"CsChrimson": "aDN2"}])
    figure_params_aDN2 = figure_params.copy()
    figure_params_aDN2["suptitle"] = "aDN2 > CsChrimson"
    figure_params_aDN2["beh_name"] = "groom"
    figure_params_aDN2["ylim"] = [-3,7]
    fig_aDN2 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(tmpdata_path, f"headless_aDN2.pkl"),
                                    overwrite=overwrite)
    # additional kinematics parameters
    figure_params_aDN2["return_var"] = "frtita_neck_dist"
    figure_params_aDN2["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_aDN2["beh_response_ylabel"] = "front leg tita - head dist (um)"
    figure_params_aDN2["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_aDN2["return_var_flip"] = False 
    figure_params_aDN2["ylim"] = [-175,100]
    fig_aDN2_1 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(tmpdata_path, f"headless_aDN2_frtita_dist.pkl"),
                                    overwrite=overwrite)

    # PR
    df_PR = summarydf.get_selected_df(df, [{"CsChrimson": "PR"}])
    figure_params_PR = figure_params.copy()
    figure_params_PR["suptitle"] = "__ > CsChrimson"
    figure_params_PR["beh_name"] = "rest"
    figure_params_PR["ylim"] = [-3,7]
    fig_PR = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(tmpdata_path, f"headless_PR.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["beh_name"] = "back"
    fig_PR_1 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(tmpdata_path, f"headless_PR.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["beh_name"] = "walk"
    fig_PR_2 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(tmpdata_path, f"headless_PR.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["return_var"] = "anus_dist"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "anal plate (um)"
    figure_params_PR["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_PR["ylim"] = [-75,25]
    fig_PR_3 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(tmpdata_path, f"headless_PR_anus.pkl"),
                                    overwrite=overwrite)

    figure_params_PR["return_var"] = "frtita_neck_dist"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "front leg tita - head dist (px)"
    figure_params_PR["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_PR["return_var_flip"] = False 
    figure_params_PR["ylim"] = [-175,100]
    fig_PR_4 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(tmpdata_path, f"headless_PR_frtita_dist.pkl"),
                                    overwrite=overwrite)

    
    figs = [fig_MDN, fig_DNp09, fig_DNp09_1, fig_aDN2, fig_aDN2_1,\
            fig_PR, fig_PR_1, fig_PR_2, fig_PR_3, fig_PR_4]


    with PdfPages(os.path.join(figures_path, f"fig_headless_summary_ball{add_str}.pdf")) as pdf:
        _ = [pdf.savefig(fig) for fig in figs]
    _ = [plt.close(fig) for fig in figs]

    
def test_stats_pre_post(all_flies, i_beh, GAL4, beh_name, var_name="v", i_0=500, i_1=750):
    """
    Perform statistical tests on behavioral data before and after decapitation.

    Args:
        all_flies (list): List of fly data dictionaries.
        i_beh (int): Index of the behavioral class to compare.
        GAL4 (str): GAL4 genotype identifier.
        beh_name (str): Name of the behavior being analyzed.
        var_name (str): Name of the behavioral variable.
        i_0 (int): Start index for the time window of comparison. 
        i_1 (int): End index for the time window of comparison.

    Returns:
        None
    """
    v_pre = []
    v_pre_paired = []
    p_pre = []
    p_pre_paired = []
    v_post = []
    v_post_paired = []
    p_post = []
    p_post_paired = []
    for fly in all_flies:
        if fly["beh_responses_pre"] is not None and not isinstance(fly["beh_responses_pre"], float):
            v_pre.append(np.mean(fly["beh_responses_pre"][i_0:i_1], axis=0))
            p_pre.append(np.mean(fly["beh_class_responses_pre"][i_0:i_1] == i_beh, axis=0))
            v_pre_paired.append(np.mean(fly["beh_responses_pre"][i_0:i_1]))
            p_pre_paired.append(np.mean(fly["beh_class_responses_pre"][i_0:i_1] == i_beh))
        if fly["beh_responses_post"] is not None and not isinstance(fly["beh_responses_post"], float):
            v_post.append(np.mean(fly["beh_responses_post"][i_0:i_1], axis=0))
            p_post.append(np.mean(fly["beh_class_responses_post"][i_0:i_1] == i_beh, axis=0))
            v_post_paired.append(np.mean(fly["beh_responses_post"][i_0:i_1]))
            p_post_paired.append(np.mean(fly["beh_class_responses_post"][i_0:i_1] == i_beh))
    v_pre = np.concatenate(v_pre).flatten()
    v_post = np.concatenate(v_post).flatten()
    v_pre_paired = np.array(v_pre_paired).flatten()
    v_post_paired = np.array(v_post_paired).flatten()
    p_pre = np.concatenate(p_pre).flatten()
    p_post = np.concatenate(p_post).flatten()
    p_pre_paired = np.array(p_pre_paired).flatten()
    p_post_paired = np.array(p_post_paired).flatten()
    # print(f"{GAL4} {var_name} pre (n={len(v_pre)}) post (n={len(v_post)}). {i_0}-{i_1}: on trials", mannwhitneyu(v_pre, v_post))
    print(f"{GAL4} {var_name} pre (n={len(v_pre_paired)}) post (n={len(v_post_paired)}). {i_0}-{i_1}: on flies ", mannwhitneyu(v_pre_paired, v_post_paired))
    
    if beh_name is not None:
        # print(f"{GAL4} {beh_name} beh class pre (n={len(p_pre)}) post (n={len(p_post)}). {i_0}-{i_1}: on trials", mannwhitneyu(p_pre, p_post))
        print(f"{GAL4} {beh_name} beh class pre (n={len(p_pre_paired)}) post (n={len(p_post_paired)}). {i_0}-{i_1}: on flies ", mannwhitneyu(p_pre_paired, p_post_paired))

def test_stats_beh_control(all_flies, all_flies_control, GAL4, beh_name, i_0=500, i_1=750):
    """
    Perform statistical tests comparing the behavioral responses of a group of flies to a control group.

    Args:
        all_flies (list): List of fly data dictionaries for the experimental group.
        all_flies_control (list): List of fly data dictionaries for the control group.
        GAL4 (str): GAL4 genotype identifier.
        beh_name (str): Name of the behavior being analyzed.
        i_0 (int): Start index for the time window of interest.
        i_1 (int): End index for the time window of interest.

    Returns:
        None
    """
    beh = []
    beh_paired = []
    beh_control = []
    beh_control_paired = []
    for fly in all_flies:
        if fly["beh_responses_post"] is not None and not isinstance(fly["beh_responses_post"], float):
            beh.append(np.mean(fly["beh_responses_post"][i_0:i_1], axis=0))
            beh_paired.append(np.mean(fly["beh_responses_post"][i_0:i_1]))
    for fly in all_flies_control:
        if fly["beh_responses_post"] is not None and not isinstance(fly["beh_responses_post"], float):
            beh_control.append(np.mean(fly["beh_responses_post"][i_0:i_1], axis=0))
            beh_control_paired.append(np.mean(fly["beh_responses_post"][i_0:i_1]))
    beh = np.concatenate(beh).flatten()
    beh_control = np.concatenate(beh_control).flatten()
    beh_paired = np.array(beh_paired).flatten()
    beh_control_paired = np.array(beh_control_paired).flatten()
    # print(f"{GAL4} (n={len(beh)}) vs. control (n={len(beh_control)}) {beh_name}. {i_0}-{i_1}: on trials", mannwhitneyu(beh, beh_control))
    print(f"{GAL4} (n={len(beh_paired)}) vs. control (n={len(beh_control_paired)}) {beh_name}. {i_0}-{i_1}: on flies", mannwhitneyu(beh_paired, beh_control_paired))

def headless_stat_test(tmpdata_path=None):
    """
    Perform statistical tests on behavioral data for different fly genotypes and experimental conditions.

    Args:
        tmpdata_path (str): Path to the directory containing temporary data files.

    Returns:
        None
    """
    if tmpdata_path is None:
        tmpdata_path = params.plotdata_base_dir
    headless_files = {
        "MDN": os.path.join(tmpdata_path, "headless_MDN3.pkl"),
        "DNp09": os.path.join(tmpdata_path, "headless_DNp09.pkl"),
        "aDN2": os.path.join(tmpdata_path, "headless_aDN2.pkl"),
        "PR": os.path.join(tmpdata_path, "headless_PR.pkl"),
    }
    with open(headless_files["MDN"], "rb") as f:
        MDN = pickle.load(f)
    with open(headless_files["DNp09"], "rb") as f:
        DNp09 = pickle.load(f)
    with open(headless_files["aDN2"], "rb") as f:
        aDN2 = pickle.load(f)
    with open(headless_files["PR"], "rb") as f:
        PR = pickle.load(f)
        
    test_stats_pre_post(MDN, i_beh=3, GAL4="MDN", beh_name="back")
    test_stats_pre_post(DNp09, i_beh=1, GAL4="DNp09", beh_name="walk")
    test_stats_pre_post(aDN2, i_beh=4, GAL4="aDN2", beh_name="groom")
    test_stats_pre_post(PR, i_beh=2, GAL4="PR", beh_name="rest")

    detailled_files = {
        "DNp09_anus": os.path.join(tmpdata_path, "headless_DNp09_anus.pkl"),
        "PR_anus": os.path.join(tmpdata_path, "headless_PR_anus.pkl"),
        "aDN2_dist_tita": os.path.join(tmpdata_path, "headless_aDN2_frtita_dist.pkl"),
        "PR_dist_tita": os.path.join(tmpdata_path, "headless_PR_frtita_dist.pkl"),
    }
    with open(detailled_files["DNp09_anus"], "rb") as f:
        DNp09_anus = pickle.load(f)
    with open(detailled_files["PR_anus"], "rb") as f:
        PR_anus = pickle.load(f)
    with open(detailled_files["aDN2_dist_tita"], "rb") as f:
        aDN2_dist_tita = pickle.load(f)
    with open(detailled_files["PR_dist_tita"], "rb") as f:
        PR_dist_tita = pickle.load(f)

    test_stats_beh_control(DNp09_anus, PR_anus, GAL4="DNp09", beh_name="anus", i_0=500, i_1=750)
    test_stats_beh_control(aDN2_dist_tita, PR_dist_tita, GAL4="aDN2", beh_name="dist tita", i_0=500, i_1=750)



if __name__ == "__main__":
    summarise_all_headless(overwrite=True, allflies_only=False)
    summarise_all_headless(overwrite=False, allflies_only=True)
    headless_stat_test()
    revisions_stat_test()