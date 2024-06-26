"""
Module to make figure for predictions data
Author: jonas.braun@epfl.ch (with input from femke.hurtak@epfl.ch)
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
from scipy.stats import mannwhitneyu

import params, summarydf, loaddata, stimulation, behaviour, plotpanels, fig_headless, filters

from twoppp import plot as myplt

mosaic_predictions_panel = """
..............................
VVVV.WWWW.XXXX..BBBB.CCCC.DDDD
VVVV.WWWW.XXXX..BBBB.CCCC.DDDD
VVVV.WWWW.XXXX..BBBB.CCCC.DDDD
VVVV.WWWW.XXXX..BBBB.CCCC.DDDD
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

def load_data_one_genotype(exp_df, figure_params, predictions_save=None, overwrite=False):
    """
    Load stimulus response data for a specific genotype and return a list of dictionaries with sitmulus responses for this genotype.

    Parameters:
    - exp_df (DataFrame): summary DataFrame containing experimental data.
    - figure_params (dict): Dictionary containing figure parameters.
    - predictions_save (str): Path to save or load predictions data.
    - overwrite (bool): Whether to overwrite existing predictions data.

    Returns:
    - all_fly_data (list): List of dictionaries containing stimulus response fly data.
    """
    if predictions_save is not None and os.path.isfile(predictions_save) and not overwrite:
        with open(predictions_save, "rb") as f:
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
            headless_trial_exists = False
            intact_trial_exists = False
            for index, trial_df in fly_df.iterrows():
                if (
                    not trial_df.walkon == "ball"
                    and not figure_params["include_noball_data"]
                ):
                    continue  # TODO: make no ball analysis
                else:
                    if trial_df["head"] == "True" or trial_df["head"] == "TRUE" or trial_df["head"] == "1" or trial_df["head"] == True:
                        beh_key = "beh_responses_pre"
                        beh_class_key = "beh_class_responses_pre"
                        intact_trial_exists = True
                    elif trial_df["head"] == "False" or trial_df["head"] == "FALSE" or trial_df["head"] == "0" or trial_df["head"] == False:
                        beh_key = "beh_responses_post"
                        beh_class_key = "beh_class_responses_post"
                        headless_trial_exists = True
                    else:
                        print(trial_df)
                        print("Error! Could not read 'head'.")
                        print("head was", trial_df["head"])
                        raise(NotImplementedError)
                beh_df = loaddata.load_beh_data_only(fly_data["fly_dir"], all_trial_dirs=[trial_df.trial_name]) 
            
                beh_responses = stimulation.get_beh_responses(
                    beh_df,
                    trigger=figure_params["trigger"],
                    trials=[trial_df.trial_name],
                    stim_p=figure_params["stim_p"],
                    beh_var=figure_params["return_var"],
                    baseline_zero=figure_params["zero_baseline"],
                )
                beh_class_responses = stimulation.get_beh_class_responses(
                    beh_df,
                    trigger=figure_params["trigger"],
                    trials=[trial_df.trial_name],
                    stim_p=figure_params["stim_p"],
                )

                if figure_params["return_var_flip"]:
                    beh_responses = -1 * beh_responses  # make the presentation more intuitive, e.g. front leg height
                if figure_params["return_var_abs"]:
                    beh_responses = np.abs(beh_responses)
                if figure_params["return_var_multiply"] is not None:
                    beh_responses = (
                        figure_params["return_var_multiply"] * beh_responses
                    )  # multiple to convert: e.g. pixels -> um

                if (
                    figure_params["return_var_change"] is not None
                ):  # compute change upon stimulation
                    beh_responses = beh_responses - np.mean(
                        beh_responses[
                            figure_params["return_var_change"][0] : figure_params[
                                "return_var_change"
                            ][1],
                            :,
                        ],
                        axis=0,
                    )

                fly_data[beh_key] = beh_responses
                fly_data[beh_class_key] = beh_class_responses
                del beh_df

            if (
                not headless_trial_exists
                and figure_params["accept_headless_only_flies"]
            ):
                fly_data["beh_responses_post"] = (
                    np.zeros_like(fly_data["beh_responses_pre"]) * np.nan
                )
                fly_data["beh_class_responses_post"] = (
                    np.zeros_like(fly_data["beh_class_responses_pre"]) * np.nan
                )
                print(
                    f"WARNING: Setting headless behavioural response to NaN because no data for fly {fly_data['fly_dir']}"
                )
            elif not headless_trial_exists:
                del fly_data
                continue

            if (
                not intact_trial_exists
                and figure_params["accept_intact_only_flies"]
            ):
                fly_data["beh_responses_pre"] = (
                    np.zeros_like(fly_data["beh_responses_post"]) * np.nan
                )
                fly_data["beh_class_responses_pre"] = (
                    np.zeros_like(fly_data["beh_class_responses_post"]) * np.nan
                )
                print(
                    f"WARNING: Setting intact behavioural response to NaN because no data for fly {fly_data['fly_dir']}"
                )
            elif not intact_trial_exists:
                del fly_data
                continue
    

            # filter data if needed
            if figure_params["filter_pre_stim_beh"] is not None:
                fly_data = filters.remove_trials_beh_pre(
                    fly_data, figure_params["filter_pre_stim_beh"]
                )

            all_fly_data.append(fly_data.copy())
            del fly_data

        if predictions_save is not None:
            with open(predictions_save, "wb") as f:
                pickle.dump(all_fly_data, f)

    return all_fly_data

def concatenate(
    all_fly_data,
    column: str,
    ):
    """
    Concatenate data from all flies for a specific column.

    Parameters:
    - all_fly_data (list): List of dictionaries containing fly data.
    - column (str): Column name to concatenate.

    Returns:
    - concatenated_data (numpy.ndarray): Concatenated data.
    """
    data_list = [fly_data[column] for fly_data in all_fly_data
        if (
        (fly_data[column] is not None and not np.all(np.isnan(fly_data[column])))
        )]
    return np.concatenate(
        data_list, axis=-1)

def plot_data_one_genotype(figure_params, all_fly_data):
    """
    Plot data for a specific genotype.

    Parameters:
    - figure_params (dict): Dictionary containing figure parameters.
    - all_fly_data (list): List of dictionaries containing fly data.

    Returns:
    - fig (matplotlib.figure.Figure): Matplotlib figure object.
    """
    nrows = len(all_fly_data) + 1 if not figure_params["allflies_only"] else 1
    fig = plt.figure(
        figsize=(figure_params["panel_size"][0], figure_params["panel_size"][1] * nrows)
    )  # layout="constrained"
    subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=False)[:, 0]
    mosaic = figure_params["mosaic"]
    axds = [subfig.subplot_mosaic(mosaic) for subfig in subfigs]

    if not figure_params["allflies_only"]:
        for fly_data, axd in tqdm(zip(all_fly_data, axds)):
            fig_headless.get_one_fly_headless_panel(fig, axd, fly_data, figure_params)

    # summarise data
    summary_fly_data = base_fly_data.copy()
    summary_fly_data["fly_df"] = all_fly_data[0]["fly_df"]
    summary_fly_data["fly_df"].date = ""
    summary_fly_data["fly_df"].fly_number = "all"
    summary_fly_data["beh_responses_pre"] = concatenate(all_fly_data, "beh_responses_pre")
    summary_fly_data["beh_responses_post"] = concatenate(all_fly_data, "beh_responses_post")
    summary_fly_data["beh_class_responses_pre"] = concatenate(all_fly_data, "beh_class_responses_pre")
    summary_fly_data["beh_class_responses_post"] = concatenate(all_fly_data, "beh_class_responses_post")
    
    fig_headless.get_one_fly_headless_panel(fig, axds[-1], summary_fly_data, figure_params, make_sourcedata=True, prediction=True)

    fig.suptitle(
        figure_params["suptitle"], fontsize=30, y=1 - 0.01 * 5 / (len(subfigs))
    )
    return fig


def summarise_predictions_one_genotype(
    GAL4,
    overwrite=False,
    allflies_only=False,
    beh_name="walk",
    return_var="v_forw",
    return_var_flip=False,
    return_var_multiply=None,
    return_var_abs=False,
    return_var_ylim=None,
    return_var_baseline=None,
    return_var_ylabel=r"$v_{||}$ (mm/s)",
    data_save_location=params.predictionsdata_base_dir,
    plot_save_location=params.predictionsplot_base_dir,
    accept_headless_only_flies=True,
    accept_intact_only_flies=True,
    include_noball_data=False,
    filter_pre_stim_beh=None,
    zero_baseline=False,
    stats_period=(500,750),
    dataset="prediction"
):
    """make a figure for one genotype and one behavioural response, before and after head cutting

    Parameters
    ----------
    GAL4 : str
        which GAL4 line to analyse
    overwrite : bool, optional
        whether to overwrite previously saved intermediate results, by default False
    allflies_only : bool, optional
        whether to plot only the summary of all flies. If False, every individual fly will be plotted., by default False
    beh_name : str, optional
        which behavioural classification to compare before and after head cutting, by default "walk"
    return_var : str, optional
        which behavioural variable to compare before and after head cutting. Must be inside "beh_df.pkl", by default "v_forw"
    return_var_flip : bool, optional
        whether to multiply behavioural variable by -1 for better visualisation, by default False
    return_var_abs : bool, optional
        whether to take the absolute value of the return variable, by default False
    return_var_ylim : list, optional
        provide an external ylim, e.g. [0,1], by default None
    return_var_multiply : float, optional
        whether to multiply return variable by a factor, e.g. 4.8 to convert from pixels to um, by default None
    return_var_baseline : list, optional
        whether to plot return variable relative to a baseline. [400,500] would take the 1s (100 samples) before stimulus onset as baseline, by default None
    return_var_ylabel : str, optional
        ylabel to plot for the behavioural variable, by default r"{||}$ (mm/s)"
    data_save_location : str, optional
        base directory where to save intermediate results, by default params.predictionsdata_base_dir
    plots_save_location : str, optional
        base directory where to save plots, by default params.predictionsplots_base_dir
    accept_headless_only_flies : bool, optional
        whether to also consider flies that have only headless experiments and no intact experiments.
        This will set the intact data to zero when it encounters a fly with only headless data, by default True
    accept_intact_only_flies : bool, optional
        whether to also consider flies that have only intact experiments and no headless experiments.
        This will set the headless data to zero when it encounters a fly with only intact data, by default True
    filter_pre_stim_beh : str, optional
        whether to filter the pre-stimulus behaviour.
        E.g. "rest" will only consider trials where the fly was resting before
        the stimulus onset with p > 0.75, by default 'None',
    include_noball_data : bool, optional
        whether to include trials without ball, by default False
    zero_baseline : bool, optional
        whether to set the signal to zero just at stimulation onset to zero, by
        default False
    stats_period : tuple, optional
        which period to use for statistics, by default (500,750)
    dataset : str, optional
        which dataset to use. Should be either 'prediction' or 'headless', default 'prediction'

    Returns
    -------
    fig
        Matplotlib figure
    """
    if dataset == "prediction":
        df = summarydf.get_predictions_df()
    elif dataset == "headless":
        df = summarydf.get_headless_df()
    elif dataset == "revisions":
        df = summarydf.get_revisions_df()
    else:
        raise(NotImplementedError)
    df = summarydf.get_selected_df(df, select_dicts=[{"CsChrimson": GAL4}])
    # df = summarydf.get_selected_df(df, select_dicts=[{"experimenter": 'FH'}])
    # df = summarydf.get_selected_df(df, select_dicts=[{"date": 230704}])

    figure_params = {
        "trigger": "laser_start",
        "stim_p": [10, 20],
        "response_name": None,
        "beh_name": beh_name,  # which coarse behaviour to look at more.
        "return_var": return_var,  # which variable from beh_df.pkl to compare
        "return_var_flip": return_var_flip,  # whether to multiply return variable with -1
        "return_var_abs": return_var_abs,  # whether to take absolute value of return variable
        "return_var_change": return_var_baseline,  # could be [400,500] for 1s baseline before stimulation
        "return_var_multiply": return_var_multiply,  # could be 4.8 for pixels -> um
        "beh_response_ylabel": return_var_ylabel,
        "suptitle": f"{GAL4} > CsChrimson",
        "panel_size": (20,3) if not allflies_only else (3,4),
        "mosaic": mosaic_predictions_panel if not allflies_only else fig_headless.mosaic_headless_summary_panel,
        "allflies_only": allflies_only,
        "ylim": return_var_ylim,
        "accept_headless_only_flies": accept_headless_only_flies,
        "accept_intact_only_flies": accept_intact_only_flies,
        "filter_pre_stim_beh": filter_pre_stim_beh,
        "include_noball_data": include_noball_data,
        "zero_baseline":zero_baseline,
        "stats_period":stats_period,
    }
    add_str = "_allflies_only" if allflies_only else ""

    predictions_save = os.path.join(
        data_save_location, f"predictions_{GAL4}_{return_var}.pkl"
    )
    all_fly_data = load_data_one_genotype(
        df, figure_params, predictions_save, overwrite=overwrite
    )
    fig = plot_data_one_genotype(figure_params, all_fly_data)
    if plot_save_location is not None:
        fig.savefig(
            os.path.join(
                plot_save_location, f"{GAL4}_predictions_{return_var}{add_str}.pdf"
            ),
            transparent=True,
        )
    return fig

def predictions_stats_tests(tmpdata_path=None):
    """
    Perform statistical tests on predictions data.
    
    Parameters:
    - tmpdata_path (str): Path to temporary data.

    Returns:
    - None
    """
    if tmpdata_path is None:
        tmpdata_path = params.predictionsdata_base_dir
    tests_pre_post = [
        {"GAL4": "DNa01", "var": "v_turn", "name": "turn vel"},
        {"GAL4": "aDN1", "var": "frtita_neck_dist", "name": "approach"},
        {"GAL4": "aDN1", "var": "mef_tita", "name": "front motion"},
        {"GAL4": "aDN1", "var": "ang_frtibia", "name": "tibia angle"},
        {"GAL4": "DNa02", "var": "v_turn", "name": "turn vel"},
        {"GAL4": "DNb02", "var": "v_turn", "name": "turn vel"},
        {"GAL4": "DNb02", "var": "v_forw", "name": "forward vel"},
        {"GAL4": "DNg14", "var": "anus_y_rel_neck", "name": "abd dip"},
        {"GAL4": "mute", "var": "ovum_x_rel_neck", "name": "ovum"},
    ]
    tests_pre_post_control = [
        {"GAL4": "PR", "var": "v_turn", "name": "turn vel"},
        {"GAL4": "PR", "var": "frtita_neck_dist", "name": "approach"},
        {"GAL4": "PR", "var": "mef_tita", "name": "front motion"},
        {"GAL4": "PR", "var": "ang_frtibia", "name": "tibia angle"},
        {"GAL4": "PR", "var": "v_turn", "name": "turn vel"},
        {"GAL4": "PR", "var": "v_turn", "name": "turn vel"},
        {"GAL4": "PR", "var": "v_forw", "name": "forward vel"},
        {"GAL4": "PR", "var": "anus_y_rel_neck", "name": "abd dip"},
        {"GAL4": "PR", "var": "ovum_x_rel_neck", "name": "ovum"},
    ]

    for to_test in tests_pre_post + tests_pre_post_control:
        file_name = os.path.join(tmpdata_path, f"predictions_{to_test['GAL4']}_{to_test['var']}.pkl")
        with open(file_name, "rb") as f:
            fly_data = pickle.load(f)
        fig_headless.test_stats_pre_post(fly_data, i_beh=0, GAL4=to_test['GAL4'], beh_name=None, var_name=to_test["name"], i_0=500, i_1=750 if "t2" not in to_test.keys() else to_test["t2"])

    for to_test_exp, to_test_control in zip(tests_pre_post, tests_pre_post_control):
        assert to_test_exp["name"] == to_test_control["name"]
        file_name_exp = os.path.join(tmpdata_path, f"predictions_{to_test_exp['GAL4']}_{to_test_exp['var']}.pkl")
        file_name_control = os.path.join(tmpdata_path, f"predictions_{to_test_control['GAL4']}_{to_test_control['var']}.pkl")
        with open(file_name_exp, "rb") as f:
            fly_data_exp = pickle.load(f)
        with open(file_name_control, "rb") as f:
            fly_data_control = pickle.load(f)

        fig_headless.test_stats_beh_control(fly_data_exp, fly_data_control, GAL4=to_test_exp['GAL4'], beh_name=to_test_exp["name"], i_0=500, i_1=750 if "t2" not in to_test_exp.keys() else to_test["t2"])

def make_all_predictions_figures(allflies_only=True, tmpdata_path=None, figures_path=None, overwrite=False):
    """
    Generate all predictions figures.

    Parameters:
    - allflies_only (bool): Whether to generate figures for all flies or for individual flies.
    - tmpdata_path (str): Path to temporary data.
    - figures_path (str): Path to save figures.
    - overwrite (bool): Whether to overwrite existing temporary data.

    Returns:
    - None
    """
    ylim_v_turn = [-100,600]
    ylim_v_forw = [-1,1]
    ylim_mef = [-0.3,1.3]
    ylim_ang_frti = [-15,40]
    ylim_dist_frtita = [-175,100]
    ylim_abd_dip = [-150,75]
    ylim_ovi_ext = [-100,50]

    fig = summarise_predictions_one_genotype("aDN1", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_dist_frtita, overwrite=overwrite, beh_name="groom",
                                             return_var="frtita_neck_dist", return_var_ylabel="front leg tita - head dist (um)", return_var_baseline=[400,500], return_var_multiply=4.8, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("aDN1", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_mef, overwrite=overwrite, beh_name="groom", return_var="mef_tita",
                                             return_var_ylabel="front leg speed (mm/s)", return_var_baseline=[400,500], return_var_multiply=4.8/10, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("aDN1", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_ang_frti, overwrite=overwrite, beh_name="groom", return_var="ang_frtibia",
                                             return_var_ylabel="tibia angle (°)", return_var_baseline=[400,500], allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("DNa01", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_v_turn, overwrite=overwrite, return_var="v_turn",
                                             return_var_ylabel=r"$v_{T}$ (°/s)", return_var_abs=True, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("DNb02", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_v_turn, overwrite=overwrite, return_var="v_turn",
                                             return_var_ylabel=r"$v_{T}$ (°/s)", return_var_abs=True, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("DNb02", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_v_forw, overwrite=overwrite, allflies_only=True)
    fig = summarise_predictions_one_genotype("DNa02", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_v_turn, overwrite=overwrite, return_var="v_turn",
                                             return_var_ylabel=r"$v_{T}$ (°/s)", return_var_abs=True, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("DNg14", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_abd_dip, overwrite=overwrite, return_var="anus_y_rel_neck",
                                             return_var_ylabel=r"anus y (um)", return_var_baseline=[400,500], return_var_multiply=4.8, return_var_flip=True, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("mute", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_ovi_ext, overwrite=overwrite, return_var="ovum_x_rel_neck",
                                             return_var_ylabel=r"ovum x", return_var_baseline=[400,500], return_var_multiply=4.8, allflies_only=allflies_only)
    
    fig = summarise_predictions_one_genotype("PR",dataset="headless", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_v_forw, overwrite=overwrite, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("PR",dataset="headless", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_v_turn, overwrite=overwrite, return_var="v_turn",
                                             return_var_ylabel=r"$v_{T}$ (°/s)", return_var_abs=True, allflies_only=allflies_only)  
    fig = summarise_predictions_one_genotype("PR",dataset="headless", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_abd_dip, overwrite=overwrite, return_var="anus_y_rel_neck",
                                             return_var_ylabel=r"anus y (um)", return_var_baseline=[400,500], return_var_multiply=4.8, return_var_flip=True, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("PR",dataset="headless", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_ovi_ext, overwrite=overwrite, return_var="ovum_x_rel_neck",
                                             return_var_ylabel=r"ovum x", return_var_baseline=[400,500], return_var_multiply=4.8, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("PR",dataset="headless", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_dist_frtita, overwrite=overwrite, beh_name="groom", return_var="frtita_neck_dist",
                                             return_var_ylabel="front leg tita - head dist (um)", return_var_baseline=[400,500], return_var_multiply=4.8, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("PR",dataset="headless", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_mef, overwrite=overwrite, beh_name="groom", return_var="mef_tita",
                                             return_var_ylabel="front leg speed (mm/s)", return_var_baseline=[400,500], return_var_multiply=4.8/10, allflies_only=allflies_only)
    fig = summarise_predictions_one_genotype("PR",dataset="headless", data_save_location=tmpdata_path, plot_save_location=figures_path,
                                             return_var_ylim=ylim_ang_frti, overwrite=overwrite, beh_name="groom", return_var="ang_frtibia",
                                             return_var_ylabel="tibia angle (°)", return_var_baseline=[400,500], allflies_only=allflies_only)
    
    _ = summarise_predictions_one_genotype("CantonS",return_var_ylim=[55,95],return_var='ang_frtibia_neck',return_var_ylabel=r"ang_{frtita}",
                                            return_var_multiply=1, accept_headless_only_flies=False, accept_intact_only_flies=False,
                                            zero_baseline=False, dataset="revisions", overwrite=True, allflies_only=allflies_only)
    _ = summarise_predictions_one_genotype("CantonS",return_var_ylim=[-1.5,0.5],return_var='v_forw',
                                            return_var_multiply=1, accept_headless_only_flies=False, accept_intact_only_flies=False,
                                            zero_baseline=False, dataset="revisions", overwrite=True, allflies_only=allflies_only)
    _ = summarise_predictions_one_genotype("CantonS",return_var_ylim=[-350,10],return_var='ovum_y',return_var_ylabel=r"ovum y",
                                            return_var_multiply=4.8, accept_headless_only_flies=False, accept_intact_only_flies=False,
                                            zero_baseline=True, dataset="revisions", overwrite=True, allflies_only=allflies_only)
    _ = summarise_predictions_one_genotype("DNg11",return_var_ylim=[55,95],return_var='ang_frtibia_neck',return_var_ylabel=r"ang_{frtita}",
                                            return_var_multiply=1, accept_headless_only_flies=True, accept_intact_only_flies=True,
                                            zero_baseline=False, dataset="revisions", overwrite=True, allflies_only=allflies_only)
    _ = summarise_predictions_one_genotype("DNp42",return_var_ylim=[-1.5,0.5],return_var='v_forw',
                                            return_var_multiply=1, accept_headless_only_flies=False, accept_intact_only_flies=False,
                                            zero_baseline=False, dataset="revisions", overwrite=True, allflies_only=allflies_only)
    _ = summarise_predictions_one_genotype("oviDN",return_var_ylim=[-350,10],return_var='ovum_y',return_var_ylabel=r"ovum y",
                                            return_var_multiply=4.8, accept_headless_only_flies=False, accept_intact_only_flies=False,
                                            zero_baseline=True, dataset="revisions", overwrite=True, allflies_only=allflies_only)
    

if __name__ == "__main__":
    make_all_predictions_figures()
    predictions_stats_tests()
    
    