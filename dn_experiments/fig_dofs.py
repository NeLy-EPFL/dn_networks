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
from fig_predictions import concatenate

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


def load_data_one_genotype_amputation(
    exp_df, figure_params, predictions_save=None, overwrite=False
):
    if (
        predictions_save is not None
        and os.path.isfile(predictions_save)
        and not overwrite
    ):
        with open(predictions_save, "rb") as f:
            all_fly_data = pickle.load(f)
    else:
        # load data for all flies
        all_fly_data = []

        for i_fly, (fly_id, fly_df) in enumerate(exp_df.groupby("fly_id")):
            fly_data = base_fly_data.copy()
            fly_data["fly_df"] = fly_df
            fly_data["fly_dir"] = np.unique(fly_df.fly_dir)[0]
            fly_data["trial_names"] = fly_df.trial_name.values
            amputation_trial_exists = False
            for index, trial_df in fly_df.iterrows():
                if (
                    not trial_df.walkon == "ball"
                    and not figure_params["include_noball_data"]
                ):
                    continue  # TODO: make no ball analysis
                else:
                    if not trial_df["leg_amp"] in ["HL", "ML", "FL"]:
                        beh_key = "beh_responses_pre"
                        beh_class_key = "beh_class_responses_pre"
                        amputation_trial_exists = True
                    else:
                        beh_key = "beh_responses_post"
                        beh_class_key = "beh_class_responses_post"
                beh_df = loaddata.load_beh_data_only(
                    fly_data["fly_dir"], all_trial_dirs=[trial_df.trial_name]
                )

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
                    beh_responses = (
                        -1 * beh_responses
                    )  # make the presentation more intuitive, e.g. front leg height

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
                not amputation_trial_exists
                and figure_params["accept_amputated_only_flies"]
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
            elif not amputation_trial_exists:
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


def plot_data_one_genotype(figure_params, all_fly_data):
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
    summary_fly_data["beh_responses_pre"] = concatenate(
        all_fly_data, "beh_responses_pre"
    )
    summary_fly_data["beh_responses_post"] = concatenate(
        all_fly_data, "beh_responses_post"
    )  # np.concatenate([fly_data["beh_responses_post"] for fly_data in all_fly_data], axis=-1)
    summary_fly_data["beh_class_responses_pre"] = concatenate(
        all_fly_data, "beh_class_responses_pre"
    )
    summary_fly_data["beh_class_responses_post"] = concatenate(
        all_fly_data, "beh_class_responses_post"
    )
    fig_headless.get_one_fly_headless_panel(
        fig, axds[-1], summary_fly_data, figure_params
    )

    fig.suptitle(
        figure_params["suptitle"], fontsize=30, y=1 - 0.01 * 5 / (len(subfigs))
    )
    return fig


def summarise_predictions_dofs(
    GAL4,
    specific_joint="TiTa",
    specific_leg="HL",
    overwrite=False,
    allflies_only=False,
    beh_name="walk",
    return_var="v_forw",
    return_var_flip=False,
    return_var_multiply=None,
    return_var_baseline=None,
    return_var_ylabel=r"$v_{||}$ (mm/s)",
    data_save_location=params.predictionsdata_base_dir,
    plot_save_location=params.predictionsplot_base_dir,
    accept_amputated_only_flies=True,
    filter_pre_stim_beh=None,
    zero_baseline=False,
    stats_period=(500,750),
    ylim=None,

):
    """make a figure for one genotype and one behavioural response,
     before and after leg cutting

    Parameters
    ----------
    GAL4 : str
        which GAL4 line to analyse
    specific_joint : str, optional
        which joint to restrict the analysis to
    specific_leg : str, optional
        which leg to restrict the analysis to
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
    accept_amputated_only_flies : bool, optional
        whether to also consider flies that have only leg cutting experiments and no intact experiments.
        This will set the intact data to zero when it encounters a fly with only leg cutting data, by default True
    filter_pre_stim_beh : str, optional
        whether to filter the pre-stimulus behaviour.
        E.g. "rest" will only consider trials where the fly was resting before
        the stimulus onset with p > 0.75, by default 'None',

    Returns
    -------
    fig
        Matplotlib figure
    """
    df = summarydf.get_predictions_df()
    df = summarydf.get_selected_df(df, select_dicts=[{"CsChrimson": GAL4}])
    #print(df[['fly_dir', 'joint_amp', 'leg_amp']])
    # intact flies
    df_intact = summarydf.get_selected_df(df, select_dicts=[{"leg_amp": 'FALSE', "head": True}])
    # need to filter for flies that have had specific amputations, but keep the controls
    df_target = summarydf.get_selected_df(
        df, select_dicts=[{"joint_amp": specific_joint}]
    )
    df_target = summarydf.get_selected_df(
        df_target, select_dicts=[{"leg_amp": specific_leg}]
    )
    df = df[df.trial_dir.isin(df_target.trial_dir.unique()) | df.trial_dir.isin(df_intact.trial_dir.unique())]

    # df = summarydf.get_selected_df(df, select_dicts=[{"experimenter": 'FH'}])
    # df = summarydf.get_selected_df(df, select_dicts=[{"date": 230704}])

    figure_params = {
        "trigger": "laser_start",
        "stim_p": [10, 20],
        "response_name": None,
        "beh_name": beh_name,  # which coarse behaviour to look at more.
        "return_var": return_var,  # which variable from beh_df.pkl to compare
        "return_var_flip": return_var_flip,  # whether to multiply return variable with -1
        "return_var_change": return_var_baseline,  # could be [400,500] for 1s baseline before stimulation
        "return_var_multiply": return_var_multiply,  # could be 4.8 for pixels -> um
        "beh_response_ylabel": return_var_ylabel,
        "suptitle": f"{GAL4} > CsChrimson",
        "panel_size": (20, 3) if not allflies_only else (3, 4),
        "mosaic": mosaic_predictions_panel,
        "allflies_only": allflies_only,
        "ylim": ylim,
        "accept_amputated_only_flies": accept_amputated_only_flies,
        "filter_pre_stim_beh": filter_pre_stim_beh,
        "zero_baseline": zero_baseline,
        "stats_period":stats_period,

    }
    add_str = "_allflies_only" if allflies_only else ""
    add_str += "_{}_{}".format(specific_joint, specific_leg)

    predictions_save = os.path.join(
        data_save_location, f"predictions_{GAL4}_{specific_joint}_{specific_leg}_{return_var}.pkl"
    )
    all_fly_data = load_data_one_genotype_amputation(
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

def make_standard_dof_panel(GAL4, specific_joint, default_params, ylim=None):
    """
    Generate a standard panel for a given genotype, joint and variable
    """
    _ = summarise_predictions_dofs(
        GAL4,
        specific_joint=specific_joint,
        beh_name=default_params['beh_name'],
        return_var=default_params['variable'],
        return_var_ylabel=default_params['label'],
        overwrite=True,
        accept_amputated_only_flies=default_params['accept_amputated_only_flies'],
        return_var_flip=default_params['return_var_flip'],
        filter_pre_stim_beh=default_params['filter_pre_stim_beh'],
        zero_baseline=default_params['zero_baseline'],
        stats_period=default_params['stats_period'],
        ylim=ylim
    )

def make_all_dof_panels():
    """
    Generate all panels for all genotypes, joints and legs
    """
    default_params = {
        'joint' : "TiTa",
        'variable' : "integrated_forward_movement",
        'label' : r"$d_{forward}$",
        'accept_amputated_only_flies':True,
        'return_var_flip':False,
        'filter_pre_stim_beh':None,
        'zero_baseline':True,
        'stats_period':(999,1000),
        'beh_name':'walk',
    }

    make_standard_dof_panel('MDN', 'HL', default_params, ylim=[-7.5,0.5])
    make_standard_dof_panel('DNp09', 'HL', default_params, ylim=[-2,40])
    make_standard_dof_panel('DNp09', 'ML', default_params, ylim=[-2,40])
    make_standard_dof_panel('DNp09', 'FL', default_params, ylim=[-2,40])


if __name__ == "__main__":
    make_all_dof_panels()
