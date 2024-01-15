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


def load_data_one_genotype(
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
        summary_fly_data = base_fly_data.copy()

        for i_fly, (fly_id, fly_df) in enumerate(exp_df.groupby("fly_id")):
            fly_data = base_fly_data.copy()
            fly_data["fly_df"] = fly_df
            fly_data["fly_dir"] = np.unique(fly_df.fly_dir)[0]
            fly_data["trial_names"] = fly_df.trial_name.values
            headless_trial_exists = False
            for index, trial_df in fly_df.iterrows():
                if (
                    not trial_df.walkon == "ball"
                    and not figure_params["include_noball_data"]
                ):
                    continue  # TODO: make no ball analysis
                else:
                    if trial_df["head"]:
                        beh_key = "beh_responses_pre"
                        beh_class_key = "beh_class_responses_pre"
                        headless_trial_exists = True
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
                not headless_trial_exists
                and figure_params["accept_headless_only_flies"]
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
            elif not headless_trial_exists:
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
    data_list = [
        fly_data[column]
        for fly_data in all_fly_data
        if ((fly_data[column] is not None and not np.all(np.isnan(fly_data[column]))))
    ]
    return np.concatenate(data_list, axis=-1)


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


def summarise_predictions_one_genotype(
    GAL4,
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
    accept_headless_only_flies=True,
    include_noball_data=False,
    filter_pre_stim_beh=None,
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
        "ylim": None,
        "accept_headless_only_flies": accept_headless_only_flies,
        "filter_pre_stim_beh": filter_pre_stim_beh,
        "include_noball_data": include_noball_data,
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


if __name__ == "__main__":
    fig = summarise_predictions_one_genotype(
        "CantonS",
        beh_name="back",
        return_var='v_forw',  # "me_front",
        return_var_ylabel=r"$v_{||}$ (mm/s)",
        overwrite=True,
        accept_headless_only_flies=True,
        return_var_flip=False,
        include_noball_data=False,
        filter_pre_stim_beh=None,  # 'rest'
    )
