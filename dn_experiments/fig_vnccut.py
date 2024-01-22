"""
Module to generate figures related to functional imaging during optogenetic stimulation of flies for which the VNC was cut.
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

import params, summarydf, loaddata, stimulation, behaviour, plotpanels, fig_functional

revision_figures_path = os.path.join(params.plot_base_dir, "revision")

vnc_cut_flies = {
    "DNp09": [89,90,91],
    "PR": [95,96,97],
}
vnc_cut_presentation_flies = {
    "DNp09": [90],
    "PR": [95],
}

def summarise_vnccut_resp(overwrite=False, mode="pdf", figures_path=None, tmpdata_path=None):
    """
    Summarize all neuronal and behavioural responses for flies that have the VNC cut.

    Parameters:
        overwrite (bool, optional): Whether to overwrite cached data if it exists. Default is False.
        mode (str, optional): The mode for generating summary figures (e.g., "pdf"). Default is "pdf".
        figures_path (str, optional): Path to save generated figures. Default is None.
        tmpdata_path (str, optional): Path to save/load cached data. Default is None.

    Returns:
        None
    """
    if figures_path is None:
        figures_path = revision_figures_path  # params.plot_base_dir
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
        "normalisation_type": "qmin_qmax",
        "return_var": None,
        "beh_response_ylabel": None,
        "neural_response_ylabel": r"$\Delta$F/F",
        "clabel": r"$\Delta$F/F",
        "response_clim": 0.8,
        "response_beh_lim": None,
        "response_q_max": 0.95,
        "suptitle": "",
        "pre_stim": None,
        "panel_size": (15,4),
        "mosaic": fig_functional.mosaic_vnccut_stim_resp_panel,
        "min_resp": 10,
        "min_resp_2": 5,
        "mode": mode,
        "pres_fly": None,
        "selected_fly_ids": None,
        "vnccut": True,
    }

    if mode == "presentation":
        figure_params["mosaic"] = fig_functional.mosaic_vnccut_stim_resp_panel_presentation
    elif mode == "presentationsummary":
        figure_params["mosaic"] = fig_functional.mosaic_vnccut_stim_resp_panel_presentationsummary
        figure_params["panel_size"] = (4,8)
    
    # DNp09
    df_Dfd_DNp09 = summarydf.get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "DNp09", "walkon": "no"}])
    figure_params_DNp09 = figure_params.copy()
    figure_params_DNp09["suptitle"] = f"Dfd response to DNp09 stimulation after cut in VNC"
    figure_params_DNp09["pres_fly"] = vnc_cut_presentation_flies["DNp09"]
    figure_params_DNp09["selected_fly_ids"] = vnc_cut_flies["DNp09"]
    fig_DNp09 = fig_functional.summarise_stim_resp(df_Dfd_DNp09, figure_params_DNp09, stim_resp_save=os.path.join(tmpdata_path, f"vnccut_stim_resp_DNp09.pkl"),
                                    overwrite=overwrite)
    
    # PR
    df_Dfd_PR = summarydf.get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "PR", "walkon": "no"}])
    figure_params_PR = figure_params.copy()
    figure_params_PR["suptitle"] = f"Dfd response to PR stimulation after cut in VNC"
    figure_params_PR["pres_fly"] = vnc_cut_presentation_flies["PR"]
    figure_params_PR["selected_fly_ids"] = vnc_cut_flies["PR"]
    fig_PR = fig_functional.summarise_stim_resp(df_Dfd_PR, figure_params_PR, stim_resp_save=os.path.join(tmpdata_path, f"vnccut_stim_resp_PR.pkl"),
                                    overwrite=overwrite)
    
    figs = [fig_DNp09, fig_PR]
    with PdfPages(os.path.join(figures_path, f"fig_vnccut_func_summary_{mode}.pdf")) as pdf:
        _ = [pdf.savefig(fig, transparent=True) for fig in figs if fig is not None]
    _ = [plt.close(fig) for fig in figs if fig is not None]

if __name__ == "__main__":
    summarise_vnccut_resp(overwrite=False, mode="pdf", figures_path=revision_figures_path)