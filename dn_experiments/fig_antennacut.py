"""
Functions to generate plots to analyse the antennae cutting experiments.
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

import params, summarydf, loaddata, stimulation, behaviour, plotpanels, fig_headless

from twoppp import plot as myplt

def summarise_all_antennacut(overwrite=False, allflies_only=True, tmpdata_path=None, figures_path=None):
    """
    Summarize and visualize behavioral responses before and after antenna cutting and save them as a PDF.

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
        "mosaic": fig_headless.mosaic_headless_panel,
        "allflies_only": False,
        "ylim": None,
    }
    if allflies_only:
        figure_params["mosaic"] = fig_headless.mosaic_headless_summary_panel
        figure_params["allflies_only"] = True
        figure_params["panel_size"] = (3,4)
        add_str = "_allflies_only"
    else:
        add_str = ""
    
    # aDN2
    df_aDN2 = summarydf.get_selected_df(df, [{"CsChrimson": "aDN2", "plot_appendix": "antennacut"}])
    figure_params_aDN2 = figure_params.copy()
    figure_params_aDN2["suptitle"] = "aDN2 > CsChrimson"
    figure_params_aDN2["beh_name"] = "groom"
    figure_params_aDN2["ylim"] = [-3,7]
    fig_aDN2 = fig_headless.summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(tmpdata_path, f"antennacut_aDN2.pkl"),
                                    overwrite=overwrite)
    # additional kinematics parameters
    figure_params_aDN2["return_var"] = "frtita_neck_dist"
    figure_params_aDN2["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_aDN2["beh_response_ylabel"] = "front leg tita - head dist (um)"
    figure_params_aDN2["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_aDN2["return_var_flip"] = False 
    figure_params_aDN2["ylim"] = [-200,100]
    fig_aDN2_1 = fig_headless.summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(tmpdata_path, f"antennacut_aDN2_frtita_dist.pkl"),
                                    overwrite=overwrite)

    
    figs = [fig_aDN2_1, fig_aDN2]

    with PdfPages(os.path.join(figures_path, f"fig_antennacut_summary_ball{add_str}.pdf")) as pdf:
        _ = [pdf.savefig(fig) for fig in figs]
    _ = [plt.close(fig) for fig in figs]

def antennacut_stat_test(tmpdata_path=None):
    """
    Perform statistical tests on behavioral data before and after antenna cutting.

    Args:
        tmpdata_path (str): Path to the directory containing temporary data files.

    Returns:
        None
    """
    if tmpdata_path is None:
        tmpdata_path = params.plotdata_base_dir
    headless_files = {
        "aDN2": os.path.join(tmpdata_path, "antennacut_aDN2_frtita_dist.pkl"),
    }
    with open(headless_files["aDN2"], "rb") as f:
        aDN2 = pickle.load(f)
        
    fig_headless.test_stats_pre_post(aDN2, i_beh=4, GAL4="aDN2", beh_name="groom", var_name="frtita_dist")


if __name__ == "__main__":
    summarise_all_antennacut(overwrite=False, allflies_only=False, figures_path=os.path.join(params.plot_base_dir, "revision"))
    summarise_all_antennacut(overwrite=False, allflies_only=True, figures_path=os.path.join(params.plot_base_dir, "revision"))

    antennacut_stat_test()