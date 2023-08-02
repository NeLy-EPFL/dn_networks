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

def get_one_fly_headless_panel(fig, axd, fly_data, figure_params):

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
    plotpanels.plot_ax_behavioural_response(fly_data["beh_responses_pre"], ax=axd["X"], x="beh",
            response_name=response_name,
            response_ylabel=figure_params["beh_response_ylabel"] if figure_params["allflies_only"] else None,
            beh_responses_2=fly_data["beh_responses_post"], beh_response_2_color=behaviour.get_beh_color(figure_params["beh_name"]))


    if not figure_params["allflies_only"]:
        # V: volocity response pre head cut
        plotpanels.plot_ax_behavioural_response(fly_data["beh_responses_pre"], ax=axd["V"], x="beh",
                response_name="head intact", response_ylabel=figure_params["beh_response_ylabel"])

        # W: volocity response post head cut
        plotpanels.plot_ax_behavioural_response(fly_data["beh_responses_post"], ax=axd["W"], x="beh",
                response_name="no head", response_ylabel=None)
    
    
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


def summarise_all_headless(overwrite=False, allflies_only=False):
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
    fig_MDN = summarise_headless(df_MDN, figure_params_MDN, headless_save=os.path.join(params.plotdata_base_dir, f"headless_MDN3.pkl"),
                                    overwrite=overwrite)
    
    # DNp09
    df_DNp09 = summarydf.get_selected_df(df, [{"CsChrimson": "DNp09"}])
    figure_params_DNp09 = figure_params.copy()
    figure_params_DNp09["suptitle"] = "DNp09 > CsChrimson"
    figure_params_DNp09["beh_name"] = "walk"
    fig_DNp09 = summarise_headless(df_DNp09, figure_params_DNp09, headless_save=os.path.join(params.plotdata_base_dir, f"headless_DNp09.pkl"),
                                    overwrite=overwrite)
    figure_params_DNp09["return_var"] = "anus_dist"
    figure_params_DNp09["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_DNp09["beh_response_ylabel"] = "anal plate (um)"
    figure_params_DNp09["return_var_multiply"] = 4.8  # pixeles -> um
    fig_DNp09_1 = summarise_headless(df_DNp09, figure_params_DNp09, headless_save=os.path.join(params.plotdata_base_dir, f"headless_DNp09_anus.pkl"),
                                    overwrite=overwrite)
    figure_params_DNp09["return_var"] = "ovum_dist"
    figure_params_DNp09["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_DNp09["beh_response_ylabel"] = "ovipositor (um)"
    figure_params_DNp09["return_var_multiply"] = 4.8  # pixeles -> um
    fig_DNp09_2 = summarise_headless(df_DNp09, figure_params_DNp09, headless_save=os.path.join(params.plotdata_base_dir, f"headless_DNp09_ovum.pkl"),
                                    overwrite=overwrite)
    
    # aDN2
    df_aDN2 = summarydf.get_selected_df(df, [{"CsChrimson": "aDN2"}])
    figure_params_aDN2 = figure_params.copy()
    figure_params_aDN2["suptitle"] = "aDN2 > CsChrimson"
    figure_params_aDN2["beh_name"] = "groom"
    fig_aDN2 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(params.plotdata_base_dir, f"headless_aDN2.pkl"),
                                    overwrite=overwrite)
    figure_params_aDN2["return_var"] = "frleg_height"
    figure_params_aDN2["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_aDN2["beh_response_ylabel"] = "front leg height (um)"
    figure_params_aDN2["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_aDN2["return_var_flip"] = True  # make the positive direction more intuitive
    fig_aDN2_2 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(params.plotdata_base_dir, f"headless_aDN2_frleg_height.pkl"),
                                    overwrite=overwrite)
    figure_params_aDN2["return_var"] = "ang_frfemur"
    figure_params_aDN2["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_aDN2["beh_response_ylabel"] = "femur angle (°)"
    figure_params_aDN2["return_var_multiply"] = None
    figure_params_aDN2["return_var_flip"] = False 
    fig_aDN2_3 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(params.plotdata_base_dir, f"headless_aDN2_femur_angle.pkl"),
                                    overwrite=overwrite)
    figure_params_aDN2["return_var"] = "ang_frtibia"
    figure_params_aDN2["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_aDN2["beh_response_ylabel"] = "tibia angle (°)"
    figure_params_aDN2["return_var_flip"] = False 
    fig_aDN2_4 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(params.plotdata_base_dir, f"headless_aDN2_tibia_angle.pkl"),
                                    overwrite=overwrite)
    figure_params_aDN2["return_var"] = "mef_tita"
    figure_params_aDN2["return_var_change"] = None
    figure_params_aDN2["beh_response_ylabel"] = "front leg speed (um)"
    figure_params_aDN2["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_aDN2["return_var_flip"] = False 
    fig_aDN2_5 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(params.plotdata_base_dir, f"headless_aDN2_mef.pkl"),
                                    overwrite=overwrite)
    # additional kinematics parameters
    figure_params_aDN2["return_var"] = "frtita_neck_dist"
    figure_params_aDN2["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_aDN2["beh_response_ylabel"] = "front leg tita - head dist (um)"
    figure_params_aDN2["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_aDN2["return_var_flip"] = False 
    fig_aDN2_6 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(params.plotdata_base_dir, f"headless_aDN2_frtita_dist.pkl"),
                                    overwrite=overwrite)
    figure_params_aDN2["return_var"] = "frfeti_neck_dist"
    figure_params_aDN2["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_aDN2["beh_response_ylabel"] = "front leg feti - head dist (um)"
    figure_params_aDN2["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_aDN2["return_var_flip"] = False 
    fig_aDN2_7 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(params.plotdata_base_dir, f"headless_aDN2_frfeti_dist.pkl"),
                                    overwrite=overwrite)
    figure_params_aDN2["return_var"] = "ang_frtibia_neck"
    figure_params_aDN2["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_aDN2["beh_response_ylabel"] = "tibia - neck angle (°)"
    figure_params_aDN2["return_var_multiply"] = None
    figure_params_aDN2["return_var_flip"] = False 
    fig_aDN2_8 = summarise_headless(df_aDN2, figure_params_aDN2, headless_save=os.path.join(params.plotdata_base_dir, f"headless_aDN2_tibia_neck_angle.pkl"),
                                    overwrite=overwrite)

    # PR
    df_PR = summarydf.get_selected_df(df, [{"CsChrimson": "PR"}])
    figure_params_PR = figure_params.copy()
    figure_params_PR["suptitle"] = "__ > CsChrimson"
    figure_params_PR["beh_name"] = "rest"
    fig_PR = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["beh_name"] = "back"
    fig_PR1 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["beh_name"] = "walk"
    fig_PR2 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["return_var"] = "anus_dist"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "anal plate (um)"
    figure_params_PR["return_var_multiply"] = 4.8  # pixeles -> um
    fig_PR3 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR_anus.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["return_var"] = "ovum_dist"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "ovipositor (px)"
    figure_params_PR["return_var_multiply"] = 4.8  # pixeles -> um
    fig_PR4 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR_ovum.pkl"),
                                    overwrite=overwrite)

    figure_params_PR["beh_name"] = "groom"
    figure_params_PR["return_var"] = "frleg_height"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "front leg height (px)"
    figure_params_PR["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_PR["return_var_flip"] = True  # make the positive direction more intuitive
    fig_PR5 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR_frleg_height.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["return_var"] = "ang_frfemur"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "femur angle (°)"
    figure_params_PR["return_var_multiply"] = None
    figure_params_PR["return_var_flip"] = False 
    fig_PR6 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR_femur_angle.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["return_var"] = "ang_frtibia"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "tibia angle (°)"
    figure_params_PR["return_var_flip"] = False 
    fig_PR7 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR_tibia_angle.pkl"),
                                    overwrite=overwrite)

    figure_params_PR["return_var"] = "mef_tita"
    figure_params_PR["return_var_change"] = None
    figure_params_PR["beh_response_ylabel"] = "front leg speed (px)"
    figure_params_PR["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_PR["return_var_flip"] = False 
    fig_PR8 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR_mef.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["return_var"] = "frtita_neck_dist"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "front leg tita - head dist (px)"
    figure_params_PR["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_PR["return_var_flip"] = False 
    fig_PR9 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR_frtita_dist.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["return_var"] = "frfeti_neck_dist"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "front leg feti - head dist (px)"
    figure_params_PR["return_var_multiply"] = 4.8  # pixeles -> um
    figure_params_PR["return_var_flip"] = False 
    fig_PR10 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR_frfeti_dist.pkl"),
                                    overwrite=overwrite)
    figure_params_PR["return_var"] = "ang_frtibia_neck"
    figure_params_PR["return_var_change"] = [400,500]  # show relative changes by computing baseline of 1s before
    figure_params_PR["beh_response_ylabel"] = "tibia - neck angle (°)"
    figure_params_PR["return_var_multiply"] = None
    figure_params_PR["return_var_flip"] = False 
    fig_PR11 = summarise_headless(df_PR, figure_params_PR, headless_save=os.path.join(params.plotdata_base_dir, f"headless_PR_tibia_neck_angle.pkl"),
                                    overwrite=overwrite)

    
    figs = [fig_MDN, fig_DNp09, fig_DNp09_1, fig_DNp09_2, fig_aDN2, fig_aDN2_2, fig_aDN2_3, fig_aDN2_4, fig_aDN2_5, fig_aDN2_6, fig_aDN2_7, fig_aDN2_8,\
            fig_PR, fig_PR1, fig_PR2, fig_PR3, fig_PR4, fig_PR5, fig_PR6, fig_PR7, fig_PR8, fig_PR9, fig_PR10, fig_PR11]


    with PdfPages(os.path.join(params.plot_base_dir, f"fig_headless_summary_ball{add_str}.pdf")) as pdf:
        _ = [pdf.savefig(fig) for fig in figs]
    _ = [plt.close(fig) for fig in figs]


def headless_stat_test():
    headless_files = {
        "MDN": os.path.join(params.plotdata_base_dir, "headless_MDN3.pkl"),
        "DNp09": os.path.join(params.plotdata_base_dir, "headless_DNp09.pkl"),
        "aDN2": os.path.join(params.plotdata_base_dir, "headless_aDN2.pkl"),
        "PR": os.path.join(params.plotdata_base_dir, "headless_PR.pkl"),
    }
    with open(headless_files["MDN"], "rb") as f:
        MDN = pickle.load(f)
    with open(headless_files["DNp09"], "rb") as f:
        DNp09 = pickle.load(f)
    with open(headless_files["aDN2"], "rb") as f:
        aDN2 = pickle.load(f)
    with open(headless_files["PR"], "rb") as f:
        PR = pickle.load(f)
    
    def test_stats_pre_post(all_flies, i_beh, GAL4, beh_name, i_0=500, i_1=750):
        v_pre = []
        p_pre = []
        v_post = []
        p_post = []
        for fly in all_flies:
            v_pre.append(np.mean(fly["beh_responses_pre"][i_0:i_1], axis=0))
            v_post.append(np.mean(fly["beh_responses_post"][i_0:i_1], axis=0))
            p_pre.append(np.mean(fly["beh_class_responses_pre"][i_0:i_1] == i_beh, axis=0))
            p_post.append(np.mean(fly["beh_class_responses_post"][i_0:i_1] == i_beh, axis=0))
        v_pre = np.concatenate(v_pre).flatten()
        v_post = np.concatenate(v_post).flatten()
        p_pre = np.concatenate(p_pre).flatten()
        p_post = np.concatenate(p_post).flatten()
        print(f"{GAL4} v:", mannwhitneyu(v_pre, v_post))
        print(f"{GAL4} {beh_name} beh class:", mannwhitneyu(p_pre, p_post))
        
    test_stats_pre_post(MDN, i_beh=3, GAL4="MDN", beh_name="back")
    test_stats_pre_post(DNp09, i_beh=1, GAL4="DNp09", beh_name="walk")
    test_stats_pre_post(aDN2, i_beh=4, GAL4="aDN2", beh_name="groom")
    test_stats_pre_post(PR, i_beh=2, GAL4="PR", beh_name="rest")

    detailled_files = {
        "DNp09_anus": os.path.join(params.plotdata_base_dir, "headless_DNp09_anus.pkl"),
        "PR_anus": os.path.join(params.plotdata_base_dir, "headless_PR_anus.pkl"),
        "aDN2_height": os.path.join(params.plotdata_base_dir, "headless_aDN2_frleg_height.pkl"),
        "PR_height": os.path.join(params.plotdata_base_dir, "headless_PR_frleg_height.pkl"),
        "aDN2_angle": os.path.join(params.plotdata_base_dir, "headless_aDN2_tibia_angle.pkl"),
        "PR_angle": os.path.join(params.plotdata_base_dir, "headless_PR_tibia_angle.pkl"),
        "aDN2_dist_tita": os.path.join(params.plotdata_base_dir, "headless_aDN2_frtita_dist.pkl"),
        "PR_dist_tita": os.path.join(params.plotdata_base_dir, "headless_PR_frtita_dist.pkl"),
        "aDN2_dist_feti": os.path.join(params.plotdata_base_dir, "headless_aDN2_frfeti_dist.pkl"),
        "PR_dist_feti": os.path.join(params.plotdata_base_dir, "headless_PR_frfeti_dist.pkl"),
    }
    with open(detailled_files["DNp09_anus"], "rb") as f:
        DNp09_anus = pickle.load(f)
    with open(detailled_files["PR_anus"], "rb") as f:
        PR_anus = pickle.load(f)
    with open(detailled_files["aDN2_height"], "rb") as f:
        aDN2_height = pickle.load(f)
    with open(detailled_files["PR_height"], "rb") as f:
        PR_height = pickle.load(f)
    with open(detailled_files["aDN2_angle"], "rb") as f:
        aDN2_angle = pickle.load(f)
    with open(detailled_files["PR_angle"], "rb") as f:
        PR_angle = pickle.load(f)
    with open(detailled_files["aDN2_dist_tita"], "rb") as f:
        aDN2_dist_tita = pickle.load(f)
    with open(detailled_files["PR_dist_tita"], "rb") as f:
        PR_dist_tita = pickle.load(f)
    with open(detailled_files["aDN2_dist_feti"], "rb") as f:
        aDN2_dist_feti = pickle.load(f)
    with open(detailled_files["PR_dist_feti"], "rb") as f:
        PR_dist_feti = pickle.load(f)

    def test_stats_beh_control(all_flies, all_flies_control, GAL4, beh_name, i_0=500, i_1=750):
        beh = []
        beh_control = []
        for fly, fly_control in zip(all_flies, all_flies_control):
            beh.append(np.mean(fly["beh_responses_post"][i_0:i_1], axis=0))
            beh_control.append(np.mean(fly_control["beh_responses_post"][i_0:i_1], axis=0))
        beh = np.concatenate(beh).flatten()
        beh_control = np.concatenate(beh_control).flatten()
        print(f"{GAL4} {beh_name}:", mannwhitneyu(beh, beh_control))

    test_stats_beh_control(DNp09_anus, PR_anus, GAL4="DNp09", beh_name="anus", i_0=500, i_1=750)

    # test_stats_beh_control(aDN2_height, PR_height, GAL4="aDN2", beh_name="height", i_0=750, i_1=1000)
    # test_stats_beh_control(aDN2_angle, PR_angle, GAL4="aDN2", beh_name="angle", i_0=750, i_1=1000)
    test_stats_beh_control(aDN2_dist_tita, PR_dist_tita, GAL4="aDN2", beh_name="dist tita", i_0=500, i_1=750)
    test_stats_beh_control(aDN2_dist_feti, PR_dist_feti, GAL4="aDN2", beh_name="dist feti", i_0=500, i_1=750)


if __name__ == "__main__":
    summarise_all_headless(overwrite=True, allflies_only=False)
    summarise_all_headless(overwrite=False, allflies_only=True)
    headless_stat_test()