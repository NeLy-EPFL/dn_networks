import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import pickle

from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter


from twoppp import load, rois, utils
from twoppp import plot as myplt
from twoppp.behaviour import synchronisation

# sys.path.append(os.path.dirname(__file__))
import params, summarydf, baselines, loaddata, stimulation, plotpanels, behaviour

def get_mosaic_from_panel_style(panel_style):
    if panel_style is None:
        mosaic = """
        EEEEEEEEEEEEEEEE.CCCCCCCCDAAAAAAAAAAAAAAAAAAAA
        BBBBBBBBBBBBBBBB.CCCCCCCCDAAAAAAAAAAAAAAAAAAAA
        BBBBBBBBBBBBBBBB.CCCCCCCCDAAAAAAAAAAAAAAAAAAAA
        """
    elif panel_style == "withbehclass":
        mosaic = """
        EEEEEEEEEEEEEEEE..FFFFFFFF.CCCCCCCCDAAAAAAAAAAAAAAAAAAAA
        BBBBBBBBBBBBBBBB..FFFFFFFF.CCCCCCCCDAAAAAAAAAAAAAAAAAAAA
        BBBBBBBBBBBBBBBB..FFFFFFFF.CCCCCCCCDAAAAAAAAAAAAAAAAAAAA
        """
    else:
        raise NotImplementedError
    return mosaic

def get_plots_from_panel_style(fig, axd, panel_style, figure_params, beh_responses, beh_class_responses, stim_responses, response_values, response_name,
                               background_image, roi_centers, fly_df, clim, sort_ind, comparison=False):
    if panel_style is None or panel_style == "withbehclass" and not comparison:
        # behavioural response
        plotpanels.plot_ax_behavioural_response(beh_responses, ax=axd["E"],
            response_name=response_name, response_ylabel=figure_params["beh_response_ylabel"])

        # all neurons response
        plotpanels.plot_ax_allneurons_response(stim_responses, ax=axd["B"],
            response_ylabel=figure_params["neural_response_ylabel"])

        # all neurons matrix with confidence interval
        plotpanels.plot_ax_allneurons_confidence(stim_responses, ax=axd["C"],
            clim=clim, sort_ind=sort_ind)

        # colour bar
        plotpanels.plot_ax_cbar(fig=fig, ax=axd["D"], clim=clim, clabel=figure_params["clabel"])

        # neuron summary over std image
        plotpanels.plot_ax_response_summary(background_image=background_image, roi_centers=roi_centers,
            ax=axd["A"], response_values=response_values, response_name=response_name,
            fly_name=f"{fly_df.date.values[0]} Fly {fly_df.fly_number.values[0]}",
            q_max=figure_params["response_q_max"], clim=clim)
    if panel_style == "withbehclass" and not comparison:
        plotpanels.plot_ax_behclass(beh_class_responses, ax=axd["F"])
    if comparison:
        axd["E"].axis("off")
        axd["B"].axis("off")
        axd["F"].axis("off")
        axd["C"].axis("off")
        plotpanels.plot_ax_cbar(fig=fig, ax=axd["D"], clim=clim, clabel=figure_params["clabel"])
        plotpanels.plot_ax_response_summary(background_image=background_image, roi_centers=roi_centers,
            ax=axd["A"], response_values=None, response_name=None,
            response_values_left=response_values[0], response_values_right=response_values[1],
            response_name_left=response_name[0], response_name_right=response_name[1],
            fly_name=f"{fly_df.date.values[0]} Fly {fly_df.fly_number.values[0]}",
            q_max=figure_params["response_q_max"], clim=clim)

def summarise_one_fly_one_modality(fly_df, figure_params):  # panel_style=None, walkrest=True):
    nrows = 3 if figure_params["walkrest"] else 1
    fig = plt.figure(figsize=(figure_params["figsize"][0],figure_params["figsize"][1]*nrows))  # layout="constrained"
    subfigs = fig.subfigures(nrows=nrows, ncols=1)
    mosaic = get_mosaic_from_panel_style(figure_params["panel_style"])
    axds = [subfig.subplot_mosaic(mosaic) for subfig in subfigs]

    fly_dir = np.unique(fly_df.fly_dir)[0]
    trial_names = fly_df.trial_name.values
    # TODO: work out difference between all trials and selected trials
    background_image = loaddata.get_background_image(fly_dir)
    roi_centers = loaddata.get_roi_centers(fly_dir)
    twop_df, beh_df = loaddata.load_data(fly_dir, all_trial_dirs=trial_names)  # TODO: change this such that it loads all trials that are good enough

    all_stim_responses, all_beh_responses = stimulation.get_neural_responses(twop_df, figure_params["trigger"],
                                                                 trials=trial_names,
                                                                 stim_p=figure_params["stim_p"],
                                                                 return_var=figure_params["return_var"])
    all_beh_class_responses = stimulation.get_beh_class_responses(beh_df, figure_params["trigger"],
                                                                 trials=trial_names,
                                                                 stim_p=figure_params["stim_p"])
    n_all_responses = all_stim_responses.shape[-1]
    walk_pre, rest_pre = behaviour.get_pre_stim_beh(beh_df, trigger=figure_params["trigger"],
                                                    stim_p=figure_params["stim_p"],
                                                    n_pre_stim=params.pre_stim_n_samples_beh,
                                                    trials=trial_names)  # selected_trials

    # assert(n_all_responses == len(walk_pre))
    if n_all_responses != len(walk_pre):
        if len(walk_pre) + 1 == n_all_responses:
            print(f"Warning will ignore last 2p response because there was a trial number mismatch.")
            print(f"Please fix this for fly {fly_dir}")
            all_stim_responses = all_stim_responses.copy()[:,:,:-1]
            all_beh_responses = all_beh_responses.copy()[:,:-1]
            n_all_responses -= 1
            print(len(walk_pre), len(rest_pre), n_all_responses, all_beh_class_responses.shape)
        elif len(walk_pre) - 1 == n_all_responses:
            print(f"Warning will ignore last 2p response because there was a trial number mismatch.")
            print(f"Please fix this for fly {fly_dir}")
            walk_pre = walk_pre.copy()[:-1]
            rest_pre = rest_pre.copy()[:-1]
            all_beh_class_responses = all_beh_class_responses.copy()[:,:-1]
            print(len(walk_pre), len(rest_pre), n_all_responses, all_beh_class_responses.shape)
        else:
            raise AssertionError
    print("number of responses", n_all_responses)
    print("number of walk/rest before stim", np.sum(walk_pre), np.sum(rest_pre))

    if figure_params["walkrest"]:
        _stim_responses = [all_stim_responses, all_stim_responses[:,:,walk_pre], all_stim_responses[:,:,rest_pre]]
        _beh_responses = [all_beh_responses, all_beh_responses[:,walk_pre], all_beh_responses[:,rest_pre]]
        _response_names = ["all trials", "walk pre", "rest pre"]
        _beh_class_responses = [all_beh_class_responses, all_beh_class_responses[:,walk_pre], all_beh_class_responses[:,rest_pre]]
    else:
        _stim_responses = [all_stim_responses]
        _beh_responses = [all_beh_responses]
        _response_names = ["all trials"]
        _beh_class_responses = [all_beh_class_responses]

    for i_r, (stim_responses, beh_responses, response_name, beh_class_responses) in \
        enumerate(zip(_stim_responses, _beh_responses, _response_names, _beh_class_responses)):
        response_values = stimulation.summarise_responses(stim_responses)

        if i_r == 0:
            sort_ind = np.argsort(response_values)
            if figure_params["response_clim"] is None:
                try:
                    clim = np.quantile(np.abs(response_values), q=figure_params["response_q_max"])
                except IndexError:
                    clim = 1
            else:
                clim = figure_params["response_clim"]

        get_plots_from_panel_style(fig=fig, axd=axds[i_r], panel_style=figure_params["panel_style"], figure_params=figure_params,
            beh_responses=beh_responses, stim_responses=stim_responses, response_values=response_values,
            beh_class_responses=beh_class_responses, response_name=response_name, background_image=background_image,
            roi_centers=roi_centers, fly_df=fly_df, clim=clim, sort_ind=sort_ind)

    return fig

def summarise_one_fly_natbeh(fly_df, figure_params):
    nrows = 3
    fig = plt.figure(figsize=(figure_params["figsize"][0],figure_params["figsize"][1]*nrows))  # layout="constrained"
    subfigs = fig.subfigures(nrows=nrows, ncols=1)
    mosaic = get_mosaic_from_panel_style(figure_params["panel_style"])
    axds = [subfig.subplot_mosaic(mosaic) for subfig in subfigs]

    fly_dir = np.unique(fly_df.fly_dir)[0]
    trial_names = fly_df.trial_name.values
    # TODO: work out difference between all trials and selected trials
    background_image = loaddata.get_background_image(fly_dir)
    roi_centers = loaddata.get_roi_centers(fly_dir)
    twop_df, beh_df = loaddata.load_data(fly_dir, all_trial_dirs=trial_names)  # TODO: change this such that it loads all trials that are good enough

    clim_set = False
    if beh_df.index.get_level_values("Date")[0] == 221115:
        a = 0

    for i_r, trigger in enumerate(["back", "walk", "rest"]):
        stim_responses, beh_responses = stimulation.get_neural_responses(twop_df, f"{trigger}_trig_start",
                                                                 trials=trial_names,
                                                                 stim_p=[],
                                                                 return_var=figure_params["return_var"])
        beh_class_responses = stimulation.get_beh_class_responses(beh_df, f"{trigger}_trig_start",
                                                                 trials=trial_names,
                                                                 stim_p=[])
        n_responses = stim_responses.shape[-1]
        response_values = stimulation.summarise_responses(stim_responses)

        if (i_r == 0 or not clim_set) and n_responses:
            sort_ind = np.argsort(response_values)
            if figure_params["response_clim"] is None:
                try:
                    clim = np.quantile(np.abs(response_values), q=figure_params["response_q_max"])
                except IndexError:
                    clim = 1
            else:
                clim = figure_params["response_clim"]
            clim_set = True
        elif i_r == 0:
            clim = 0
            sort_ind = np.arange(len(response_values))

        get_plots_from_panel_style(fig=fig, axd=axds[i_r], panel_style=figure_params["panel_style"], figure_params=figure_params,
            beh_responses=beh_responses, stim_responses=stim_responses, response_values=response_values,
            beh_class_responses=beh_class_responses, response_name=trigger, background_image=background_image,
            roi_centers=roi_centers, fly_df=fly_df, clim=clim, sort_ind=sort_ind)
    
    return fig

def summarise_one_fly_compare(fly_df, figure_params):
    nrows = 3
    fig = plt.figure(figsize=(figure_params["figsize"][0],figure_params["figsize"][1]*nrows))  # layout="constrained"
    subfigs = fig.subfigures(nrows=nrows, ncols=1)
    mosaic = get_mosaic_from_panel_style(figure_params["panel_style"])
    axds = [subfig.subplot_mosaic(mosaic) for subfig in subfigs]

    fly_dir = np.unique(fly_df.fly_dir)[0]
    trial_names = fly_df.trial_name.values
    # TODO: work out difference between all trials and selected trials
    background_image = loaddata.get_background_image(fly_dir)
    roi_centers = loaddata.get_roi_centers(fly_dir)
    twop_df, beh_df = loaddata.load_data(fly_dir, all_trial_dirs=trial_names)  # TODO: change this such that it loads all trials that are good enough


    stim_responses_1, beh_responses_1 = stimulation.get_neural_responses(twop_df, figure_params["trigger"],
                                                                 trials=trial_names,
                                                                 stim_p=figure_params["stim_p"],
                                                                 return_var=figure_params["return_var"])
    beh_class_responses_1 = stimulation.get_beh_class_responses(beh_df, figure_params["trigger"],
                                                                 trials=trial_names,
                                                                 stim_p=figure_params["stim_p"])

    stim_responses_2, beh_responses_2 = stimulation.get_neural_responses(twop_df, figure_params["trigger_2"],
                                                                 trials=trial_names,
                                                                 stim_p=figure_params["stim_p_2"],
                                                                 return_var=figure_params["return_var"])
    beh_class_responses_2 = stimulation.get_beh_class_responses(beh_df, figure_params["trigger_2"],
                                                                 trials=trial_names,
                                                                 stim_p=figure_params["stim_p_2"])

    _stim_responses = [stim_responses_1, stim_responses_2]  # , None]
    _beh_responses = [beh_responses_1, beh_responses_2]  # , None]
    _response_names = [figure_params["response_name"], figure_params["response_name_2"]]  # , None]
    _beh_class_responses = [beh_class_responses_1, beh_class_responses_2]  # , None]

    _response_values = []

    for i_r, (stim_responses, beh_responses, response_name, beh_class_responses) in \
        enumerate(zip(_stim_responses, _beh_responses, _response_names, _beh_class_responses)):
        response_values = stimulation.summarise_responses(stim_responses)
        _response_values.append(response_values)

        if i_r == 0:
            sort_ind = np.argsort(response_values)
            if figure_params["response_clim"] is None:
                try:
                    clim = np.quantile(np.abs(response_values), q=figure_params["response_q_max"])
                except IndexError:
                    clim = 1
            else:
                clim = figure_params["response_clim"]

        get_plots_from_panel_style(fig=fig, axd=axds[i_r], panel_style=figure_params["panel_style"], figure_params=figure_params,
            beh_responses=beh_responses, stim_responses=stim_responses, response_values=response_values,
            beh_class_responses=beh_class_responses, response_name=response_name, background_image=background_image,
            roi_centers=roi_centers, fly_df=fly_df, clim=clim, sort_ind=sort_ind)
    # now left & right summary plot
    get_plots_from_panel_style(fig=fig, axd=axds[2], panel_style=figure_params["panel_style"], figure_params=figure_params,
        beh_responses=None, stim_responses=None, response_values=_response_values,
        beh_class_responses=None, response_name=_response_names, background_image=background_image,
        roi_centers=roi_centers, fly_df=fly_df, clim=clim, sort_ind=sort_ind,
        comparison=True)

    return fig
    
def summarise_one_modality(exp_df, file_name, figure_params, base_folder=params.plot_base_dir):
    n_flies = len(exp_df.groupby("fly_id"))
    figs = []
    for i_fig, (fly_id, fly_df) in enumerate(exp_df.groupby("fly_id")):
        if figure_params["trigger"] == "natbeh":
            figs.append(summarise_one_fly_natbeh(fly_df, figure_params=figure_params))
        elif figure_params["trigger_2"] is not None:
            figs.append(summarise_one_fly_compare(fly_df, figure_params=figure_params))
        else:
            figs.append(summarise_one_fly_one_modality(fly_df, figure_params=figure_params))

    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    with PdfPages(os.path.join(params.plot_base_dir, f"{file_name}_summary_{figure_params['normalisation_type']}.pdf")) as pdf:
        _ = [pdf.savefig(fig) for fig in figs]
    _ = [plt.close(fig) for fig in figs]

def summarise_all_modalities(exp_dfs, file_names, all_figure_params, base_folder=params.plot_base_dir):
    for exp_df, file_name, figure_params in zip(exp_dfs, file_names, all_figure_params):
        summarise_one_modality(exp_df, file_name, base_folder=base_folder, figure_params=figure_params)

if __name__ == "__main__":
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
        "beh_response_ylabel": r"$v_{||}$ (mm/s)",
        "neural_response_ylabel": r"$\Delta$F/F",
        "clabel": r"$\Delta$F/F",
        "panel_style": "withbehclass",
        "response_clim": None,
        "response_q_max": 0.95,
        "figsize": [15,3],
    }
    
    df = summarydf.load_data_summary()
    df = summarydf.filter_data_summary(df, no_co2=False)
    dfs, df_names = summarydf.get_genotype_dfs_of_interest()
    df_Dfd_MDN, df_Dfd_DNp09, df_Dfd_aDN, df_Dfd_PR, df_Scr, df_BO = dfs
    df_Dfd_MDN_ball = summarydf.get_stim_ball_df(df_Dfd_MDN)
    df_Dfd_MDN_ball_inc_nostim = summarydf.get_ball_df(df_Dfd_MDN)
    df_Dfd_MDN_wheel = summarydf.get_stim_wheel_df(df_Dfd_MDN)
    df_Dfd_MDN_wheel_inc_nostim = summarydf.get_selected_df(df, [{"GCaMP": "Dfd", "CsChrimson": "MDN3", "walkon": "wheel"}])
    # summarydf.get_wheel_df(df_Dfd_MDN)
    df_Dfd_DNp09_ball = summarydf.get_stim_ball_df(df_Dfd_DNp09)
    df_Dfd_DNp09_wheel = summarydf.get_stim_wheel_df(df_Dfd_DNp09)
    df_Dfd_PR_ball = summarydf.get_stim_ball_df(df_Dfd_PR)
    df_Dfd_olfac = summarydf.get_olfac_df()
    df_wheel = summarydf.get_wheel_df()
    df_ball = summarydf.get_ball_df()
    df_Dfd_aDN_olfac = summarydf.get_aDN2_olfac_df()

    df_new_230508 = summarydf.get_new_flies_df(df=None, date=230401, no_co2=False,
                                               q_thres_neural=None, q_thres_beh=None, q_thres_stim=None)

    df_new_230514 = summarydf.get_new_flies_df(df=None, date=230509, no_co2=False,
                                               q_thres_neural=None, q_thres_beh=None, q_thres_stim=None)
    
    
    # new data from 01.04.2023 to 10.05.2023
    # summarise_one_modality(exp_df=df_new_230508, file_name="Dfd_new_230508", figure_params=figure_params)
    # summarise_one_modality(exp_df=df_new_230514, file_name="Dfd_new_230514", figure_params=figure_params)
    
    """
    # PR
    summarise_one_modality(exp_df=df_Dfd_PR_ball, file_name="Dfd_PR_ball", figure_params=figure_params)

    
    # MDN
    summarise_one_modality(exp_df=df_Dfd_MDN_ball, file_name="Dfd_MDN_ball", figure_params=figure_params)
    summarise_one_modality(exp_df=df_Dfd_MDN_wheel, file_name="Dfd_MDN_wheel", figure_params=figure_params)

    # DNp09
    
    summarise_one_modality(exp_df=df_Dfd_DNp09_ball, file_name="Dfd_DNp09_ball", figure_params=figure_params)
    summarise_one_modality(exp_df=df_Dfd_DNp09_wheel, file_name="Dfd_DNp09_wheel", figure_params=figure_params)
    
    # Scr & BO
    summarise_one_modality(exp_df=df_Scr, file_name="Scr_MDN_ball", figure_params=figure_params)
    summarise_one_modality(exp_df=df_BO, file_name="BO_MDN_ball", figure_params=figure_params)
    
    # aDN
    figure_params["stim_p"] = [20]
    summarise_one_modality(exp_df=df_Dfd_aDN, file_name="Dfd_aDN2_ball", figure_params=figure_params)
    
    # olfac = natural grooming
    figure_params["stim_p"] = []
    figure_params["trigger"] = "olfac_start"
    summarise_one_modality(exp_df=df_Dfd_olfac, file_name="Dfd_olfac", figure_params=figure_params)
    
    # natural backward walking
    figure_params["stim_p"] = []
    figure_params["trigger"] = "back_trig_start"
    summarise_one_modality(exp_df=df_wheel, file_name="Dfd_natback_wheel", figure_params=figure_params)
    
    figure_params["stim_p"] = []
    figure_params["trigger"] = "walk_trig_start"
    summarise_one_modality(exp_df=df_ball, file_name="Dfd_natwalk_ball", figure_params=figure_params)
    
    figure_params["stim_p"] = []
    figure_params["trigger"] = "rest_trig_start"
    summarise_one_modality(exp_df=df_wheel, file_name="Dfd_natrest_wheel", figure_params=figure_params)
    
    
    figure_params["trigger"] = "natbeh"
    # summarise_one_modality(exp_df=df_wheel, file_name="Dfd_natbeh_wheel", figure_params=figure_params)
    summarise_one_modality(exp_df=df_ball, file_name="Dfd_natbeh_ball", figure_params=figure_params)
    
    
    # compare aDN stimulation with olfac stimulation
    figure_params["trigger"] = "laser_start"
    figure_params["stim_p"] = [20]
    figure_params["trigger_2"] = "olfac_start"
    figure_params["stim_p_2"] = None
    figure_params["response_name"] = "aDN2 stim"
    figure_params["response_name_2"] = "olfac stim"
    summarise_one_modality(exp_df=df_Dfd_aDN_olfac, file_name="Dfd_comp_aDN2_olfac_ball", figure_params=figure_params)
    
    # compare DNp09 stimulation with natural walking
    figure_params["trigger"] = "laser_start"
    figure_params["stim_p"] = [10, 20]
    figure_params["trigger_2"] = "walk_trig_start"
    figure_params["stim_p_2"] = None
    figure_params["response_name"] = "DNp09 stim"
    figure_params["response_name_2"] = "natural walk"
    summarise_one_modality(exp_df=df_Dfd_DNp09_ball, file_name="Dfd_comp_DNp09_natwalk_ball", figure_params=figure_params)
    summarise_one_modality(exp_df=df_Dfd_DNp09_wheel, file_name="Dfd_comp_DNp09_natwalk_wheel", figure_params=figure_params)
    """
    # compare MDN stimulation with natural backward walking
    figure_params["trigger"] = "laser_start"
    figure_params["stim_p"] = [10, 20]
    figure_params["trigger_2"] = "back_trig_start"
    figure_params["stim_p_2"] = None
    figure_params["response_name"] = "MDN stim"
    figure_params["response_name_2"] = "natural backwalk"
    # summarise_one_modality(exp_df=df_Dfd_MDN_ball_inc_nostim, file_name="Dfd_comp_MDN_natbackwalk_ball", figure_params=figure_params)
    summarise_one_modality(exp_df=df_Dfd_MDN_wheel_inc_nostim, file_name="Dfd_comp_MDN_natbackwalk_wheel", figure_params=figure_params)
    
    """
    # compare aDN stimulation with lower power aDN stim
    figure_params["trigger"] = "laser_start"
    figure_params["stim_p"] = [10]
    figure_params["trigger_2"] = "laser_start"
    figure_params["stim_p_2"] = [20]
    figure_params["response_name"] = "aDN2 stim 10uW"
    figure_params["response_name_2"] = "aDN2 stim 20uW"
    summarise_one_modality(exp_df=df_Dfd_aDN, file_name="Dfd_comp_aDN2_aDN2_ball", figure_params=figure_params)
    
    """
 