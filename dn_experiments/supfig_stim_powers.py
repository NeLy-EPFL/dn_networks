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
from scipy.ndimage import gaussian_filter1d

import params, summarydf, loaddata, stimulation, behaviour, plotpanels

from twoppp import plot as myplt
from twoppp import utils


def plot_stim_p_effect(figures_path=None):
    if figures_path is None:
        figures_path = params.plot_base_dir

    df = summarydf.load_data_summary()
    # df = summarydf.filter_data_summary(df, no_co2=True)  # , q_thres_neural=None)
    df = summarydf.get_selected_df(df, [{"laser_power": "1_5_10_20", "GCaMP": "Dfd",
                                         "walkon": "ball", "stim_location": "cc"}])

    p_levels = [1,5,10,20]
    p_colors = [myplt.DARKGREEN, myplt.DARKYELLOW, myplt.DARKORANGE, myplt.DARKRED]

    base_fly_data = {
            "GAL4": None,
            "fly_id": None,
            "fly_dir": None,
            "trial_names": None,
            "beh_responses": None,
            "beh_class_responses": None,
        }
    all_fly_data = []

    for i_fly, (fly_id, fly_df) in enumerate(df.groupby("fly_id")):
        fly_data = base_fly_data.copy()
        fly_data["fly_id"] = fly_id
        fly_data["fly_dir"] = np.unique(fly_df.fly_dir)[0]
        fly_data["trial_names"] = fly_df.trial_name.values
        fly_data["GAL4"] = np.unique(fly_df.CsChrimson)[0]
        if fly_data["GAL4"] == "MDN":
            fly_data["GAL4"] = "MDN3"
        elif fly_data["GAL4"] == "DNP9":
            fly_data["GAL4"] = "DNp09"

        beh_df = loaddata.load_beh_data_only(fly_data["fly_dir"], all_trial_dirs=fly_data["trial_names"]) 
        beh_responses = []
        beh_class_responses = []
        for i_p, p in enumerate(p_levels):
            beh_responses.append(stimulation.get_beh_responses(beh_df, trigger="laser_start", trials=fly_data["trial_names"],
                                              beh_var="v_forw", stim_p=[p]))
            beh_class_responses.append(stimulation.get_beh_responses(beh_df, trigger="laser_start", trials=fly_data["trial_names"],
                                              beh_var="beh_class", stim_p=[p]))
        
        if any([beh_response.shape[-1] % 10 for beh_response in beh_responses]):
            print(f"uneven trials: {fly_data['fly_dir']}")
            for i_p, (beh_response, beh_class_response) in enumerate(zip(beh_responses, beh_class_responses)):
                if beh_response.shape[-1] % 10:
                    continue
                else:  # remove the last trial to make sure that all conditions have an even trial number
                    beh_responses[i_p] = beh_response[:,:-1]
                    beh_class_responses[i_p] = beh_class_response[:,:-1]
        if any([not beh_response.size or np.std(beh_response) > 10 for beh_response in beh_responses]):
            print(f"no responses: {fly_data['fly_dir']}")
            continue
        fly_data["beh_responses"] = beh_responses
        fly_data["beh_class_responses"] = beh_class_responses
        all_fly_data.append(fly_data)

    MDN_flies = [fly_data for fly_data in all_fly_data if fly_data["GAL4"] == "MDN3"]
    DNp09_flies = [fly_data for fly_data in all_fly_data if fly_data["GAL4"] == "DNp09"]
    aDN2_flies = [fly_data for fly_data in all_fly_data if fly_data["GAL4"] == "aDN2"]
    PR_flies = [fly_data for fly_data in all_fly_data if fly_data["GAL4"] == "PR"]
    all_flies = [DNp09_flies, aDN2_flies, MDN_flies, PR_flies]
    genotypes = ["DNp09", "aDN2", "MDN", "PR"]
    i_behs = [1,4,3,2]

    fig, axs = plt.subplots(2,4, figsize=(9.5,6), sharex="row", sharey="row")
    x = np.arange(np.sum(params.response_t_params_beh))/params.fs_beh-params.response_t_params_2p_label[0]

    for i_f, fly in enumerate(PR_flies):
        if i_f == 0:
            PR_beh_class_responses = np.array(fly["beh_class_responses"])
        else:
            PR_beh_class_responses = np.concatenate((PR_beh_class_responses, np.array(fly["beh_class_responses"])), axis=-1)


    for i_g, (genotype, flies, i_beh) in enumerate(zip(genotypes, all_flies, i_behs)):
        beh_name = behaviour.beh_mapping[i_beh]

        for i_f, fly in enumerate(flies):
            if i_f == 0:
                beh_responses = np.array(fly["beh_responses"])
                beh_class_responses = np.array(fly["beh_class_responses"])
            else:
                beh_responses = np.concatenate((beh_responses, np.array(fly["beh_responses"])), axis=-1)
                beh_class_responses = np.concatenate((beh_class_responses, np.array(fly["beh_class_responses"])), axis=-1)

        for i_p, (p, c) in enumerate(zip(p_levels, p_colors)):
            myplt.plot_mu_sem(mu=np.mean(beh_responses[i_p], axis=-1), err=utils.conf_int(beh_responses[i_p], axis=-1),
                            x=x, ax=axs[0, i_g], color=c, linewidth=plotpanels.linewidth)
            axs[1, i_g].plot(x, gaussian_filter1d(np.mean(beh_class_responses[i_p] == i_beh, axis=-1), sigma=10), color=c)
        axs[1, i_g].plot(x, gaussian_filter1d(np.mean(PR_beh_class_responses[-2:] == i_beh, axis=(0,-1)), sigma=10), color=myplt.BLACK)  # average of PR flies at 10/20uW
        
        myplt.shade_categorical(catvar=np.concatenate((np.zeros((params.response_t_params_beh[0])),
                                                           np.ones((params.response_t_params_beh[1])),
                                                           np.zeros((params.response_t_params_beh[2])))),
                                    x=x,colors=[myplt.WHITE, myplt.BLACK], ax=axs[0, i_g])
        axs[0, i_g].set_title(f"{genotype} {len(flies)} flies {beh_responses.shape[-1]} trials")
        axs[0, i_g].set_ylabel(r"$v_{||}$ (mm/s)")
        axs[0, i_g].set_ylim([-2.5, 7.5])
        axs[0, i_g].set_yticks([-2.5, 0, 2.5, 5, 7.5])
        axs[0, i_g].set_xticks([0,params.response_t_params_2p_label[1]])
        axs[0, i_g].set_xticklabels(["", ""])
        plotpanels.make_nice_spines(axs[0, i_g])
        myplt.shade_categorical(catvar=np.concatenate((np.zeros((params.response_t_params_beh[0])),
                                                           np.ones((params.response_t_params_beh[1])),
                                                           np.zeros((params.response_t_params_beh[2])))),
                                    x=x,colors=[myplt.WHITE, myplt.BLACK], ax=axs[1, i_g])
        axs[1, i_g].set_ylabel(f"{beh_name} probability")
        axs[1, i_g].set_xlabel("t (s)")
        axs[1, i_g].set_xticks([0,params.response_t_params_2p_label[1]])
        axs[1, i_g].set_ylim([-0.01,1.01])
        axs[1, i_g].set_yticks([0,1])
        plotpanels.make_nice_spines(axs[1, i_g])

    fig.tight_layout()
    fig.savefig(os.path.join(figures_path, "supfig_stimp.pdf"), dpi=300)


if __name__ == "__main__":
    plot_stim_p_effect()