
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, pearsonr
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm

import params, summarydf, loaddata, stimulation, behaviour, plotpanels, fig_functional

from twoppp import plot as myplt

def analyse_natbehaviour_responses_DNp09(figures_path=None, tmpdata_path=None, pre_stim="not_walk", min_n_resp=params.min_n_resp_natbeh, n_neurons_select=10):
    if figures_path is None:
        figures_path = params.plot_base_dir
    if tmpdata_path is None:
        tmpdata_path = params.plotdata_base_dir
    data_file = os.path.join(tmpdata_path, f"natbeh_{pre_stim}_resp_DNp09.pkl")
    with open(data_file, "rb") as f:
        all_fly_data = pickle.load(f)
    all_fly_data = [fly_data for fly_data in all_fly_data if fly_data["nat_n_sel_responses"] > min_n_resp and fly_data["n_sel_responses"] > min_n_resp]

    fig1, ax = plt.subplots(1,1,figsize=(6,4))

    n_bins = 100
    bins_stim_sum = np.zeros((n_bins))
    bins_nat_sum = np.zeros((n_bins))
    n_per_bin = np.zeros((n_bins))

    min_n_neurons = n_bins
    max_n_neurons = 0

    for i_fly, alpha in enumerate(np.linspace(0.3,0.2, len(all_fly_data))):
        n_neurons = len(all_fly_data[i_fly]["response_values"])
        if n_neurons < min_n_neurons:
            min_n_neurons = n_neurons
        if n_neurons > max_n_neurons:
            max_n_neurons = n_neurons
        sorted_stim_resp = np.array(all_fly_data[i_fly]["response_values"][np.flip(all_fly_data[i_fly]["sort_ind"])])
        sorted_nat_resp = np.array(all_fly_data[i_fly]["nat_response_values"][np.flip(all_fly_data[i_fly]["sort_ind"])])
        ax.plot(np.arange(n_neurons), sorted_stim_resp, color=myplt.BLACK, alpha=alpha)
        ax.plot(np.arange(n_neurons), sorted_nat_resp, color=myplt.DARKGREEN, alpha=alpha)
        bins_stim_sum[:n_neurons] += sorted_stim_resp
        bins_nat_sum[:n_neurons] += sorted_nat_resp
        n_per_bin[:n_neurons] += 1
        
    ax.plot(np.arange(min_n_neurons), bins_stim_sum[:min_n_neurons] / len(all_fly_data), color=myplt.BLACK, linewidth=4)
    ax.plot(np.arange(min_n_neurons), bins_nat_sum[:min_n_neurons] / len(all_fly_data), color=myplt.DARKGREEN, linewidth=4)

    print("DNp09 3 flies:", pearsonr(bins_stim_sum[:min_n_neurons], bins_nat_sum[:min_n_neurons]))
    print("omitting first 10 neurons:", pearsonr(bins_stim_sum[10:min_n_neurons], bins_nat_sum[10:min_n_neurons]))


    ax.axvspan(0, n_neurons_select, alpha=0.2, color=myplt.DARKORANGE, ec=None)
    ax.set_xlim([0,max_n_neurons])
    ax.set_ylim([-1,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])
    ax.set_xlabel("neurons sorted by stimulus response")
    ax.set_ylabel(r"$\Delta$F/F")
    plotpanels.make_nice_spines(ax)
    fig1.tight_layout()

    fig2, axs = plt.subplots(2,len(all_fly_data), figsize=(9.5,4), sharex="row", sharey="row")
    x = np.arange(np.sum(params.response_t_params_2p))/params.fs_int-params.response_t_params_2p_label[0]

    for i_fly, alpha in enumerate(np.linspace(1,0.5, len(all_fly_data))):
        n_neurons = len(all_fly_data[i_fly]["response_values"])
        sel_neurons = np.flip(all_fly_data[i_fly]["sort_ind"])[:n_neurons_select]
        
        neurons_y = np.array(all_fly_data[i_fly]["roi_centers"])[sel_neurons][:,0]
        neurons_y = neurons_y[neurons_y.argsort()]
        sel_neurons = sel_neurons[neurons_y.argsort()]
        plotpanels.plot_ax_response_summary(background_image=all_fly_data[i_fly]["background_image"],
                                            roi_centers=np.array(all_fly_data[i_fly]["roi_centers"])[sel_neurons],
                                            ax=axs[0, i_fly],
                                            response_values_left=all_fly_data[i_fly]["response_values"][sel_neurons],
                                            response_values_right=all_fly_data[i_fly]["nat_response_values"][sel_neurons],
                                            title=f"fly ID {all_fly_data[i_fly]['fly_df'].fly_id.values[0]}")
        
        stim_response = np.mean(all_fly_data[i_fly]["stim_responses"][:,sel_neurons,:], axis=-1)
        i_0 = int(params.response_t_params_2p[0])
        i_b = int(i_0 - params.response_n_baseline)
        rel_stim_response = stim_response-np.mean(stim_response[i_b:i_0], axis=0)
        _ = [axs[1,i_fly].plot(x, rel_stim_response[:,i], color=myplt.BLACK, alpha=a)
            for i, a in enumerate(np.linspace(1,0.5, len(sel_neurons)))]
        nat_response = np.mean(all_fly_data[i_fly]["nat_responses"][:,sel_neurons,:], axis=-1)
        i_0 = int(params.response_t_params_2p[0])
        i_b = int(i_0 - params.response_n_baseline)
        rel_nat_response = nat_response-np.mean(nat_response[i_b:i_0], axis=0)
        _ = [axs[1,i_fly].plot(x, rel_nat_response[:,i], color=myplt.DARKGREEN, alpha=a)
            for i, a in enumerate(np.linspace(1,0.5, len(sel_neurons)))]
        myplt.shade_categorical(catvar=np.concatenate((np.zeros((params.response_t_params_2p[0])),
                                                            np.ones((params.response_t_params_2p[1])),
                                                            np.zeros((params.response_t_params_2p[2])))),
                                        x=x,colors=[myplt.WHITE, myplt.BLACK], ax=axs[1,i_fly])
        if i_fly == 0:
            axs[1,i_fly].set_xlabel("Time (s)")
            axs[1,i_fly].set_ylabel(r"$\Delta$F/F")
            axs[1,i_fly].set_xlim([-5,10])
            axs[1,i_fly].set_ylim(-0.25,1)
            axs[1,i_fly].set_yticks([0,0.5,1])
        plotpanels.make_nice_spines(axs[1, i_fly])
        axs[1,i_fly].set_xticks([0,params.response_t_params_2p_label[1]])
        axs[1,i_fly].spines['bottom'].set_position('zero')
    fig2.tight_layout()

    # make density plots
    fig3 = plt.figure(figsize=(14,4))
    mosaic = """
    AAEEEECC
    BBEEEEDD
    """
    axd = fig3.subplot_mosaic(mosaic)
    all_roi_centers = np.concatenate([fig_functional.align_roi_centers(fly_data["roi_centers"]) for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)
    all_response_values = np.concatenate([fly_data["response_values"] for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)
    all_response_values_nat = np.concatenate([fly_data["nat_response_values"] for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)
    all_response_values_diff = np.concatenate([fly_data["response_values"] - fly_data["nat_response_values"] for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)
    all_n_response = np.array([fly_data["n_sel_responses"] for fly_data in all_fly_data if len(fly_data["roi_centers"])])

    # only the top 10 neurons
    sel_roi_centers = np.concatenate([fig_functional.align_roi_centers(np.array(fly_data["roi_centers"])[fly_data["sort_ind"][-n_neurons_select:]]) for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)
    sel_response_values_diff = np.concatenate([fly_data["response_values"][fly_data["sort_ind"][-n_neurons_select:]] - fly_data["nat_response_values"][fly_data["sort_ind"][-n_neurons_select:]] for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)

    # all other neurons
    not_sel_roi_centers = np.concatenate([fig_functional.align_roi_centers(np.array(fly_data["roi_centers"])[fly_data["sort_ind"][:-n_neurons_select]]) for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)
    not_sel_response_values_diff = np.concatenate([fly_data["response_values"][fly_data["sort_ind"][:-n_neurons_select]] - fly_data["nat_response_values"][fly_data["sort_ind"][:-n_neurons_select]] for fly_data in all_fly_data if len(fly_data["roi_centers"])], axis=0)

    plotpanels.plot_ax_multi_fly_response_density(
        roi_centers=all_roi_centers,
        response_values=all_response_values,
        background_image=np.zeros_like(all_fly_data[0]["background_image"]),
        ax=axd["A"],
        n_flies=len(all_n_response),
        clim=0.8,)
    axd["A"].set_title("stimulation", fontsize=16)
    plotpanels.plot_ax_multi_fly_response_density(
        roi_centers=all_roi_centers,
        response_values=all_response_values_nat,
        background_image=np.zeros_like(all_fly_data[0]["background_image"]),
        ax=axd["B"],
        n_flies=len(all_n_response),
        clim=0.8,)
    axd["B"].set_title("natural", fontsize=16)
    
    plotpanels.plot_ax_multi_fly_response_density(
        roi_centers=all_roi_centers,
        response_values=all_response_values_diff,
        background_image=np.zeros_like(all_fly_data[0]["background_image"]),
        ax=axd["E"],
        n_flies=len(all_n_response),
        clim=0.8,)
    axd["E"].set_title("difference", fontsize=16)
    
    plotpanels.plot_ax_multi_fly_response_density(
        roi_centers=sel_roi_centers,
        response_values=sel_response_values_diff,
        background_image=np.zeros_like(all_fly_data[0]["background_image"]),
        ax=axd["C"],
        n_flies=len(all_n_response),
        clim=0.8,)
    axd["C"].set_title("difference top 10 only", fontsize=16)
    plotpanels.plot_ax_multi_fly_response_density(
        roi_centers=not_sel_roi_centers,
        response_values=not_sel_response_values_diff,
        background_image=np.zeros_like(all_fly_data[0]["background_image"]),
        ax=axd["D"],
        n_flies=len(all_n_response),
        clim=0.8,)
    axd["D"].set_title("difference all other neurons", fontsize=16)

    fig3.tight_layout()

    with PdfPages(os.path.join(figures_path, f"supfig_DNp09_natbeh_{pre_stim}_to_stim.pdf")) as pdf:
        _ = [pdf.savefig(fig, transparent=True) for fig in [fig1, fig2, fig3]]
    _ = [plt.close(fig) for fig in [fig1, fig2, fig3]]

def analyse_natbehaviour_responses_singlefly(GAL4="MDN3", tmpdata_path=None, min_n_resp=params.min_n_resp_natbeh, pre_stim=None, fly_id=None, contrast_color=myplt.DARKCYAN):
    if tmpdata_path is None:
        tmpdata_path = params.plotdata_base_dir
    data_file = os.path.join(tmpdata_path, f"natbeh_{pre_stim}_resp_{GAL4}.pkl")
    with open(data_file, "rb") as f:
        all_fly_data = pickle.load(f)
    if fly_id is None:
        all_fly_data = [fly_data for fly_data in all_fly_data if fly_data["nat_n_sel_responses"] > min_n_resp and fly_data["n_sel_responses"] > min_n_resp]
    else:
        all_fly_data = [fly_data for fly_data in all_fly_data if fly_data["fly_df"].fly_id.values[0] == fly_id]
    fly_data = all_fly_data[0]

    stim_response_values, stim_response_conf_int = stimulation.summarise_responses(fly_data["stim_responses"],
                                                                     only_confident=0, return_conf_int=True)
    nat_response_values, nat_response_conf_int = stimulation.summarise_responses(fly_data["nat_responses"],
                                                                        only_confident=0, return_conf_int=True)

    fig, axs = plt.subplots(1,2,figsize=(9.5,4))

    n_neurons = len(stim_response_values)
    sort_ind = np.flip(np.argsort(stim_response_values))
    corr_coef = np.corrcoef(stim_response_values, nat_response_values)[0,1]
    print(GAL4, corr_coef, pearsonr(stim_response_values, nat_response_values))
    print("omitting first 10 neurons:", pearsonr(stim_response_values[sort_ind[10:]], nat_response_values[sort_ind[10:]]))

    stim_labels = (stim_response_values>stim_response_conf_int).astype(int) - \
              (stim_response_values<-1*stim_response_conf_int).astype(int)
    nat_labels = (nat_response_values>nat_response_conf_int).astype(int) - \
                (nat_response_values<-1*nat_response_conf_int).astype(int)
    conf_matrix = np.flip(confusion_matrix(y_true=stim_labels, y_pred=nat_labels))

    myplt.plot_mu_sem(mu=stim_response_values[sort_ind], err=stim_response_conf_int[sort_ind], ax=axs[0], color=myplt.BLACK, linewidth=2)
    myplt.plot_mu_sem(mu=nat_response_values[sort_ind], err=nat_response_conf_int[sort_ind], ax=axs[0], color=contrast_color, linewidth=2,
                    label=f"r = {corr_coef:.3}")
    axs[0].legend(frameon=False, fontsize=16)
    axs[0].set_xlim([0,n_neurons])
    axs[0].set_ylim([-1,1])
    axs[0].set_yticks([-1,-0.5,0,0.5,1])
    axs[0].set_xlabel("neurons sorted by stimulus response")
    axs[0].set_ylabel(r"$\Delta$F/F")
    plotpanels.make_nice_spines(axs[0])

    axs[1].imshow(conf_matrix, cmap=plt.cm.Greys, clim=[0,len(stim_labels)])
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            axs[1].text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    axs[1].set_ylabel('Optogenetic activation', fontsize=16, color=myplt.BLACK)
    axs[1].set_yticks([0,1,2])
    axs[1].set_yticklabels(["more active", "no change", "less active"])
    axs[1].set_xlabel('Natural behaviour', fontsize=16, color=contrast_color)
    axs[1].set_xticks([0,1,2])
    axs[1].set_xticklabels(["more active", "no change", "less active"])
    axs[1].spines["top"].set_linewidth(0)
    axs[1].spines["right"].set_linewidth(0)
    axs[1].spines["bottom"].set_linewidth(0)
    axs[1].spines["left"].set_linewidth(0)
    axs[1].tick_params(width=0)
    fig.tight_layout()

    return fig

def analyse_natbehaviour_responses_all_genotypes(figures_path=None, tmpdata_path=None, min_n_resp=params.min_n_resp_natbeh):
    if figures_path is None:
        figures_path = params.plot_base_dir
    if tmpdata_path is None:
        tmpdata_path = params.plotdata_base_dir
    fig_DNp09 = analyse_natbehaviour_responses_singlefly(GAL4="DNp09", min_n_resp=min_n_resp, pre_stim="not_walk",  # "rest",
                                                        fly_id=fig_functional.presentation_natbeh_flies["DNp09"],
                                                        contrast_color=myplt.DARKGREEN)
    fig_aDN2 = analyse_natbehaviour_responses_singlefly(GAL4="aDN2", min_n_resp=min_n_resp, pre_stim=None,
                                                        fly_id=fig_functional.presentation_natbeh_flies["aDN2"],
                                                        contrast_color=myplt.DARKRED)
    fig_MDN = analyse_natbehaviour_responses_singlefly(GAL4="MDN3", min_n_resp=min_n_resp, pre_stim=None,
                                                        fly_id=fig_functional.presentation_natbeh_flies["MDN"],
                                                        contrast_color=myplt.DARKCYAN)
        
    with PdfPages(os.path.join(figures_path, f"supfig_singlefly_natbeh.pdf")) as pdf:
        _ = [pdf.savefig(fig, transparent=True) for fig in [fig_DNp09, fig_aDN2, fig_MDN]]
    _ = [plt.close(fig) for fig in [fig_DNp09, fig_aDN2, fig_MDN]]


if __name__ == "__main__":
    analyse_natbehaviour_responses_DNp09()
    analyse_natbehaviour_responses_all_genotypes()
    
