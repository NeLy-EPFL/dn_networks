"""
Functions to generate plots to analyse the imaging of command neurons in intact and headless flies.
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




def get_calcium_transients(twop_df):
    """
    Calculate the mean calcium transients in response to stimulation.
    Takes the mean across all neurons (2 for DNp09, 4 for MDN)

    Parameters:
    - twop_df (DataFrame): DataFrame containing two-photon imaging data.

    Returns:
    - ndarray: Mean calcium responses across all neurons.
    """
    all_stim_responses, all_beh_responses = stimulation.get_neural_responses(twop_df, "laser_start",
                                                                        trials=None,
                                                                        stim_p=[10],
                                                                        return_var=None,
                                                                        neural_regex=params.neurons_filt_regex)
    return np.mean(all_stim_responses, axis=1)  # take the mean across all neurons

def get_beh_data(beh_df, var2, var2_rel=[400,500]):
    """
    Extract behavioral responses to stimulation.

    Parameters:
    - beh_df (DataFrame): DataFrame containing behavioral data.
    - var2 (str): Behavioral variable of interest.
    - var2_rel (list, optional): Time window for baseline correction. Defaults to [400, 500].

    Returns:
    - ndarray: Velocity responses and behavioral responses to stimulation.
    """
    beh_responses = stimulation.get_beh_responses(beh_df, trigger="laser_start", trials=None,
                                           stim_p=[10],beh_var=var2)
    try:
        v_responses = stimulation.get_beh_responses(beh_df, trigger="laser_start", trials=None,
                                            stim_p=[10],beh_var="v_forw")
    except KeyError:  # off the ball, there is no v_forw, because fictrac was not run
        v_responses = np.zeros_like(beh_responses)
    if var2 == "anus_dist":
        beh_responses *= 4.8 # pixeles -> um
    elif var2 == "meh_tita":
        beh_responses *= (4.8*100/1000) # pixeles/frame -> um/s -> mm/s
    if var2_rel is not None:
        beh_responses = beh_responses - np.mean(beh_responses[var2_rel[0]:var2_rel[1],:], axis=0)

    return v_responses, beh_responses


def get_headless_imaging_data(genotypes=["MDN", "DNp09"], special_vars=["meh_tita", "anus_dist"]):
        """
    Extract headless imaging data for specified genotypes and conditions.

    Parameters:
    - genotypes (list, optional): List of genotypes to include. Defaults to ["MDN", "DNp09"].
    - special_vars (list, optional): List of special variables for behavioral analysis. Defaults to ["meh_tita", "anus_dist"].

    Returns:
    - tuple: Tuple containing arrays of calcium data, velocity data, and behavioral data.
      The shape of each array is (N_genotypes, N_flies_per_genotype, 2, 2, N_samples, N_stims), where:
      - N_genotypes: Number of specified genotypes.
      - N_flies_per_genotype: Number of flies per genotype. (Hardcoded to 3)
      - 2 (first dimension): Represents the presence (0) or absence (1) of head.
      - 2 (second dimension): Represents the presence (0) or absence (1) of ball.
      - N_samples: Number of samples.
      - N_stims: Number of stimulations per condition. (Hardcoded to 10)
    """
    N_genotypes = len(genotypes)
    N_flies_per_genotype = 3
    N_stims = 10
    N_samples = 243  # TODO
    N_samples_beh = 1500  # TODO

    all_calcium_data = np.zeros((N_genotypes, N_flies_per_genotype, 2, 2, N_samples, N_stims))
    all_v_data = np.zeros((N_genotypes, N_flies_per_genotype, 2, 2, N_samples_beh, N_stims))
    all_beh_data = np.zeros((N_genotypes, N_flies_per_genotype, 2, 2, N_samples_beh, N_stims))

    df = summarydf.get_headless_df()

    for i_gen, (genotype, special_var) in enumerate(zip(genotypes, special_vars)):
        exp_df = summarydf.get_selected_df(df, [{"GCaMP": genotype}])
        for i_fly, (fly_id, fly_df) in enumerate(exp_df.groupby("fly_id")):
            fly_dir = np.unique(fly_df.fly_dir)[0]
            for index, trial_df in fly_df.iterrows():
                trial_dir = trial_df["trial_dir"]
                twop_df = loaddata.get_filtered_twop_df(fly_dir=fly_dir, all_trial_dirs=[trial_df.trial_name])
                beh_df = loaddata.get_beh_df_with_me(fly_dir=fly_dir, all_trial_dirs=[trial_df.trial_name], add_sleap=True, add_me=False)

                this_calcium_data = get_calcium_transients(twop_df)
                this_v_data, this_beh_data = get_beh_data(beh_df, special_var, var2_rel=[400,500])

                if trial_df["head"] and trial_df["walkon"] == "ball":  # head + ball
                    all_calcium_data[i_gen, i_fly, 0, 0, :, :] = this_calcium_data
                    all_v_data[i_gen, i_fly, 0, 0, :, :] = this_v_data
                    all_beh_data[i_gen, i_fly, 0, 0, :, :] = this_beh_data
                elif trial_df["head"]:  # head + no ball
                    all_calcium_data[i_gen, i_fly, 0, 1, :, :] = this_calcium_data
                    all_v_data[i_gen, i_fly, 0, 1, :, :] = this_v_data
                    all_beh_data[i_gen, i_fly, 0, 1, :, :] = this_beh_data
                elif trial_df["walkon"] == "ball":  # no head + ball
                    all_calcium_data[i_gen, i_fly, 1, 0, :, :] = this_calcium_data
                    all_v_data[i_gen, i_fly, 1, 0, :, :] = this_v_data
                    all_beh_data[i_gen, i_fly, 1, 0, :, :] = this_beh_data
                else:  # no head + no ball
                    all_calcium_data[i_gen, i_fly, 1, 1, :, :] = this_calcium_data
                    all_v_data[i_gen, i_fly, 1, 1, :, :] = this_v_data
                    all_beh_data[i_gen, i_fly, 1, 1, :, :] = this_beh_data

    return all_calcium_data, all_v_data, all_beh_data

def make_one_headless_imaging_panel(fig, axd, calcium_data, v_data, beh_data, ylim_beh=None, ylim_v=None, ylabel=None, title=None):
    """
    Create a panel in the headless imaging figure with comparisons of neural, velocity, and behavioral responses.

    Parameters:
    - fig: Figure object.
    - axd: Dictionary of Axes objects.
    - calcium_data (list): List of calcium response data before and after stimulation.
    - v_data (list): List of velocity response data before and after stimulation.
    - beh_data (list): List of behavioral response data before and after stimulation.
    - ylim_beh (list, optional): Y-axis limits for behavioral response plot. Defaults to None.
    - ylim_v (list, optional): Y-axis limits for velocity response plot. Defaults to None.
    - ylabel (str, optional): Y-axis label for behavioral response plot. Defaults to None.
    - title (str, optional): Title of the panel. Defaults to None.
    """
    # N: neural response comparison
    plotpanels.plot_ax_behavioural_response(calcium_data[0], ax=axd["N"], x="2p", ylim=[-0.2, 0.8],
            response_name=title,
            response_ylabel=r"$\Delta$F/F",
            beh_responses_2=calcium_data[1], beh_response_2_color=myplt.DARKRED)
    # V: velocity response comparison
    plotpanels.plot_ax_behavioural_response(v_data[0], ax=axd["V"], x="beh", ylim=ylim_v,
            response_name=None,
            response_ylabel=r"$v_{||}$ (mm/s)",
            beh_responses_2=v_data[1], beh_response_2_color=myplt.DARKRED)
    
    # B: behavioural variable response comparison
    plotpanels.plot_ax_behavioural_response(beh_data[0], ax=axd["B"], x="beh", ylim=ylim_beh,
            response_name=None,
            response_ylabel=ylabel,
            beh_responses_2=beh_data[1], beh_response_2_color=myplt.DARKRED)

def make_headless_imaging_figure(figures_path=None):
    """
    Create a figure summarizing headless imaging data for specific hardcoded genotypes and conditions.

    Parameters:
    - figures_path (str, optional): Path to save the figure. Defaults to None.
    """
    if figures_path is None:
        figures_path = os.path.join(params.plot_base_dir, "revision")

    genotypes = ["MDN", "DNp09"]  # fly ids in headless_df: MDN: 164, 165, 166; DNp09: 159, 172, 175
    conditions = ["onball", "hanging"]
    special_vars = ["meh_tita", "anus_dist"]
    special_vars_ylabel = [r"$\Delta$ Hind leg motion (mm/s)", "anal plate (um)"]
    ylim_v = [-2,5]
    ylims_beh = [[-0.5,1.5], [-50,25]]

    fig = plt.figure(figsize=(15,10))
    mosaic = """
    NNN
    VVV
    BBB
    """
    subfigs = fig.subfigures(nrows=1, ncols=4, squeeze=True)
    axds = [subfig.subplot_mosaic(mosaic) for subfig in subfigs]

    all_calcium_data, all_v_data, all_beh_data = get_headless_imaging_data(genotypes=genotypes, special_vars=special_vars)
    # quantile normalise calcium data for each fly
    for i_gen in range(all_calcium_data.shape[0]):
        for i_fly in range(all_calcium_data.shape[1]):
            q_max = np.quantile(all_calcium_data[i_gen, i_fly], params.baseline_qmax)
            q_min = np.quantile(all_calcium_data[i_gen, i_fly], 1-params.baseline_qmax)
            all_calcium_data[i_gen, i_fly] -= q_min
            all_calcium_data[i_gen, i_fly] /= (q_max - q_min)
    # compute change upon stimulation by subtracting baseline
    baseline = np.mean(all_calcium_data[:,:,:,:,params.n_s_2p_5s-params.n_s_2p_1s:params.n_s_2p_5s,:], axis=4, keepdims=True)
    all_calcium_data -= np.repeat(baseline, all_calcium_data.shape[4], axis=4)

    for i_axd, axd in enumerate(axds):
        i_gen = i_axd // 2
        genotype = genotypes[i_gen]
        # if genotype == "DNp09":  # TODO
        #     continue
        special_var_ylabel = special_vars_ylabel[i_gen]
        ylim_beh = ylims_beh[i_gen]

        onball = i_axd % 2
        condition = conditions[onball]
        title = f"{genotype} {condition}"

        calcium_data_pre = np.concatenate(all_calcium_data[i_gen, :, 0, onball, :, :], axis=-1)  # concatenate flies: N_samples x (N_trials x N_flies) matrix
        calcium_data_post = np.concatenate(all_calcium_data[i_gen, :, 1, onball, :, :], axis=-1)
        v_data_pre = np.concatenate(all_v_data[i_gen, :, 0, onball, :, :], axis=-1)
        v_data_post = np.concatenate(all_v_data[i_gen, :, 1, onball, :, :], axis=-1)
        beh_data_pre = np.concatenate(all_beh_data[i_gen, :, 0, onball, :, :], axis=-1)
        beh_data_post = np.concatenate(all_beh_data[i_gen, :, 1, onball, :, :], axis=-1)
        
        make_one_headless_imaging_panel(fig, axd, 
                                        calcium_data=[calcium_data_pre, calcium_data_post],
                                        v_data=[v_data_pre, v_data_post],
                                        beh_data=[beh_data_pre, beh_data_post],
                                        ylim_beh=ylim_beh, ylim_v=ylim_v,
                                        ylabel=special_var_ylabel, title=title)
    
    with PdfPages(os.path.join(figures_path, f"fig_headless_imaging_summary.pdf")) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

if __name__ == "__main__":
    make_headless_imaging_figure()