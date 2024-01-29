"""
2023.09.11
author: jonas.braun@epfl.ch
Plotting functions to analyse which behaviours and VNC projections are included in which cluster
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import tqdm

import params

linewidth = 2
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelpad'] = 5

def make_nice_spines(ax):
    """
    Customize the appearance of axis spines and ticks for a given matplotlib axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to customize.

    Returns:
    - None
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 2*linewidth))
    ax.spines['bottom'].set_position(('outward',2*linewidth))
    ax.tick_params(width=linewidth)
    ax.tick_params(length=2.5*linewidth)
    ax.tick_params(labelsize=16)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)

def add_matched_dns_to_df(df, df_known):
    """
    Add matched phenotype class information from df_known to df.

    Parameters:
    - df (pd.DataFrame): The DataFrame to add matched information to.
    - df_known (pd.DataFrame): The DataFrame containing known phenotype class information.

    Returns:
    - pd.DataFrame: The modified DataFrame df with added phenotype class information.
    """
    for index, row in df_known.iterrows():
        dn_name = row["name"]
        dn_beh = row["phenotype class"]
        df.loc[df.name == dn_name, ("phenotype class")] = dn_beh
    return df

def get_beh_for_cluster(df, i_cluster, behaviours):
    """
    Calculate the number of neurons per behavior for a specific cluster.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing cluster and behavior information.
    - i_cluster (int): The cluster number to analyze.
    - behaviours (list of str): List of behavior names.

    Returns:
    - tuple: A tuple containing:
        - n_neurons_per_beh (numpy.ndarray): Array of neuron counts per behavior.
        - total_neurons (int): Total number of neurons in the cluster.
        - n_neurons_other (int): Number of neurons not accounted for by specified behaviors.
    """
    cluster_df = df[df["cluster nb in figure"] == i_cluster]
    n_neurons = np.sum(cluster_df["count"].values)
    
    n_neurons_per_beh = np.zeros_like(behaviours, dtype=int)
    
    for i_beh, behaviour in enumerate(behaviours):
        beh_df = cluster_df[cluster_df["phenotype class"] == behaviour]
        n_neurons_per_beh[i_beh] = np.sum(beh_df["count"].values)
    return n_neurons_per_beh, np.sum(n_neurons_per_beh), n_neurons - np.sum(n_neurons_per_beh)

def get_vnc_for_cluster(df, df_vnc, i_cluster, vnc_names):
    """
    Calculate the number of neurons per VNC neuropil target by a specific cluster.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing cluster and neuron information.
    - df_vnc (pd.DataFrame): The DataFrame containing VNC neuropil projection information.
    - i_cluster (int): The cluster number to analyze.
    - vnc_names (list of str): List of VNC neuropil target names.

    Returns:
    - tuple: A tuple containing:
        - n_neurons_per_vnc (numpy.ndarray): Array of neuron counts per VNC neuropil target.
        - total_neurons_known (int): Total number of neurons in the cluster with known VNC projections.
        - n_neurons_unknown (int): Number of neurons in the cluster with unknown VNC projections.
    """

    cluster_df = df[df["cluster nb in figure"] == i_cluster]
    n_neurons = np.sum(cluster_df["count"].values)
    
    n_neurons_per_vnc = np.zeros_like(vnc_names, dtype=int)
    n_neurons_unknown = 0
    
    for index, row in cluster_df.iterrows():
        n = row["count"]
        name = row["name"]
        if not np.sum(df_vnc["name"]==name):
            n_neurons_unknown += n
            continue
        neuron_df = df_vnc[df_vnc["name"] == name]
        for i_vnc, vnc_name in enumerate(vnc_names):
            if neuron_df[vnc_name].values[0]:
                n_neurons_per_vnc[i_vnc] += n
            
    n_neurons_known = n_neurons - n_neurons_unknown
    return n_neurons_per_vnc, n_neurons_known, n_neurons_unknown

def load_data_from_supp_file():
    """
    Load data from supplementary files and create DataFrames.

    Returns:
    - tuple: A tuple containing:
        - df (pd.DataFrame): DataFrame containing cluster and behavior information.
        - df_vnc (pd.DataFrame): DataFrame containing VNC neuropil projection information.
        - df_known (pd.DataFrame): DataFrame containing known phenotype class information.
        - df_studied (pd.DataFrame): DataFrame containing studied DN clusters information.
    """
    df = pd.read_excel(params.SUPP_FILE_2,sheet_name="DN cluster behaviors")
    df_vnc = pd.read_excel(params.SUPP_FILE_2,sheet_name="DN cluster VNC projections")
    df_vnc = df_vnc.fillna(0)
    df_known = pd.read_excel(params.SUPP_FILE_2,sheet_name="DN litterature aggregation")
    df_studied = pd.read_excel(params.SUPP_FILE_2,sheet_name="Investigated DN clusters")
    df_studied = df_studied.fillna(0)
    
    df = add_matched_dns_to_df(df, df_known)
    return df, df_vnc, df_known, df_studied

def plot_vnc_projections_on_ax(ax, cluster_vnc_norm, cluster_ids, vnc_names):
    """
    Plot VNC neuropil projections on a given matplotlib axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - cluster_vnc_norm (numpy.ndarray): Normalized VNC neuropil projections for clusters.
    - cluster_ids (list of int): List of cluster IDs.
    - vnc_names (list of str): List of VNC neuropil target names.

    Returns:
    - matplotlib.image.AxesImage: The plotted image.
    """
    im = ax.imshow(cluster_vnc_norm, cmap=plt.cm.get_cmap("Greys"), clim=[0,1])
    ax.set_xticks(np.arange(len(vnc_names)))
    ax.set_yticks(np.arange(len(cluster_ids)))
    ax.set_yticklabels(cluster_ids)
    make_nice_spines(ax)
    ax.set_xticklabels(vnc_names, rotation=45, ha="right")
    ax.set_xlabel("VNC neuropil targeted")
    ax.set_ylabel("cluster")
    return im

def plot_behaviors_on_ax(ax, cluster_behaviour_norm, cluster_ids, behaviours):
    """
    Plot behavior information on a given matplotlib axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - cluster_behaviour_norm (numpy.ndarray): Normalized behavior information for clusters.
    - cluster_ids (list of int): List of cluster IDs.
    - behaviours (list of str): List of behavior names.

    Returns:
    - matplotlib.image.AxesImage: The plotted image.
    """
    im = ax.imshow(cluster_behaviour_norm, cmap=plt.cm.get_cmap("Greys"), clim=[0,1])
    ax.set_xticks(np.arange(len(behaviours)))
    ax.set_yticks(np.arange(len(cluster_ids)))
    ax.set_yticklabels(cluster_ids)
    ax.set_xlabel("known behaviors")
    ax.set_ylabel("cluster")
    make_nice_spines(ax)
    ax.set_xticklabels(behaviours, rotation=45, ha="right")
    return im

def plot_known_dns(ax, dn_values_norm, cluster_ids, dn_names):
    """
    Plot known DN information on a given matplotlib axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - dn_values_norm (numpy.ndarray): Normalized known DN information for clusters.
    - cluster_ids (list of int): List of cluster IDs.
    - dn_names (list of str): List of DN names.

    Returns:
    - matplotlib.image.AxesImage: The plotted image.
    """
    im = ax.imshow(dn_values_norm[:,:-1].T, cmap=plt.cm.get_cmap("Greys"), clim=[0,1])
    ax.set_xticks(np.arange(len(dn_names)))
    ax.set_yticks(np.arange(len(cluster_ids)))
    ax.set_yticklabels(cluster_ids)
    make_nice_spines(ax)
    ax.set_xticklabels(dn_names, rotation=45, ha="right")
    ax.set_xlabel("DN of interest")
    ax.set_ylabel("cluster")
    return im



def make_cluster_analysis_plots():
    """
    Plot VNC, behaviour, and studied DN distribution across clusters based on data from supplementary file.

    Returns:
    - None
    """
    df, df_vnc, df_known, df_studied = load_data_from_supp_file()

    cluster_ids = [int(this_id) for this_id in np.unique(df["cluster nb in figure"] )if not np.isnan(this_id)]
    n_cluster = len(cluster_ids)
    behaviours = ["anterior", "takeoff", "landing", "walking", "flight"]
    n_beh = len(behaviours)
    vnc_names = ['T1', 'T2', 'T3', 'LTct', 'IntTct', 'NTct', 'WTct', 'HTct', 'Ov', 'ANm', 'mVAC']  # df_vnc.columns.values[1:]
    n_vnc = len(vnc_names)

    cluster_behaviour = np.zeros((n_cluster, n_beh))
    cluster_behaviour_known = np.zeros((n_cluster))
    cluster_behaviour_unknown = np.zeros((n_cluster))
    for i, i_cluster in enumerate(cluster_ids):
        cluster_behaviour[i], cluster_behaviour_known[i], cluster_behaviour_unknown[i] = get_beh_for_cluster(df, i_cluster, behaviours)
    cluster_behaviour_norm = cluster_behaviour / np.repeat(cluster_behaviour_known[:,np.newaxis], repeats=n_beh, axis=-1)

    cluster_vnc = np.zeros((n_cluster, n_vnc))
    cluster_vnc_known = np.zeros((n_cluster))
    cluster_vnc_unknown = np.zeros((n_cluster))

    for i, i_cluster in enumerate(cluster_ids):
        cluster_vnc[i,:], cluster_vnc_known[i], cluster_vnc_unknown[i] = get_vnc_for_cluster(df, df_vnc, i_cluster, vnc_names)
    cluster_vnc_norm = cluster_vnc / np.repeat(cluster_vnc_known[:,np.newaxis], repeats=n_vnc, axis=-1)

    print("Number of DNs with known behaviour per cluster")
    print(cluster_behaviour_known)
    print("Number of DNs with unknown behaviour per cluster")
    print(cluster_behaviour_unknown)
    
    print("Number of DNs with known VNC projection per cluster")
    print(cluster_vnc_known)
    print("Number of DNs with unknown VNC projection per cluster")
    print(cluster_vnc_unknown)

    fig, axs = plt.subplots(1,2, figsize=(9.5,5))  # , sharey=True)
    im = plot_vnc_projections_on_ax(ax=axs[0], cluster_vnc_norm=cluster_vnc_norm, cluster_ids=cluster_ids, vnc_names=vnc_names)
    im = plot_behaviors_on_ax(ax=axs[1], cluster_behaviour_norm=cluster_behaviour_norm, cluster_ids=cluster_ids, behaviours=behaviours)

    cbar = plt.colorbar(im, ax=axs[1], shrink=1, ticks=[0,0.5,1])
    make_nice_spines(cbar.ax)
    cbar.outline.set_edgecolor('#ffffff')
    fig.tight_layout()
    fig.savefig(os.path.join(params.FIGURES_DIR, "clusters_vnc_behaviour.pdf"), dpi=300)

    dn_names = df_studied.name.values
    dn_values = (df_studied.values[:,1:]).astype(float)
    dn_values_norm = dn_values / np.repeat(np.sum(dn_values, axis=1, keepdims=True), n_cluster+1, axis=1)

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    im = plot_known_dns(ax, dn_values_norm=dn_values_norm, cluster_ids=cluster_ids, dn_names=dn_names)
    cbar = plt.colorbar(im, ax=ax, shrink=0.73, ticks=[0,0.5,1])
    cbar.outline.set_edgecolor('#ffffff')
    make_nice_spines(cbar.ax)
    cbar.outline.set_edgecolor('#ffffff')
    fig.tight_layout()
    fig.savefig(os.path.join(params.FIGURES_DIR, "clusters_studied_dns.pdf"), dpi=300)


if __name__ == "__main__":
    make_cluster_analysis_plots()