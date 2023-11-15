"""
2023.08.30
author: femke.hurtak@epfl.ch
Script to compute the recurrence of specific subnetworks.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import pickle
import math

from loaddata import load_graph_and_matrices, load_nodes_and_edges
from statistics_utils import connectivity_stats_to_dict
from graph_plot_utils import get_downstream_specs
import plot_params
import neuron_params


def sigmoid(x):
    return ((1 / (1 + math.exp(-x))) - 0.5) * 2


def topness(recurrence_index, feedback, n_downstream):
    """
    Compute the topness metric. The topness metric is defined
    as the ratio between the downstream density and the number of feedback connections.

    Parameters
    ----------
    recurrence_index : float
        Recurrence index of the neuron.
    feedback : int
        Number of feedback connections.
    n_downstream : int
        Number of neurons downstream.

    Returns
    -------
    topness_score : float
        Topness score.
    """
    if n_downstream < 5:
        return np.NaN
    else:
        if feedback == 0:
            return 1  #  sigmoid(+inf) == 1
        return sigmoid(recurrence_index / (feedback / n_downstream))


def compute_topness_metric(
    connectivity_dict: dict,
    edges: pd.DataFrame,
    equiv_rid_index: pd.DataFrame,
):
    """
    For each neuron, compute the topness metric. The topness metric is defined
    as the ratio between the downstream density and the number of feedback connections.

    Parameters
    ----------
    connectivity_dict : dict
        Dictionary with the number of neurons connected for each neuron as one
        of the values added to the value dict.
        It Assumes that "density" is already computed.
    edges : pd.DataFrame
        Dataframe containing the edges of the network.

    Returns
    -------
    connectivity_dict : dict
        Dictionary with the infromation on each neuron.
    """
    list_dns = equiv_rid_index.root_id.values

    for key, value in tqdm(connectivity_dict.items()):
        rid = {value["root_id"]: {"name": "neuron", "color": "k"}}
        _, network_downstream = get_downstream_specs(
            edges, rid, list_dns, feedback_layer_1=False, feedback_layer_2=True
        )
        edges_downstream = network_downstream["edges"]
        # count the number of edges (tuple of 2 neurons) where the second neuron
        # is the root_id of the neuron of interest
        n_feedback_connections = sum(
            [1 for edge in edges_downstream if edge[1] == value["root_id"]]
        )
        topness_score = topness(
            value["downstream_density"],
            n_feedback_connections,
            value["n_neurons_downstream"],
        )
        connectivity_dict[key]["topness"] = topness_score

    return connectivity_dict


def compute_downstream_density(
    connectivity_dict: dict,
    edges: pd.DataFrame,
    equiv_rid_index: pd.DataFrame,
):
    """
    For each neuron, compute the density of the downstream graph. The density
    is defined as the number of connections divided by the number of possible
    connections.

    Parameters
    ----------
    connectivity_dict : dict
        Dictionary with the number of neurons connected for each neuron as one
        of the values added to the value dict.
    edges : pd.DataFrame
        Dataframe containing the edges of the network.

    Returns
    -------
    connectivity_dict : dict
        Dictionary with the infromation on each neuron.
    """
    list_dns = equiv_rid_index.root_id.values

    for key, value in tqdm(connectivity_dict.items()):
        rid = {value["root_id"]: {"name": "neuron", "color": "k"}}
        _, network_downstream = get_downstream_specs(edges, rid, list_dns)
        # Number of neurons downstream
        n_neurons_downstream = value["n_neurons_downstream"]
        # Number of possible connections
        n_possible_connections = n_neurons_downstream * (
            n_neurons_downstream - 1  # no self-connections
        )
        # Number of actual connections
        n_actual_connections = len(network_downstream["edges"])
        # Density
        if n_neurons_downstream == 0 or n_neurons_downstream == 1:
            density = 0
        else:
            density = n_actual_connections / n_possible_connections
        connectivity_dict[key]["downstream_density"] = density

    return connectivity_dict


def connectivity_vs_recurrence_plots():
    # Draw a 2d plot with number of neurons downstream vs density of the
    # downstream graph

    # Load the data
    graph_selected = "dn"  # "dn" # "dn_gng" # "central_an_dn"
    _, _, unn_matrix, _, equiv_index_rootid = load_graph_and_matrices(
        graph_selected
    )
    _, edges = load_nodes_and_edges()

    working_folder = plot_params.STATS_ARGS["folder"]

    # Create a dictionary with the number of neurons downstream for each neuron
    # in the graph
    connectivity_dict = {
        index: {"root_id": rid, "color": "grey", "name": "neuron"}
        for index, rid in zip(
            equiv_index_rootid.index, equiv_index_rootid.root_id
        )
    }
    for key, value in connectivity_dict.items():
        if value["root_id"] in neuron_params.KNOWN_DNS.keys():
            connectivity_dict[key]["color"] = neuron_params.KNOWN_DNS[
                value["root_id"]
            ]["color"]
            connectivity_dict[key]["name"] = neuron_params.KNOWN_DNS[
                value["root_id"]
            ]["name"]

    connectivity_dict = connectivity_stats_to_dict(
        connectivity_dict, unn_matrix.todense()
    )
    connectivity_dict = compute_downstream_density(
        connectivity_dict, edges, equiv_index_rootid
    )

    connectivity_dict = compute_topness_metric(
        connectivity_dict, edges, equiv_index_rootid
    )

    # Save the dictionary
    with open(
        os.path.join(working_folder, "connectivity_dict.pkl"), "wb"
    ) as f:
        pickle.dump(connectivity_dict, f)

    ## ----------------- Plot the results ----------------- ##

    # plot the number of neurons downstream vs density of the downstream graph
    # for each neuron
    fig, ax = plt.subplots()
    ax.scatter(
        [
            connectivity_dict[key]["n_excit_neurons_downstream"]
            for key in connectivity_dict.keys()
        ],
        [
            connectivity_dict[key]["downstream_density"]
            for key in connectivity_dict.keys()
        ],
        c=[
            connectivity_dict[key]["color"] for key in connectivity_dict.keys()
        ],
        alpha=0.5,
    )
    ax.set_xlabel("Number of excited neurons downstream")
    ax.set_ylabel("Density of the downstream graph")
    fig.savefig(
        os.path.join(
            working_folder,
            "n_excit_neurons_downstream_vs_recurrence_index.pdf",
        )
    )
    fig.savefig(
        os.path.join(
            working_folder,
            "n_excit_neurons_downstream_vs_recurrence_index.png",
        )
    )

    ## --- Plot the topness metric --- ##
    fig, ax = plt.subplots()
    ax.scatter(
        [
            connectivity_dict[key]["n_neurons_downstream"]
            for key in connectivity_dict.keys()
        ],
        [
            connectivity_dict[key]["topness"]
            for key in connectivity_dict.keys()
        ],
        c=[
            connectivity_dict[key]["color"] for key in connectivity_dict.keys()
        ],
        alpha=0.5,
    )
    ax.set_xlabel("Number of neurons downstream")
    ax.set_ylabel("Topness metric")
    fig.savefig(
        os.path.join(
            working_folder,
            "feedforward_metric_vs_n_neurons_downstream.pdf",
        )
    )
    fig.savefig(
        os.path.join(
            working_folder,
            "feedforward_metric_vs_n_neurons_downstream.png",
        )
    )

    # Violin plot of the topness metric
    fig, ax = plt.subplots()
    sns.violinplot(
        data=[
            connectivity_dict[key]["topness"]
            for key in connectivity_dict.keys()
        ],
        palette=["grey"],
        linewidht=1,
        cut=0,
        ax=ax,
    )
    ax.set_ylabel("Topness metric")
    fig.savefig(os.path.join(working_folder, "feedforward_metric_violin.pdf"))


if __name__ == "__main__":
    connectivity_vs_recurrence_plots()
