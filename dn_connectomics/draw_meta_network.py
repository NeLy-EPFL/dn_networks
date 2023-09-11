"""
2023.08.30
author: femke.hurtak@epfl.ch
script to draw the meta network, obtained by clustering the neurons. Each node 
of the meta network is a cluster of neurons, and the edges are the connections
between the clusters.
"""

import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

import plot_params

from louvain_clustering import confusion_matrix_communities
from loaddata import load_graph_and_matrices, get_name_from_rootid
from common import convert_index_root_id
from graph_plot_utils import make_nice_spines, add_edge_legend


def load_communities(
    path: str,
    return_type: str = "set",
    threshold: int = 2,
    data_type: str = "node_index",
):
    """
    Load the communities from a file.

    Parameters
    ----------
    path : str
        The path to the file containing the communities.
    return_type : str
        The type of the return value. Can be either "set" or "list".

    Returns
    -------
    communities : list[return_type[int]]
        The list of communities.
    """
    df = pd.read_csv(os.path.join(path, "clustering.csv"))
    if data_type not in df.columns:
        data_type = "node_index"
    if return_type == "set":
        communities_ = [
            set(df[df["cluster"] == cluster][data_type].values)
            for cluster in df["cluster"].unique()
        ]
    elif return_type == "list":
        communities_ = [
            df[df["cluster"] == cluster][data_type].values
            for cluster in df["cluster"].unique()
        ]
    else:
        raise ValueError("return_type must be either set or list")

    communities_ = [
        community for community in communities_ if len(community) > threshold
    ]
    return communities_


def define_meta_graph(
    matrix: np.ndarray,
    communities: list[set[int]] = None,
    size_threshold: int = 2,
):
    """
    Define a meta graph from the confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The confusion matrix.

    Returns
    -------
    meta_graph : nx.DiGraph
        The meta graph.
    """
    graph_ = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph)
    if communities is not None:
        communities_used = [
            community_
            for community_ in communities
            if len(community_) > size_threshold
        ]
        for node in graph_.nodes():
            graph_.nodes[node]["composition"] = communities_used[node]
    return graph_


def add_names_to_nodes(graph_: nx.DiGraph, equiv_index_rootid_):
    """
    Add the names of the neurons to the attributes of the nodes.
    """
    for node in graph_.nodes():
        list_rootids = convert_index_root_id(
            equiv_index_rootid_, graph_.nodes[node]["composition"]
        )
        list_of_names = [get_name_from_rootid(x_) for x_ in list_rootids]
        graph_.nodes[node]["names"] = list_of_names
    return graph_


def plot_meta_graph(
    meta_graph: nx.DiGraph,
    working_folder: str,
    details: dict = None,
    extension: str = ".pdf",
    normalisation: int = 1000,
    graph_name: str = "meta_graph",
):
    """
    Plot the meta graph.

    Parameters
    ----------
    meta_graph : nx.DiGraph
        The meta graph.
    working_folder : str
        The path to the working folder.
    details : dict
        The details of the meta graph, used for plotting.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.circular_layout(meta_graph)
    edges, weights = zip(*nx.get_edge_attributes(meta_graph, "weight").items())

    node_colors = details["colors"] if details is not None else "black"
    labels = (
        {node: details["labels"][node] for node in meta_graph.nodes()}
        if (details is not None and details["show_labels"])
        else {node: str(int(node) + 1) for node in meta_graph.nodes()}
    )
    edge_colors = [
        plot_params.INHIB_COLOR if w_ < 0 else plot_params.EXCIT_COLOR
        for w_ in weights
    ]
    widths = [np.abs(w_) / normalisation for w_ in weights]
    nx.draw(
        meta_graph,
        pos,
        node_size=[
            len(meta_graph.nodes[node]["composition"])
            * plot_params.META_GRAPH["scale_nodes"]
            for node in meta_graph.nodes()
        ],
        node_color=node_colors,
        alpha=0.5,
        edgelist=edges,
        edge_color=edge_colors,
        width=widths,
        labels=labels,
        with_labels=True,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )
    ax = add_edge_legend(
        ax, [w_ for w_ in widths if w_ >= 1], edge_colors, 1 / normalisation
    )
    plt.tight_layout()
    labels = (
        "_no_labels" if (details is None or not details["show_labels"]) else ""
    )
    plt.savefig(
        os.path.join(working_folder, graph_name + labels + extension),
        dpi=300,
    )
    return


def draw_meta_network():
    working_folder = os.path.join(
        plot_params.CLUSTERING_ARGS["folder"], "data"
    )

    (
        _,
        _,
        unn_matrix,
        _,
        _,
    ) = load_graph_and_matrices("dn")
    graph = nx.from_scipy_sparse_array(unn_matrix, create_using=nx.DiGraph)

    size_threshold = plot_params.CLUSTERING_ARGS[
        "confusion_mat_size_threshold"
    ]
    communities = load_communities(working_folder, threshold=size_threshold)
    confusion_matrix, ax = confusion_matrix_communities(
        graph,
        communities,
        connection_type=plot_params.CLUSTERING_ARGS["confusion_mat_values"],
        size_threshold=plot_params.CLUSTERING_ARGS[
            "confusion_mat_size_threshold"
        ],
        count_synapses=plot_params.CLUSTERING_ARGS[
            "confusion_mat_count_synpases"
        ],
        normalise_by_size=plot_params.CLUSTERING_ARGS[
            "confusion_mat_normalise"
        ],
        return_ax=True,
    )
    make_nice_spines(ax)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_folder, "confusion_matrix_normalised_cluster_size.pdf"
        ),
        dpi=300,
    )

    meta_graph = define_meta_graph(
        confusion_matrix, communities, size_threshold
    )

    extensions = [".pdf", ".png", ".eps"]
    for extension in extensions:
        plot_meta_graph(
            meta_graph,
            working_folder,
            details=None,
            extension=extension,
            normalisation=plot_params.META_GRAPH["edge_normalisation"],
            graph_name="meta_graph_literature_analysis_normalised_cluster_size",
        )


if __name__ == "__main__":
    draw_meta_network()
