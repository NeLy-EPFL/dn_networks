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

from louvain_clustering import confusion_matrix_communities, load_communities
from loaddata import load_graph_and_matrices, get_name_from_rootid
from common import convert_index_root_id
from graph_processing_utils import (
                    add_names_to_nodes,
                    add_cluster_information,
                    add_node_colors,
                    add_weight_to_edges,
                    add_genetic_match_to_nodes,
                    remove_inhbitory_connections,
                    remove_excitatory_connections)
from graph_plot_utils import make_nice_spines, add_edge_legend, draw_graph_selfstanding


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

def add_names_to_indexed_nodes(graph_: nx.DiGraph, equiv_index_rootid_):
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
    Plot the meta graph, i.e. the graph where each node is a cluster of neurons.

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

def draw_clusters_interactions(
        graph: nx.DiGraph,
        filename: str,
        restricted_connections: str = None,
        position_reference: str = None,
        ):
    """
    Draw the clusters interactions. The plot resembles a large flower,
    where each petal is a cluster of neurons. Each petal is coloured according
    to the type of the neurons it contains.Each petal is made of three
    concentric circles, where the neurons are placed. The neurons are placed
    according to their connectiivty within the cluster. The inner circle
    contains only neurons that output to other neurons within the cluster
    without receiving any input from them. The middle circle contains neurons
    that receive inputs and outputs from other neurons within the cluster. The
    outer circle is the rest (only inputs or no connections within the cluster
    in case we restrict to a subset).
    In the middle of the flower, there is a circle containing the neurons that
    are not part of any cluster.
    The positions can be defined with regards to a specific type of connection
    (inhibitory or excitatory) or all connections. 
    """

    # count the number of clusters defined
    cluster_list = []
    for graph_node in graph.nodes():
        if 'cluster' in graph.nodes[graph_node].keys() and graph.nodes[graph_node]['cluster'] == graph.nodes[graph_node]['cluster']:
            cluster_list.append(graph.nodes[graph_node]['cluster'])
    
    clusters = list(set(cluster_list))

    if clusters == []:
        raise ValueError('No clusters defined in the graph')
    nb_clusters = len(clusters)


    cluster_centers = [
        ((1+1.2*nb_clusters)*np.cos(2*np.pi*cluster_nb/nb_clusters),
        (1+1.2*nb_clusters)*np.sin(2*np.pi*cluster_nb/nb_clusters))
        for cluster_nb in range(nb_clusters)
        ]

    positions = {}
    for cluster_idx, cluster_nb in enumerate(clusters):
        cluster_nb
        # get a subset of the graph known_graph, where the 'cluster' attribute is equal to cluster_of_interest
        subgraph = graph.subgraph(
            [
                node
                for node in graph.nodes()
                if graph.nodes[node]['cluster'] == cluster_nb
            ]
            )
        # use the network with specific connectivity to define the positions of the neurons
        if position_reference == 'inhibitory': 
            subgraph = remove_excitatory_connections(subgraph)
        elif position_reference == 'excitatory':
            subgraph = remove_inhbitory_connections(subgraph)

        # draw the graph
        pos = draw_graph_selfstanding(subgraph,center=cluster_centers[cluster_idx],output='pos')
        positions = {**positions, **pos}

    # additional nodes that are not clustered
    additional_nodes = [node for node in graph.nodes() if not graph.nodes[node]['cluster'] in clusters]
    subgraph = graph.subgraph(additional_nodes)
    if position_reference == 'inhibitory': 
        subgraph = remove_excitatory_connections(subgraph)
    elif position_reference == 'excitatory':
        subgraph = remove_inhbitory_connections(subgraph)
    pos = draw_graph_selfstanding(subgraph,center=(0,0),output='pos',radius_scaling=2)
    positions = {**positions, **pos}


    # draw the graph
    fig, ax = plt.subplots(figsize=plot_params.NETWORK_PLOT_ARGS["fig_size"])

    if restricted_connections == 'inhibitory': 
        graph = remove_excitatory_connections(graph)
    elif restricted_connections == 'excitatory':
        graph = remove_inhbitory_connections(graph)

    edge_norm = max([np.abs(graph.edges[e]["weight"]) for e in graph.edges]) / 5
    widths = [np.abs(graph.edges[e]["weight"]) / edge_norm for e in graph.edges]
    edges_colors = [
            plot_params.EXCIT_COLOR
            if graph.edges[e]["weight"] > 0
            else plot_params.INHIB_COLOR
            for e in graph.edges
        ]
    node_labels = {
            n: graph.nodes[n]["node_label"]
            if "node_label" in graph.nodes[n].keys()
            else ""
            for n in graph.nodes
    }
    node_colors = [
            graph.nodes[n]["node_color"]
            if "node_color" in graph.nodes[n].keys()
            else "grey"
            for n in graph.nodes
        ]
    nx.draw(
        graph,
        pos=positions,
        nodelist=graph.nodes,
        with_labels=True,
        labels=node_labels,
        alpha=0.5,
        node_size=plot_params.NETWORK_PLOT_ARGS["node_size"],
        node_color=node_colors,
        edge_color=edges_colors,
        width=widths,
        connectionstyle="arc3,rad=0.1",
        font_size=2,
        font_color="black",
        ax=ax,
    )
    add_edge_legend(ax, normalized_weights=widths,
                    color_list=edges_colors,
                    arrow_norm=1 / edge_norm,)
    
    # check if savign folder exists
    if not os.path.exists(plot_params.NETWORK_PLOT_ARGS["folder"]):
        os.makedirs(plot_params.NETWORK_PLOT_ARGS["folder"])
    
    plt.savefig(os.path.join(
        plot_params.NETWORK_PLOT_ARGS["folder"],
        filename + '.png'
    ),  
    dpi=300)
    plt.savefig(os.path.join(
        plot_params.NETWORK_PLOT_ARGS["folder"],
        filename + '.pdf'
    ),  
    dpi=300)
    plt.savefig(os.path.join(
        plot_params.NETWORK_PLOT_ARGS["folder"],
        filename + '.eps'
    ),  
    dpi=300)
    return ax


def draw_network_organised_by_clusters(
    restricted_nodes: str = 'known_only',
    restricted_clusters: list[int] = None,
    restricted_connections: str = None,
    position_reference: str = None,):
    """
    Draw the network organised by clusters. 
    Definition of the part of the graph to plot is done by the arguments:
        restricted_nodes: 'known_only' or 'all'
        restricted_clusters: list of clusters to plot, or None
        restricted_connections: 'inhibitory', 'excitatory', or None
        position_reference: None, 'inhibitory', 'excitatory', 'all'. Base the 
            position of the neurons on the connections within the cluster
            taking into account only the connections of the type specified. If
            None, it is the same as restricted_connections.
    """
    (
        dn_graph,
        _,
        _,
        _,
        _,
    ) = load_graph_and_matrices("dn")

    # add cluster information to the graph
    dn_graph = add_cluster_information(dn_graph)
    # add a colour depending on the type of the neuron
    dn_graph = add_node_colors(dn_graph, mode = 'cluster')
    # add the names of the neurons to the graph
    dn_graph = add_names_to_nodes(dn_graph)
    # add a 'weight' attribute to the edges, equal to the effectve weight
    dn_graph = add_weight_to_edges(dn_graph)
    # add a field 'genetic_match' to the nodes, True if the node has a potential genetic match
    dn_graph = add_genetic_match_to_nodes(dn_graph)

    filename = 'network_in_clusters'
    # restricting nodes:
    if restricted_nodes == 'known_only':
        dn_graph = dn_graph.subgraph(
            [node for node in dn_graph.nodes() if dn_graph.nodes[node]["has_genetic_match"]]
            )
        filename += '_known_neurons_only'
    if restricted_clusters is not None:
        dn_graph = dn_graph.subgraph(
            [node for node in dn_graph.nodes() if dn_graph.nodes[node]["cluster"] in restricted_clusters]
            )
        filename += '_restricted_clusters'
        filename += '_'.join([str(cluster) for cluster in restricted_clusters])
    if restricted_connections == 'inhibitory':
    #    dn_graph = remove_excitatory_connections(dn_graph)
        filename += '_inhibitory_connections_only'
    elif restricted_connections == 'excitatory':
    #    dn_graph = remove_inhbitory_connections(dn_graph)
        filename += '_excitatory_connections_only'
    if position_reference is None:
        position_reference = restricted_connections
    if position_reference == None:
        filename += '_position_reference_all'
    else:
        filename += '_position_reference_' + position_reference

    ax = draw_clusters_interactions(
        dn_graph,
        filename=filename,
        restricted_connections=restricted_connections,
        position_reference=position_reference,
        )


if __name__ == "__main__":
    draw_meta_network()
    draw_network_organised_by_clusters(
        restricted_nodes = plot_params.NETWORK_PLOT_ARGS["restricted_nodes"],
        restricted_clusters = plot_params.NETWORK_PLOT_ARGS["restricted_clusters"],
        restricted_connections = plot_params.NETWORK_PLOT_ARGS["restricted_connections"],
        position_reference = plot_params.NETWORK_PLOT_ARGS["position_reference"],
        )
