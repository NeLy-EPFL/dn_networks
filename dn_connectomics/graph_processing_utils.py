import numpy as np

import plot_params
import neuron_params
import params
from louvain_clustering import load_communities
from common import identify_name_from_rid



def name_has_probably_a_genetic_match(name: str):
    """
    This is a very simple heuristic to determine whether a neuron has a genetic line.
    """
    has_match = False
    # Namiki line
    if len(name) == 5 and name[:2] == 'DN' and name[2].isalpha() and name[3:].isdigit():
        has_match = True
    # manual annotations: 'DN' + one letter + '_'
    if name[:2] == 'DN' and name[2].isalpha() and name[3] == '_':
        has_match = True
    return has_match

def add_cluster_information(graph):
    """
    Add cluster information to the graph.
    """
    data_folder = plot_params.CLUSTERING_ARGS["data_folder"]
    communities = load_communities(
        data_folder,
        return_type="list",
        data_type="root_id",
        threshold = plot_params.CLUSTERING_ARGS["confusion_mat_size_threshold"],
        )
    
    published_cluster = { # correct for the shuffeling TODO: find a stable loading technique
        0: 9,
        1: 3,
        2: 10,
        3: 1,
        4: 4,
        5: 2,
        6: 7,
        7: 12,
        8: 8,
        9: 11,
        10: 6,
        11: 5,
    }

    for node in graph.nodes():
        cluster_assigned = False
        for count, c_ in enumerate(communities):
            if node in c_:
                graph.nodes[node]["cluster"] = published_cluster[count]
                cluster_assigned = True
        if not cluster_assigned:
            graph.nodes[node]["cluster"] = np.NaN
    return graph
    

def add_names_to_nodes(graph):
    """
    Add names to the nodes of the graph.
    """
    # add the names to the nodes
    for node in graph.nodes():
        graph.nodes[node]["node_label"] = identify_name_from_rid(node)
        graph.nodes[node]["name"] = identify_name_from_rid(node)
    return graph

def get_color_cluster(cluster):
    """
    Get the color of the cluster.
    """
    if cluster == cluster:             # check if a float is not nan
        return plot_params.CLUSTERING_ARGS['cluster_colors'][int(cluster)-1]
    else:
        return plot_params.DEFAULT_NODE_COLOR

def add_node_colors(graph, mode='defined'):
    """
    Add color to the nodes of the graph.
    """
    if mode == 'defined':
        for node in graph.nodes():
            if node in neuron_params.KNOWN_DNS.keys():
                graph.nodes[node]["node_color"] = neuron_params.KNOWN_DNS[node]["color"]
    elif mode == 'genetic':
        for node in graph.nodes():
            graph.nodes[node]["node_color"] = plot_params.NODE_GENETIC_COLORS[0] if name_has_probably_a_genetic_match(graph.nodes[node]["name"]) else plot_params.NODE_GENETIC_COLORS[1]
    elif mode == 'cluster':
        for node in graph.nodes():
            if "cluster" in graph.nodes[node].keys():
                graph.nodes[node]["node_color"] = get_color_cluster(graph.nodes[node]["cluster"])
            else:
                graph.nodes[node]["node_color"] = plot_params.DEFAULT_NODE_COLOR

    else:
        for node in graph.nodes():
            graph.nodes[node]["node_color"] = plot_params.DEFAULT_NODE_COLOR
    return graph

def add_weight_to_edges(graph, selected_field = 'eff_weight', cutoff = True):
    """
    Add 'weight' attribute to the edges of the graph, equal to the selected field.
    Necessary for parsing in the networkx drawing functions.
    """
    edges_to_remove = []
    for edge in graph.edges():
        graph.edges[edge]["weight"] = graph.edges[edge][selected_field]
        if graph.edges[edge]["syn_count"] < params.SYNAPSE_CUTOFF and cutoff:
            edges_to_remove.append(edge)
    graph.remove_edges_from(edges_to_remove)
    return graph


def add_genetic_match_to_nodes(graph):
    """
    Add a boolean to the nodes of the graph, indicating whether the neuron has a genetic match.
    Warning: this is a very simple heuristic to determine whether a neuron has a genetic line.
    """
    for node in graph.nodes():
        graph.nodes[node]["has_genetic_match"] = name_has_probably_a_genetic_match(graph.nodes[node]["name"])
    return graph

def remove_inhbitory_connections(graph):
    """
    Remove inhibitory connections from the graph.
    """
    edges_to_remove = []
    for edge in graph.edges():
        if graph.edges[edge]["eff_weight"] < 0:
            edges_to_remove.append(edge)
    graph.remove_edges_from(edges_to_remove)
    return graph

def remove_excitatory_connections(graph):
    """
    Remove excitatory connections from the graph.
    """
    edges_to_remove = []
    for edge in graph.edges():
        if graph.edges[edge]["eff_weight"] > 0:
            edges_to_remove.append(edge)
    graph.remove_edges_from(edges_to_remove)
    return graph
    