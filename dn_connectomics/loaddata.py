"""
2023.08.30
author: femke.hurtak@epfl.ch
Helping script to load the data and interact with it
"""

import pickle
import os
import pandas as pd
import numpy as np

import params


def load_graph_and_matrices(subset="dn"):
    """
    Load the graph from the data directory.
    """
    accepted_subsets = ["dn", "dn_gng", "central_an_dn"]
    assert subset in accepted_subsets, "Subset must be one of: {}".format(
        accepted_subsets
    )

    filename = f"graph_{subset}.pkl"
    filepath = os.path.join(params.DATA_DIR, filename)

    with open(filepath, "rb") as f:
        data_dump = pickle.load(f)

    unn_matrix = data_dump["mat_unnorm"].asfptype()
    syncount_matrix = data_dump["mat_syncount"].asfptype()
    nn_matrix = data_dump["mat_norm"]
    graph = data_dump["nx_graph"]
    equiv_index_rootid = data_dump["lookup"]

    return graph, syncount_matrix, unn_matrix, nn_matrix, equiv_index_rootid


def load_nodes_and_edges():
    """
    Load the nodes and edges from the data directory.
    """
    nodes_file = params.NODES_FILE
    edges_file = params.EDGES_FILE

    nodes_file = os.path.join(params.DATA_DIR, nodes_file)
    edges_file = os.path.join(params.DATA_DIR, edges_file)

    with open(nodes_file, "rb") as f:
        nodes = pd.read_pickle(f)

    with open(edges_file, "rb") as f:
        edges = pd.read_pickle(f)

    # nb: edges are fine-grained such that a single row corresponds to a single
    # (pre-post root id) pair and a single neuropil. This means that the
    # synapse count is not the total number of synapses between the two neurons
    # but the number of synapses in the given neuropil.

    return nodes, edges


def load_names():
    """
    Load the names from the data directory.
    """
    names_file = params.NAMES_FILE
    names_file = os.path.join(params.DATA_DIR, names_file)

    # read the csv file as a pd dataframe
    names_ = pd.read_csv(names_file)
    names_ = names_[["root_id", "name_taken"]]
    names_ = names_.rename(columns={"name_taken": "name"})
    return names_


def get_name_from_rootid(id: int, empty_type: str = np.NaN):
    """
    Get the name of the neuron from the root id.
    To find the match the names dataframe is loaded based on the preset
    parameters.

    Parameters
    ----------
    id : int
        The root id of the neuron.
    empty_type : str
        The type of empty value to return if the neuron has no name.

    Returns
    -------
    name : str
        The name of the neuron.
    """
    names = load_names()
    name_ = names.loc[names.root_id == id]
    if len(name_) == 0:
        return empty_type
    return name_.name.values[0]


def get_rootids_from_name(name: str):
    """
    Get the root ids of the neuron from the name.
    To find the match the names dataframe is loaded based on the preset
    parameters.

    Parameters
    ----------
    name : str
        The name of the neuron.

    Returns
    -------
    root_id : list[int]
        The root ids of the neuron.
    """
    names = load_names()
    root_id = names.loc[names.name == name]
    if len(root_id) == 0:
        return np.NaN
    return root_id.root_id.values
