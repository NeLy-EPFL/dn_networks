import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm


def connectivity_stats(matrix, info):
    """
    input: sparse matrix, dataframe with info on neurons
    output: identical dataframe with statistical information on number of
    neurons connected
    """
    id_keys = ["all", "GNG_DN", "DN"]
    nt_keys = ["excit", "inhib", "connected"]
    syn_keys = ["in", "out"]

    for id_key in id_keys:
        selection = info[id_key]
        for syn_key in syn_keys:
            if syn_key == "in":
                subset_connections = np.multiply(
                    matrix.T.todense(), np.matrix(selection)
                )
            else:
                subset_connections = np.multiply(
                    matrix.todense(), np.matrix(selection)
                )
            for nt_key in nt_keys:
                if nt_key == "connected":
                    info[f"{id_key}_{nt_key}_{syn_key}"] = [
                        np.count_nonzero(subset_connections[index, :])
                        for index in tqdm(info.index)
                    ]
                elif nt_key == "excit":
                    info[f"{id_key}_{nt_key}_{syn_key}"] = [
                        np.count_nonzero(subset_connections[index, :] > 0)
                        for index in tqdm(info.index)
                    ]
                elif nt_key == "inhib":
                    info[f"{id_key}_{nt_key}_{syn_key}"] = [
                        np.count_nonzero(subset_connections[index, :] < 0)
                        for index in tqdm(info.index)
                    ]

    for id_key in id_keys:
        for syn_key in syn_keys:
            for nt_key in nt_keys:
                info.sort_values(
                    by=f"{id_key}_{nt_key}_{syn_key}",
                    ascending=False,
                    inplace=True,
                    ignore_index=True,
                )
                info.reset_index(
                    inplace=True, names=f"{id_key}_{nt_key}_{syn_key}_sorting"
                )

    return info


def connectivity_stats_to_dict(
    storage: dict,
    matrix: np.ndarray,
) -> dict:
    """
    Enter the connectivity staistics information for one given graph in a
    dictionary, for compatibility with the nx library.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Adjacency matrix of the graph.
    storage : dict
        Dictionary to store the information.
        each value is a dict with information concerning the neuron.

    Returns
    -------
    storage : dict
        Dictionary with the information.
    """
    if storage is None:
        storage = {}

    size_matrix = matrix.shape[0]
    for index in tqdm(range(size_matrix)):
        # number of neurons downstream
        storage[index]["n_neurons_downstream"] = np.count_nonzero(
            matrix[index, :]
        )
        storage[index]["n_excit_neurons_downstream"] = np.count_nonzero(
            matrix[index, :] > 0
        )
        storage[index]["n_inhib_neurons_downstream"] = np.count_nonzero(
            matrix[index, :] < 0
        )
        # synapse count
        storage[index]["n_synapses_downstream"] = np.sum(
            np.abs(matrix[index, :])
        )

    return storage


def synapse_stats(matrix, info):
    """
    input: sparse matrix, dataframe with info on neurons
    output: identical dataframe with statistical information on number of
    synapses connected or effective connection strength for normalised cases
    """
    id_keys = ["all", "GNG_DN", "DN"]
    nt_keys = ["excit", "inhib", "connected"]
    syn_keys = ["in", "out"]

    for id_key in id_keys:
        selection = info[id_key]
        for syn_key in syn_keys:
            if syn_key == "in":
                subset_connections = np.multiply(
                    matrix.T.todense(), np.matrix(selection)
                )
            else:
                subset_connections = np.multiply(
                    matrix.todense(), np.matrix(selection)
                )
            for nt_key in nt_keys:
                if nt_key == "connected":
                    info[f"{id_key}_{nt_key}_{syn_key}"] = [
                        np.sum(np.abs(subset_connections[index, :]))
                        for index in tqdm(info.index)
                    ]
                elif nt_key == "excit":
                    info[f"{id_key}_{nt_key}_{syn_key}"] = [
                        np.sum(
                            np.multiply(
                                subset_connections[index, :],
                                subset_connections[index, :] > 0,
                            )
                        )
                        for index in tqdm(info.index)
                    ]
                elif nt_key == "inhib":
                    info[f"{id_key}_{nt_key}_{syn_key}"] = [
                        -1
                        * np.sum(
                            np.multiply(
                                subset_connections[index, :],
                                subset_connections[index, :] < 0,
                            )
                        )
                        for index in tqdm(info.index)
                    ]

    for id_key in id_keys:
        for syn_key in syn_keys:
            for nt_key in nt_keys:
                info.sort_values(
                    by=f"{id_key}_{nt_key}_{syn_key}",
                    ascending=False,
                    inplace=True,
                    ignore_index=True,
                )
                info.reset_index(
                    inplace=True, names=f"{id_key}_{nt_key}_{syn_key}_sorting"
                )

    return info
