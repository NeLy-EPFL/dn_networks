import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
from tqdm import tqdm
import scipy as sc
import scipy.cluster.hierarchy as sch

from loaddata import get_name_from_rootid


def connections_up_to_n_hops(matrix, n):
    """
    input: (sparse) matrix

    computes: recursively computes all the connections possible in less than
    n hops.

    output: (sparse) matrix
    From an input (sparse) matrix compute the matrix representing the
    connectivity
    """
    return_mat = matrix
    for i in range(2, n + 1):
        return_mat += return_mat @ matrix
    return return_mat


def generate_random(matrix):
    """
    Generates a random matrix by shuffeling the elements of the matrix in
    argument
    """
    rng = np.random.default_rng()
    (n, m) = matrix.shape
    full_mat = matrix.todense()
    arr = np.array(full_mat.flatten())
    rng.shuffle(arr)  # , axis=1)
    arr = arr.reshape((n, m))
    return sc.sparse.csr_matrix(arr)


def density(matrix):
    """
    input: sparse matrix
    output: number of existing edges divided by edges on a fully-comnected
    graph of identical size
    """
    mat = matrix
    den = mat.nnz / np.prod(mat.shape)
    return den


def select_subset_matrix(matrix, sub_indices):
    """
    Extract a submatrix from a sparse matrix given by specific indices
    """
    mat_tmp = matrix[sub_indices, :]
    mat = mat_tmp[:, sub_indices]
    return mat


def connection_degree_n_hops(matrix, n, dn_indices=[]):
    """
    inputs:
        sparse matrix
        number of maximimal hops (n)
        indices to evaluate
    output:
        array of size n with densities for increasing number of hops allowed
    """

    degree = []
    for deg in range(1, n + 1):
        mat = connections_up_to_n_hops(matrix, deg)
        if len(dn_indices) == 0:
            degree.append(density(mat))
        else:
            dn_mat = select_subset_matrix(mat, dn_indices)
            degree.append(density(dn_mat))
    return degree


def cluster_matrix(matrix):
    """
    inputs: scipy sparse correlation matrix

    processing: clusters the matrix using scipy hierarchical clustering
        order indices such that the underlying dendrogram is a tree

    outputs: hierarchical clustered sparse matrix
    """
    # if the matrix is a scipy sparse matrix, convert to dense
    if sc.sparse.issparse(matrix):
        matrix = matrix.todense()
    dendrogram = sch.dendrogram(
        sch.linkage(matrix, method="ward"), no_plot=True
    )
    # get the order of the indices
    order = dendrogram["leaves"]
    # reorder the matrix
    clustered = matrix[order, :]
    clustered = clustered[:, order]
    return clustered, order


def convert_rootid_index(equiv_index_rootid, neuron_subset):
    """
    define which indices in the matrix correspond to the neurons
    of interest
    """
    indices = [
        equiv_index_rootid.loc[equiv_index_rootid["root_id"] == neuron].index[
            0
        ]
        for neuron in neuron_subset
    ]
    return indices


def convert_index_root_id(equiv_index_rootid, indices_subset):
    """
    define which rids correspond to the neurons
    of interest in the matrix
    """
    if isinstance(indices_subset, int):
        indices_subset = [indices_subset]
    rids = [
        equiv_index_rootid.loc[
            equiv_index_rootid.index == neuron
        ].root_id.values[0]
        for neuron in indices_subset
    ]
    return rids


def identify_rids_from_name(neuron_name: str, set_of_neurons: dict):
    """
    Return the dictionary of root_ids corresponding to the name of the neuron.

    Parameters
    ----------
    neuron_name : str
        Name of the neuron.
    neurons_of_interest : dict
        Dictionary of neurons of interest. Keys are root_ids, values is a
        dictionary with 'name', 'color', 'root_id'

    Returns
    -------
    dict
        Dictionary of neurons of interest.
    """
    neurons_of_interest = {
        neuron_id: neuron_info
        for neuron_id, neuron_info in set_of_neurons.items()
        if neuron_name in neuron_info["name"]
    }
    return neurons_of_interest


def identify_name_from_rid(root_id: int, set_of_neurons: dict = None):
    """
    Return name corresponding to the root_id of the neuron.

    Parameters
    ----------
    neuron_name : int
        Root_id of the neuron.
    neurons_of_interest : dict, optional
        Dictionary of neurons of interest. Keys are root_ids, values is a
        dictionary with 'name', 'color', 'root_id'.
        This allows to get the name in a certain subset.
        The default is None, in which case the name is retrieved from the
        database.

    Returns
    -------
    str
        Name of the neuron.
    """
    if not set_of_neurons is None:
        neuron_name = (
            set_of_neurons[root_id]["name"]
            if root_id in set_of_neurons.keys()
            else ""
        )
    else:
        neuron_name = get_name_from_rootid(root_id)
    return neuron_name


def plot_matrix_simple(
    matrix, savefig: str = None, ax: plt.Axes = None, vmax=50, cmap="RdBu_r"
):
    """
    Plot a matrix with a colorbar and save it.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    if sc.sparse.issparse(matrix):
        matrix_ = matrix.todense()
    else:
        matrix_ = matrix
    c = ax.imshow(
        matrix_,
        cmap=cmap,
        vmax=vmax,
        vmin=-1 * vmax,
    )
    ax.set_xlabel("postsynaptic neuron")
    ax.set_ylabel("presynaptic neuron")
    cbar = plt.colorbar(c, ax=ax)
    if savefig is not None:
        plt.savefig(savefig)
    return ax
