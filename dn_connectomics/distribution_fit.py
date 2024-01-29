"""
2023.08.30
author: femke.hurtak@epfl.ch
Script to compare the network statistics with know graph models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import optimize
import seaborn as sns

import plot_params

from loaddata import load_graph_and_matrices
from connectivity_stats import connectivity_stats, prepare_table
from graph_plot_utils import make_nice_spines


def get_avg_degree(graph_):
    degree_sequence = sorted((d for n, d in graph_.degree()), reverse=True)
    return np.mean(degree_sequence)


def get_random_graph_degree_seq(graph_):
    """
    Get the degree sequence of a random graph with the same number of nodes and
    edges as the input graph.
    """
    avg_degree = get_avg_degree(graph_)
    nb_nodes = len(graph_.nodes)
    random_graph = nx.gnm_random_graph(
        nb_nodes, avg_degree * nb_nodes / 2, directed=True
    )
    degree_sequence_random = sorted(
        (d for _, d in random_graph.degree()), reverse=True
    )
    return degree_sequence_random


def r2_score(y_true, y_pred):
    """
    Compute the R2 score between two arrays.
    """
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum(
        (y_true - np.mean(y_true)) ** 2
    )


def get_fit(
    down_stats, func_type="exponential", range_fit=None, initial_guess=None
):
    """
    Fit a distribution to the data, based on the scipy.optimize.curve_fit
    function.
    """

    # Define the function to fit
    def func(x, a, b):
        if func_type == "exponential":
            return a * np.exp(-b * x)
        elif func_type == "power_law":
            return a * (x + 1) ** (-b)
        raise ValueError(f"Unknown function type: {func_type}")

    if range_fit is not None:
        X_fit = np.arange(*range_fit)
        Y_fit = down_stats[range_fit[0] : range_fit[1]]
    else:
        X_fit = np.arange(len(down_stats))
        Y_fit = down_stats

    # Fit the function
    popt, pcov = optimize.curve_fit(func, X_fit, Y_fit, p0=initial_guess)

    X = np.arange(len(down_stats))
    fitted_distribution = func(X, *popt)
    return fitted_distribution, r2_score(Y_fit, fitted_distribution)


def plot_ridges(list_distributions):
    """
    Plot the distribution of the number of outgoing connections for the
    biological data and the fitted graphs. Display as a ridge plot.

    Parameters
    ----------
    list_distributions : list[dict]
        List of dictionaries containing the distributions to plot.

    Returns
    -------
    None.
    Adds the R2 value to the dictionary of the first distribution compared with
    the other distributions.
    """
    fig, ax = plt.subplots(len(list_distributions), 1, figsize=(8, 6))
    pal = sns.cubehelix_palette(4, rot=-0.25, light=0.7)
    binwidth = 5
    max_data = int(max(list_distributions[0]["data"]))
    for index, distribution_dict in enumerate(list_distributions):
        if index == 0:
            label = distribution_dict["label"]
        else:
            label = (
                distribution_dict["label"]
                + "\n"
                + f"R2 = {distribution_dict['r2']:.2f}"
            )
        ax[index].hist(
            distribution_dict["data"],
            bins=range(0, max_data + binwidth, binwidth),
            density=False,
            label=label,
            alpha=1,
            color=pal[index],
            edgecolor="white",
            # range=(0, percentile),
        )
        if index < len(list_distributions) - 1:
            ax[index].spines["bottom"].set_visible(False)
            ax[index].set_xticks([])
        ax[index].spines["left"].set_visible(True)
        ax[index] = make_nice_spines(ax[index])
        ax[index].set_yscale("log")
        ax[index].set_ylim(5e-1, 10e2)
        ax[index].set_xlim(-10e-1, 10e1)

        ax[index].set_yticks([10e-1, 10e0, 10e1])

        ax[index].axhline(0, color="k", clip_on=False)
        ax[index].legend(loc="center right", facecolor="white", framealpha=1)

    # fig.subplots_adjust(hspace=0.1)
    return ax


def plot_histograms(list_distributions, nb_bins=20):
    """
    Plot the distribution of the number of outgoing connections for the
    biological data and the fitted graphs.

    Parameters
    ----------
    list_distributions : list[dict]
        List of dictionaries containing the distributions to plot.
    nb_bins : int, optional
        Number of bins for the histogram. The default is 25.

    Returns
    -------
    None.
    Adds the R2 value to the dictionary of the first distribution compared with
    the other distributions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # get the 90% percentile of the data list_distributions[0]["data"]
    percentile = np.percentile(list_distributions[0]["data"], 95)
    for distribution_dict in list_distributions:
        ax.hist(
            distribution_dict["data"],
            bins=nb_bins,
            density=True,
            label=distribution_dict["label"],
            alpha=0.5,
            color=distribution_dict["color"],
            range=(0, percentile),
        )
    ax.set_xlabel("Number of outgoing connections")
    ax.set_ylabel("Probability density")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_curves(list_distributions):
    """
    Plot the curves of the number of outgoing connections for the
    biological data and the rfitted graphs.

    Parameters
    ----------
    list_distributions : list[dict]
        List of dictionaries containing the distributions to plot.
    nb_bins : int, optional
        Number of bins for the histogram. The default is 25.

    Returns
    -------
    None.
    Adds the R2 value to the dictionary of the first distribution compared with
    the other distributions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for distribution_dict in list_distributions:
        ax.semilogy(
            distribution_dict["data"],
            label=distribution_dict["label"],
            alpha=0.5,
            color=distribution_dict["color"],
        )
    ax.set_ylabel("Number of outgoing connections")
    ax.set_ylabel("DNs")
    ax.legend()
    fig.tight_layout()
    plt.show()

    # TODO: add R2 value to the dictionary of the first distribution compared


def compare_distributions():
    """
    Compare the distribution of the number of outgoing connections for the
    biological data and models for graphs.

    Parameters
    ----------
    working_folder : str
        Path to the working folder.
    unn_matrix : scipy.sparse.csr_matrix
        Unnormalized adjacency matrix of the graph.
    equiv_index_rootid : pd.DataFrame
        Dictionary mapping the index of the unn_matrix to the root_id of the
        corresponding neuron.

    Returns
    -------
    None.
    """
    working_folder = plot_params.STATS_ARGS["folder"]
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    # load data
    (
        _,
        syncount_matrix,
        _,
        _,
        equiv_index_rootid,
    ) = load_graph_and_matrices("dn")

    dn_graph = nx.from_scipy_sparse_matrix(
        syncount_matrix, create_using=nx.DiGraph
    )

    dns_info = prepare_table(equiv_index_rootid, marker="DNg")
    dns_info = connectivity_stats(syncount_matrix, dns_info)

    list_distributions = []

    # Biological data
    down_stats = dns_info["DN_connected_out"]
    list_distributions.append(
        {
            "data": down_stats,
            "label": "Biological data",
            "color": "tab:blue",
        }
    )

    # Random graph
    random_distribution = get_random_graph_degree_seq(dn_graph)
    list_distributions.append(
        {
            "data": random_distribution,
            "label": "Random graph",
            "color": "tab:orange",
            "r2": r2_score(down_stats, random_distribution),
        }
    )

    # Exponential fit
    exponential_distribution = get_fit(
        down_stats,
        "exponential",
        range_fit=None,
        initial_guess=(100, 1),
    )
    list_distributions.append(
        {
            "data": exponential_distribution[0],
            "label": "Exponential fit",
            "color": "tab:green",
            "r2": exponential_distribution[1],
        }
    )

    # Power law fit
    powerlaw_distribution = get_fit(
        down_stats, "power_law", range_fit=None, initial_guess=(100, 1)
    )
    list_distributions.append(
        {
            "data": powerlaw_distribution[0],
            "label": "Power law fit",
            "color": "tab:red",
            "r2": powerlaw_distribution[1],
        }
    )

    # Plot
    _ = plot_ridges(list_distributions)
    plt.savefig(
        os.path.join(working_folder, "dn_outgoing_connections_fitted.pdf")
    )
    return


if __name__ == "__main__":
    compare_distributions()
