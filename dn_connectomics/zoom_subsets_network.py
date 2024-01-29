"""
2023.08.30
author: femke.hurtak@epfl.ch
Script for drawing subsets of the network, zooming in on clusters.
"""

import os
import matplotlib.pyplot as plt

import neuron_params
import plot_params

from loaddata import load_graph_and_matrices
from draw_meta_network import load_communities
from common import (
    select_subset_matrix,
    plot_matrix_simple,
    convert_index_root_id,
    identify_name_from_rid,
)
from graph_plot_utils import make_nice_spines


def replace_ticks_with_names(
    ax, equiv_index_rootid, subset_indices, labeled_subset: dict = None
):
    """
    Identify the names of the neurons in the subset_indices and replace the
    ticks with the names.
    """
    rids = convert_index_root_id(equiv_index_rootid, subset_indices)
    names = [identify_name_from_rid(rid, labeled_subset) for rid in rids]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    return ax


def draw_matrix_plots_subsets(unn_matrix, subset_indices, equiv_index_rootid):
    """
    Draw the matrix of the subset of the network.
    """
    matrix_subset = select_subset_matrix(unn_matrix, subset_indices)
    ax = plot_matrix_simple(matrix_subset)  # , savefig=savedir)
    make_nice_spines(ax)
    replace_ticks_with_names(
        ax, equiv_index_rootid, subset_indices, neuron_params.KNOWN_DNS
    )
    plt.tight_layout()
    return ax


def make_matrix_plots_subsets(
    unn_matrix, communities, equiv_index_rootid, folder
):
    """
    Make the matrix plots for the subsets of the communities.
    """
    os.makedirs(folder, exist_ok=True)

    for i, subset_indices in enumerate(communities):
        savedir = os.path.join(folder, f"cluster_{i+1}.pdf")
        ax = draw_matrix_plots_subsets(
            unn_matrix, subset_indices, equiv_index_rootid
        )
        ax.set_title(f"Cluster {i+1}")
        plt.savefig(savedir, dpi=300)


def draw_subsets_network():
    working_folder = plot_params.CLUSTERING_ARGS["folder"]
    data_folder = plot_params.CLUSTERING_ARGS["data_folder"]
    (
        _,
        _,
        unn_matrix,
        _,
        equiv_index_rootid,
    ) = load_graph_and_matrices("dn")

    communities = load_communities(data_folder, return_type="list")
    communities = [
        c
        for c in communities
        if len(c) > plot_params.CLUSTERING_ARGS["confusion_mat_size_threshold"]
    ]

    plots_folder = os.path.join(
        working_folder,
        "cluster_zoom",
    )
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    make_matrix_plots_subsets(
        unn_matrix, communities, equiv_index_rootid, plots_folder
    )


if __name__ == "__main__":
    draw_subsets_network()
