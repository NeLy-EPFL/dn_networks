import os
import matplotlib.pyplot as plt

import params
import neuron_params

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


def make_matrix_plots_subsets(
    unn_matrix, communities, equiv_index_rootid, folder
):
    """
    Make the matrix plots for the subsets of the communities.
    """
    os.makedirs(folder, exist_ok=True)

    for i, subset_indices in enumerate(communities):
        savedir = os.path.join(folder, f"cluster_{i+1}.pdf")

        matrix_subset = select_subset_matrix(unn_matrix, subset_indices)
        ax = plot_matrix_simple(matrix_subset)  # , savefig=savedir)
        make_nice_spines(ax)
        ax.set_title(f"Cluster {i+1}")
        replace_ticks_with_names(
            ax, equiv_index_rootid, subset_indices, neuron_params.KNOWN_DNS
        )
        plt.tight_layout()
        plt.savefig(savedir, dpi=300)


if __name__ == "__main__":
    working_folder = os.path.join(
        params.FIGURES_DIR,
        "network_visualisations",
        "whole_network",
        "louvain",
    )

    (
        graph,
        unn_matrix,
        _,
        equiv_index_rootid,
    ) = load_graph_and_matrices("dn")

    communities = load_communities(working_folder, return_type="list")
    communities = [c for c in communities if len(c) > 10]

    plots_folder = os.path.join(
        working_folder,
        "cluster_zoom",
    )

    make_matrix_plots_subsets(
        unn_matrix, communities, equiv_index_rootid, plots_folder
    )
