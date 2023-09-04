"""
2023.08.30
author: femke.hurtak@epfl.ch
Script to overlay the neurotransmitter type on the connectivity matrix.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plot_params

from loaddata import (
    load_graph_and_matrices,
    load_nodes_and_edges,
)
from common import (
    select_subset_matrix,
    plot_matrix_simple,
    convert_index_root_id,
)


def def_matrix_neurotransmitter(edges, rid_order, nt_type="GLUT"):
    """
    Create matrix with only a neurotransmitter type
    """
    relevant_edges = edges[
        (edges["nt_type"] == nt_type)
        & (edges["pre_root_id"].isin(rid_order))
        & (edges["post_root_id"].isin(rid_order))
    ]
    nt_matrix = np.zeros((len(rid_order), len(rid_order)))
    # iterate over the rows of the dataframe 'edges'
    for _, row in relevant_edges.iterrows():
        # get the index of the source and target neurons
        source = rid_order.index(row["pre_root_id"])
        target = rid_order.index(row["post_root_id"])
        # add the value of the edge to the matrix
        nt_matrix[source, target] += row["syn_count"]
    return nt_matrix


def draw_nt_specific_matrices(
    edges_, matrix_, equiv_index_rootid, order_, working_folder: str = None
):
    """
    Draw the matrix with only a neurotransmitter type
    """

    ordered_matrix = select_subset_matrix(matrix_, order_)
    rid_order = convert_index_root_id(equiv_index_rootid, order_)

    # plot
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    for i in range(3):
        ax[i] = plot_matrix_simple(
            ordered_matrix, vmax=10, ax=ax[i], cmap="Greys"
        )

    # create matrix with only a neurotransmitter type

    nt_matrix = def_matrix_neurotransmitter(edges_, rid_order, nt_type="GLUT")
    ax[0].imshow(
        nt_matrix,
        cmap="Purples",
        alpha=0.5,
        vmax=1,
        vmin=0,
    )
    ax[0].set_title(
        "Glutamatergic", color=plot_params.NT_TYPES["GLUT"]["color"]
    )

    nt_matrix = def_matrix_neurotransmitter(edges_, rid_order, nt_type="GABA")
    ax[1].imshow(
        nt_matrix,
        cmap="Blues",
        alpha=0.5,
        vmax=1,
        vmin=0,
    )
    ax[1].set_title("GABA", color=plot_params.NT_TYPES["GABA"]["color"])

    nt_matrix = def_matrix_neurotransmitter(edges_, rid_order, nt_type="ACH")
    ax[2].imshow(
        nt_matrix,
        cmap="Reds",
        alpha=0.5,
        vmax=1,
        vmin=0,
    )
    ax[2].set_title(
        "Acetylcholine", color=plot_params.NT_TYPES["ACH"]["color"]
    )
    if working_folder is not None:
        plt.savefig(os.path.join(working_folder, "nt_matrix_glut.pdf"))
    return


def get_fraction_neuropil(edges_, neuropil="GNG", subset: list = None):
    """
    Compute the fraction of connections in a given neuropil
    """
    if subset is not None:
        relevant_edges = edges_[
            (edges_["pre_root_id"].isin(subset))
            & (edges_["post_root_id"].isin(subset))
            & (edges_["nt_type"].isin(["GLUT", "GABA", "ACH"]))
        ]
    else:
        relevant_edges = edges_
    # get the total number of connections
    total = relevant_edges["syn_count"].sum()
    # get the number of connections in the neuropil
    neuropil_edges = relevant_edges[(relevant_edges["neuropil"] == neuropil)]
    neuropil_count = neuropil_edges["syn_count"].sum()
    # compute the fraction
    fraction = neuropil_count / total

    if not os.path.exists(plot_params.STATS_ARGS["folder"]):
        os.makedirs(plot_params.STATS_ARGS["folder"])
    filename = os.path.join(
        plot_params.STATS_ARGS["folder"], "fraction_neuropil.txt"
    )
    with open(filename, "w") as f:
        f.write(
            "Fraction of connections in {}: {} / {} = {}".format(
                neuropil, neuropil_count, total, fraction
            )
        )
    f.close()
    return fraction


def detail_neurotransmitters_matrix():
    working_folder = plot_params.CLUSTERING_ARGS["folder"]
    data_folder = plot_params.CLUSTERING_ARGS["data_folder"]

    ## --- load data --- ##
    clustering = pd.read_csv(
        os.path.join(data_folder, "clustering.csv"), index_col=0
    )
    order = clustering["node_index"]
    (
        _,
        unn_matrix,
        _,
        equiv_index_rootid,
    ) = load_graph_and_matrices("dn")
    _, edges = load_nodes_and_edges()

    ## --- Draw the matrix with only a neurotransmitter type --- ##
    draw_nt_specific_matrices(
        edges, unn_matrix, equiv_index_rootid, order, working_folder
    )

    ## --- Compute the fraction of connections in a given neuropil --- ##
    list_dns = convert_index_root_id(equiv_index_rootid, order)
    get_fraction_neuropil(edges, neuropil="GNG", subset=list_dns)


if __name__ == "__main__":
    detail_neurotransmitters_matrix()
