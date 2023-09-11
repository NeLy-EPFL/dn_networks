import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from loaddata import load_nodes_and_edges
import plot_params
from draw_meta_network import load_communities
from graph_plot_utils import make_nice_spines


def get_synapses_in_brain_neuropil(edges, communities):
    """
    Get the number of synapses in each neuropil for each cluster.

    Parameters
    ----------
    edges : pd.DataFrame
        The dataframe containing the edges.
    communities : list[list[int]]
        The list of communities.

    Returns
    -------
    data_df : pd.DataFrame
        The dataframe containing the number of synapses in each neuropil for each cluster.
    """

    data_df = pd.DataFrame()

    for ci, community in enumerate(communities):
        community_data = {"n_neurons": len(community), "n_synapses": 0}
        relevant_edges = edges[edges["post_root_id"].isin(community)]
        # merge dataframe on 'region' column, sum 'syn_count' column
        relevant_edges = relevant_edges.groupby("neuropil").sum(
            numeric_only=True
        )
        relevant_edges = relevant_edges.reset_index()
        community_data["n_synapses"] = relevant_edges["syn_count"].sum(
            numeric_only=True
        )
        for specific_region in relevant_edges["neuropil"].unique():
            community_data[specific_region] = relevant_edges[
                relevant_edges["neuropil"] == specific_region
            ]["syn_count"].values[0]
        community_df = pd.DataFrame.from_dict(
            community_data, orient="index"
        ).transpose()
        community_df["cluster"] = ci
        data_df = pd.concat([data_df, community_df], ignore_index=True)

    data_df = data_df.fillna(0)

    normalise = plot_params.BRAIN_SYNAPSES_DISTRIBUTION_ARGS["normalise"]
    if normalise:
        # divide all columns except 'n_neurons', 'cluster' and 'n_synapses'
        # by n_synapses
        data_df[
            data_df.columns.difference(["n_neurons", "cluster", "n_synapses"])
        ] = data_df[
            data_df.columns.difference(["n_neurons", "cluster", "n_synapses"])
        ].div(
            data_df["n_synapses"], axis=0
        )

    return data_df


def plot_brain_synapses_distribution():
    """
    Plot the distribution of synapses in the brain neuropils for each cluster.
    """

    ## --- load data
    _, edges = load_nodes_and_edges()

    working_folder = os.path.join(
        plot_params.CLUSTERING_ARGS["folder"], "data"
    )
    size_threshold = plot_params.CLUSTERING_ARGS[
        "confusion_mat_size_threshold"
    ]
    communities = load_communities(
        working_folder, threshold=size_threshold, data_type="root_id"
    )

    ## --- get the number of synapses in each neuropil for each cluster
    data_df = get_synapses_in_brain_neuropil(edges, communities)

    ## --- plot the data using imshow
    data_columns = data_df.columns.difference(
        ["n_neurons", "cluster", "n_synapses"]
    )
    # y axis: cluster
    # x axis: neuropil name
    # color: fraction of synapses in neuropil
    fig, ax = plt.subplots(figsize=(30, 5))
    logscale = plot_params.BRAIN_SYNAPSES_DISTRIBUTION_ARGS["logscale"]
    if logscale:
        plotted_data = np.log10(data_df[data_columns].values)
    else:
        plotted_data = data_df[data_columns].values
    im = ax.imshow(
        plotted_data,
        cmap=plot_params.BRAIN_SYNAPSES_DISTRIBUTION_ARGS["cmap"],
        interpolation=None,
        aspect="auto",
        # vmax=0.5,
        # vmin=0,
    )
    make_nice_spines(ax)
    ax.set_xticks(range(len(data_columns)))
    ax.set_xticklabels(data_columns, rotation=90)
    ax.set_yticks(range(len(communities)))
    ax.set_yticklabels(range(1, len(communities) + 1))
    ax.set_xlabel("neuropil")
    ax.set_ylabel("cluster")
    ax.set_title("Fraction of synapses in neuropil")
    cbar = fig.colorbar(im, ax=ax, shrink=0.73, ticks=[-1, -3, -5])
    make_nice_spines(cbar.ax)
    cbar.ax.set_yticklabels(["1e-1", "1e-3", "1e-5"])
    cbar.outline.set_edgecolor("#ffffff")
    cbar.outline.set_edgecolor("#ffffff")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plot_params.CLUSTERING_ARGS["data_folder"],
            "synapses_in_brain_neuropil.pdf",
        ),
        dpi=300,
    )


if __name__ == "__main__":
    plot_brain_synapses_distribution()
