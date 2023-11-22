"""
2023.08.30
author: femke.hurtak@epfl.ch
Script to draw the neurons directly downstream of a given neuron.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


import plot_params
import neuron_params
from loaddata import load_nodes_and_edges, get_name_from_rootid
from common import identify_rids_from_name
from graph_plot_utils import get_downstream_specs, plot_downstream_network


def make_flower_plot(
    neuron_name: str,
    edges: pd.DataFrame = None,
    neurons_of_interest: dict = None,
    layout_args: dict = None,
    plot_all_neurons: bool = True,
    display_names: bool = False,
):
    """
    Make a flower plot for a given neuron, i.e. the neuron of interest in the
    centre, and all neurons directly connected around it.

    Parameters
    ----------
    neuron : str
        Name of the neuron to plot.
    edges : pd.DataFrame
        Dataframe containing the edges of the network.
    neurons_of_interest : Dict
        Dictionary containing the neurons of interest. The keys are the rids,
        the values is a dict with 'color' and 'name' as keys.
    layout_args: dict
        Dictionary of design-related options
        Includes:
            - level : int
        Number of hops to include in the plot. Default is 1, i.e. only the
        connections from the neuron of interest. If level is 2, then the
        connections to the neuron of interest and the connections of those
        neurons are included.
            - arrow_norm : float
        Normalization factor for the arrow width. Default is 1.0.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    # Format data necessary
    rids = identify_rids_from_name(neuron_name, neurons_of_interest)
    if rids is {}:
        raise ValueError(f"Neuron {neuron_name} not found in neurons of interest.")
    direct_plot = True if layout_args["level"] == 1 else False
    list_nodes = neurons_of_interest.keys()

    if plot_all_neurons:
        nbr_graphs = len(rids) + 1
        fig_size_factor = (
            layout_args["fig_size"] if "fig_size" in layout_args else 5
        )
        fig, ax = plt.subplots(
            1,
            nbr_graphs,
            figsize=(nbr_graphs * fig_size_factor, fig_size_factor),
        )
        for i, neuron in enumerate(rids):
            center = {neuron: rids[neuron]}
            network_1, network_2 = get_downstream_specs(
                edges,
                center,
                list_dns=list_nodes,
                feedback_layer_1=layout_args["feedback_direct_plot"],
                feedback_layer_2=layout_args["feedback_downstream_plot"],
                named_dns=neurons_of_interest,
            )
            _ = plot_downstream_network(
                network_1,
                network_2,
                neurons_of_interest=center,
                ax=ax[i],
                first_layer_opaque=direct_plot,
                other_layer_visible=layout_args["other_layer_visible"]
                if "other_layer_visible" in layout_args
                else False,
                arrow_norm=layout_args["arrow_norm"]
                if "arrow_norm" in layout_args
                else 0.1,
                display_names=display_names,
            )
            ax[i].set_title(neuron_name + " downstream DNs")
            ax[i].set_aspect("equal")

        # if ax is a list, then the last element is the overall plot
        if nbr_graphs > 1:
            ax_overall = ax[nbr_graphs - 1]
        else:
            ax_overall = ax

    else:
        fig_size_factor = (
            layout_args["fig_size"] if "fig_size" in layout_args else 5
        )
        fig, ax_overall = plt.subplots(
            1, 1, figsize=(fig_size_factor, fig_size_factor)
        )

    ax_overall.set_aspect("equal")
    network_1, network_2 = get_downstream_specs(
        edges,
        rids,
        list_dns=list_nodes,
        feedback_layer_1=layout_args["feedback_direct_plot"],
        feedback_layer_2=layout_args["feedback_downstream_plot"],
        named_dns=neurons_of_interest,
    )
    _ = plot_downstream_network(
        network_1,
        network_2,
        neurons_of_interest=rids,
        ax=ax_overall,
        first_layer_opaque=direct_plot,
        other_layer_visible=layout_args["other_layer_visible"]
        if "other_layer_visible" in layout_args
        else False,
        arrow_norm=layout_args["arrow_norm"]
        if "arrow_norm" in layout_args
        else 0.1,
        display_names=display_names,
    )
    ax_overall.set_title("overall downstream DNs")
    # plt.tight_layout()
    return fig


def draw_all_flower_plots():
    """
    Draw all flower plots for the neurons of interest. Create plots with the
    direct connections and plots with the downstream connections.
    """
    # Load the data
    nodes, edges = load_nodes_and_edges()
    dn_mask = ~nodes["name_taken"].isna() & nodes["name_taken"].str.contains(
        "DN"
    )
    nodes = nodes[dn_mask]
    edges = edges[
        (
            edges["pre_root_id"].isin(nodes["root_id"])
            & edges["post_root_id"].isin(nodes["root_id"])
            & (edges["syn_count"] >= 5)
        )
    ]
    # if there are pairs of ('pre_root_id', 'post_root_id') that have multiple
    # entries in the edge_info table, sum up their synapse counts and assign the
    #  'nt_type' and neuropil with the highest initial weight.
    edges = edges.groupby(["pre_root_id", "post_root_id"]).agg(
        {
            "neuropil": lambda x: x.value_counts().index[0],
            "syn_count": "sum",
            "nt_type": lambda x: x.value_counts().index[0],
        }
    )
    edges = edges.reset_index()

    working_folder = plot_params.FLOWER_PLOT_PARAMS["folder"]
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    dictionary_dns = {
        root_id: {
            "root_id": root_id,
            "color": plot_params.DARKORANGE
            if (not pd.isna(get_name_from_rootid(root_id)) and "DNg" in get_name_from_rootid(root_id))
            else plot_params.DARKPURPLE,
            "name": get_name_from_rootid(root_id),
        }
        for root_id in nodes["root_id"]
        if not pd.isna(get_name_from_rootid(root_id))
    }
    # Replace the neurons where the root_ids are already defined
    dictionary_dns.update(neuron_params.REF_DNS)
    neuron_names = neuron_params.FLOWER_PLOTS_TO_MAKE

    for neuron_name in tqdm(neuron_names):
        # ------------------- Flower plot for direct connections -------------------
        layout_args = plot_params.FLOWER_PLOT_PARAMS["direct_layout_args"]
        fig1 = make_flower_plot(
            neuron_name,
            edges=edges,
            neurons_of_interest=dictionary_dns,
            layout_args=layout_args,
            plot_all_neurons=plot_params.FLOWER_PLOT_PARAMS[
                "plot_each_neuron"
            ],
            display_names=plot_params.FLOWER_PLOT_PARAMS["display_names"],
        )
        fig1.savefig(
            os.path.join(
                working_folder, f"{neuron_name}_flower_plot_direct.pdf"
            )
        )
        plt.close(fig1)

        # ------------------- Flower plot for downstream order connections ---------
        layout_args = plot_params.FLOWER_PLOT_PARAMS["indirect_layout_args"]

        fig2 = make_flower_plot(
            neuron_name,
            edges=edges,
            neurons_of_interest=dictionary_dns,
            layout_args=layout_args,
            plot_all_neurons=plot_params.FLOWER_PLOT_PARAMS[
                "plot_each_neuron"
            ],
            display_names=plot_params.FLOWER_PLOT_PARAMS["display_names"],
        )
        fig2.savefig(
            os.path.join(
                working_folder, f"{neuron_name}_flower_plot_indirect.pdf"
            )
        )
        plt.close(fig2)
    return


if __name__ == "__main__":
    draw_all_flower_plots()
