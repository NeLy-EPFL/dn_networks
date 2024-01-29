"""
2023.08.30
author: femke.hurtak@epfl.ch
Script to compute how for a signal can propagate in the network starting from
a specific neuron.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from scipy.stats import bootstrap

import neuron_params
import plot_params
from graph_plot_utils import (
    plot_graph_diffusion,
    make_nice_spines,
    define_graph_attributes,
    add_layer_reached_by_neuron,
)
from loaddata import load_nodes_and_edges, load_names


def draw_diffusion_from_ref_neurons(folder: str, df: pd.DataFrame):
    """
    Plot the diffusion from the reference neurons. It corresponds to the number
    of neurons reached at a certain number of hops.
    """
    summary_ref_neurons = {}
    for _, v in neuron_params.REF_DNS.items():
        if v["name"] not in summary_ref_neurons.keys():
            summary_ref_neurons[v["name"]] = {"rids": [], "color": v["color"]}
        summary_ref_neurons[v["name"]]["rids"].append(v["root_id"])

    for k, v in summary_ref_neurons.items():
        _, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax = plot_graph_diffusion(
            df,
            start_neurons=v["rids"],
            named_dns={},
            drawing_style="density",
            ax=ax,
            neuron_color=v["color"],
        )
        ax.legend(loc="lower right", bbox_to_anchor=(1, 0))
        make_nice_spines(ax)
        ax.set_title(k)
        ax.set_xlim(0, 300)

        plt.savefig(
            os.path.join(folder, f"diffusion_{k}.png"),
            bbox_inches="tight",
        )


def compute_cumulative_diffusion(df: pd.DataFrame):
    """
    Plot the cumulative diffusion from the reference neurons. It corresponds to the number
    of neurons reached at a certain number of hops.
    """
    graph, _ = define_graph_attributes(
        df,
        neuron_params.KNOWN_DNS,
    )

    layer_cumulated = {}
    for start_node in tqdm(graph.nodes()):
        layer_df = pd.DataFrame()
        layer_df, _ = add_layer_reached_by_neuron(graph, start_node, layer_df)
        layer_cumulated[start_node] = {
            "root_id": graph.nodes[start_node]["root_id"],
            "per_layer": layer_df["layer_count"].values,
            "cumulated": layer_df["layer_count"].values.cumsum(),
        }
    return layer_cumulated


def plot_cumulative_diffusion(
    layers,
    working_folder,
    all_lines: bool = False,
    method: str = "mean",
    range_stats: float = 0.25,
):
    """
    Plot the cumulative diffusion from the reference neurons. It corresponds to the number
    of neurons reached at a certain number of hops.

    Parameters
    ----------
    layers: dict
        Dictionary of layers: keys are root_ids, values is a dictionary including
        the number of neurons reached saved in a list.
    working_folder: str
        Folder where to save the figure
    all_lines: bool
        If True, plot all the lines. If False, plot the mean and envelope.

    Returns
    -------
    None
        saves the figure in the working_folder
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    nb_layers = max(
        [
            len(layers[list(layers.keys())[i]]["cumulated"])
            for i in range(len(layers))
        ]
    )
    if all_lines:
        for k, v in layers.items():
            if plot_params.NETWORK_STATS_ARGS["extend_lines"]:
                if len(v["cumulated"]) == 0:
                    v["cumulated"] = np.zeros(nb_layers)
                else:
                    while len(v["cumulated"]) < nb_layers:
                        v["cumulated"] = np.append(
                            v["cumulated"], v["cumulated"][-1]
                        )
                plt.plot(
                    v["cumulated"], linewidth=0.5, color="grey", alpha=0.1
                )

    if (
        not all_lines or plot_params.NETWORK_STATS_ARGS["overlay_method"]
    ):  # compute the mean and enveloppe
        array = []
        for k, v in layers.items():
            array.append(v["cumulated"])
        df = pd.DataFrame(array).T
        df = df.fillna(method="ffill")
        df = df.fillna(0)
        if method == "median":
            median = df.median(axis=1)
            ax.plot(median, linewidth=2, color="black", label="median")
            if not all_lines:
                q1 = df.quantile(q=range_stats, axis=1)
                q2 = df.quantile(q=1 - range_stats, axis=1)
                ax.fill_between(
                    df.index,
                    q1,
                    q2,
                    color="grey",
                    alpha=0.2,
                    label=f"{int(range_stats*10)}-{int(100-range_stats*10)} percentile",
                )
        elif method == "mean":
            mean = df.mean(axis=1)
            ax.plot(mean, linewidth=2, color="black", label="mean")
            if not all_lines:
                std = df.std(axis=1)
                ax.fill_between(
                    df.index,
                    mean - std,
                    mean + std,
                    color="grey",
                    alpha=0.2,
                    label="std",
                )
        elif method == "bootstrap":
            mean = df.mean(axis=1)
            all_r = []
            for i_d in range(df.values.shape[0]):
                r = bootstrap((df.values[i_d, :],), statistic=np.mean)
                all_r.append(r)
            lower_bound = np.array([r.confidence_interval.low for r in all_r])
            upper_bound = np.array([r.confidence_interval.high for r in all_r])
            ax.plot(mean, linewidth=2, color="black", label="mean")
            ax.fill_between(
                df.index,
                lower_bound,
                upper_bound,
                color="grey",
                alpha=0.2,
                label="bootstrap of the mean",
            )

    for k, v in layers.items():
        if v["root_id"] in neuron_params.REF_DNS.keys():
            ax.plot(
                v["cumulated"],
                linewidth=2,
                color=neuron_params.REF_DNS[v["root_id"]]["color"],
                # label=neurons.REF_DNS[v["root_id"]]["name"],
            )

    ax.set_xlabel("Number of hops")
    ax.set_xticks(np.arange(0, nb_layers, 1))
    ax.set_ylabel("Number of neurons reached")
    make_nice_spines(ax)
    plt.legend()
    plt.savefig(os.path.join(working_folder, "cumulative_diffusion.png"))
    plt.savefig(os.path.join(working_folder, "cumulative_diffusion.pdf"))


def cumulative_diffusion_plot():
    _, edges = load_nodes_and_edges()
    list_dns = load_names()["root_id"].values

    working_folder = plot_params.NETWORK_STATS_ARGS["layers_folder"]
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    connection_df = edges[
        (edges["syn_count"] > 5)
        & (edges["pre_root_id"].isin(list_dns))
        & (edges["post_root_id"].isin(list_dns))
    ]

    layers = compute_cumulative_diffusion(connection_df)
    plot_cumulative_diffusion(
        layers,
        working_folder,
        method=plot_params.NETWORK_STATS_ARGS["method"],
        all_lines=plot_params.NETWORK_STATS_ARGS["all_lines"],
        range_stats=0.25,
    )


if __name__ == "__main__":
    cumulative_diffusion_plot()
