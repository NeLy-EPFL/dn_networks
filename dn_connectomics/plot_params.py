"""
2023.08.30
author: femke.hurtak@epfl.ch
File containing parameters for the plots.
"""

import os
import params

# --- Global plotting parameters --- #

DARKRED = "#9c0d0b"
DARKGREEN = "#0e6117"
DARKCYAN = "#0bbfb3"
DARKBLUE = "#0f0b87"
DARKORANGE = "#d17c04"
DARKPURPLE = "#730d71"
LIGHTGREY = "#c3c3c3"
DARKPINK = "#C20C6D"

INHIB_COLOR = "navy"
EXCIT_COLOR = "darkred"

NT_TYPES = {
    "GABA": {"color": INHIB_COLOR, "linestyle": "-"},  # ":"
    "ACH": {"color": EXCIT_COLOR, "linestyle": "-"},  # "--"
    "GLUT": {"color": DARKPINK, "linestyle": "-"},
    "SER": {"color": LIGHTGREY, "linestyle": "-"},
    "DA": {"color": LIGHTGREY, "linestyle": "-"},
    "OCT": {"color": LIGHTGREY, "linestyle": "-"},
}

# --- Plotting parameters for the flower plots --- #
FLOWER_PLOT_PARAMS = {}
FLOWER_PLOT_PARAMS["folder"] = os.path.join(
    params.FIGURES_DIR,
    "network_visualisations",
    "flower_plots",
)
FLOWER_PLOT_PARAMS["direct_layout_args"] = {
    "level": 1,
    "arrow_norm": 0.1,
    "other_layer_visible": False,
    "fig_size": 5,
    "include_legend": True,
    "feedback_direct_plot": False,
    "feedback_downstream_plot": False,
}
FLOWER_PLOT_PARAMS["indirect_layout_args"] = {
    "level": 2,
    "arrow_norm": 0.1,
    "other_layer_visible": False,
    "fig_size": 5,
    "include_legend": True,
    "feedback_direct_plot": False,
    "feedback_downstream_plot": True,
}
FLOWER_PLOT_PARAMS["plot_each_neuron"] = False

# --- Plotting parameters for the clustering --- #
CLUSTERING_ARGS = {}
CLUSTERING_ARGS["positive_connections"] = "excitatory"
CLUSTERING_ARGS["time_limit"] = 3  # in seconds
CLUSTERING_ARGS["iterations"] = 100
CLUSTERING_ARGS["cutoff"] = 0.25  # cluster thresholding
CLUSTERING_ARGS["folder"] = os.path.join(
    params.FIGURES_DIR,
    "network_visualisations",
    "whole_network",
    "louvain",
)
CLUSTERING_ARGS["data_folder"] = os.path.join(
    CLUSTERING_ARGS["folder"],
    "data",
)
CLUSTERING_ARGS["control_folder"] = os.path.join(
    CLUSTERING_ARGS["folder"],
    "shuffled_control",
)
CLUSTERING_ARGS["confusion_mat_values"] = "relative"
CLUSTERING_ARGS["confusion_mat_size_threshold"] = 10
CLUSTERING_ARGS["confusion_mat_count_synpases"] = True
CLUSTERING_ARGS["confusion_mat_normalise"] = True

META_GRAPH = {}
META_GRAPH["edge_normalisation"] = 5
META_GRAPH["scale_nodes"] = 10

BRAIN_SYNAPSES_DISTRIBUTION_ARGS = {}
BRAIN_SYNAPSES_DISTRIBUTION_ARGS["cmap"] = "Purples"
BRAIN_SYNAPSES_DISTRIBUTION_ARGS["normalise"] = True
BRAIN_SYNAPSES_DISTRIBUTION_ARGS["logscale"] = True

# --- Plotting parameters for the statistics --- #
STATS_ARGS = {}
STATS_ARGS["folder"] = os.path.join(
    params.FIGURES_DIR,
    "statistics",
)
STATS_ARGS["measured_feature"] = "DN_connected_out"
STATS_ARGS["measured_feature_label"] = "Number of connected DNs"

# --- Network-wide stats --- #
NETWORK_STATS_ARGS = {}
NETWORK_STATS_ARGS["folder"] = os.path.join(
    params.FIGURES_DIR,
    "network_visualisations",
)
NETWORK_STATS_ARGS["layers_folder"] = os.path.join(
    NETWORK_STATS_ARGS["folder"],
    "layers",
)
NETWORK_STATS_ARGS["method"] = "bootstrap"
