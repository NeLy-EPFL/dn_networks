"""
2023.08.30
author: femke.hurtak@epfl.ch
Script to make all the plots for the paper.
"""
import os

from flower_plots import draw_all_flower_plots
from connectivity_stats import compute_connectivity_stats, compare_connectivity_stats
from distribution_fit import compare_distributions
from louvain_clustering import draw_louvain_custering
from louvain_communities_control import compare_clustering_with_random
from draw_meta_network import draw_meta_network, draw_network_organised_by_clusters
from zoom_subsets_network import draw_subsets_network
from detail_nt_matrix import detail_neurotransmitters_matrix
from connectivity_vs_recurrence import connectivity_vs_recurrence_plots
from layer_reaching_plot import cumulative_diffusion_plot
from brain_synapses_distribution import plot_brain_synapses_distribution
from cluster_behavior_analysis import make_cluster_analysis_plots
import params
import plot_params


if __name__ == "__main__":
    figures_path = params.FIGURES_DIR
    os.makedirs(figures_path, exist_ok=True)

    # --- subnetwork visualisations --- #
    print("=== Drawing flower plots, fig. 3 c-d, fig. 5 e, ED fig. 6, 7, 8 & 9")
    draw_all_flower_plots()

    # --- connectivity statistics --- #
    print("=== Computing connectivity statistics, fig 5 a-c")
    compute_connectivity_stats()
    print("=== Comparing distributions with graph models, fig 6 a")
    compare_distributions()
    print("=== Computing cumulative diffusion, fig. 3 e")
    cumulative_diffusion_plot()
    print("== Comparing distributions with different underlying graph models, ED fig. 5")
    compare_connectivity_stats(
        direction = plot_params.INTERNEURON_STATS_ARGS["measured_feature"],
        connection = plot_params.INTERNEURON_STATS_ARGS["connection_type"],
        reference_sorting = plot_params.INTERNEURON_STATS_ARGS["reference_sorting"],
    )

    # --- clustering visualisations --- #
    print("=== Drawing Louvain clustering, fig 6")
    draw_louvain_custering()
    print("=== Comparing Louvain clustering with random")
    compare_clustering_with_random()
    print("=== Drawing meta-network, fig 6 i")
    draw_meta_network()
    print("=== Drawing neurotransmitters matrix")
    detail_neurotransmitters_matrix()
    print("=== Showing where clusters get their input from, fig 6 h")
    plot_brain_synapses_distribution()
    print("=== Showing where clusters output to in the VNC, which behaviors & DNs are represented, fig 6 f,g,j")
    make_cluster_analysis_plots()
    print("=== Detail of inter-cluster connectivity at neuron-level resolution, ED fig. 10")
    draw_network_organised_by_clusters(
        restricted_nodes = plot_params.NETWORK_PLOT_ARGS["restricted_nodes"],
        restricted_clusters = plot_params.NETWORK_PLOT_ARGS["restricted_clusters"],
        restricted_connections = plot_params.NETWORK_PLOT_ARGS["restricted_connections"],
        position_reference = plot_params.NETWORK_PLOT_ARGS["position_reference"],
    )
