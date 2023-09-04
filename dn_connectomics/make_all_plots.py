"""
2023.08.30
author: femke.hurtak@epfl.ch
Script to make all the plots for the paper.
"""

from flower_plots import draw_all_flower_plots
from connectivity_stats import compute_connectivity_stats
from distribution_fit import compare_distributions
from louvain_clustering import draw_louvain_custering
from louvain_communities_control import compare_clustering_with_random
from draw_meta_network import draw_meta_network
from zoom_subsets_network import draw_subsets_network
from detail_nt_matrix import detail_neurotransmitters_matrix
from connectivity_vs_recurrence import connectivity_vs_recurrence_plots
from layer_reaching_plot import cumulative_diffusion_plot

import params


if __name__ == "__main__":
    figures_path = params.FIGURES_DIR
    if not figures_path.exists():
        figures_path.mkdir(parents=True)

    # --- subnetwork visualisations --- #
    print("=== Drawing flower plots")
    draw_all_flower_plots()

    # --- connectivity statistics --- #
    print("=== Computing connectivity statistics")
    compute_connectivity_stats()
    print("=== Comparing distributions with graph models")
    compare_distributions()
    print("=== Computing cumulative diffusion")
    cumulative_diffusion_plot()

    # --- clustering visualisations --- #
    print("=== Drawing Louvain clustering")
    draw_louvain_custering()
    print("=== Comparing Louvain clustering with random")
    compare_clustering_with_random()
    print("=== Drawing meta-network")
    draw_meta_network()
    print("=== Drawing neurotransmitters matrix")
    detail_neurotransmitters_matrix()
