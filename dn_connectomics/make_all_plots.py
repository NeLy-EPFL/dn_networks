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


if __name__ == "__main__":
    # --- subnetwork visualisations --- #
    draw_all_flower_plots()

    # --- connectivity statistics --- #
    compute_connectivity_stats()
    compare_distributions()
    connectivity_vs_recurrence_plots()
    cumulative_diffusion_plot()

    # --- clustering visualisations --- #
    draw_louvain_custering()
    compare_clustering_with_random()
    draw_meta_network()
    draw_subsets_network()
    detail_neurotransmitters_matrix()
