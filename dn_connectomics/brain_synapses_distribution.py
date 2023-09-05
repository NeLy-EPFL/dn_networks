import pandas as pd
import os

from loaddata import load_nodes_and_edges
import plot_params
from draw_meta_network import load_communities

if __name__ == "__main__":
    nodes, edges = load_nodes_and_edges()

    working_folder = os.path.join(
        plot_params.CLUSTERING_ARGS["folder"], "data"
    )
    size_threshold = plot_params.CLUSTERING_ARGS[
        "confusion_mat_size_threshold"
    ]
    communities = load_communities(working_folder, threshold=size_threshold)

    print(communities)
