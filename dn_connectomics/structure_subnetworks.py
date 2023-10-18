from scipy import sparse, linalg
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx

import plot_params
from common import select_subset_matrix
from loaddata import load_graph_and_matrices
from draw_meta_network import load_communities
from zoom_subsets_network import draw_matrix_plots_subsets
from common import convert_index_root_id
from graph_plot_utils import draw_graph_selfstanding, positions_by_degree


def svd_decomposition(matrix):
    """
    Returns: U, S, Vt, the SVD decomposition of the matrix
    """

    if sparse.issparse(matrix):
        matrix = matrix.todense()
    U, S, Vt = linalg.svd(matrix)
    return U, S, Vt


def embedding(matrix, method="tsne", n_components=2, perplexity=None):
    """
    Parameters
    ----------
    matrix : np.array of shape (n_samples, n_samples)
        The adjacency matrix of the graph
    method : str, optional
        The method to use for embedding, by default "tsne"
    n_components : int, optional
        The number of components of the embedding, by default 2
    perplexity : int, optional
        The perplexity for the tsne method, by default 100

    Returns
    -------
    embedding : np.array of shape (n_samples, n_components) where
    n_samples correpsonds to the U matrix of the SVD decomposition
    and n_components is the number of components of the embedding

    """
    U, _, _ = svd_decomposition(matrix)
    scaler = StandardScaler()
    data = scaler.fit_transform(U)
    if method == "tsne":
        if perplexity is None:
            perplexity = int(np.sqrt(data.shape[0]))
        return TSNE(
            n_components=n_components, perplexity=perplexity
        ).fit_transform(data)
    elif method == "umap":
        fit = umap.UMAP()
        return fit.fit_transform(data)
    elif method == "pca":
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)
    else:
        raise ValueError("Method not supported")


def visualize_embedding(embedding, highlighted_pts=None, title=None):
    """
    Visualize the embedding of the data in 2D

    Parameters
    ----------
    embedding : np.array of shape (n_samples, 2)
        The embedding of the data
    highlighted_pts : list, optional
        The indices of the points to highlight, by default None
    title : str, optional
        The title of the plot, by default None

    Returns
    -------
    None
    """
    plt.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5, c="grey")
    if highlighted_pts is not None:
        plt.scatter(
            embedding[highlighted_pts, 0],
            embedding[highlighted_pts, 1],
            s=20,
            c="k",
            alpha=0.5,
            edgecolors="white",
        )
    plt.title(title)
    plt.show()


def define_subgraph(
    matrix, indices, equiv_index_rootid, information_dict=None
):
    """
    Define the subgraph of the network corresponding to the indices
    """
    submatrix = select_subset_matrix(matrix, indices)
    subgraph = nx.from_scipy_sparse_array(submatrix, create_using=nx.DiGraph)

    # Add the information to the nodes
    for node_ind, index in enumerate(indices):
        subgraph.nodes[node_ind]["initial_index"] = index
        root_id = convert_index_root_id(equiv_index_rootid, index)
        subgraph.nodes[node_ind]["root_id"] = root_id
        # add any other field defined in information_dict[root_id] explicitely
        if information_dict is not None:
            for key, value in information_dict[root_id].items():
                subgraph.nodes[node_ind][key] = value
    return subgraph


if __name__ == "__main__":
    working_folder = os.path.join(
        plot_params.CLUSTERING_ARGS["folder"], "data"
    )
    (
        _,
        _,
        unn_matrix,
        _,
        equiv_index_rootid,
    ) = load_graph_and_matrices("dn")
    size_threshold = plot_params.CLUSTERING_ARGS[
        "confusion_mat_size_threshold"
    ]
    communities = load_communities(
        working_folder, return_type="list", threshold=size_threshold
    )
    root_ids = load_communities(
        working_folder,
        return_type="list",
        threshold=size_threshold,
        data_type="root_id",
    )

    subgraph = define_subgraph(unn_matrix, communities[2], equiv_index_rootid)
    edges_to_remove = []
    for edge in subgraph.edges:
        if np.abs(subgraph.edges[edge]["weight"]) < 10:
            edges_to_remove.append(edge)
    subgraph.remove_edges_from(edges_to_remove)
    for edge in subgraph.edges:
        subgraph.edges[edge]["abs_weight"] = np.abs(
            subgraph.edges[edge]["weight"]
        )
    # remove untouched nodes
    untouched_nodes = []
    for node in subgraph.nodes:
        if subgraph.degree(node) == 0:
            untouched_nodes.append(node)
    subgraph.remove_nodes_from(untouched_nodes)

    # pos = positions_by_degree(subgraph)
    pos = nx.kamada_kawai_layout(subgraph, weight="abs_weight")
    draw_graph_selfstanding(subgraph, edge_norm=50, pos=pos)
    plt.show()

    # embedding = embedding(matrix, method="pca")
    # visualize_embedding(embedding)
    # closeness_centrality = nx.closeness_centrality(subgraph)
    # draw_matrix_plots_subsets(unn_matrix, communities[0], equiv_index_rootid)
