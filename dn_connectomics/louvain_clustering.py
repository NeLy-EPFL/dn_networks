import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import multiprocessing as mp

mp.set_start_method("spawn")

import pandas as pd
from loaddata import (
    load_graph_and_matrices,
    get_name_from_rootid,
)
from processing_utils import run_function_with_timeout
import params
import plot_params
from common import cluster_matrix, generate_random
from graph_plot_utils import cluster_graph_louvain, make_nice_spines


def arrange_matrix(matrix_, modules_, sorting=None):
    """
    Arrange a matrix according the clustering modules identified.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        Matrix to be arranged.
    modules : dict
    sorting: str
        'size': sort by module size
        'density': sort by connection density within modules

    Returns
    -------
    mat : scipy.sparse.csr_matrix
        Arranged matrix.
    """

    # sorting step
    if sorting is None:
        pass
    elif sorting == "size":
        modules_ = sorted(modules_, key=len, reverse=True)
    elif sorting == "density":
        pass  # TODO: implement
    else:
        raise ValueError("Invalid sorting method.")

    # arrange matrix
    ordering = [x for x in modules_[0]]
    for module in modules_[1:]:
        ordering.extend([x for x in module])
    matrix_ = matrix_[ordering, :]
    matrix_ = matrix_[:, ordering]
    return matrix_


def run_clustering_n_times(
    args_: dict,
    working_folder_: str,
    iterations_: int = 3,
    time_limit_: float = 3,
):
    """
    Run the clustering algorithm n times, saving the results to n files in
    the working folder.

    Parameters
    ----------
    args : dict
        Arguments to be passed to the clustering function.
    working_folder : str
        Path to the folder where the clustering results will be stored.
    iterations : int
        Number of times to run the clustering algorithm.
    time_limit : float
        Time limit for each clustering run.

    Returns
    -------
    None
        Saves the clustering results to files in the working folder.
    """
    if not os.path.exists(working_folder_):
        os.makedirs(working_folder_)

    for i in range(iterations_):
        print(f"Running iteration {i}")
        file_name = os.path.join(
            working_folder_, f"iterated_modules_louvain_{i}.pkl"
        )
        modules_louvain_ = run_function_with_timeout(
            target=cluster_graph_louvain,
            args=args_,
            time_limit=time_limit_,
        )
        pickle.dump(modules_louvain_, open(file_name, "wb"))

    return None


def summarise_clustering_results(
    working_folder: str,
    graph_: nx.DiGraph = None,
):
    """
    Summarise the clustering results from the working folder.

    Parameters
    ----------
    working_folder : str
        Path to the folder where the clustering results are stored.
    graph : nx.DiGraph, optional
        Graph of the neurons. The default is None.
        If not None, the function will compute additional statistics on the
        modularity of the clustering.

    Returns
    -------
    clustering_summary : dict
        Dictionary where the key is the node_id, and the value is a list of the
        module numbers it belongs to.
    """
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    clustering_stats_df = pd.DataFrame()
    undirected_graph = nx.Graph(graph_)

    clustering_summary_ = {}
    for file_name in os.listdir(working_folder):
        if file_name.endswith(".pkl"):
            modules_louvain = pickle.load(
                open(os.path.join(working_folder, file_name), "rb")
            )
            # modules_louvain is a list of sets with node_ids inside.
            # assign a number to each set.
            # create a dictionary where the key is the node_id,
            # and the value is a list of the module numbers it belongs to.
            for i, module in enumerate(modules_louvain):
                for node_id in module:
                    if node_id in clustering_summary_.keys():
                        clustering_summary_[node_id].append(i)
                    else:
                        clustering_summary_[node_id] = [i]

            # compute the number of modules, the size of the largest module,
            # and the average module size, the average number of neurons in the
            #  10 largests module. Save all these values in the dataframe.
            if graph_ is not None:
                communities_ = [set(cluster) for cluster in modules_louvain]
                modularity_ = nx.community.modularity(
                    undirected_graph, communities_, weight=None
                )
            stats_dict = {
                "nb_modules": len(modules_louvain),
                "largest_module": max(
                    [len(module) for module in modules_louvain]
                ),
                "average_module_size": np.mean(
                    [len(module) for module in modules_louvain]
                ),
                "sum_ten_largest_modules": np.sum(
                    sorted(
                        [len(module) for module in modules_louvain],
                        reverse=True,
                    )[:10]
                ),
                "sum_five_largest_modules": np.sum(
                    sorted(
                        [len(module) for module in modules_louvain],
                        reverse=True,
                    )[:5]
                ),
            }
            if graph_ is not None:
                stats_dict["modularity"] = modularity_
            data = pd.DataFrame.from_dict(
                stats_dict,
                orient="index",
            ).T
            clustering_stats_df = pd.concat(
                [clustering_stats_df, data],
                ignore_index=True,
            )

    clustering_stats_df.to_csv(
        os.path.join(working_folder, "clustering_stats.csv")
    )
    return clustering_summary_


def calculate_similarity(
    list1: list[int],
    list2: list[int],
):
    """
    Calculate the similarity between two lists.
    The similarity is defined by the number of elements that are exactly the
    same for both lists at the same time.
    """
    assert len(list1) == len(list2)
    similarity = 0
    for i, l1_i in enumerate(list1):
        if l1_i == list2[i]:
            similarity += 1
    similarity /= len(list1)
    return similarity


def define_meta_similarity_matrix(
    clustering_summary_: dict,
):
    """
    Define a meta-similarity matrix based on the clustering summary.
    For each pair of nodes, calculate the similarity between the two nodes
    based on the modules they belong to.
    The similarity is defined by the number of modules that are exactly the
    same for both nodes at the same time.

    Parameters
    ----------
    clustering_summary : dict
        Dictionary where the key is the node_id, and the value is a list of the
        module numbers it belongs to.

    Returns
    -------
    meta_similarity_matrix : np.ndarray
        Meta-similarity matrix.
    """
    similarity = np.zeros(
        (len(clustering_summary_.keys()), len(clustering_summary_.keys()))
    )
    for node_id1 in clustering_summary_.keys():
        for node_id2 in clustering_summary_.keys():
            similarity_score = calculate_similarity(
                clustering_summary_[node_id1],
                clustering_summary_[node_id2],
            )
            similarity[node_id1, node_id2] = similarity_score

    return similarity


def detect_clusters(sim_mat, cutoff: float = 0.5, draw_edges: bool = False):
    """
    Detect clusters in a similarity matrix.

    Parameters
    ----------
    sim_mat : np.ndarray
        Similarity matrix, with values between 0 and 1.
    cutoff : float
        Cutoff value for elements to be considered part of the same cluster.

    Returns
    -------
    clusters : list[list[int]]
        List of clusters, where each cluster is a list of node_ids.
    """
    clustered_mat, order = cluster_matrix(sim_mat)
    clusters = []
    start_cluster = 0
    for i in range(clustered_mat.shape[0]):
        average_similarity = np.mean(clustered_mat[i, start_cluster:i])
        if average_similarity < cutoff:
            clusters.append(order[start_cluster:i])
            start_cluster = i
    clusters.append(order[start_cluster:])
    if not draw_edges:
        return clusters
    else:
        # draw a matrix where all entries are white except for the boundaries
        # between clusters as defined by the number of elements in each list in
        # the clusters list.

        # create a matrix of zeros
        mat = np.zeros((sim_mat.shape[0], sim_mat.shape[1]))
        # draw the boundaries between clusters
        min = 0
        max = 0
        for cluster in clusters:
            max += len(cluster)
            mat[min:max, min:max] = 1
            min = max
        return clusters, mat


def plot_clustered_network(
    clusters_, edges_matrix_, matrix_, save_dir: str = None
):
    """
    Arrange the connectivity matrix according to the clustering, and plot it.
    """
    clustered_matrix = arrange_matrix(matrix_, clusters_)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    c = ax.imshow(
        clustered_matrix.todense(),
        cmap="RdBu_r",
        vmax=10,
        vmin=-10,
    )
    ax.imshow(
        edges_matrix_,
        cmap="Greys",
        vmax=1,
        vmin=0,
        alpha=0.1,
        interpolation="none",
    )
    ax.set_title("DN network")
    ax.set_xlabel("postsynaptic neuron")
    ax.set_ylabel("presynaptic neuron")
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("synaptic weight")
    if save_dir is not None:
        plt.savefig(save_dir)
    return


def save_clustering_in_table(
    meta_clusters_: list[list[int]], working_folder_: str
):
    """
    Save clustering in a table.
    Define one column with the root_id, one with the cluster number, and one
    with the name of the neuron when it's known.

    Parameters
    ----------
    meta_clusters : list[list[int]]
        List of clusters, where each cluster is a list of node_ids.
    working_folder : str
        Path to the working folder.

    Returns
    -------
    None.
        Saves the data as .csv in the working folder.
    """
    # Create a dataframe where each row is a node_id, and the columns are:
    # root_id, cluster number, and name.
    df = pd.DataFrame(columns=["node_index", "root_id", "cluster", "name"])
    for i, cluster in enumerate(meta_clusters_):
        for node_id in cluster:
            root_id = equiv_index_rootid.loc[
                equiv_index_rootid.index == node_id
            ].root_id.values[0]
            name = get_name_from_rootid(root_id, empty_type="DN")
            data = pd.DataFrame.from_dict(
                {
                    "node_index": node_id,
                    "root_id": root_id,
                    "cluster": i,
                    "name": name,
                },
                orient="index",
            ).T
            df = pd.concat(
                [df, data],
                ignore_index=True,
            )
    df = df.sort_values(by=["cluster", "name"])
    df.to_csv(os.path.join(working_folder_, "clustering.csv"))

    # Cluster list
    df_cluster_list = df.groupby(["cluster", "name"]).count()
    df_cluster_list = df_cluster_list.drop(columns=["node_index"])
    df_cluster_list = df_cluster_list.rename({"root_id": "count"}, axis=1)
    df_cluster_list = df_cluster_list.sort_values(by=["cluster", "name"])
    df_cluster_list.to_csv(
        os.path.join(working_folder_, "clustering_list.csv")
    )
    return


def confusion_matrix_communities(
    G: nx.DiGraph,
    modules_louvain: list[set[int]],
    ax: plt.Axes = None,
    connection_type: str = "connection",
    size_threshold=2,
    count_synapses: bool = False,
    normalise_by_size: bool = False,
    return_ax: bool = False,
):
    """
    Connection confusion matrix:
    confusion matrix of the connections between the communities
    rows: communities
    columns: communities
    values: number of connections between the communities
    the diagonal is the number of connections within the communities
    the off-diagonal is the number of connections between the communities

    Parameters
    ----------
    G : nx.DiGraph
        directed graph of the neurons
    modules_louvain : list[set[int]]
        list of the communities
    ax : plt.Axes, optional
        axes to plot the confusion matrix, by default None
    connection_type : str, optional
        type of connections to consider, by default 'connection'.
        Can be 'excitatory' or 'inhibitory', or 'relative' for the difference
    count_synapses : bool, optional
        whether to count the number of synapses or the number of connections, by default False
    normalise_by_size : bool, optional
        whether to normalise the confusion matrix by the size of the communities, by default False

    Returns
    -------
    np.ndarray
        confusion matrix
    """
    # plotting parameters
    color_map = {
        "connection": "Greys",
        "excitatory": "Reds",
        "inhibitory": "Blues",
    }
    # nb_modules is the number of communities where there are at least 2 neurons
    nb_modules = 0
    modules_louvain_ = []
    for community in modules_louvain:
        if len(community) > size_threshold:
            nb_modules += 1
            modules_louvain_.append(community)
    # compute the confusion matrix
    confusion_matrix = np.zeros((nb_modules, nb_modules))
    for d_in, community_in in enumerate(modules_louvain_):
        for d_out, community_out in enumerate(modules_louvain_):
            # define the graph of the connections between the communities
            G_sub = G.subgraph(community_in.union(community_out))
            for e in G_sub.edges:
                if e[0] in community_in and e[1] in community_out:
                    if connection_type == "connection":
                        confusion_matrix[d_in, d_out] += (
                            1
                            if not count_synapses
                            else abs(G_sub.edges[e]["weight"])
                        )
                    elif (connection_type == "excitatory") & (
                        G_sub.edges[e]["weight"] > 0
                    ):
                        confusion_matrix[d_in, d_out] += (
                            1
                            if not count_synapses
                            else G_sub.edges[e]["weight"]
                        )
                    elif (connection_type == "inhibitory") & (
                        G_sub.edges[e]["weight"] < 0
                    ):
                        confusion_matrix[d_in, d_out] += (
                            1
                            if not count_synapses
                            else abs(G_sub.edges[e]["weight"])
                        )
                    elif connection_type == "relative":
                        confusion_matrix[d_in, d_out] += (
                            (1 if G_sub.edges[e]["weight"] > 0 else -1)
                            if not count_synapses
                            else G_sub.edges[e]["weight"]
                        )
            if normalise_by_size:
                confusion_matrix[d_in, d_out] /= len(community_in)

    if not normalise_by_size:
        confusion_matrix = confusion_matrix.astype(int)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    if connection_type != "relative":
        m = ax.imshow(confusion_matrix, cmap=color_map[connection_type])
        for i in range(nb_modules):
            for j in range(nb_modules):
                if confusion_matrix[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        confusion_matrix[i, j],
                        ha="center",
                        va="center",
                        color="grey",
                        fontsize=6,
                    )

    else:
        extr = 75  # np.max(np.abs(confusion_matrix))
        m = ax.imshow(confusion_matrix, cmap="RdBu_r", vmin=-extr, vmax=extr)

    # colorbar
    plt.colorbar(m, ax=ax)
    ax.set_xticks(np.arange(nb_modules))
    ax.set_yticks(np.arange(nb_modules))
    ax.set_xticklabels(np.arange(1, nb_modules + 1))
    ax.set_yticklabels(np.arange(1, nb_modules + 1))
    ax.set_ylabel("Presynaptic community")
    ax.set_xlabel("Postsynaptic community")
    connection_title = (
        " (number of synapses)"
        if count_synapses
        else " (number of connections)"
    )
    ax.set_title(f"{connection_type} confusion matrix" + connection_title)

    if return_ax:
        return confusion_matrix, ax
    return confusion_matrix


def run_louvain_custering(control: bool = False):
    ## -- Data Loading -- ##
    (
        _,
        unn_matrix,
        _,
        _,
    ) = load_graph_and_matrices("dn")

    DNgraph = nx.from_scipy_sparse_array(unn_matrix, create_using=nx.DiGraph)

    if control:
        shuffled_mat = generate_random(unn_matrix)
        shuffled_graph = nx.from_scipy_sparse_array(
            shuffled_mat, create_using=nx.DiGraph
        )
        # save the shuffled matrix
        shuffled_dir = os.path.join(
            plot_params.CLUSTERING_ARGS["folder"],
            "shuffled_control",
        )
        if not os.path.exists(shuffled_dir):
            os.makedirs(shuffled_dir)
        np.save(
            os.path.join(
                shuffled_dir,
                "shuffled_matrix.npy",
            ),
            shuffled_mat,
        )
    # Control or not
    network_graph = shuffled_graph if control else DNgraph
    network_matrix = shuffled_mat if control else unn_matrix
    network_folder = "shuffled_control" if control else "data"

    # Parameters
    args = {
        "graph": network_graph,
        "visualise_matrix": False,
        "connection_type": plot_params.CLUSTERING_ARGS["positive_connections"],
    }

    time_limit = plot_params.CLUSTERING_ARGS["time_limit"]
    iterations = plot_params.CLUSTERING_ARGS["iterations"]
    cutoff = plot_params.CLUSTERING_ARGS["cutoff"]

    working_folder = plot_params.CLUSTERING_ARGS["folder"]
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)
    processing_folder = os.path.join(working_folder, network_folder)
    if not os.path.exists(processing_folder):
        os.makedirs(processing_folder)

    ## -- Clustering -- ##

    run_clustering_n_times(args, processing_folder, iterations, time_limit)
    clustering_summary = summarise_clustering_results(
        processing_folder, network_graph
    )
    meta_similarity_matrix = define_meta_similarity_matrix(clustering_summary)
    meta_clusters, edges_matrix = detect_clusters(
        meta_similarity_matrix, cutoff=cutoff, draw_edges=True
    )
    save_clustering_in_table(meta_clusters, processing_folder)

    # compute additional statistics on the clustered graph
    communities = [set(cluster) for cluster in meta_clusters]
    # modularity = nx.community.modularity(DNgraph, communities)
    modularity = nx.community.modularity(
        network_graph, communities, weight=None
    )
    print(f"Modularity score as implemented in networkx: {modularity}")

    ## --- plot the clustered graph --- ##
    savedir = os.path.join(
        processing_folder,
        "dn_network_meta_clustered.pdf",
    )
    plot_clustered_network(
        meta_clusters, edges_matrix, network_matrix, savedir
    )

    ## --- plot the confusion matrix --- ##
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    _, ax = confusion_matrix_communities(
        network_graph,
        communities,
        ax,
        plot_params.CLUSTERING_ARGS["confusion_mat_values"],
        size_threshold=plot_params.CLUSTERING_ARGS[
            "confusion_mat_size_threshold"
        ],
        count_synapses=plot_params.CLUSTERING_ARGS[
            "confusion_mat_count_synpases"
        ],
        normalise_by_size=plot_params.CLUSTERING_ARGS[
            "confusion_mat_normalise"
        ],
        return_ax=True,
    )
    make_nice_spines(ax)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            processing_folder,
            "confusion_matrix_synapse_normalised_cluster_size.pdf",
        ),
        dpi=300,
    )


def draw_louvain_custering():
    run_louvain_custering()
    run_louvain_custering(control=True)


if __name__ == "__main__":
    draw_louvain_custering()
