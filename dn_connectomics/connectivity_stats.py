"""
2023.08.30
author: femke.hurtak@epfl.ch
Script to compute the connectivity statistics for specific neuron sets.
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import numpy as np
from scipy import stats, sparse

import plot_params
import neuron_params
from loaddata import load_graph_and_matrices, load_nodes_and_edges, load_names, get_list_neurons_from_type
from statistics_utils import connectivity_stats
from graph_plot_utils import make_nice_spines, make_axis_disappear
from common import connections_up_to_n_hops

def pearson(y_true, y_pred):
    """
    Compute the R2 score between two lists.
    """
    y_true_ = np.array(y_true)
    y_pred_ = np.array(y_pred)
    return stats.pearsonr(y_true_, y_pred_)


def prepare_table(
    equiv_index_rootid_,
    marker: str = "DNg",
):
    """
    Prepare the table with the information about the neurons, and whether
    they are in the category given by the marker.

    Parameters
    ----------
    marker : str
        Marker to use to identify the neurons of interest.
    equiv_index_rootid_ : pd.DataFrame
        Dataframe containing the equivalence between the root id and the
        index in the matrix.

    Returns
    -------
    dns_info : pd.DataFrame
        Dataframe containing the information about the neurons of interest.
    """
    info = equiv_index_rootid_.copy()
    root_id_names_df = load_names()
    dn_names = root_id_names_df[
        root_id_names_df["name"].str.contains('DN')
    ]["root_id"].values
    highlighted_neurons = root_id_names_df[
        root_id_names_df["name"].str.contains(marker)
    ]["root_id"].values
    # formatting to reuse the code
    info["DN"] = info["root_id"].isin(dn_names)
    info["all"] = True
    info["GNG_DN"] = info["root_id"].isin(highlighted_neurons)
    return info


def scatter_plot(ax, x, y, args):
    ax.scatter(
        x,
        y,
        c=args["color"] if "color" in args.keys() else "grey",
        label=args["label"] if "label" in args.keys() else "",
        marker=args["marker"] if "marker" in args.keys() else "o",
        alpha=args["alpha"] if "alpha" in args.keys() else 1,
        s=args["s"] if "s" in args.keys() else 20,
        edgecolors=args["edgecolors"] if "edgecolors" in args.keys() else None,
    )
    ax.set_xlabel(args["x_label"] if "x_label" in args.keys() else "")
    ax.set_ylabel(args["y_label"] if "y_label" in args.keys() else "")
    make_nice_spines(ax)
    return ax


def plot_connection_stats(
    ax: plt.Axes,
    stats_df: pd.DataFrame,
    highlighted_neurons: dict = None,
    displayed_neurons: dict = None,
    plotting_args: dict = None,
    show_insert: bool = True,
):
    """
    Draw the connection stats.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Dataframe containing the stats.
    highlighted_neurons : dict
        Dictionary containing the highlighted neurons.
    plotting_args : dict
        Dictionary containing the plotting arguments.
        Includes: 'x_arg', 'y_arg'

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    """
    # Data selection
    selected_stats = stats_df
    highlighted_df = stats_df[
        stats_df["root_id"].isin(highlighted_neurons.keys())
    ]
    displayed_df = stats_df[stats_df["root_id"].isin(displayed_neurons.keys())]

    # Plotting
    # --- Scatter plot (left): Background
    ax[0] = scatter_plot(
        ax[0],
        selected_stats[plotting_args["x_arg"]],
        selected_stats[plotting_args["y_arg"]],
        args={
            "label": plotting_args["data_label"],
            "color": plotting_args["scatter_color"],
            "x_label": plotting_args["x_label"],
            "y_label": plotting_args["y_label"],
        },
    )

    ymax = 40

    # --- Inset plot (left): distribution
    if show_insert:
        axins0 = inset_axes(
            ax[0],
            width="50%",
            height="55%",
            loc="lower right",
        )
        axins0.patch.set_alpha(0)
        make_axis_disappear(axins0)
        axins1 = inset_axes(axins0, width="40%", height="50%", loc="upper left")
        sns.violinplot(
            ax=axins1,
            data=[selected_stats[plotting_args["y_arg"]]],
            palette=[plotting_args["scatter_color"]],
            inner="box",
            linewidth=1,
            cut=0,
            scale="width",
            width=0.8,
            saturation=1,
            bw=0.2,
        )
        # axins1.set_xticklabels([plotting_args["data_label"]])
        axins1.set(ylim=(0, ymax))
        make_nice_spines(axins1)
        # axins1.set_ylabel('Number of connected DNs within 2 hops')

        # --- Inset plot (left): highlighted neurons single points
        if len(highlighted_df) > 0:
            axins2 = inset_axes(
                axins0, width="40%", height="50%", loc="upper right"
            )
            sns.stripplot(
                ax=axins2,
                data=highlighted_df,
                y=plotting_args["y_arg"],
                palette={k: v["color"] for k, v in highlighted_neurons.items()},
                hue="root_id",
                legend=False,
                size=10,
                marker="s",
                edgecolor="white",
            )
            axins2.set_xticklabels(["ref DNs"])
            make_nice_spines(axins2)
            axins2.set(ylim=(0, 100))
            axins2.spines["left"].set_visible(False)
            axins2.spines["left"].set_linewidth(0)
            axins2.tick_params(width=0)
            axins2.set_yticklabels([])
            axins2.set_ylabel("")
            axins2.set(ylim=(0, ymax))

    # --- Scatter plot (left): highlighted neurons
    # The highlighted_neurons is a dict where keys are root_ids, and the values
    # are dicts containing a field 'name'.
    # make a dict with the name as key and the list of root_ids having this name
    if len(highlighted_neurons) > 0:
        highlighted_summary = {}
        for n_ in highlighted_neurons:
            if highlighted_neurons[n_]["name"] not in highlighted_summary:
                highlighted_summary[highlighted_neurons[n_]["name"]] = {
                    "rids": [],
                    "color": highlighted_neurons[n_]["color"],
                }
            highlighted_summary[highlighted_neurons[n_]["name"]][
                "rids"
            ].append(n_)

        for ref_neurons, ref_ids in highlighted_summary.items():
            ax[0] = scatter_plot(
                ax[0],
                x=selected_stats[plotting_args["x_arg"]][
                    selected_stats["root_id"].isin(ref_ids["rids"])
                ],
                y=selected_stats[plotting_args["y_arg"]][
                    selected_stats["root_id"].isin(ref_ids["rids"])
                ],
                args={
                    "x_label": plotting_args["x_label"],
                    "y_label": plotting_args["y_label"],
                    "marker": "s",
                    "s": 100,
                    "edgecolors": "white",
                    "color": ref_ids["color"],
                    "label": ref_neurons,
                },
            )

    # --- Violin plot (right)
    sns.violinplot(
        ax=ax[1],
        data=[
            selected_stats[plotting_args["y_arg"]],
            displayed_df[plotting_args["y_arg"]],
        ]
        if len(displayed_neurons) > 0
        else [selected_stats[plotting_args["y_arg"]]],
        palette=[plotting_args["scatter_color"], "k"]
        if len(displayed_neurons) > 0
        else [plotting_args["scatter_color"]],
        inner=None,
        linewidth=1,
        cut=0,
        scale="width",
        width=0.8,
        saturation=1,
    )
    ax[1].set_xticklabels(["", ""] if len(displayed_neurons) > 0 else [""])
    make_nice_spines(ax[1])

    # --- Scatter plot (left): displayed neurons in black
    if len(displayed_neurons) > 0:
        ax[0] = scatter_plot(
            ax[0],
            displayed_df[plotting_args["x_arg"]],
            displayed_df[plotting_args["y_arg"]],
            args={
                "label": plotting_args["black_dots_label"],
                "color": "k",
                "marker": "o",
                "alpha": 1,
                "s": 50,
                "edgecolors": "white",
            },
        )
    ax[0].legend(frameon=False, fontsize=14)
    return ax


def compare_ans_dns(edges, nodes, working_folder):
    """
    Compare the ANs and DNs contribution as presynaptic partners to DNs.
    """
    list_dns = get_list_neurons_from_type(type='descending')
    list_ans = get_list_neurons_from_type(type='ascending')
    ans_to_dns = edges[
        edges["pre_root_id"].isin(list_ans)
        & edges["post_root_id"].isin(list_dns)
    ]
    dns_to_dns = edges[
        edges["pre_root_id"].isin(list_dns)
        & edges["post_root_id"].isin(list_dns)
    ]
    all_to_dns = edges[edges["post_root_id"].isin(list_dns)]
    dns_to_all = edges[edges["pre_root_id"].isin(list_dns)]
    # open a file to write the stats
    file_name = os.path.join(working_folder, "ans_dns_stats.txt")
    with open(file_name, "w") as f:
        # fraction of DNs as outputs of DNs
        f.write(
            "Fraction of DNs as outputs of DNs: {}\n".format(
                len(dns_to_dns) / len(dns_to_all)
            )
        )
        # fraction of ANs as inputs to DNs
        f.write(
            "Fraction of ANs as inputs to DNs: {}\n".format(
                len(ans_to_dns) / len(all_to_dns)
            )
        )
        # fraction of DNs as inputs to DNs
        f.write(
            "Fraction of DNs as inputs to DNs: {}\n".format(
                len(dns_to_dns) / len(all_to_dns)
            )
        )
        # number of existing ANs and DNs
        f.write("Number of ANs: {}\n".format(len(list_ans)))
        f.write("Number of DNs: {}\n".format(len(list_dns)))
        # number of connections given by the unique (pre_root_id, post_root_id)
        f.write(
            "Number of ANs to DNs connections: {}\n".format(
                len(ans_to_dns.groupby(["pre_root_id", "post_root_id"]))
            )
        )
        f.write(
            "Number of DNs to DNs connections: {}\n".format(
                len(dns_to_dns.groupby(["pre_root_id", "post_root_id"]))
            )
        )
        # sum of the number of synapses
        f.write(
            "Number of ANs to DNs synapses: {}\n".format(
                ans_to_dns["syn_count"].sum()
            )
        )
        f.write(
            "Number of DNs to DNs synapses: {}\n".format(
                dns_to_dns["syn_count"].sum()
            )
        )
        # number of DNs receiving input from ANs vs DNs
        f.write(
            "Number of DNs receiving input from ANs: {}\n".format(
                len(ans_to_dns["post_root_id"].unique())
            )
        )
        f.write(
            "Number of DNs receiving input from DNs: {}\n".format(
                len(dns_to_dns["post_root_id"].unique())
            )
        )
        neurons_reported = neuron_params.KNOWN_DNS.keys()
        for neuron in neurons_reported:
            # how many ANs and DNs are connected to this neuron
            n_ans = len(
                ans_to_dns[ans_to_dns["post_root_id"] == neuron][
                    "pre_root_id"
                ].unique()
            )
            n_dns = len(
                dns_to_dns[dns_to_dns["post_root_id"] == neuron][
                    "pre_root_id"
                ].unique()
            )
            name = neuron_params.KNOWN_DNS[neuron]["name"]
            f.write(
                "Number of ANs connected to {} ({}) : {}\n".format(
                    name, neuron, n_ans
                )
            )
            f.write(
                "Number of DNs connected to {} ({}) : {}\n".format(
                    name, neuron, n_dns
                )
            )

    f.close()
    return


def make_gng_dn_plot(dns_info):
    """
    Draw a plot showing the number of connected DNs as a function of the
    connectivity of the DNs.
    """
    _, ax = plt.subplots(1, 2, figsize=(8, 6), width_ratios=[3, 1])
    plotting_args_gng = {
        "x_arg": plot_params.STATS_ARGS["measured_feature"] + "_sorting",
        "y_arg": "GNG_" + plot_params.STATS_ARGS["measured_feature"],
        "x_label": "DNs sorted by connectivity",
        "y_label": plot_params.STATS_ARGS["measured_feature_label"],
        "data_label": "GNG DNs",
        "black_dots_label": "DNs tested",
        "scatter_color": plot_params.DARKORANGE,
    }
    ax = plot_connection_stats(
        ax,
        dns_info,
        highlighted_neurons={},
        displayed_neurons={},
        plotting_args=plotting_args_gng,
    )
    plotting_args_dn = {
        "x_arg": plot_params.STATS_ARGS["measured_feature"] + "_sorting",
        "y_arg": plot_params.STATS_ARGS["measured_feature"],
        "x_label": "DNs sorted by connectivity",
        "y_label": plot_params.STATS_ARGS["measured_feature_label"],
        "data_label": "all DNs",
        "black_dots_label": "DNs tested",
        "scatter_color": plot_params.LIGHTGREY,
    }
    ax = plot_connection_stats(
        ax,
        dns_info,
        highlighted_neurons=neuron_params.REF_DNS,
        displayed_neurons={},
        plotting_args=plotting_args_dn,
    )


def write_relevant_stats(working_folder, dns_info, edges):
    """
    Write the stats used for the manuscript to a file.
    """
    file_name = os.path.join(working_folder, "stats.txt")
    with open(file_name, "w") as f:
        # number of connections
        f.write("Number of DNs: {}\n".format(len(dns_info)))
        n_zero_input = dns_info[dns_info["DN_connected_in"] == 0]
        f.write("Number of DNs with no input: {}\n".format(len(n_zero_input)))
        n_one_input = dns_info[dns_info["DN_connected_in"] == 1]
        f.write("Number of DNs with one input: {}\n".format(len(n_one_input)))
        f.write("\n")

        dn_edges = edges[
            edges["pre_root_id"].isin(dns_info["root_id"])
            & edges["post_root_id"].isin(dns_info["root_id"])
        ]

        for nt_type in dn_edges["nt_type"].unique():
            f.write(
                "Number of {} synapses: {}\n".format(
                    nt_type, len(dn_edges[dn_edges["nt_type"] == nt_type])
                )
            )
            f.write(
                "Number of {} neurons: {}\n".format(
                    nt_type,
                    len(
                        dn_edges[dn_edges["nt_type"] == nt_type][
                            "pre_root_id"
                        ].unique()
                    ),
                )
            )
        f.write(
            "Verification: total number of synapses: {}\n".format(
                len(dn_edges)
            )
        )
        f.write(
            "Verification: total number of neurons: {}\n".format(
                len(dn_edges["pre_root_id"].unique())
            )
        )
        f.write("\n")

        ## --- Check that comDNs are statistically different from the rest --- ##
        feature = plot_params.STATS_ARGS["measured_feature"]
        down_neurons = dns_info[feature][
            ~dns_info["root_id"].isin(neuron_params.REF_DNS.keys())
        ].values
        # get mean and std of the distribution
        mean = down_neurons.mean()
        median = np.median(down_neurons)
        std = down_neurons.std()
        f.write(
            "Overall DNs downstream excited mean: {}, std: {}, median: {}\n".format(
                mean, std, median
            )
        )
        f.write("\n")

        for neuron in neuron_params.REF_DNS.keys():
            n_down = dns_info[dns_info["root_id"] == neuron][feature].values[0]
            name = neuron_params.REF_DNS[neuron]["name"]
            f.write(
                "Number of connections for {} ({}) : {}\n".format(
                    name, neuron, n_down
                )
            )
            # compute the z-score
            z_score = (n_down - mean) / std
            # compute the p-value
            p_value = 1 - stats.norm.cdf(z_score)
            f.write("p-value on z-score: {}\n".format(p_value))
    f.close()
    return


def make_specific_neurons_plot(
        dns_info,
        specific_neurons=neuron_params.REF_DNS,
        show_insert = True):
    _, ax = plt.subplots(1, 2, figsize=(8, 6), width_ratios=[3, 1])

    plotting_args_dn = {
        "x_arg": plot_params.STATS_ARGS["measured_feature"] + "_sorting",
        "y_arg": plot_params.STATS_ARGS["measured_feature"],
        "x_label": "DNs sorted by connectivity",
        "y_label": plot_params.STATS_ARGS["measured_feature_label"],
        "data_label": "all DNs",
        "black_dots_label": "DNs tested",
        "scatter_color": plot_params.LIGHTGREY,
    }
    ax = plot_connection_stats(
        ax,
        dns_info,
        highlighted_neurons=specific_neurons,  # REF_DNS,
        displayed_neurons={},  # neurons.VALIDATION_DNS,
        plotting_args=plotting_args_dn,
        show_insert = show_insert,
    )


def compute_connectivity_stats():
    # Load the data
    nodes, edges = load_nodes_and_edges()
    (
        _,
        syncount_matrix,
        _,
        _,
        equiv_index_rootid,
    ) = load_graph_and_matrices("dn")

    working_folder = plot_params.STATS_ARGS["folder"]
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    dns_info = prepare_table(equiv_index_rootid, marker="DNg")
    dns_info = connectivity_stats(syncount_matrix, dns_info)
    dns_info.to_csv(os.path.join(working_folder, "dns_info.csv"))

    ## --- Plot the number of connected DNs as a function of the connectivity of the DNs --- ##
    make_gng_dn_plot(dns_info)
    plt.savefig(os.path.join(working_folder, "gng_dn_plot.pdf"))

    make_specific_neurons_plot(dns_info, specific_neurons=neuron_params.VALIDATION_DNS)
    plt.savefig(os.path.join(working_folder, "tested_neurons_plot.pdf"))

    make_specific_neurons_plot(dns_info, specific_neurons=neuron_params.BEHAVIOUR_DNS, show_insert = False)
    plt.savefig(os.path.join(working_folder, "behaviour_neurons_plot.pdf"))

    make_specific_neurons_plot(dns_info, specific_neurons=neuron_params.STRONGLY_CONNECTED_DNS, show_insert = False)
    plt.savefig(os.path.join(working_folder, "strongly_connected_neurons_plot.pdf"))

    ## --- Print some stats for the paper to a file --- ##
    write_relevant_stats(working_folder, dns_info, edges)

    ## --- Compare ANs to DNs --- ##
    compare_ans_dns(edges, nodes, working_folder)

    ## --- For the neurons studied in the paper, print the number of connections --- ##
    neurons_reported = neuron_params.KNOWN_DNS.keys()
    info_to_share = ["root_id", "DN_connected_in", "DN_connected_out"]
    dns_info_reported = dns_info[dns_info["root_id"].isin(neurons_reported)][
        info_to_share
    ]
    dns_info_reported["name"] = dns_info_reported["root_id"].apply(
        lambda x: neuron_params.KNOWN_DNS[x]["name"]
    )
    dns_info_reported.to_csv(
        os.path.join(working_folder, "stats_reported_neurons.csv")
    )

    return

def get_stat_feature(
    direction: str = "inputs",
    connection: str = "all",
    graph_type: str = "dn",
    hops: int = 1,
    include_smaller_hops: bool = True,
):
    """
    Compute the stats for a given feature and a given graph.

    Parameters
    ----------
    direction : str
        Direction of the connection to consider. can be 'inputs', 'outputs', 'connections'.
    connection : str
        Type of connection to consider. Can be 'all', 'excitatory', 'inhibitory'.
    graph_type : str
        Type of graph to consider. Can be 'dn', 'central_an_dn'.
    hops : int
        Number of hops to consider.
    include_smaller_hops : bool
        Whether to include allt he paths with fewer hops than the specified number.

    Returns
    -------
    stats : dict
        Dictionary containing the number of connections for each neuron. The 
        keys are the root ids.
    """
    # List of neurons to consider as final results: dns
    list_dns = get_list_neurons_from_type(type='descending')
    # Load the data
    (_, _, matrix, _, equiv_index_rootid) = load_graph_and_matrices(graph_type)
    # Remove all unwanted connections
    if connection == "all":
        pass
    elif connection == "excitatory":
        matrix[matrix < 0] = 0
    elif connection == "inhibitory":
        matrix[matrix > 0] = 0
    matrix.eliminate_zeros()
    # Get the effective matrix
    if hops == 1:
        effective_matrix = matrix.copy()
    else:
        if include_smaller_hops:
            effective_matrix = connections_up_to_n_hops(matrix, hops)
        else:
            effective_matrix = matrix ** hops
    # Define a scipy sparse diagonal matrix with element (i,i) being 1 if the index i
    # in equiv_index_rootid corresopnds to a root_id in list_dns, 0 otherwise.
    # This allows to select only the DNs in the matrix.
    dn_indexing_list = np.zeros(equiv_index_rootid.shape[0])
    for i, index in enumerate(equiv_index_rootid.index):
        if equiv_index_rootid.loc[index, "root_id"] in list_dns:
            dn_indexing_list[i] = 1
    dn_indexing_matrix = sparse.diags(dn_indexing_list, format = 'csr')

    if direction == "inputs":
        effective_matrix = effective_matrix.transpose()
        
    # Get the effective connections to DNs
    effective_matrix = effective_matrix @ dn_indexing_matrix
    

    # Make a dictionary where the keys are the root ids and the values are the number
    # of connections to/from DNs.
    stats = {}
    for i, index in enumerate(equiv_index_rootid.index):
        stats[equiv_index_rootid.loc[index, "root_id"]] = np.count_nonzero(
            effective_matrix.getrow(i).toarray()
            )
    return stats
    


def compare_connectivity_stats(
        direction:str = 'outputs',
        connection:str = 'all',
        reference_sorting:str = 'direct',
        ):
    """
    Compare the connectivity from DNs to DNs accounting for intermediates or not.
    """
    # Get dictionaries with the number of outputs for each DN taking into account
    # intermediates or not.
    direct_dn_stats = get_stat_feature(
        direction = direction,
        connection = connection,
        graph_type = 'dn',
        hops = 1,)
    indirect_dn_stats = get_stat_feature(
        direction = direction,
        connection = connection,
        graph_type = 'dn',
        hops = 2,)
    indirect_interneuron_stats = get_stat_feature(
        direction = direction,
        connection = connection,
        graph_type = "central_an_dn",
        hops = 2,)
    # Plot the results in 3 graphs.
    # Assign a color to each neuron such that the number of connections in the
    # direct_dn_stats is a gradient from red to blue. Keep the same color for
    # the same neuron in the other graphs.
    # Get the list of neurons in the 3 graphs
    match reference_sorting:
        case 'direct':
            reference_stats = direct_dn_stats
        case 'indirect':
            reference_stats = indirect_dn_stats
        case 'all_interneurons':
            reference_stats = indirect_interneuron_stats
    
    neurons = list(reference_stats.keys())
    reference_connections = [reference_stats[n] for n in neurons]
    # sort the neurons by the number of direct connections
    sorted_neurons = [x for _, x in sorted(zip(reference_connections, neurons))][::-1]
    # get the color gradient
    colors = sns.color_palette(
        plot_params.INTERNEURON_STATS_ARGS['color_schema'],
        len(neurons),
        )
    # make a dict with the color for each neuron
    color_dict = {n: c for n, c in zip(sorted_neurons, colors)}
    # make the graphs
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    label = ['direct', '2 hops', '2 hops with brain interneurons']
    for i, stats in enumerate([direct_dn_stats, indirect_dn_stats, indirect_interneuron_stats]):
        # get the number of connections for each neuron
        connections = [stats[n] for n in sorted_neurons]
        # quantification of correlation
        ref_connection = [reference_stats[n] for n in sorted_neurons]
        r, _ = pearson(connections, ref_connection)
        # make the plot
        label_ = label[i] + '\n' + '(r2={:.2f})'.format(r**2)
        ax[i].scatter(
            range(len(neurons)),
            connections,
            c=[color_dict[n] for n in sorted_neurons],
            s=20,
            label=label_,
        )
        ax[i].set_ylabel('Number of DNs connected')
        ax[i].set_xlabel('DNs')
        make_nice_spines(ax[i])
        ax[i].legend()
    plt.tight_layout()

    # Save the figure
    working_folder = plot_params.INTERNEURON_STATS_ARGS["folder"]
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)
    figtitle = "compare_connectivity_stats_{}_{}_connections_sorting_{}.pdf".format(direction, connection, reference_sorting)
    plt.savefig(os.path.join(working_folder, figtitle), dpi=300)
    return

if __name__ == "__main__":
    #compute_connectivity_stats()
    compare_connectivity_stats(
        direction = plot_params.INTERNEURON_STATS_ARGS["measured_feature"],
        connection = plot_params.INTERNEURON_STATS_ARGS["connection_type"],
        reference_sorting = plot_params.INTERNEURON_STATS_ARGS["reference_sorting"],
    )
