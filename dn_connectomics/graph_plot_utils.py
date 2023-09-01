import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.sparse.linalg import svds
from scipy import sparse
import seaborn as sns
from collections import Counter
from tqdm import tqdm

import plot_params

NT_TYPES = plot_params.NT_TYPES


def get_network_specs(
    connectivity_df: pd.DataFrame,
    neurons_of_interest: dict = None,
    root_id_indexing: bool = False,
) -> dict:
    """
    == Adapted from connectools.util.plot_network.get_network_specs
    from Gizem Ozdil (gizem.ozdil@epfl.ch) ==

    From a connectivity table, obtains the edges and visual network properties.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        Connectivity table that includes `pre_root_id`, `post_root_id`,
        `nt_type`, `syn_count` as columns.
    neurons_of_interest : Dict, optional
        Nested dictionary containing segment ID as keys,
        a dictionary of color and segment name as values
        ,by default None

    Returns
    -------
    Dict
        Dictionary containing the network specs such as edges, colors etc.
    """

    unique_neurons = set(connectivity_df["pre_root_id"]).union(
        set(connectivity_df["post_root_id"])
    )
    if not root_id_indexing:
        seg_idx_lookup = {
            neuron_id: i for i, neuron_id in enumerate(unique_neurons)
        }

        edges = list(
            connectivity_df.apply(
                lambda row: (
                    seg_idx_lookup[row["pre_root_id"]],
                    seg_idx_lookup[row["post_root_id"]],
                ),
                axis=1,
            )
        )
    else:
        seg_idx_lookup = {
            neuron_id: neuron_id for i, neuron_id in enumerate(unique_neurons)
        }
        edges = list(
            connectivity_df.apply(
                lambda row: (
                    row["pre_root_id"],
                    row["post_root_id"],
                ),
                axis=1,
            )
        )

    edge_weights = connectivity_df.syn_count.to_numpy()
    edge_type = connectivity_df.nt_type.to_numpy()
    edge_colors = [NT_TYPES[nt_name]["color"] for nt_name in edge_type]
    edge_ls = [NT_TYPES[nt_name]["linestyle"] for nt_name in edge_type]

    if neurons_of_interest is not None:
        node_colors = [
            neurons_of_interest[neuron_id]["color"]
            if neuron_id in neurons_of_interest
            else "lightgrey"
            for neuron_id in seg_idx_lookup
        ]
        node_labels = {
            idx: neurons_of_interest[neuron_id]["name"]  # .replace("_", " ")
            if neuron_id in neurons_of_interest
            else ""
            for neuron_id, idx in seg_idx_lookup.items()
        }
    else:
        node_colors = ["lightgrey"] * len(seg_idx_lookup)
        node_labels = {idx: "" for _, idx in seg_idx_lookup.items()}

    return {
        "edges": edges,
        "edge_type": edge_type,
        "edge_colors": edge_colors,
        "edge_linestyle": edge_ls,
        "weights": edge_weights,
        "lookup": seg_idx_lookup,
        "node_colors": node_colors,
        "node_labels": node_labels,
    }


def sort_existing_pos(
    pos: dict,
    edges_data: dict,
):
    """
    Sort the nodes in the existing pos dictionary by the weigths of the connections.
    Modifies the pos dictionary in place.
    """
    weights_list = edges_data["weights"]
    edges_list = edges_data["edges"]

    # sort the weights list and save the indices
    sorted_weights_idx = np.argsort(weights_list)
    # sort the edges list by the indices
    sorted_edges = [edges_list[idx] for idx in sorted_weights_idx[::-1]]
    # Get the list of nodes in the order of the sorted edges, taking only the second node in the tuple
    sorted_nodes = [edge[1] for edge in sorted_edges]
    # remove duplicates from the sorted edges list, keeping only the first value
    unique_nodes = []
    for node in sorted_nodes:
        if node not in unique_nodes:
            unique_nodes.append(node)
    # Get the list of positions in the pos dictionary as a list of tuples
    pos_list = list(pos.values())

    new_pos = {
        node: position for node, position in zip(unique_nodes, pos_list)
    }
    pos.update(new_pos)
    return pos


def add_edge_legend(
    ax: plt.Axes,
    normalized_weights: list,
    color_list: list,
    arrow_norm: float,
):
    """
    Add a legend to the plot with the edge weights.
    """
    color = Counter(color_list).most_common(1)[0][0]
    lines = []
    edges_weight_list = sorted(normalized_weights)
    for _, width in enumerate(edges_weight_list):
        lines.append(Line2D([], [], linewidth=width, color=color))
    # keep only 3 legend entries, evenly spaced, including the first and last
    if len(edges_weight_list) > 3:
        edges_weight_list = [
            edges_weight_list[0],
            edges_weight_list[len(edges_weight_list) // 2],
            edges_weight_list[-1],
        ]
        lines = [lines[0], lines[len(lines) // 2], lines[-1]]

    edges_weight_list = [
        f"{int(weight/arrow_norm)}" for weight in edges_weight_list
    ]
    ax.legend(
        lines,
        edges_weight_list,
        bbox_to_anchor=(0.1, 0.25),
        frameon=False,
    )
    return ax


def plot_downstream_network(
    network_1: dict,  # direct connection ref -> DNs
    network_2: dict,  # connections down DNs -> down DNs
    neurons_of_interest: dict,
    node_size: int = 300,
    ax: plt.Axes = None,
    first_layer_opaque: bool = False,
    other_layer_visible: bool = False,
    arrow_norm: float = 0.1,
) -> dict:
    """
    Make a circular plot where the refernce neurons are in the center, and their
    downstream partners are in the outer circle.
    Visualise the connections between the downstream partners to look at
    layer interconnectivity.
    """
    if first_layer_opaque:
        alpha_1 = 1
        alpha_2 = 0.1 if other_layer_visible else 0
    else:
        alpha_1 = 0.1 if other_layer_visible else 0
        alpha_2 = 1

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8), dpi=120)

    if len(network_1["edges"]) == 0:  # no direct connections
        pos = {}
        for node in neurons_of_interest.keys():
            pos[node] = np.array([0, 0])
            G = nx.MultiDiGraph()
            G.add_nodes_from(network_2["lookup"].values())
            nx.draw(
                G,
                ax=ax,
            )
        return pos

    G = nx.MultiDiGraph()
    G.add_edges_from(network_2["edges"])
    G.add_edges_from(network_1["edges"])
    # define G_down as the subgraph of G that contains all the nodes of G
    # except those in the neurons_of_interest
    G_down = G.copy()
    G_down.remove_nodes_from(neurons_of_interest.keys())

    pos = nx.circular_layout(G_down)
    pos = sort_existing_pos(pos, network_1)
    for (
        node
    ) in neurons_of_interest.keys():  # define a position for each neuron
        pos[node] = np.array([0, 0])

    # plot direct connections
    normalized_weights = network_1["weights"] * arrow_norm
    G.remove_edges_from(network_2["edges"])

    # Add the labels for the edges
    ax = add_edge_legend(
        ax, normalized_weights, network_1["edge_colors"], arrow_norm
    )

    nx.draw(
        G,
        pos,
        nodelist=network_1["lookup"].values(),
        with_labels=False,
        # labels=network_1["node_labels"],
        width=normalized_weights,
        alpha=alpha_1,
        node_size=node_size,
        node_color=network_1["node_colors"],
        edge_color=network_1["edge_colors"],
        style=network_1["edge_linestyle"],
        # font_size=5,
        # font_color="black",
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    if len(network_2["edges"]) == 0:
        normalized_weights = []
    else:
        normalized_weights = network_2["weights"] * arrow_norm  # (
        #   network_2["weights"] / network_2["weights"].max()
        # ) * 5
    # plot downstream interconnections
    G.add_edges_from(network_2["edges"])
    G.remove_edges_from(network_1["edges"])

    nx.draw(
        G,
        pos,
        nodelist=network_2["lookup"].values(),
        # with_labels=True,
        # labels=network_2["node_labels"],
        width=normalized_weights,
        alpha=alpha_2,
        node_size=node_size,
        node_color=network_2["node_colors"],
        edge_color=network_2["edge_colors"],
        style=network_2["edge_linestyle"],
        connectionstyle="arc3,rad=0.1",
        # font_size=5,
        # font_color="black",
        ax=ax,
    )

    return pos


def get_downstream_specs(
    connectivity_df: pd.DataFrame,
    neurons_of_interest: dict,
    list_dns: list[int] = None,
    feedback_layer_1: bool = False,
    feedback_layer_2: bool = False,
    named_dns: dict = None,
) -> dict:
    """
    Extract the edges from ref neurons to their downtream partners,
    and their mutual connections.

    Inputs:
        connectivity_df: pd.DataFrame
            connectivity table
        neurons_of_interest: Dict
            dict of neurons of interest to use as first layer
        list_dns: list[int]
            list of downstream neurons among which to look for connections
        feedback: bool
            whether to include feedback connections to the first layer

    Returns:
        network1: Dict
            edges from ref neurons to their downstream partners
        network2: Dict
            edges between downstream partners and back to the ref neuron


    """
    pd.options.mode.chained_assignment = None

    if list_dns is None:
        logged_neurons = set(connectivity_df["pre_root_id"]).union(
            set(connectivity_df["post_root_id"])
        )
        list_dns = logged_neurons

    mask = (
        (connectivity_df["pre_root_id"].isin(neurons_of_interest.keys()))
        & (connectivity_df["post_root_id"].isin(list_dns))
        & (connectivity_df["syn_count"] > 5)
        & (connectivity_df["nt_type"].isin(["ACH", "GABA", "GLUT"]))
    )

    layer1_df = connectivity_df[mask]

    # Collapse the connectivity table to sum the number of synapses between
    # the same neurons. Here the input group 'neurons_of_interest' is understood
    # as one unit and therefore connections are also summed

    layer1_df["pre_root_id"] = layer1_df[
        "pre_root_id"
    ].replace(  # agregate neurons of interest
        neurons_of_interest.keys(), next(iter(neurons_of_interest))
    )
    layer1_df = (  # sum synapses between neurons of interest
        layer1_df.groupby(["pre_root_id", "post_root_id", "nt_type"])
        .agg({"syn_count": "sum"})
        .reset_index()
    )

    list_layer_2_dns = list(
        set(layer1_df["post_root_id"].unique())
        - set(neurons_of_interest.keys())
    )

    if named_dns is not None:
        table_connected_dns = {
            root_id: (
                {"root_id": root_id, "color": "k", "name": "DN"}
                if root_id not in named_dns.keys()
                else named_dns[root_id]
            )
            for root_id in list_layer_2_dns
        }
    else:
        table_connected_dns = {
            root_id: {"root_id": root_id, "color": "k", "name": "DN"}
            for root_id in list_layer_2_dns
        }

    plotted_neurons = {**table_connected_dns, **neurons_of_interest}

    downstream_mask = (
        (connectivity_df["pre_root_id"].isin(list_layer_2_dns))
        & (connectivity_df["post_root_id"].isin(list_layer_2_dns))
        & (connectivity_df["syn_count"] > 5)
        & (connectivity_df["nt_type"].isin(["ACH", "GABA", "GLUT"]))
    )

    connected_down = connectivity_df[mask]["post_root_id"].unique()
    feedback_mask = (connectivity_df["pre_root_id"].isin(connected_down)) & (
        connectivity_df["post_root_id"].isin(neurons_of_interest.keys())
        & (~connectivity_df["pre_root_id"].isin(neurons_of_interest.keys()))
        & (connectivity_df["syn_count"] > 5)
    )
    if feedback_layer_2:
        downstream_mask = downstream_mask | feedback_mask
    if feedback_layer_1:
        feedback_df = connectivity_df[feedback_mask]
        layer1_df = pd.concat([layer1_df, feedback_df])

    layer2_df = connectivity_df[downstream_mask]
    layer2_df["post_root_id"] = layer2_df["post_root_id"].replace(
        neurons_of_interest.keys(), next(iter(neurons_of_interest))
    )

    layer2_df = (
        layer2_df.groupby(["pre_root_id", "post_root_id", "nt_type"])
        .agg({"syn_count": "sum"})
        .reset_index()
    )

    network_1 = get_network_specs(
        layer1_df, neurons_of_interest=plotted_neurons, root_id_indexing=True
    )
    network_2 = get_network_specs(
        layer2_df,
        neurons_of_interest=plotted_neurons,  # table_connected_dns,
        root_id_indexing=True,
    )
    pd.options.mode.chained_assignment = "warn"
    return network_1, network_2


def plot_network(
    network_specs: dict,
    ax: plt.Axes = None,
    pos: dict = None,
    node_size: int = 300,
    title: str = "",
    export_path: str = None,
    activate_click: bool = False,
) -> dict:
    """Plots the network using the network specs and
    returns the positions of the nodes.

    Parameters
    ----------
    network_specs : Dict
        Dictionary containing the network characteristics.
    ax : plt.Axes, optional
        Ax that the network will be plotted on, by default None
    node_size : int, optional
        Size of the network nodes, by default 300
    title : str, optional
        Title of the figure, by default ''
    export_path : str, optional
        Path to save the figure, by default None
    activate_click : bool, optional
        Prints out the click location on the figure, by default False

    Returns
    -------
    Dict
        Dictionary containing node ID as keys and positions of nodes
        on the figure as values.


    Example Usage:
    >>> connectivity_df = connectivity_df[
        connectivity_df['pre_root_id'] == 720575940624319124
        )
    >>> neurons_of_interest = {
        720575940624319124: {'name': 'aDN1_R', 'color': 'seagreen'}
        }
    >>> network_specs = get_network_specs(
        connectivity_df, neurons_of_interest=neurons_of_interes
        t)
    >>> plot_network(
            network_specs, title='Connectivity of aDN1 right',
            export_path='./adn1_conn.png'
        )
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    # Create graph from adjacency matrix
    G = nx.MultiDiGraph()

    G.add_edges_from(network_specs["edges"])

    # These are algorithms to set the positions. Otherwise, you can set  the position by having:
    # "pos = {0:(0,0), 1:(1,1), 2:(2,2)}"
    if pos is None:
        pos = nx.kamada_kawai_layout(G, scale=1)

    # Normalize the edge width of the network
    normalized_weights = (
        network_specs["weights"] / network_specs["weights"].max()
    ) * 5

    # Plot graph
    nx.draw(
        G,
        pos,
        nodelist=network_specs["lookup"].values(),
        with_labels=True,
        labels=network_specs["node_labels"],
        width=normalized_weights,
        alpha=0.75,
        node_size=node_size,
        node_color=network_specs["node_colors"],
        edge_color=network_specs["edge_colors"],
        style=network_specs["edge_linestyle"],
        connectionstyle="arc3,rad=0.1",
        font_size=5,
        font_color="black",
        ax=ax,
    )

    if activate_click:
        clicked = fig.canvas.mpl_connect("button_press_event", onclick)

    if title:
        ax.set_title(title)

    if export_path is not None:
        plt.savefig(export_path)

    return pos


def sort_left_right(
    rid_list: list[int],  # contains rids of the neurons
    graph: nx.MultiDiGraph,
    x_list: list[float],  # contains the x positions of the neurons
) -> list[float]:
    """
    Sorts the nodes in the x axis such that left and rights neurons are
    on the left and right side of the graph, respectively.
    """
    left = [
        i for i, x in enumerate(rid_list) if ("L" in graph.nodes[x]["label"])
    ]
    right = [
        i for i, x in enumerate(rid_list) if ("R" in graph.nodes[x]["label"])
    ]
    middle = set(np.arange(len(x_list))) - set(left) - set(right)
    sorted_indices = left + list(middle) + right

    tmp_x_list = x_list.copy()
    for i, idx in enumerate(sorted_indices):
        x_list[idx] = tmp_x_list[
            i
        ]  # inverse transformation of the sorting permutation
    return sorted_indices, x_list


def sort_by_cluster(
    layer_rids_list: list[int],  # rids to sort
    graph: nx.MultiDiGraph,  # connectivity data
    x_list: list[float],  # x positions to reattribute,
    ref_rids: list[int],  # rids of neurons in the next layer
    ref_ordering: list[int],  # order of the neurons in the next layer
    inverse: bool = False,  # sort from right to left
) -> list[float]:
    """
    sort the neurons in the x axis such that neurons in the same cluster
    connect to the same neuron on the closest layer.
    """
    if inverse:
        x_list = [-x for x in x_list]
    sorted_indices = []
    for i, idx in enumerate(ref_ordering):
        ref_neuron = ref_rids[idx]
        input_nodes = list(graph.predecessors(ref_neuron))
        layer_input_nodes = [x for x in layer_rids_list if x in input_nodes]
        output_nodes = list(graph.successors(ref_neuron))
        layer_output_nodes = [x for x in layer_rids_list if x in output_nodes]
        added_input_indices = [
            layer_rids_list.index(x)
            for x in layer_rids_list
            if (x in layer_input_nodes)
            and (layer_rids_list.index(x) not in sorted_indices)
        ]
        sorted_indices.extend(added_input_indices)
        added_output_indices = [
            layer_rids_list.index(x)
            for x in layer_rids_list
            if (x in layer_output_nodes)
            and (layer_rids_list.index(x) not in sorted_indices)
        ]
        sorted_indices.extend(added_output_indices)

    unconnected = set(np.arange(len(x_list))) - set(sorted_indices)
    sorted_indices.extend(list(unconnected))
    tmp_x_list = x_list.copy()
    for i, idx in enumerate(sorted_indices):
        x_list[idx] = tmp_x_list[i]
    return x_list


def define_color_based_layers(
    G: nx.MultiDiGraph,
) -> dict:
    """Defines the positions of the classes in the network."""

    # list the existing colors in the graph
    colors = list(set(nx.get_node_attributes(G, "color").values()))
    network_height = len(colors)

    # for each color, get a list of nodes with that color
    color_nodes = {
        color: [
            node for node, data in G.nodes(data=True) if data["color"] == color
        ]
        for color in colors
    }

    # create a dictionary of y postions for each color
    y_positions = {
        color: i + 1
        for i, color in enumerate(set(colors) - {"k", "lightgrey"})
    }
    y_positions["k"] = 0
    y_positions["lightgrey"] = network_height - 1

    # create a dictionary of x positions list for each color
    x_positions = {
        color: np.linspace(
            -0.6 * network_height,
            0.6 * network_height,
            len(color_nodes[color]),
        )
        for color in set(colors) - {"k", "lightgrey"}
    }
    x_positions["k"] = np.linspace(
        -1 * network_height, 1 * network_height, len(color_nodes["k"])
    )
    x_positions["lightgrey"] = np.linspace(
        -1 * network_height, 1 * network_height, len(color_nodes["lightgrey"])
    )
    # for each node in each color set, move L and R neurons to their respective
    # sides
    ordering = {}
    for color in set(colors) - {"k", "lightgrey"}:
        ordering[color], x_positions[color] = sort_left_right(
            color_nodes[color], G, x_positions[color]
        )

    # sort grey and black neurons by clustering based on the connections with
    # the closest layer
    color_near_k = [color for color in colors if y_positions[color] == 1][0]
    color_near_grey = [
        color for color in colors if y_positions[color] == network_height - 2
    ][0]
    x_positions["k"] = sort_by_cluster(
        color_nodes["k"],
        G,
        x_positions["k"],
        color_nodes[color_near_k],
        ordering[color_near_k],
        inverse=network_height % 2 == 0,
    )
    x_positions["lightgrey"] = sort_by_cluster(
        color_nodes["lightgrey"],
        G,
        x_positions["lightgrey"],
        color_nodes[color_near_grey],
        ordering[color_near_grey],
    )

    # create a dictionary of [x,y] positions for each node
    pos = {
        node: [x_positions[color][i], y_positions[color]]
        for color in colors
        for i, node in enumerate(color_nodes[color])
    }
    return pos


def class_based_plotting(
    network_specs: dict,
    predefined_function: str = None,
    node_size: int = 300,
    ax: plt.Axes = None,
    export_path: str = None,
) -> dict:
    """Plots the network with flexible placement and
    returns the positions of the nodes."""

    # Create graph from adjacency matrix
    G = nx.MultiDiGraph()

    G.add_edges_from(network_specs["edges"])
    node_data = [
        (
            node,
            {
                "color": network_specs["node_colors"][i],
                "label": list(network_specs["node_labels"].values())[i],
            },
        )
        for i, node in enumerate(list(network_specs["lookup"].keys()))
    ]
    G.add_nodes_from(node_data)
    # These are algorithms to set the positions.
    match predefined_function:
        case None:
            pos = define_color_based_layers(G)
        case "multipartite":
            pos = nx.multipartite_layout(
                G, subset_key="color", scale=1, align="horizontal"
            )
        case "kamada_kawai":
            pos = nx.kamada_kawai_layout(G, scale=1)
        case "spectral":
            pos = nx.spectral_layout(G, scale=1)
        case "spring":
            pos = nx.spring_layout(G, scale=1)
        case "shell":
            pos = nx.shell_layout(G, scale=1)
        case "circular":
            pos = nx.circular_layout(G, scale=1)

    # Normalize the edge width of the network
    normalized_weights = (
        network_specs["weights"] / network_specs["weights"].max()
    ) * 5

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    # Plot graph
    nx.draw(
        G,
        pos,
        nodelist=network_specs["lookup"].values(),
        with_labels=True,
        labels=network_specs["node_labels"],
        width=normalized_weights,
        alpha=0.75,
        node_size=node_size,
        node_color=network_specs["node_colors"],
        edge_color=network_specs["edge_colors"],
        style=network_specs["edge_linestyle"],
        connectionstyle="arc3,rad=0.1",
        font_size=5,
        font_color="black",
        ax=ax,
    )

    if export_path is not None:
        plt.savefig(export_path)

    return pos


def define_graph_attributes(
    connection_df: pd.DataFrame,
    named_dns: dict = None,
):
    """
    Build the graph with relevant nodes and edges, add the basic attributes.
    positions are not defined.
    """

    # extract relevant information for building the graph
    network_specs = get_network_specs(
        connectivity_df=connection_df,
        neurons_of_interest=named_dns,
        root_id_indexing=False,  # use renumbering to parse dense arrays
    )
    normalised_weights = (
        network_specs["weights"] / network_specs["weights"].max()
    ) * 5

    # construct the graph
    G = nx.DiGraph()
    graph_data = [
        (
            u,
            v,
            {
                "weight": w,
                "edge_colors": c,
                "edge_linestyle": l,
            },
        )
        for (u, v), w, c, l in zip(
            network_specs["edges"],
            normalised_weights,
            network_specs["edge_colors"],
            network_specs["edge_linestyle"],
        )
    ]
    G.add_edges_from(graph_data)
    for node in G.nodes:
        G.nodes[node]["node_color"] = network_specs["node_colors"][node]
        G.nodes[node]["node_label"] = network_specs["node_labels"][node]
        G.nodes[node]["root_id"] = list(network_specs["lookup"].keys())[node]

    return G, network_specs


def plot_network_diffusion(
    G: nx.DiGraph,
    start_neurons: int,
    node_size: int = 10,
    ax: plt.Axes = None,
):
    """
    Represent the diffusion of the network from the start neurons to the
    downstream neurons, represented as a network with each hop in a new layer.
    Neurons are represented only once, back connection can exist.

    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    # define the positions of the nodes
    for i, start_neuron in enumerate(start_neurons):
        # find the index given to the node where the lookup dictionary contains the
        # start neuron
        G_sub = G.copy()
        start_node = [
            n for n in G.nodes if G.nodes[n]["root_id"] == start_neuron
        ][0]
        G_sub.nodes[start_node][f"pos_{i}"] = (0, 0)

        touched_nodes = [start_node]
        downstream_nodes = list(G_sub.successors(start_node))
        row = 0
        # while there exist nodes in the downstream nodes that are not in the touched nodes
        while len(set(downstream_nodes) - set(touched_nodes)) > 0:
            row -= 1
            col = 0
            new_downstream_nodes = []
            for node in downstream_nodes:
                if node not in touched_nodes:
                    touched_nodes.append(node)
                    G_sub.nodes[node][f"pos_{i}"] = ((-1) ** i * col, row)
                    new_downstream_nodes += list(G_sub.successors(node))
                    col += 1
            downstream_nodes = list(
                set(new_downstream_nodes) - set(touched_nodes)
            )

        # define a subgraph restricted to the touched nodes
        G_sub = G_sub.subgraph(touched_nodes)

        # plot the original network
        # for axis in ax:
        args = {
            "edge_color": nx.get_edge_attributes(
                G_sub, "edge_colors"
            ).values(),
            "style": list(
                nx.get_edge_attributes(G_sub, "edge_linestyle").values()
            ),
            "connectionstyle": "arc3,rad=0.1",
        }

        nx.draw(
            G_sub,
            pos=nx.get_node_attributes(G_sub, f"pos_{i}"),
            nodelist=list(G_sub.nodes),
            with_labels=False,
            labels=nx.get_node_attributes(G_sub, "node_label"),
            alpha=1,
            node_size=node_size,
            node_color=nx.get_node_attributes(G_sub, "node_color").values(),
            font_size=5,
            font_color="black",
            ax=ax,
            **args,
        )

    return ax


def assign_layer_positions(G: nx.DiGraph, start_node: int):
    """
    Assign the positions of the nodes in the graph, based on the start node.
    Each layer corresponds to the neurons that are downstream of the previous
    layer.
    """
    G_sub = G.copy()
    G_sub.nodes[start_node]["pos"] = (0, 0)

    touched_nodes = [start_node]
    downstream_nodes = list(G_sub.successors(start_node))
    row = 0
    # while there exist nodes in the downstream nodes that are not in the touched nodes
    while len(set(downstream_nodes) - set(touched_nodes)) > 0:
        row += 1
        col = 0
        new_downstream_nodes = []
        for node in downstream_nodes:
            if node not in touched_nodes:
                touched_nodes.append(node)
                G_sub.nodes[node]["pos"] = (col, row)
                new_downstream_nodes += list(G_sub.successors(node))
                col += 1
        downstream_nodes = list(set(new_downstream_nodes) - set(touched_nodes))
    G_sub = G_sub.subgraph(touched_nodes)
    return G_sub


def add_layer_reached_by_neuron(
    G: nx.DiGraph, start_node: int, df_ref: pd.DataFrame
):
    """
    Add the layer reached by each neuron to the dataframe.
    """

    G_sub = assign_layer_positions(G, start_node)

    ref_data = [
        (i, len([n for n in G_sub.nodes if G_sub.nodes[n]["pos"][1] == i]))
        for i in range(
            max(
                nx.get_node_attributes(G_sub, "pos").values(),
                key=lambda x: x[1],
            )[1]
        )
    ]
    local_layer_counts = pd.DataFrame(
        ref_data,
        columns=["layer", "layer_count"],
    )
    if not df_ref.empty:
        local_layer_counts.index = np.arange(
            df_ref.index[-1] + 1,
            df_ref.index[-1] + 1 + len(local_layer_counts),
        )

    df_ref = pd.concat(
        [df_ref, local_layer_counts],
        axis=0,
        join="outer",
    )
    return df_ref, local_layer_counts


def plot_density_diffusion(
    G: nx.DiGraph,
    start_neurons: list[int],
    show_all_traces: bool = False,
    ax: plt.Axes = None,
    neuron_color: str = "black",
):
    """
    Represent the diffusion of the network from the start neurons to the
    downstream neurons, represented as a density plot with each hop in a new layer.
    Neurons are represented only once, back connection can exist.
    """

    # find the index given to the node where the lookup dictionary contains the
    # start neuron
    df_ref = pd.DataFrame()
    for start_neuron in tqdm(start_neurons):
        start_node = [
            n for n in G.nodes if G.nodes[n]["root_id"] == start_neuron
        ][0]
        df_ref, _ = add_layer_reached_by_neuron(G, start_node, df_ref)
    df_ref["DN"] = "reference neuron"

    # define the reference for the graph: the layers reached starting from
    # any random neuron
    layer_count_df = pd.DataFrame()
    if show_all_traces:
        _, ax0 = plt.subplots(figsize=(10, 8), dpi=120)
    for node in tqdm(G.nodes):
        # add to layer_count_df a column with the name being the node and the
        # values being the number of nodes in each layer
        # layer data
        layer_count_df, local_layer_counts = add_layer_reached_by_neuron(
            G, node, layer_count_df
        )

        if show_all_traces:
            ax0.plot(
                local_layer_counts["layer"],
                local_layer_counts["layer_count"],
                alpha=0.1,
                color="k",
                linestyle="-",
            )
    layer_count_df["DN"] = "average DN-DN connectivity"

    df_ref = pd.concat([df_ref, layer_count_df], axis=0, join="outer")

    # plot the result as a violin plot
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8), dpi=120)
    sns.set_palette(sns.color_palette([neuron_color, "grey"]))
    sns.barplot(
        data=df_ref,
        x="layer_count",
        y="layer",
        hue="DN",
        errorbar=("ci", 95),
        ax=ax,
        orient="h",
    )

    sns.despine(left=True, bottom=True)
    ax.set_xlabel("Number of DNs per layer")
    ax.set_ylabel("Layer")
    ax.legend()

    return ax


def plot_graph_diffusion(
    connection_df: pd.DataFrame,
    start_neurons: int,
    named_dns: dict = None,
    node_size: int = 10,
    neuron_color: str = "black",
    drawing_style: str = "network",  # "density",
    ax: plt.Axes = None,
):
    """
    Plots the diffusion of a signal through a network over a given depth.
    """
    # Plot graph
    graph, _ = define_graph_attributes(
        connection_df,
        named_dns,
    )

    if drawing_style == "network":
        ax = plot_network_diffusion(
            graph,
            start_neurons,
            node_size=node_size,
            ax=ax,
        )
    elif drawing_style == "density":
        ax = plot_density_diffusion(
            graph,
            start_neurons,
            ax=ax,
            neuron_color=neuron_color,
        )

    return ax


def plot_graph_latent_axis(
    connection_df: pd.DataFrame,
    depth: int = 1,
    list_relevant_neurons: list[int] = None,
    named_dns: dict = None,
    node_size: int = 100,
    use_laplacian: bool = True,
    pos: dict = None,
    draw_graph: bool = True,
) -> nx.MultiDiGraph:
    """
    Plot the neurons in a square grid.
    Represent the latent axis as a color gradient.
    """

    # extract relevant information for building the graph
    G, network_specs = define_graph_attributes(
        connection_df=connection_df,
        list_relevant_neurons=list_relevant_neurons,
        named_dns=named_dns,
    )

    # compute the Laplacian of the graph
    if use_laplacian:
        L = sparse.csr_matrix(nx.directed_laplacian_matrix(G))
    else:
        L = nx.to_scipy_sparse_array(
            G, format="csr", weight="weight", dtype=float
        )

    # SVD decomposition of the Laplacian
    U, S, Vt = svds(L, k=depth, return_singular_vectors="vh")

    # get the columns of Vt to color code the nodes
    for d in range(depth):
        # add the feature L_d to the nodes of G
        nx.set_node_attributes(
            G,
            {node: {f"L_{d}": Vt[d, node]} for node in G.nodes},
        )

    if draw_graph:
        # Plot graph
        fig, ax = plt.subplots(1, depth, figsize=(5 * depth, 5), dpi=120)

        # plot the network with the latent axis as color gradient
        for d in range(depth):
            # map the values of Vt[:,d] to a color gradient from blue to red,
            # with 0 being white, blue the 5th percentile and red the 95th
            # percentile and update network_specs["node_colors"]
            color_latent_dim = [
                plt.cm.RdBu(
                    (Vt[d, node] - np.percentile(Vt[d, :], 5))
                    / (
                        np.percentile(Vt[d, :], 95)
                        - np.percentile(Vt[d, :], 5)
                    )
                )
                for node in network_specs["lookup"].values()
            ]

            nx.draw(
                G,
                pos,
                nodelist=network_specs["lookup"].values(),
                with_labels=True,
                labels=network_specs["node_labels"],
                alpha=1,
                node_size=node_size,
                node_color=color_latent_dim,
                edge_color="white",
                width=0,
                font_size=5,
                font_color="black",
                ax=ax[d],
            )
            ax[d].set_title(f"Depth {d}")
    if draw_graph:
        return G, fig
    else:
        return G


def cluster_graph_louvain(
    graph: nx.DiGraph,
    connection_type: str = "excitatory",
    visualise_matrix: bool = False,
):
    """
    Cluster the graph using the Louvain algorithm.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to cluster.
    connection_type : str
        The type of connection to use for clustering. Must be one of
        - 'excitatory': excitatory connections are given positive weights, inhibitory
        connections are given negative weights.
        - 'inhibitory': excitatory connections are given negative weights, inhibitory
        connections are given positive weights.
        - 'absolute': all connections are given positive weights.
    visualise_matrix : bool
        Whether to visualise the matrix of weights.

    Returns
    -------
    cluster_list : list[list[int]]
        The list of clusters. Each cluster is a list of node IDs.
    """

    # Create a new graph with the same nodes, symmetric
    graph_clustered = nx.Graph()
    graph_clustered.add_nodes_from(graph.nodes)

    # Add edges with the processed weights
    for u, v, data in graph.edges(data=True):
        if connection_type == "excitatory":
            weight = data["weight"]
        elif connection_type == "inhibitory":
            weight = -data["weight"]
        elif connection_type == "absolute":
            weight = abs(data["weight"])
        graph_clustered.add_edge(u, v, weight=weight)

    # make a figure shwoing the matrix of the weights from both graphs
    if visualise_matrix:
        # make a two panel figure
        _, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
        dn_mat = nx.to_scipy_sparse_array(graph)
        ax[0].spy(dn_mat, markersize=0.05)
        dn_mat_clustered = nx.to_scipy_sparse_array(graph_clustered)
        ax[1].spy(dn_mat_clustered, markersize=0.05)
        plt.show()

    # Cluster the graph
    modules_louvain = nx.algorithms.community.louvain_communities(
        graph_clustered
    )

    # visualise the corresponding matrix
    if visualise_matrix:
        modules_louvain = sorted(modules_louvain, key=len, reverse=True)

        ordering_louvain = [x for x in modules_louvain[0]]
        for i in range(len(modules_louvain) - 1):
            ordering_louvain.extend([x for x in modules_louvain[i + 1]])
        dn_mat_louvain = nx.to_scipy_sparse_array(graph)
        dn_mat_louvain = dn_mat_louvain[ordering_louvain, :]
        dn_mat_louvain = dn_mat_louvain[:, ordering_louvain]
        plt.spy(dn_mat_louvain, markersize=0.05, color="gray")

    return modules_louvain


def plot_indivual_community_graphs(
    H: nx.DiGraph,
    ax: plt.Axes,
):
    """
    Plot a connection graph for a community, i.e. a subset of the nodes of the original graph.
    """

    if nx.number_of_edges(H) > 0:
        widths = (
            [H.edges[e]["weight"] for e in H.edges]
            / max([H.edges[e]["weight"] for e in H.edges])
            * 5
        )
    else:
        widths = 1
    nx.draw(
        H,
        ax=ax,
        pos=nx.spring_layout(H),
        with_labels=True,
        labels={n: H.nodes[n]["name"] for n in H.nodes},
        node_color=[H.nodes[n]["color"] for n in H.nodes],
        width=widths,
        edge_color=[H.edges[e]["color"] for e in H.edges],
        node_size=10,
    )


def plot_communities_graphs(
    G: nx.DiGraph,
    modules_louvain: list[list[int]],
    ax: plt.Axes = None,
    community_nbr: list[int] = [],
    plot_only_known_neurons: bool = False,
):
    """
    Plot a connection graph for each community separately using the
    nx.draw_networkx function

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph of the neurons
    modules_louvain : list[list[int]]
        The list of communities
    ax : plt.Axes, optional
        The axes to plot on, by default None
    community_nbr : list[int], optional
        The list of communities to plot, by default [] i.e. all communities
    plot_only_known_neurons : bool, optional
        Whether to plot only the neurons that have a genetic match, by default False

    Returns
    -------
    Modifies the ax object
    """
    nb_modules = 0
    for community in modules_louvain:
        if len(community) > 1:
            nb_modules += 1
    if community_nbr == []:
        list_modules = list(range(nb_modules))
    else:
        list_modules = community_nbr

    if ax is None:
        fig, ax = plt.subplots(
            1, len(list_modules), figsize=(len(list_modules) * 5, 5)
        )

    for d, community in enumerate([modules_louvain[i] for i in list_modules]):
        print(
            f"Community {d+1} has {len(community)} neurons"
            f" and {nx.number_of_edges(G.subgraph(community))} edges"
        )
        # print the list of node names that are not 'DN' in the community
        print(
            "The matched DNs in this community are: ",
            [
                G.nodes[n]["name"]
                for n in community
                if G.nodes[n]["name"] != "DN"
            ],
        )
        if plot_only_known_neurons:
            community = [n for n in community if G.nodes[n]["name"] != "DN"]
        G_sub = G.subgraph(community)
        H = G_sub.to_directed()
        if len(list_modules) > 1:
            ax[d].set_title(f"Community {d+1}")
            plot_indivual_community_graphs(H, ax[d])
        else:
            ax.set_title(f"Community {d+1}")
            plot_indivual_community_graphs(H, ax)
    return ax


def make_nice_spines(ax, linewidth=2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 2 * linewidth))
    ax.spines["bottom"].set_position(("outward", 2 * linewidth))
    ax.tick_params(width=linewidth)
    ax.tick_params(length=2.5 * linewidth)
    ax.tick_params(labelsize=16)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)
    return ax


def make_axis_disappear(ax):
    sides = ["top", "left", "right", "bottom"]
    for side in sides:
        ax.spines[side].set_visible(False)
        ax.spines[side].set_linewidth(0)
        ax.set_xticks([])
        ax.set_yticks([])
