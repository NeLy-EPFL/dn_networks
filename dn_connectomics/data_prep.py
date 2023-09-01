"""
This script sonstructs the following from CSV files from the Codex data dump:
1. Graph representations of:
     (i)   The full FlyWire dataset;
     (ii)  The subset of it containing only central, descending, and ascending
           neurons;
     (iii) The subset of it containing only descending neurons and GNG neurons;
     (iv)  The subset of it containing only descending neurons.
   Each of the graph is saved as a pickle file of a dictionary containing:
     (i)   'nx_graph': The NetworkX DiGraph representation;
     (ii)  'mat_norm': The SciPy sparse matrix representation (CSR format) of
                       the normalized weights (see below);
     (iii) 'mat_unnorm': The SciPy sparse  matrix representation (CSR format)
                         of the unormalized weights;
     (iv)  'lookup': A Pandas DataFrame with columns 'index' and 'root_id' that
                     maps the index of each neuron in the matrix representation
                     to the root ID of the neuron.
2. The node and edge attribute tables in the form of a pandas DataFrame, each
   saved as a pickle file.

Prerequisites
=============
1. CSV files from the Codex data dump must be downloaded to `codex_dump_dir`
   first.
2. Under `codex_dump_dir`, there should also be a file `manual_cell_names.csv`
   containing manually supplied cell names. Please run this script first to get
   a list of root IDs that need to be manually checked (it will be printed in
   the log). Then, manually supply the cell names in the CSV file, and run this
   script again. 

Notes
=====
The edges have the following attributes:
1. 'syn_count': Number of synapses between the two neurons
2. 'nt_type': Neurotransmitter type; note that this is not the same as nt_type
    in the "Connections" table from Codex; instead, it is the neurotransmitter
    type of the presynaptic partner (taken from the "Neurons" table) so that
    each neuron only has one type of output by neurotransmitter.
3. 'eff_weight': Effective weight of the synapse, defined as the synapse count
    multiplied by the sign of the neurotransmitter.
4. 'syn_count_norm' and 'eff_weight_norm': Normalized versions of 'syn_count'
    and 'eff_weight' calculated by dividing the unnormalized values by the total
    number of incoming synapses of the postsynaptic neuron.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from functools import reduce
import os

import params


# Configs -- change these as needed
data_dir = params.RAW_DATA_DIR
codex_dump_dir = params.CODEX_DUMP_DIR
graphs_dir = params.GRAPHS_DIR

# Read Codex dump
neurons = pd.read_csv(codex_dump_dir / "neurons.csv")
morphology_clusters = pd.read_csv(codex_dump_dir / "morphology_clusters.csv")
connectivity_clusters = pd.read_csv(
    codex_dump_dir / "connectivity_clusters.csv"
)
classification = pd.read_csv(codex_dump_dir / "classification.csv")
cell_stats = pd.read_csv(codex_dump_dir / "cell_stats.csv")
labels = pd.read_csv(codex_dump_dir / "labels.csv")
connections = pd.read_csv(codex_dump_dir / "connections.csv")
connectivity_tags = pd.read_csv(codex_dump_dir / "connectivity_tags.csv")
manual_cell_names = pd.read_csv(codex_dump_dir / "manual_cell_names.csv")

# Select Katharina Eichler's DN labels
our_friend = "Katharina Eichler"
helpful_tag = "Part of comprehensive neck connective tracing"
ke_labels = labels[
    (labels["user_name"] == our_friend)
    & (labels["label"].str.contains(helpful_tag))
].copy()
ke_labels["dn_label"] = [x.split(";")[0] for x in ke_labels["label"]]
ke_labels = ke_labels[ke_labels["dn_label"].str.startswith("DN")]
ke_labels = ke_labels[["root_id", "dn_label"]]

# RegEx patterns for extracting DN names
regex_patterns = {
    "strict": r"(DN[a-z]\d\d)",
    "loose": r"(DN[a-z]+\d+)",
    "mult": r"(DN[a-z]\d\d\/\d\d)",
}

# Infer DN names based on hemibrain types, cell types, and human-made labels
source_columns = {
    "hemibrain_type": classification.set_index("root_id")["hemibrain_type"],
    "cell_type": classification.set_index("root_id")["cell_type"],
    "label": labels.set_index("root_id")["label"],
}
dn_ids = classification[classification["super_class"] == "descending"][
    "root_id"
]
inferred_names_dfs = {}
for col_name, column in source_columns.items():
    for pattern_name, pattern in regex_patterns.items():
        extracted_df = column.str.extractall(pattern)
        inferred_names_dfs[f"{col_name}_{pattern_name}"] = extracted_df

# Go through these labels and make a dataframe containing all inferred names
for name, df in inferred_names_dfs.items():
    assert len(df.columns) == 1
    df.rename(columns={0: name}, inplace=True)

    # Make sure the match is unique. If not, mark it as "TO_CHECK"
    match_count = df.reset_index().groupby("root_id")[name].count()
    df = df.reset_index().set_index(["root_id"])[[name]]
    df = df[~df.index.duplicated(keep="first")]
    duplicates = match_count[match_count > 1]
    if (num_duplicates := duplicates.shape[0]) > 0:
        print(f"{name} contains {num_duplicates} duplicates")
    df.loc[duplicates.index, name] = "TO_CHECK"

    sel = df[df[name].notna()]
    sel = sel.loc[np.intersect1d(dn_ids, sel.index)]  # drop non-DNs
    print(f"{len(sel)} DN names inferred from {name}")
    inferred_names_dfs[name] = sel
inferred_names = reduce(
    lambda left, right: pd.merge(
        left, right, left_index=True, right_index=True, how="outer"
    ),
    inferred_names_dfs.values(),
)


# Now, infer neuron name from the guesses above.
def is_valid(x, /):
    return pd.notna(x) and x != "TO_CHECK"


manual_cell_names = manual_cell_names.set_index("root_id")
assert manual_cell_names.index.is_unique
names_taken = []
for root_id, etr in inferred_names.iterrows():
    if root_id in manual_cell_names.index:
        # If the cell name as been manually supplied, use it
        cell_name = manual_cell_names.loc[root_id, "cell_name"]
    elif is_valid(etr["hemibrain_type_mult"]) or is_valid(
        etr["cell_type_mult"]
    ):
        # If multiple names are mentioned, require manual check
        cell_name = "TO_CHECK"
    elif is_valid(etr["hemibrain_type_strict"]) and is_valid(
        etr["cell_type_strict"]
    ):
        # If both hemibrain type and cell type are valid, and they are the same,
        # then we have a valid name. Otherwise, flag it for manual checking
        if etr["hemibrain_type_strict"] == etr["cell_type_strict"]:
            cell_name = etr["hemibrain_type_strict"]
        else:
            cell_name = "TO_CHECK"
    else:
        # Apply priority cascade
        priority_list = [
            "hemibrain_type_strict",
            "cell_type_strict",
            "hemibrain_type_loose",
            "cell_type_loose",
        ]
        cell_name = "TO_CHECK"
        for col_name in priority_list:
            if is_valid(etr[col_name]):
                cell_name = etr[col_name]
                break

    # Additionally, if the cell type is given as DNxx000, but the hemibrain type
    # is given as DNx00, mark it as clarificaiton needed and possible enter it
    # in the manual override table.
    if is_valid(etr["cell_type_loose"]) and is_valid(
        etr["hemibrain_type_strict"]
    ):
        if etr["cell_type_loose"] != etr["hemibrain_type_strict"]:
            if not root_id in manual_cell_names.index:
                print(
                    f"* {root_id} needs clarification: "
                    f"cell_type={etr['cell_type_loose']}, "
                    f"hemibrain_type={etr['hemibrain_type_strict']}"
                )
    names_taken.append(cell_name)
inferred_names["name_taken"] = names_taken
rows_to_check = inferred_names[inferred_names["name_taken"] == "TO_CHECK"]
print(f"{rows_to_check.shape[0]} names to check manually:")
for root_id, _ in rows_to_check.iterrows():
    print(f"  {root_id}")
print(
    "Number of DN names inferred:", inferred_names["name_taken"].notna().sum()
)
ids_with_name_assigned = inferred_names[
    inferred_names["name_taken"].notna()
].index
dns_missing_names = dn_ids[~dn_ids.isin(ids_with_name_assigned)]
print("DNs still missing names:")
for dn_id in dns_missing_names:
    print(f"*  {dn_id}")
inferred_names.to_csv(graphs_dir / "inferred_names.csv")


# Build node attributes df
node_info = pd.merge(neurons, morphology_clusters, on="root_id", how="left")
node_info = pd.merge(
    node_info, connectivity_clusters, on="root_id", how="left"
)
node_info = pd.merge(node_info, classification, on="root_id", how="left")
node_info = pd.merge(node_info, cell_stats, on="root_id", how="left")
node_info = pd.merge(node_info, connectivity_tags, on="root_id", how="left")
node_info = pd.merge(node_info, ke_labels, on="root_id", how="left")
node_info = pd.merge(node_info, inferred_names, on="root_id", how="left")

# Build edge attribute df
nt_weights = params.NT_WEIGHTS
edge_info = connections.copy()

# Unify nt type per neuron
edge_info = pd.merge(
    edge_info,
    node_info[["root_id", "nt_type"]],
    left_on="pre_root_id",
    right_on="root_id",
    how="left",
    suffixes=("", "_unified"),
)
edge_info["nt_type"] = np.where(
    edge_info["nt_type_unified"].notnull(),
    edge_info["nt_type_unified"],
    edge_info["nt_type"],
)
edge_info = edge_info.drop(columns=["root_id", "nt_type_unified"])

# Assign effective weights
weight_vec = np.array([nt_weights[x] for x in edge_info["nt_type"]])
edge_info["eff_weight"] = edge_info["syn_count"] * weight_vec
edge_info = edge_info[edge_info["eff_weight"] != 0]

# Calculate effective weights normalized by total incoming synapses
in_total = edge_info.groupby("post_root_id")["syn_count"].sum()
in_total = in_total.to_frame(name="in_total")
edge_info = edge_info.merge(in_total, left_on="post_root_id", right_index=True)
edge_info["syn_count_norm"] = edge_info["syn_count"] / edge_info["in_total"]
edge_info["eff_weight_norm"] = edge_info["eff_weight"] / edge_info["in_total"]
edge_info = edge_info.drop(columns=["in_total"])

# Build full graph
g_all = nx.from_pandas_edgelist(
    edge_info,
    source="pre_root_id",
    target="post_root_id",
    edge_attr=[
        "neuropil",
        "syn_count",
        "nt_type",
        "eff_weight",
        "syn_count_norm",
        "eff_weight_norm",
    ],
    create_using=nx.DiGraph,
)
node_attr_names = [
    "group",
    "nt_type",
    "nt_type_score",
    "morphology_cluster",
    "connectivity_cluster",
    "flow",
    "super_class",
    "class",
    "sub_class",
    "cell_type",
    "hemibrain_type",
    "hemilineage",
    "side",
    "nerve",
    "length_nm",
    "area_nm",
    "size_nm",
    "dn_label",
]
node_info_sel = node_info[["root_id", *node_attr_names]].set_index("root_id")
nx.set_node_attributes(g_all, node_info_sel.to_dict(orient="index"))

# Build graph with central brain neurons, ANs, and DNs only
mask_central_an_dn = (
    (node_info["super_class"] == "central")
    | (node_info["super_class"] == "ascending")
    | (node_info["super_class"] == "descending")
)
g_central_an_dn = g_all.copy()
g_central_an_dn.remove_nodes_from(node_info[~mask_central_an_dn]["root_id"])

# Build graph with central brain neurons and DNs only
mask_central_dn = (node_info["super_class"] == "central") | (
    node_info["super_class"] == "descending"
)
g_central_dn = g_all.copy()
g_central_dn.remove_nodes_from(node_info[~mask_central_dn]["root_id"])

# Build graph with DNs and GNG interneurons only
dn_gng_mask = node_info["group"].str.contains("GNG") | (
    node_info["super_class"] == "descending"
)
g_dn_gng = g_all.copy()
g_dn_gng.remove_nodes_from(node_info[~dn_gng_mask]["root_id"])

# Build graph with DNs only
dn_mask = node_info["super_class"] == "descending"
g_dn = g_all.copy()
g_dn.remove_nodes_from(node_info[~dn_mask]["root_id"])

# Build sparse matrix representations of these graphs and save to disk
nx_graphs = {
    "all": g_all,
    "central_an_dn": g_central_an_dn,
    "central_dn": g_central_dn,
    "dn_gng": g_dn_gng,
    "dn": g_dn,
}
graphs = {}  # nx graphs, sparse matrices, and index lookup for the matrices
for name, graph in nx_graphs.items():
    nodelist = sorted(list(graph.nodes()))
    mat_norm = nx.to_scipy_sparse_array(
        graph, nodelist=nodelist, weight="eff_weight_norm", format="csr"
    )
    mat_unnorm = nx.to_scipy_sparse_array(
        graph, nodelist=nodelist, weight="eff_weight", format="csr"
    )
    lookup = pd.DataFrame(
        data={"index": np.arange(len(nodelist)), "root_id": nodelist}
    )
    entry = {
        "nx_graph": graph,
        "mat_norm": mat_norm,
        "mat_unnorm": mat_unnorm,
        "lookup": lookup,
    }

    # Save graphs
    with open(graphs_dir / f"graph_{name}.pkl", "wb") as f:
        pickle.dump(entry, f)

# Save node and edge info tables
node_info.to_pickle(os.path.join(graphs_dir, params.NODES_FILE))
edge_info.to_pickle(os.path.join(graphs_dir, params.EDGES_FILE))
