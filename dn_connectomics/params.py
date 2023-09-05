"""
2023.08.30
author: femke.hurtak@epfl.ch
File containing parameters for the project.
"""
from pathlib import Path
import os

# --- Initial data preparation --- #
RAW_DATA_DIR = os.path.join(Path(__file__).absolute().parent, "data")
CODEX_DUMP_DIR = os.path.join(RAW_DATA_DIR, "codex_dump", "v630-20230803")
GRAPHS_DIR = os.path.join(RAW_DATA_DIR, "graphs", "v630-20230803")

# --- Data storage --- #
DATA_DIR = GRAPHS_DIR

NODES_FILE = "node_info.pkl"
EDGES_FILE = "edge_info.pkl"
NAMES_FILE = "inferred_names.csv"

# --- Figure storage --- #
FIGURES_DIR = os.path.join(Path(__file__).absolute().parent, "figures")

# --- Biological parameters --- #
NT_WEIGHTS = {"ACH": +1, "GABA": -1, "GLUT": -1, "SER": 0, "DA": 0, "OCT": 0}
# nb: GLUT being inhibitory is still unclear, you can change it here before
# running the data preparation script
