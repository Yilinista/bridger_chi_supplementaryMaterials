"""
Copy this file to a directory created by get_clustering.py

In a jupyter notebook, load these variables by running `%run <path_to_this_file>` in a cell. Then `print(README)` to see the variables that were imported
"""

import os
import pickle
from pathlib import Path
import networkx as nx

_CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

README = _CURRENT_DIR.joinpath("README.txt").read_text()

_fpath = _CURRENT_DIR.joinpath('./giant_component.gpickle')
G_gc = nx.read_gpickle(str(_fpath))

_fpath = _CURRENT_DIR.joinpath('./memberships.pickle')
memberships = pickle.loads(_fpath.read_bytes())

_fpath = _CURRENT_DIR.joinpath('./cl_to_authors.pickle')
cl_to_authors = pickle.loads(_fpath.read_bytes())

cl_to_subgraph = {cl: nx.subgraph(G_gc, authors) for cl, authors in cl_to_authors.items()}

clusters = list(cl_to_subgraph.keys())

