# -*- coding: utf-8 -*-

DESCRIPTION = """Conveniently load the coauthorship graph and community membership data"""

import sys, os, time, pickle, json
from collections import defaultdict
from typing import Optional, Hashable, Container, Mapping, Dict, List, Union, Collection
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

import logging
root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

import pandas as pd
import numpy as np
import networkx as nx

from util import get_root_dir
ROOT_DIR = get_root_dir()

PATH_TO_DATADIR = os.environ.get('DATADIR') or os.path.join(ROOT_DIR, 'data')

NodeId = Hashable
PartitionId = Hashable
Component = Container[NodeId]
OverlappingPartition = Dict[NodeId, List[PartitionId]]

DEFAULTS = {
    'data_fnames': {
        'coauthorship_edgelist': os.path.join(PATH_TO_DATADIR, 'coauthor_CSsubset/coauthor_2015-2020_minpubs3_collabweighted/coauthor_edgelist.csv'),
        'memberships': os.path.join(PATH_TO_DATADIR, 'coauthor_CSsubset/coauthor_2015-2020_minpubs3_collabweighted/infomap_runs/memberships_minWeight02_seed00030.pickle'),
        'cluster_to_numPapers': os.path.join(PATH_TO_DATADIR, 'coauthor_CSsubset/coauthor_2015-2020_minpubs3_collabweighted/clusters_numPapers_for_memberships_minWeight02_seed00030.pickle'),
    },
    'min_year': 2015,
    'max_year': 2020
}

def get_cl_to_authors(memberships):
    """Get dict of cluster -> authors

    """
    cl_to_authors = defaultdict(list)
    for node, cl_membership in memberships.items():
        for cl in cl_membership:
            cl_to_authors[cl].append(node)
    return cl_to_authors

class CoauthorGraph:

    """Contains the coauthorship graph and the community membership data

    Example usage:

    ```
    from load_coauthor_graph import CoauthorGraph
    coauthor_graph = CoauthorGraph.from_defaults()  # should take <1 min. Uses ~4GB RAM

    # Networkx Graph object is in coauthor_graph.G
    num_nodes = coauthor_graph.G.number_of_nodes()
    num_edges = coauthor_graph.G.number_of_edges()
    print(f"graph has {num_nodes} nodes and {num_edges} edges")

    # dictionary of author_id to list of cluster IDs is in coauthor_graph.memberships
    # dictionary of cluster_id to list of authors is in coauthor_graph.cl_to_authors
    num_clusters = len(coauthor_graph.cl_to_authors)
    print(f"there are {num_clusters} clusters total")

    # dictionary of cluster_id to number of papers is in coauthor_graph.cluster_to_num_papers
    print(f"the largest cluster has {max(coauthor_graph.cluster_to_num_papers.values())} papers")
    ```

    """

    def __init__(self,
            min_year: int = DEFAULTS['min_year'],
            max_year: int = DEFAULTS['max_year'],
            G: Optional[nx.Graph] = None,
            memberships: Optional[OverlappingPartition] = None,
            cluster_to_num_papers: Optional[Dict[str, int]] = None
            ) -> None:
        """
        :min_year, max_year: (integers) minimum and maximum year
        :G: Coauthorship graph (weighted undirected nx.Graph). Nodes are authors, links represent coauthorship relations
        :memberships: dict mapping author IDs to IDs
        :cluster_to_num_papers: dict mapping cluster IDs to number of papers
        """
        self.min_year = min_year
        self.max_year = max_year
        self.G = G
        self.memberships = memberships
        self.cluster_to_num_papers = cluster_to_num_papers

        self.cl_to_authors = None

    @classmethod
    def from_defaults(cls, defaults=DEFAULTS, **kwargs):
        """load object from defaults (see source code)

        """
        obj = cls(min_year=defaults['min_year'],
                max_year=defaults['max_year'],
                **kwargs)
        fnames = defaults['data_fnames']
        if obj.G is None:
            obj.load_coauthorship_graph(fnames['coauthorship_edgelist'])
        if obj.memberships is None:
            obj.load_cluster_memberships(fnames['memberships'])
        if obj.cluster_to_num_papers is None:
            obj.load_cluster_to_num_papers(fnames['cluster_to_numPapers'])
        return obj

    def load_coauthorship_graph(self,
            path_to_edgelist_csv: str):
        logger.debug("loading coauthorship graph from file {}".format(path_to_edgelist_csv))
        df_edgelist = pd.read_csv(path_to_edgelist_csv)
        G = nx.from_pandas_edgelist(df_edgelist, source='AuthorId_1', target='AuthorId_2', edge_attr=True)
        # With edge weight thresholding
        weight_thresh = 0.02
        logger.debug("removing edges with weight <= {}".format(weight_thresh))
        edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w <= weight_thresh]
        G.remove_edges_from(edges_to_remove)
        logger.debug("coauthorship graph has {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
        self.G = G

    def load_cluster_memberships(self,
            path_to_pickle: str
            ) -> None:
        logger.debug("loading cluster memberships from file {}".format(path_to_pickle))
        fpath = Path(path_to_pickle)
        self.memberships = pickle.loads(fpath.read_bytes())
        self.cl_to_authors = get_cl_to_authors(self.memberships)
        logger.debug("there are {} clusters".format(len(self.cl_to_authors)))

    def load_cluster_to_num_papers(self,
            path_to_pickle: str
            ) -> None:
        logger.debug("loading dict of cluster to number of papers from file {}".format(path_to_pickle))
        fpath = Path(path_to_pickle)
        self.cluster_to_num_papers = pickle.loads(fpath.read_bytes())


        

def main(args):
    pass

if __name__ == "__main__":
    total_start = timer()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    logger.info("pid: {}".format(os.getpid()))
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
