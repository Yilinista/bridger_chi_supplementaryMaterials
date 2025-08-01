# -*- coding: utf-8 -*-

DESCRIPTION = """Run clustering and save files"""

import sys, os, time, pickle
from typing import Dict, List, Optional
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

# requires the h1-the-swan fork of the karateclub library (pull request to the main library is pending)
from karateclub import EgoNetSplitter

from clustering import run_clustering, get_cl_to_authors

class RunCluster:

    """Run clustering and save files"""

    def __init__(self, 
            path_to_edgelist, 
            outdir, 
            resolution: float = 0.001,
            weight: Optional[str] = 'weight'
            ) -> None:
        """

        :path_to_edgelist: path to co-authorship edgelist CSV file. Should have columns 'AuthorId_1' and 'AuthorId_2'. 
        :outdir: path to output directory (will be created)
        :resolution: resolution parameter to use with the clusterer (EgoNetSplitter)
        :weight: column name in the co-authorship edgelist for weights. Default is 'weight'. Will be ignored if the column does not exist. Specify None to force the graph to be treated as unweighted.

        """
        self.path_to_edgelist = path_to_edgelist
        self.outdir = Path(outdir)
        self.resolution = resolution
        self.weight = weight

    def get_readme_text(self) -> str:
        """get README text
        :returns: str

        """
        return f"""
        {self.outdir}

        Overlapping clustering using EgoNetSplitter on the giant component of the co-authorship graph
        using resolution parameter {self.resolution}

        from edgelist file: {self.path_to_edgelist}

        :G_gc: networkx Graph of the giant component for the coauthorship graph
        :memberships: dict of author to cluster
        :cl_to_authors: dict of cluster to author
        :cl_to_subgraph: dict of cluster ID to cluster subgraph (networkx Graph)
        :clusters: list of cluster IDs (in an arbitrary, but consistent order)
        """

    def get_cl_to_subgraph(self, 
            G: nx.Graph, 
            cl_to_authors: Dict[int, List[int]]
            ) -> Dict[int, nx.Graph]:
        """Get a dictionary of cluster ID to subgraph

        :G: full graph object (node IDs are authors)
        :cl_to_authors: dict of cluster ID to list of authors
        :returns: dict of cluster ID to subgraph

        """
        cl_to_subgraph = {}
        for cl, authors in cl_to_authors.items():
            subgraph = nx.subgraph(G, authors)
            cl_to_subgraph[cl] = subgraph
        return cl_to_subgraph

    def save_files(self) -> None:
        """Save files to the output directory

        """
        outfpath = self.outdir.joinpath("giant_component.gpickle")
        logger.debug("saving file {}".format(outfpath))
        nx.write_gpickle(self.G_gc, str(outfpath))

        outfpath = self.outdir.joinpath("memberships.pickle")
        logger.debug("saving file {}".format(outfpath))
        outfpath.write_bytes(pickle.dumps(self.memberships))

        outfpath = self.outdir.joinpath("cl_to_authors.pickle")
        logger.debug("saving file {}".format(outfpath))
        outfpath.write_bytes(pickle.dumps(self.cl_to_authors))

        readme_text = self.get_readme_text()
        outfpath = self.outdir.joinpath("README.txt")
        logger.debug("saving file {}".format(outfpath))
        outfpath.write_text(readme_text)

    def main(self):
        logger.debug("loading co-authorship graph from edgelist file: {}".format(self.path_to_edgelist))
        self.edgelist = pd.read_csv(self.path_to_edgelist)
        self.G = nx.from_pandas_edgelist(self.edgelist, source='AuthorId_1', target='AuthorId_2', edge_attr=True)
        logger.debug("done loading graph. {} nodes and {} edges".format(self.G.number_of_nodes(), self.G.number_of_edges()))
        self.components = list(nx.connected_components(self.G))
        # Make sure the components are sorted biggest first:
        self.components.sort(key=len, reverse=True)
        logger.debug("there are {} connected components. The biggest 5 components have sizes: {}".format(len(self.components), [len(x) for x in self.components[:5]]))
        # Get the largest component (AKA the giant component)
        logger.debug("getting the subgraph for the largest component")
        self.G_gc = nx.subgraph(self.G, self.components[0])
        logger.debug("done getting subgraph. {} nodes and {} edges".format(self.G_gc.number_of_nodes(), self.G_gc.number_of_edges()))
        logger.debug("Running clustering...")
        # Note that the below requires the h1-the-swan fork of the karateclub library (pull request to the main library is pending)
        self.clusterer = EgoNetSplitter(resolution=self.resolution, weight=self.weight)
        self.G_gc, self.memberships = run_clustering(self.G_gc, clusterer=self.clusterer)
        self.cl_to_authors = get_cl_to_authors(self.memberships)
        # self.cl_to_subgraph = self.get_cl_to_subgraph(self.G_gc, self.cl_to_authors)
        self.save_files()
        
def main(args):
    outdir = Path(args.outdir)
    logger.debug("Creating output directory {}".format(outdir))
    outdir.mkdir()
    RunCluster(args.edgelist, outdir, args.resolution).main()

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
    parser.add_argument("edgelist", help="path to co-authorship edgelist CSV file. Should have columns 'AuthorId_1' and 'AuthorId_2'. ")
    parser.add_argument("outdir", help="path to output directory (will be created)")
    parser.add_argument("--resolution", type=float, default=0.001, help="resolution parameter to use with the clusterer (EgoNetSplitter)")
    parser.add_argument("--weight", default='weight', help="column name in the co-authorship edgelist for weights. Default is 'weight'. Will be ignored if the column does not exist.")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
