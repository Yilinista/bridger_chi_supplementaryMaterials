# -*- coding: utf-8 -*-

DESCRIPTION = """For some of the high-degree nodes, local clustering is too computationally difficult. Use a sampling strategy to calculate betweenness centraility for these nodes"""

import sys, os, time, pickle, json
from pathlib import Path
from typing import Iterable, Dict, Tuple
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
from networkx.algorithms.community.centrality import girvan_newman

Components = Dict[int, Iterable[int]]  # dict mapping component index to nodes in that component

from egosplit_local_clustering import girvan_newman_local

from multiprocessing import Pool, cpu_count

def get_nodes_to_skip(fpath: Path):
    with open(fpath, 'r') as f:
        comp_data = [json.loads(line) for line in f]
    comp_nodes = [int(k) for item in comp_data for k in item.keys()]
    return comp_nodes

def calc_components_one_node(node):
    sample_size = 100
    outfpath = outdir.joinpath('components_localGirvanNewman_highDegree_sampleBetweenness{}_node{}.jsonl'.format(sample_size, node))
    logger.debug("writing to file: {}...".format(outfpath))
    with outfpath.open("w") as outf:
        try:
            start = timer()
            ego_net_minus_ego = G_gc.subgraph(G_gc.neighbors(node))
            if ego_net_minus_ego.number_of_nodes() < sample_size:
                k = ego_net_minus_ego.number_of_nodes()
            else:
                k = sample_size

            def most_central_edge_sample(G, k=k, seed=1):
                centrality = nx.edge_betweenness_centrality(G, weight='weight', k=k, seed=seed)
                return max(centrality, key=centrality.get)

            logger.debug("getting components for node: {}. ego_net_minus_ego graph has {} nodes and {} edges".format(node, ego_net_minus_ego.number_of_nodes(), ego_net_minus_ego.number_of_edges()))
            comps = girvan_newman_local(ego_net_minus_ego, most_valuable_edge=most_central_edge_sample, timeout=3600)
            outline = {node: comps}
            print(json.dumps(outline), file=outf)
            logger.debug("done getting components for node: {}. took {}".format(node, format_timespan(timer()-start)))
        except Exception as e:
            logger.debug("error encountered for node {}: {}".format(node, e))

def main(args):
    global outdir
    global G_gc
    outdir = Path(args.outdir)
    if outdir.exists():
        logger.debug("using output directory: {}".format(outdir))
    else:
        logger.debug("creating output directory {}".format(outdir))
        outdir.mkdir()
    df_edgelist = pd.read_csv(args.input)
    G = nx.from_pandas_edgelist(df_edgelist, source='AuthorId_1', target='AuthorId_2', edge_attr=True)
    logger.debug("number of nodes: {}, number of edges {}".format(G.number_of_nodes(), G.number_of_edges()))
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    logger.debug("{} components. biggest 5 are size: {}".format(len(components), [len(x) for x in components[:5]]))
    G_gc = nx.subgraph(G, components[0])
    logger.debug("largest connected component: number of nodes: {}, number of edges {}".format(G_gc.number_of_nodes(), G_gc.number_of_edges()))

    # sort the nodes by degree, lowest first
    # it will speed through all of the small ego nets, and then slow down
    deg = list(G_gc.degree())
    deg.sort(key=lambda x: x[1])
    nodes = [n for n, d in deg]

    if args.donefile is not None:
        donefile = Path(args.donefile)
        logger.debug("getting nodes to skip from file: {}".format(donefile))
        nodes_to_skip = get_nodes_to_skip(donefile)
        logger.debug("found {} nodes to skip".format(len(nodes_to_skip)))
        # nodes = [n for n in nodes if n not in nodes_to_skip]
        # above is too slow
        nodes = set(nodes).difference(set(nodes_to_skip))
        nodes = list(nodes)
        nodes.sort()
        logger.debug("number of nodes to process is now {}".format(len(nodes)))

    n_procs = args.num_process
    with Pool(processes=n_procs) as p:
        p.map(calc_components_one_node, nodes)


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
    parser.add_argument("input", help="input edgelist (CSV including columns 'AuthorId_1' and 'AuthorId_2')")
    parser.add_argument("outdir", help="output directory")
    parser.add_argument("--donefile", help="path to JSONL file containing the results for the nodes that have already been processed successfully. These will be skipped")
    parser.add_argument("--num-process", type=int, default=10, help="number of processes")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
