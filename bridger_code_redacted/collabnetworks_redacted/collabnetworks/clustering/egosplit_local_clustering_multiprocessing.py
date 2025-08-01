# -*- coding: utf-8 -*-

DESCRIPTION = """use multiprocessing to parallelize the first step of the ego-splitting clustering. Start with a weighted, undirected graph, and get a mapping of nodes to non-overlapping partitions of the nodes' ego-net-minus-ego"""

import sys, os, time, pickle
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

from multiprocessing import Pool, cpu_count
# from multiprocessing.pool import ThreadPool
from copy import deepcopy

Components = Dict[int, Iterable[int]]  # dict mapping component index to nodes in that component

def most_central_edge(G):
    centrality = nx.edge_betweenness_centrality(G, weight='weight')
    return max(centrality, key=centrality.get)

def girvan_newman_local(G: nx.Graph) -> Components:
    comp = girvan_newman(G, most_valuable_edge=most_central_edge)
    mod_scores = []
    partitions = []
    for i in range(8):
        try:
            partition = tuple(sorted(c) for c in next(comp))
        except StopIteration:
            break
        modularity_score = nx.community.modularity(G, partition, weight='weight')
        mod_scores.append(modularity_score)
        partitions.append(partition)
    mod_scores = pd.Series(mod_scores)
    t = mod_scores.max() - mod_scores.std()
    for i, p in enumerate(partitions):
        if mod_scores[i] >= t:
            return {i: n for i, n in enumerate(p)}

# def get_data(egonets: Iterable[Tuple[int, nx.Graph]]) -> Dict[int, Components]:
#     """
#     :egonets: iterable of tuples: (node_id, ego_net_minus_ego graph)
#     :returns: dict of {node_id: {component_index: node_ids}}
#     """
#     components = {}
#     logger.debug(f"looping through {len(egonets)} items...")
#     i = 0
#     for node_id, ego_net_minus_ego in egonets:
#         try:
#             components[node_id] = girvan_newman_local(ego_net_minus_ego)
#         except Exception as e:
#             logger.debug("error encountered for node {}: {}".format(node_id, e))
#
#         if i == 0:
#             logger.debug("successfully did first one. components: {}".format(components))
#         i += 1
#     return components
#
def get_data(node_ids: Iterable[int], G: nx.Graph) -> Dict[int, Components]:
    """
    :node_ids: iterable of node IDs
    :G: full graph
    :returns: dict of {node_id: {component_index: node_ids}}
    """
    components = {}
    logger.debug(f"looping through {len(node_ids)} items...")
    i = 0
    for node_id in node_ids:
        try:
            ego_net_minus_ego = G.subgraph(G.neighbors(node_id))
            components[node_id] = girvan_newman_local(ego_net_minus_ego)
        except Exception as e:
            logger.debug("error encountered for node {}: {}".format(node_id, e))

        if i == 0:
            logger.debug("successfully did first one. components: {}".format(components))
        i += 1
    return components

def chunk_args(list_of_args, num_chunks):
    # just calling np.array() on a complicated structure seems to screw things up. This might get around that
    arr = np.empty(len(list_of_args), dtype=object)
    for i, item in enumerate(list_of_args):
        arr[i] = item
    return np.array_split(arr, num_chunks)

def main_multiprocessing(args, G_gc, outfpath):
    n_procs = args.processes

    # args_mp = [(node, G_gc.subgraph(G_gc.neighbors(node))) for node in G_gc.nodes()]
    # logger.debug(f"{len(args_mp)} elements to process")
    # args_mp_chunks = chunk_args(args_mp, n_procs)

    args_mp = list(G_gc.nodes())
    logger.debug(f"{len(args_mp)} elements to process")
    # args_mp_chunks = chunk_args(args_mp, n_procs)
    args_mp_chunks = np.array_split(args_mp, n_procs)
    args_mp_chunks = [(node_subset, G_gc) for node_subset in args_mp_chunks]

    logger.debug(f"running {len(args_mp_chunks)} processes, {n_procs} at a time")
    with Pool(processes=n_procs) as p:
        # data = p.map(get_data, args_mp_chunks)
        data = p.starmap(get_data, args_mp_chunks)

    logger.debug("done processing data. consolidating...")
    all_data = {}
    for data_chunk in data:
        all_data.update(data_chunk)

    logger.debug(f"{len(all_data)} keys in output dict")

    logger.debug(f"saving to {outfpath}")
    outfpath.write_bytes(pickle.dumps(all_data))

def main(args):
    outfpath = Path(args.output)
    df_edgelist = pd.read_csv(args.input)
    G = nx.from_pandas_edgelist(df_edgelist, source='AuthorId_1', target='AuthorId_2', edge_attr=True)
    logger.debug("number of nodes: {}, number of edges {}".format(G.number_of_nodes(), G.number_of_edges()))
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    logger.debug("{} components. biggest 5 are size: {}".format(len(components), [len(x) for x in components[:5]]))
    G_gc = nx.subgraph(G, components[0])
    logger.debug("largest connected component: number of nodes: {}, number of edges {}".format(G_gc.number_of_nodes(), G_gc.number_of_edges()))

    if args.processes != 1:
        main_multiprocessing(args, G_gc, outfpath)
        return

    components = {}
    for node in G_gc.nodes():
        ego_net_minus_ego = G_gc.subgraph(G_gc.neighbors(node))
        components[node] = girvan_newman_local(ego_net_minus_ego)
    logger.debug(f"len(components) keys in output dict")
    logger.debug(f"saving to {outfpath}")
    outfpath.write_bytes(pickle.dumps(components))


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
    parser.add_argument("output", help="output path for components")
    parser.add_argument("--processes", type=int, default=1, help="number of processes, if using multiprocessing (default: 1)")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
