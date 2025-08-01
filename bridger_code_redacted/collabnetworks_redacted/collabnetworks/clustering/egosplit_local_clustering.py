# -*- coding: utf-8 -*-

DESCRIPTION = """the first step of the ego-splitting clustering. Start with a weighted, undirected graph, and get a mapping of nodes to non-overlapping partitions of the nodes' ego-net-minus-ego"""

import sys, os, time, pickle, json
from pathlib import Path
from typing import Iterable, Dict, Tuple, Optional, Union
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
from networkx.algorithms.community.quality import modularity

from multiprocessing import Pool, cpu_count

import signal

# refer to https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

Components = Dict[
    int, Iterable[int]
]  # dict mapping component index to nodes in that component


def most_central_edge(G):
    centrality = nx.edge_betweenness_centrality(G, weight="weight")
    return max(centrality, key=centrality.get)


def girvan_newman_local(
    G: nx.Graph, most_valuable_edge=most_central_edge, timeout=60, return_more=False
):
    comp = girvan_newman(G, most_valuable_edge=most_valuable_edge)
    mod_scores = []
    partitions = []
    for i in range(8):
        # Start timer, after <timeout> seconds, a SIGALRM is sent
        signal.alarm(timeout)
        try:
            # logger.debug(f"iteration {i}")
            partition = tuple(sorted(c) for c in next(comp))
        except StopIteration:
            break
        except TimeoutException:
            # logger.debug("Timeout Exception (took longer than 60 seconds) for this ego net, on iteration {}. Skipping".format(i))
            signal.alarm(0)
            if return_more is False:
                return {}
            else:
                return {"error": "timeout", "iteration": i}
        else:
            # Reset the alarm
            signal.alarm(0)
        # logger.debug("calculating modularity")
        modularity_score = modularity(G, partition, weight="weight")
        # logger.debug(f"modularity score: {modularity_score}")
        mod_scores.append(modularity_score)
        partitions.append(partition)
    stop_iteration = i
    # logger.debug(f"done getting {len(partitions)} partitions")
    mod_scores = pd.Series(mod_scores)
    t = mod_scores.max() - mod_scores.std()
    # logger.debug("calculated {} partitions. t=={}".format(len(partitions), t))
    for i, p in enumerate(partitions):
        if mod_scores[i] >= t:
            # logger.debug("selected partition ({}) has {} components with mod_score: {}".format(i, len(p), mod_scores[i]))
            selected_partition = i
            components = {j: n for j, n in enumerate(p)}
            if return_more is False:
                return components
            else:
                return {
                    "components": components,
                    "stop_iteration": stop_iteration,
                    "t": t,
                    "selected_partition": selected_partition,
                }


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
# def get_data(node_ids: Iterable[int], G: nx.Graph) -> Dict[int, Components]:
#     """
#     :node_ids: iterable of node IDs
#     :G: full graph
#     :returns: dict of {node_id: {component_index: node_ids}}
#     """
#     components = {}
#     logger.debug(f"looping through {len(node_ids)} items...")
#     i = 0
#     for node_id in node_ids:
#         try:
#             ego_net_minus_ego = G.subgraph(G.neighbors(node_id))
#             components[node_id] = girvan_newman_local(ego_net_minus_ego, most_valuable_edge=most_central_edge)
#         except Exception as e:
#             logger.debug("error encountered for node {}: {}".format(node_id, e))

#         if i == 0:
#             logger.debug("successfully did first one. components: {}".format(components))
#         i += 1
#     return components


def load_graph_and_run_local_gn_for_subset(
    fname,
    outdir,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    min_weight: Optional[float] = None,
) -> None:
    """Load graph and run local girvan newman clustering on each egonet

    Args:
        fname (str): filename for edgelist csv
        outdir (str): output directory
        start_idx (int): index of node to start with
        end_idx (int): index of node to end with
        min_weight (float): drop edges below this weight threshold
    """
    outdir = Path(outdir)
    if outdir.exists():
        logger.debug("using output directory: {}".format(outdir))
    else:
        logger.debug("creating output directory {}".format(outdir))
        outdir.mkdir()
    df_edgelist = pd.read_csv(fname)
    G = nx.from_pandas_edgelist(
        df_edgelist, source="AuthorId_1", target="AuthorId_2", edge_attr=True
    )
    logger.debug(
        "number of nodes: {}, number of edges {}".format(
            G.number_of_nodes(), G.number_of_edges()
        )
    )
    # components = list(nx.connected_components(G))
    # components.sort(key=len, reverse=True)
    # logger.debug("{} components. biggest 5 are size: {}".format(len(components), [len(x) for x in components[:5]]))
    # G_gc = nx.subgraph(G, components[0])
    # logger.debug("largest connected component: number of nodes: {}, number of edges {}".format(G_gc.number_of_nodes(), G_gc.number_of_edges()))

    if min_weight is not None:
        logger.debug("removing edges with weight <= {}".format(min_weight))
        edges_to_remove = [
            (u, v) for u, v, w in G.edges(data="weight") if w <= min_weight
        ]
        G.remove_edges_from(edges_to_remove)
        logger.debug(
            "number of nodes: {}, number of edges {}".format(
                G.number_of_nodes(), G.number_of_edges()
            )
        )

    # randomize the order of the nodes
    rng = np.random.default_rng(1)
    nodes = rng.permutation(G.nodes())
    logger.debug(f"num nodes: {len(nodes)}")

    end_idx = end_idx if end_idx is not None else len(nodes)
    # outfpath = outdir.joinpath('components_localGirvanNewman_{}-{}.pickle'.format(start_idx, end_idx))
    outfpath = outdir.joinpath(
        "components_localGirvanNewman_{}-{}.jsonl".format(start_idx, end_idx)
    )

    clustering_metadata_fpath = outdir.joinpath(
        "local_clustering_metadata_{}-{}.csv".format(start_idx, end_idx)
    )

    components = {}
    logger.debug("processing nodes {} to {}".format(start_idx, end_idx))
    logger.debug("writing to file: {}...".format(outfpath))
    logger.debug(
        "writing clustering metadata to file: {}...".format(clustering_metadata_fpath)
    )
    f_meta = clustering_metadata_fpath.open("w", buffering=1)
    metadata_columns = [
        "node_idx",
        "node_name",
        "egonet_num_nodes",
        "egonet_num_edges",
        "num_components",
        "stop_iteration",
        "selected_partition",
        "t",
        "runtime",
        "error",
    ]
    f_meta.write(",".join(metadata_columns))
    f_meta.write("\n")
    with outfpath.open("w") as outf:
        curr_idx = start_idx
        logger.debug(f"{start_idx}: {nodes[start_idx]}")
        for node in nodes[start_idx:end_idx]:
            # logger.debug(f"{curr_idx} start")
            meta_out = {"node_idx": curr_idx, "node_name": node}
            ego_net_minus_ego = G.subgraph(G.neighbors(node))
            meta_out["egonet_num_nodes"] = ego_net_minus_ego.number_of_nodes()
            meta_out["egonet_num_edges"] = ego_net_minus_ego.number_of_edges()
            # logger.debug("getting components for node: {}. ego_net_minus_ego graph has {} nodes and {} edges".format(node, ego_net_minus_ego.number_of_nodes(), ego_net_minus_ego.number_of_edges()))
            gn_start = timer()
            if (
                ego_net_minus_ego.number_of_nodes() > 2
                and ego_net_minus_ego.number_of_edges() > 1
            ):
                # components[node] = girvan_newman_local(ego_net_minus_ego)
                # logger.debug(f"running girvan newman for node {node} ({ego_net_minus_ego.number_of_nodes()} nodes; {ego_net_minus_ego.number_of_edges()} edges)")
                # comps = girvan_newman_local(ego_net_minus_ego, timeout=600)
                ret_dict = girvan_newman_local(
                    ego_net_minus_ego, timeout=600, return_more=True
                )
                if ret_dict is None:
                    logger.error(
                        f"ret_dict is None! node_idx: {curr_idx}, node_name: {node}"
                    )
                if "error" in ret_dict:
                    comps = ""
                    meta_out["stop_iteration"] = ret_dict["iteration"]
                    meta_out["selected_partition"] = ""
                    meta_out["t"] = ""
                    meta_out["error"] = ret_dict["error"]
                else:
                    comps = ret_dict["components"]
                    meta_out["stop_iteration"] = ret_dict["stop_iteration"]
                    meta_out["selected_partition"] = ret_dict["selected_partition"]
                    meta_out["t"] = ret_dict["t"]
                    meta_out["error"] = ""
            else:
                # if the ego net is tiny, just take connected components:
                # components[node] = {i: n for i, n in enumerate(nx.connected_components(ego_net_minus_ego))}
                comps = {
                    i: list(n)
                    for i, n in enumerate(nx.connected_components(ego_net_minus_ego))
                }
                meta_out["stop_iteration"] = ""
                meta_out["selected_partition"] = ""
                meta_out["t"] = ""
                meta_out["error"] = ""
            gn_time = timer() - gn_start
            meta_out["num_components"] = len(comps)
            meta_out["runtime"] = gn_time
            meta_outline = [str(meta_out[colname]) for colname in metadata_columns]
            f_meta.write(",".join(meta_outline))
            f_meta.write("\n")

            if comps != "":
                outline = {int(node): comps}
                print(json.dumps(outline), file=outf, flush=True)
            curr_idx += 1
            # logger.debug(f"{curr_idx} end")
            # logger.debug(f"done with node {node}")
    #     logger.debug(f"{len(components)} keys in output dict")
    # logger.debug(f"saving to {outfpath}")
    # outfpath.write_bytes(pickle.dumps(components))

    f_meta.close()