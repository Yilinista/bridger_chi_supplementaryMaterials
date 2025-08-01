# -*- coding: utf-8 -*-

DESCRIPTION = """For nodes that failed with a timeout error in the egosplit_local clustering procedure, find their egonet clustering using a different strategy:
Apply a harsher weight threshold to the egonet. Drop isolates and try the clustering again.
"""

import sys, os, time, pickle, json
from pathlib import Path
from typing import Iterable, Dict, Tuple, Optional, Union, Hashable, List
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

from collabnetworks.clustering.egosplit_local_clustering import girvan_newman_local

from multiprocessing import Pool, cpu_count

NodeId = Hashable


def load_graph_drop_edges_and_run_local_gn(
    fname,
    outdir,
    nodes: List[NodeId],
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    min_weight: float = 0.05,
) -> None:
    """Load graph, and run local girvan newman clustering on each egonet, after dropping weak edges and dropping isolates

    Args:
        fname (str): filename for edgelist csv
        outdir (str): output directory
        start_idx (int): index of node to start with
        end_idx (int): index of node to end with
        min_weight (float): drop edges below this weight threshold for each egonet (default: 0.05)
    """
    outdir = Path(outdir)
    if not outdir.exists():
        raise FileNotFoundError(f"outdir {outdir} does not exist")
    df_edgelist = pd.read_csv(fname)
    G = nx.from_pandas_edgelist(
        df_edgelist, source="AuthorId_1", target="AuthorId_2", edge_attr=True
    )
    logger.debug(
        "number of nodes: {}, number of edges {}".format(
            G.number_of_nodes(), G.number_of_edges()
        )
    )
    end_idx = end_idx if end_idx is not None else len(nodes)
    outfpath = outdir.joinpath(
        "components_localGirvanNewman_redo_{}-{}.jsonl".format(start_idx, end_idx)
    )
    with outfpath.open("w") as outf:
        for node in nodes[start_idx:end_idx]:
            # logger.debug(f"{curr_idx} start")
            # meta_out = {"node_idx": curr_idx, "node_name": node}
            ego_net_minus_ego = G.subgraph(G.neighbors(node)).copy()
            # meta_out["egonet_num_nodes"] = ego_net_minus_ego.number_of_nodes()
            # meta_out["egonet_num_edges"] = ego_net_minus_ego.number_of_edges()
            # logger.debug("getting components for node: {}. ego_net_minus_ego graph has {} nodes and {} edges".format(node, ego_net_minus_ego.number_of_nodes(), ego_net_minus_ego.number_of_edges()))
            edges_to_remove = [(u,v) for u,v,w in ego_net_minus_ego.edges(data='weight') if w <= min_weight]
            ego_net_minus_ego.remove_edges_from(edges_to_remove)
            ego_net_minus_ego.remove_nodes_from(list(nx.isolates(ego_net_minus_ego)))
            gn_start = timer()
            comps = girvan_newman_local(ego_net_minus_ego, timeout=10800)
            if not comps:
                raise RuntimeError("timeout error")
            gn_time = timer() - gn_start
            if comps != "":
                outline = {int(node): comps}
                print(json.dumps(outline), file=outf, flush=True)


def main_multiprocess(args, nodes):
    n_proc = args.process
    mp_args = []

    step = len(nodes) // n_proc + 1
    logger.debug(f"using step: {step}")

    start_idx = 0
    while True:
        end_idx = start_idx + step
        mp_args.append(
            (args.input, args.outdir, nodes, start_idx, end_idx, args.min_weight)
        )
        if end_idx >= len(nodes):
            break
        start_idx += step

    logger.debug(f"running {len(mp_args)} processes, {n_proc} at a time")

    with Pool(processes=n_proc) as pool:
        pool.starmap(load_graph_drop_edges_and_run_local_gn, mp_args)


def main(args):
    dirpath = Path(args.outdir)
    df_metadata = pd.concat(pd.read_csv(x) for x in dirpath.glob("*metadata*.csv"))
    df_timeout = df_metadata[df_metadata.error == "timeout"]
    nodes = df_timeout["node_name"].tolist()
    logger.debug(
        f"{len(nodes)} nodes identified that need to be rerun because of timeout errors"
    )
    if args.process == 1:
        load_graph_drop_edges_and_run_local_gn(
            args.input, args.outdir, nodes, start_idx=0, end_idx=None, min_weight=args.min_weight
        )
    else:
        main_multiprocess(args, nodes)


if __name__ == "__main__":
    total_start = timer()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(" ".join(sys.argv))
    logger.info("{:%Y-%m-%d %H:%M:%S}".format(datetime.now()))
    logger.info("pid: {}".format(os.getpid()))
    import argparse

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "input",
        help="input edgelist (CSV including columns 'AuthorId_1' and 'AuthorId_2')",
    )
    parser.add_argument(
        "outdir",
        help="output directory. should already exist and contain metadata CSV files identifying the nodes that had timeout errors",
    )
    # parser.add_argument(
    #     "--start", type=int, default=0, help="index of the first node to start with"
    # )
    # parser.add_argument(
    #     "--end", type=int, default=None, help="index of the last node to process"
    # )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
        help="remove all edges with weight less than or equal to this threshold (default: 0.05)",
    )
    parser.add_argument(
        "--process",
        type=int,
        default=1,
        help="number of processes to use for multiprocessing",
    )
    parser.add_argument("--debug", action="store_true", help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug("debug mode is on")
    main(args)
    total_end = timer()
    logger.info(
        "all finished. total time: {}".format(format_timespan(total_end - total_start))
    )
