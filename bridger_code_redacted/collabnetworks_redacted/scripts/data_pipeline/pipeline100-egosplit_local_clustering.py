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

from collabnetworks.clustering.egosplit_local_clustering import (
    load_graph_and_run_local_gn_for_subset,
)


def main_multiprocess(args):
    n_proc = args.process
    mp_args = []

    df_edgelist = pd.read_csv(args.input)
    G = nx.from_pandas_edgelist(
        df_edgelist, source="AuthorId_1", target="AuthorId_2", edge_attr=True
    )
    logger.debug(
        "number of nodes: {}, number of edges {}".format(
            G.number_of_nodes(), G.number_of_edges()
        )
    )
    step = G.number_of_nodes() // n_proc + 1
    logger.debug(f"using step: {step}")

    start_idx = args.start
    while True:
        end_idx = start_idx + step
        mp_args.append((args.input, args.outdir, start_idx, end_idx, args.min_weight))
        if end_idx >= G.number_of_nodes():
            break
        start_idx += step

    logger.debug(f"running {len(mp_args)} processes, {n_proc} at a time")

    with Pool(processes=n_proc) as pool:
        pool.starmap(load_graph_and_run_local_gn_for_subset, mp_args)


def main(args):
    if args.process == 1:
        load_graph_and_run_local_gn_for_subset(
            args.input, args.outdir, args.start, args.end, args.min_weight
        )
    else:
        main_multiprocess(args)


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
    parser.add_argument("outdir", help="output directory")
    parser.add_argument(
        "--start", type=int, default=0, help="index of the first node to start with"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="index of the last node to process"
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=None,
        help="remove all edges with weight less than or equal to this threshold",
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
