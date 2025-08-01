# -*- coding: utf-8 -*-

DESCRIPTION = """The second step of overlapping clustering using the ego-splitting framework: partitioning the persona graph"""

import sys, os, time, json, pickle
from typing import (
    Hashable,
    Iterable,
    Mapping,
    Container,
    Optional,
    Callable,
    Dict,
    List,
    Union,
)
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

# import infomap

from .clustering import load_graph_and_partitions

NodeId = Hashable
PartitionId = Hashable
Component = Container[NodeId]
Partition = Mapping[NodeId, Mapping[PartitionId, Component]]
Memberships = Mapping[NodeId, PartitionId]  # non-overlapping memberships


def calc_infomap():
    pass


# def calc_infomap(
#     G: nx.Graph,
#     infomap_args: str = "-f undirected --seed {seed} --silent",
#     weight: str = "weight",
#     seed: int = 999,
# ) -> Memberships:
#     if "{seed}" in infomap_args:
#         infomap_args = infomap_args.format(seed=seed)
#     elif "--seed" not in infomap_args:
#         infomap_args = "{infomap_args} --seed {seed}".format(
#             infomap_args=infomap_args, seed=seed
#         )
#     im = infomap.Infomap(infomap_args)
#     mapping = {}
#     reverse_mapping = {}
#     for i, n in enumerate(G.nodes()):
#         mapping[n] = i
#         reverse_mapping[i] = n

#         _node = im.add_node(mapping[n])

#     for source, target, weight in G.edges.data("weight"):
#         im.add_link(mapping[source], mapping[target], weight)
#     logger.debug("running infomap with args: {}".format(infomap_args))
#     im_start = timer()
#     im.run()
#     logger.debug("infomap ran in {}".format(format_timespan(timer() - im_start)))
#     logger.debug(
#         f"Found {im.num_top_modules} top modules with codelength: {im.codelength}"
#     )
#     memberships = {}
#     for node in im.tree:
#         if node.is_leaf:
#             node_id = node.node_id
#             node_name = reverse_mapping[node_id]
#             path_tuple = node.path
#             path_str = ":".join([str(x) for x in path_tuple])
#             memberships[node_name] = path_str[: path_str.rfind(":")]
#     return memberships


class PersonaCluster:

    """Takes a graph, and a dictionary mapping each node to a partitioning of its neighbors. Constructs a persona graph, and identifies an overlapping clustering."""

    def __init__(
        self,
        graph: nx.Graph,
        ego_partition: Partition,
        weight: str = "weight",
        method: Callable[[nx.Graph], Memberships] = calc_infomap,
        seed: Optional[int] = 999,
    ) -> None:
        """
        :graph: the original graph
        :ego_partition: mapping of each node to a dictionary of its neighbors, split into components. A node will have as many personalities as it has these components, and will end up belonging to this many clusters.
        :method: Method to use to cluster the persona graph. This should be a callable that returns a membership mapping. Default is to use hierarchical infomap.
        :seed: integer to use as random seed. if `method` does not take a 'seed' argument, specify seed=None
        """
        self.graph = graph
        self.ego_partition = ego_partition
        self.weight = weight
        self.method = method
        self.seed = seed

    def _prepare_personalities(self):
        self.personalities = {}
        self.components = {}
        idx = 0

        for node, ptn in self.ego_partition.items():
            new_mapping = {}
            personalities = []
            for v in ptn.values():
                personalities.append(idx)
                for other_node in v:
                    new_mapping[other_node] = idx
                idx += 1
            self.components[node] = new_mapping
            self.personalities[node] = personalities

    def _map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {
            p: n for n in self.graph.nodes() for p in self.personalities[n]
        }

    def _get_new_edge_ids(self, edge):
        """
        Getting the new edge identifiers.

        Arg types:
            * **edge** *(list of ints)* - Edge being mapped to the new identifiers.
        """
        try:
            if self.weight is None or edge[2] is None:
                return (
                    self.components[edge[0]][edge[1]],
                    self.components[edge[1]][edge[0]],
                )
            else:
                return (
                    self.components[edge[0]][edge[1]],
                    self.components[edge[1]][edge[0]],
                    {self.weight: edge[2]},
                )
        except KeyError:
            return None

    def _create_persona_graph(self):
        """
        Create a persona graph using the ego-net components.
        """
        self.persona_graph_edges = []
        if self.weight is None:
            _iterator = self.graph.edges()
        else:
            _iterator = self.graph.edges(data=self.weight)
        for edge in _iterator:
            new_edge = self._get_new_edge_ids(edge)
            if new_edge is not None:
                self.persona_graph_edges.append(new_edge)

        self.persona_graph = nx.from_edgelist(self.persona_graph_edges)

    def _create_partitions(self, method: Callable[[nx.Graph], Memberships]):
        """
        Creating a non-overlapping clustering of nodes in the persona graph.
        """
        if self.seed is None:
            self.partitions = method(self.persona_graph)
        else:
            self.partitions = method(self.persona_graph, seed=self.seed)

        self.overlapping_partitions = {node: [] for node in self.graph.nodes()}
        for node, membership in self.partitions.items():
            self.overlapping_partitions[self.personality_map[node]].append(membership)

    def fit(self) -> None:
        self._prepare_personalities()
        self._map_personalities()
        self._create_persona_graph()
        self._create_partitions(self.method)

    def get_memberships(self) -> Dict[NodeId, List[PartitionId]]:
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dictionary of lists)* - Cluster memberships.
        """
        return self.overlapping_partitions


def run_clustering(G, ego_partition, seed):
    clusterer = PersonaCluster(G, ego_partition, seed=args.seed)
    clusterer.fit()
    memberships = clusterer.get_memberships()
    return memberships


def main(args):
    fpath_edgelist = Path(args.input)
    outfpath = Path(args.output)
    local_partition_dirpath = Path(args.local_partition)
    G, ego_partition = load_graph_and_partitions(
        fpath_edgelist, local_partition_dirpath, args.min_weight
    )
    memberships = run_clustering(G, ego_partition, args.seed)

    logger.debug("writing memberships to output: {}".format(outfpath))
    outfpath.write_bytes(pickle.dumps(memberships))


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
        "local_partition",
        help="directory with JSONL files resulting from the local ego-splitting partitioning (precomputed)",
    )
    parser.add_argument("output", help="output path for memberships (pickle)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (int)")
    parser.add_argument(
        "--min-weight",
        type=float,
        default=None,
        help="remove all edges with weight less than or equal to this threshold",
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
