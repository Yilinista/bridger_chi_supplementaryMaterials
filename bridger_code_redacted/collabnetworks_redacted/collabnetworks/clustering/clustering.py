# -*- coding: utf-8 -*-

DESCRIPTION = """Clustering"""

import sys, os, time, json
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
from collections import defaultdict
from six import class_types
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

try:
    from karateclub import EgoNetSplitter
except ImportError:
    pass

from .. import PaperCollectionHelper

NodeId = Hashable
PartitionId = Hashable
Component = Container[NodeId]
Partition = Mapping[NodeId, Mapping[PartitionId, Component]]
Memberships = Mapping[NodeId, PartitionId]  # non-overlapping memberships

# def run_clustering(G, reset_labels=True, clusterer=EgoNetSplitter, cl_attr_name="cluster_membership"):
#     """Get cluster membership for a graph

#     :G: Networkx Graph (unipartite, undirected)
#     :reset_labels: During the clustering, node labels (IDs) will be changed to consecutive integers. If reset_labels is true, reset the labels back to the original values before finishing.
#     :clusterer: either a class or an instance of the clusterer to use
#     :returns: G, with cluster memberships as node attributes. Also returns memberships: a dict mapping node labels to cluster membership

#     """
#     if isinstance(clusterer, class_types):
#         clusterer = clusterer()
#     G = nx.convert_node_labels_to_integers(G, label_attribute='orig_label')
#     clusterer.fit(G)
#     memberships = clusterer.get_memberships()
#     nx.set_node_attributes(G, memberships, name=cl_attr_name)
#     if reset_labels is True:
#         mapping = {node: orig_label for node, orig_label in G.nodes(data='orig_label')}
#         # relabel in place
#         nx.relabel_nodes(G, mapping, copy=False)
#         memberships = {mapping[k]: v for k, v in memberships.items()}
#     return G, memberships


def get_cl_to_authors(memberships):
    """Get dict of cluster -> authors"""
    cl_to_authors = defaultdict(list)
    for node, cl_membership in memberships.items():
        for cl in cl_membership:
            cl_to_authors[cl].append(node)
    return cl_to_authors


# def calculate_paper_relevance_score(paper_id, author_ids, mag_data) -> float:
#     """
#     :paper_id: calculate score for this paper
#     :author_ids: author ids in a cluster
#     :returns: score between 0 and 1
#
#     If the paper has fewer than two authors within the cluster, the score is 0
#     Otherwise, first and last author are weighted more heavily. Normalized by number of authors in the paper.
#     """
#     paa = mag_data.paper_authors
#     paa_subset = paa[paa.PaperId==paper_id]
#     paa_subset = paa_subset[['PaperId', 'AuthorId', 'AuthorSequenceNumber', 'num_authors', 'is_last_author']].drop_duplicates()
#     this_paper_author_ids = paa_subset.AuthorId.values
# #     if len(set(author_ids).intersection(set(this_paper_author_ids))) < 2:
# #         return 0
#     score = 0
#     num_authors = paa_subset.iloc[0].num_authors
#     for _, row in paa_subset.iterrows():
#         if row['AuthorId'] in author_ids:
#             if row['AuthorSequenceNumber'] == 1 or row['is_last_author']:
#                 score += 1
#             else:
#                 score += .75
#     return score / num_authors
#
# def get_relevance_scores_for_cluster(author_ids, mag_data):
#     paa = mag_data.paper_authors
#     if 'num_authors' not in paa.columns:
#         paa['num_authors'] = paa.groupby('PaperId')['AuthorSequenceNumber'].transform('max')
#     if 'is_last_author' not in paa.columns:
#         paa['is_last_author'] = np.where(paa['num_authors']==paa['AuthorSequenceNumber'], True, False)
#     paa_subset = paa[paa.AuthorId.isin(author_ids)]
#     paa_subset = paa_subset[['PaperId', 'AuthorId']].drop_duplicates()
#     gb = paa_subset.groupby('PaperId')
#     paa_subset = gb.filter(lambda x: len(x) > 1)
#     cl_paperid_subset = paa_subset.PaperId.drop_duplicates()
#     relevance_scores = cl_paperid_subset.apply(calculate_paper_relevance_score, author_ids=author_ids)
#     return relevance_scores


def load_ego_partition(
    dirpath: Union[Path, str], glob_pattern: str = "*.jsonl"
) -> Partition:
    """TODO: load ego partition from JSONL files

    :dirpath: directory with the JSONL files
    :returns: Partition

    """
    dirpath = Path(dirpath)
    ego_partition = {}
    for fpath in dirpath.glob(glob_pattern):
        with fpath.open("r") as f:
            for line in f:
                jline = json.loads(line)
                len_before_update = len(ego_partition)
                ego_partition.update(jline)
                # each line should add one item to ego_partition
                assert len(ego_partition) == len_before_update + 1
    return ego_partition


def load_edgelist(fpath_edgelist, min_weight=None):
    logger.debug("loading edgelist from {}".format(fpath_edgelist))
    df_edgelist = pd.read_csv(fpath_edgelist)
    G = nx.from_pandas_edgelist(
        df_edgelist, source="AuthorId_1", target="AuthorId_2", edge_attr=True
    )
    logger.debug(
        "number of nodes: {}, number of edges {}".format(
            G.number_of_nodes(), G.number_of_edges()
        )
    )

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
    return G


def load_graph_and_partitions(fpath_edgelist, local_partition_dirpath, min_weight=None):
    G = load_edgelist(fpath_edgelist, min_weight)
    ego_partition = load_partition_data(local_partition_dirpath, G)

    return G, ego_partition


def _get_connected_components(G: nx.Graph) -> Partition:
    return {i: n for i, n in enumerate(nx.connected_components(G))}


def load_partition_data(local_partition_dirpath, G):
    logger.debug(
        "loading ego_partition data from directory: {}".format(local_partition_dirpath)
    )
    ego_partition = load_ego_partition(local_partition_dirpath)
    ego_partition = {int(k): v for k, v in ego_partition.items()}

    # fill in missing data
    for node, comps in ego_partition.items():
        if comps is None:
            ego_net_minus_ego = G.subgraph(G.neighbors(node))
            ego_partition[node] = _get_connected_components(ego_net_minus_ego)
    return ego_partition


class ClusterHelper(PaperCollectionHelper):

    """Class for a cluster, obtained by clustering a co-authorship network"""

    def __init__(
        self,
        cl_id: int,
        subgraph: nx.Graph,
        data: Optional["DataHelper"] = None,
        mag_data=None,
        weight: str = "weight",
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> None:
        """
        :cl_id: cluster ID
        :subgraph: The subgraph of the co-authorship network for this cluster
        :data: DataHelper object
        :mag_data: MagData object (deprecated, use DataHelper instead)
        :weight: for the subgraph, name of the edge attribute for edge weight (None for unweighted)
        """
        self.id = cl_id
        self.subgraph = subgraph
        self.author_ids = list(subgraph.nodes)
        self.data = data
        self.mag_data = mag_data
        if self.mag_data is None and self.data is not None:
            self.mag_data = self.data.mag_data
        self.weight = weight
        self.min_year = min_year
        self.max_year = max_year

        self._authors = None  # lazy loading. see property below.

        self.paper_ids = None
        self.paper_weights = None
        if self.mag_data is not None:
            relevance_scores = self.get_relevance_scores()
            self.paper_ids = relevance_scores.index.tolist()
            self.paper_weights = relevance_scores.tolist()
        super().__init__(self.paper_ids, self.paper_weights, data=self.data, id=self.id)

        self._affiliations = None  # lazy loading, see property below

    @property
    def affiliations(self) -> List[str]:
        # list of (DisplayName) affiliations according to the papers, sorted descending by frequency
        # TODO: this was copied from AuthorHelper. Review the method for clusters, and consider changing
        if self._affiliations is None:
            df = self.df_paper_authors
            df = df[df["AuthorId"].isin(self.author_ids)]
            df = df.merge(
                self.data.mag_data.affiliations, how="inner", on="AffiliationId"
            )
            affil = (
                df["AffiliationId"]
                .dropna()
                .map(
                    self.data.mag_data.affiliations.set_index("AffiliationId")[
                        "DisplayName"
                    ]
                )
            )
            self._affiliations = affil.value_counts().index.tolist()
        return self._affiliations

    @property
    def authors(self) -> pd.DataFrame:
        """
        pandas dataframe, including columns 'AuthorId', 'NormalizedName', 'pagerank' and others
        """
        if self._authors is None:
            pageranks = nx.pagerank(self.subgraph, weight=self.weight)
            authors = pd.Series(pageranks, name="pagerank")
            authors.index.name = "AuthorId"
            authors = authors.reset_index()
            authors = authors.merge(
                self.mag_data.authors[
                    [
                        "AuthorId",
                        "NormalizedName",
                        "DisplayName",
                        "LastKnownAffiliationId",
                        "PaperCount",
                        "CitationCount",
                    ]
                ],
                how="inner",
                on="AuthorId",
            )
            authors.sort_values("pagerank", ascending=False, inplace=True)
            self._authors = authors
        return self._authors

    def get_relevance_scores(self) -> pd.Series:
        """
        Relevance scores for the papers in a cluster
        If the paper has fewer than two authors within the cluster, the score is 0
        Otherwise, each paper is given a score in proportion to the sum of this cluster's authors' pagerank scores,
        combined with an indicator of whether the author was first or last author on this paper (more heavily weighted),
        or a middle author (lower weight). Normalized by number of authors in the paper.

        Returns a pandas Series with index 'PaperId' and weights (between 0 and 1) as values
        """
        paa = self.mag_data.paper_authors
        if "num_authors" not in paa.columns:
            paa["num_authors"] = paa.groupby("PaperId")[
                "AuthorSequenceNumber"
            ].transform("max")
        if "is_last_author" not in paa.columns:
            paa["is_last_author"] = np.where(
                paa["num_authors"] == paa["AuthorSequenceNumber"], True, False
            )
        paa_subset = paa[paa.AuthorId.isin(self.author_ids)].copy()
        paa_subset = paa_subset[
            ["PaperId", "AuthorId", "AuthorSequenceNumber", "is_last_author"]
        ].drop_duplicates()

        # Filter by time period
        if self.min_year is not None or self.max_year is not None:
            paa_subset = paa_subset.merge(
                self.mag_data.papers[["PaperId", "Year"]], on="PaperId", how="inner"
            )
            if self.min_year is not None:
                paa_subset = paa_subset[paa_subset["Year"] >= self.min_year]
            if self.max_year is not None:
                paa_subset = paa_subset[paa_subset["Year"] < self.max_year]

        # Filter out papers with fewer than 2 authors in this cluster
        gb = paa_subset.groupby("PaperId")
        paa_subset = gb.filter(lambda x: len(x) > 1)

        # merge in pagerank data
        # if pagerank data is already cached, use that, otherwise calculate it
        if self._authors is not None:
            paa_subset = paa_subset.merge(
                self.authors[["AuthorId", "pagerank"]], on="AuthorId", how="inner"
            )
        else:
            paa_subset["pagerank"] = paa_subset["AuthorId"].map(
                nx.pagerank(self.subgraph, weight=self.weight)
            )

        multiplier_first_or_last_author = 1.0
        cond1 = paa_subset["AuthorSequenceNumber"] == 1
        multiplier_middle_author = 0.75
        cond2 = paa_subset["is_last_author"] == True
        multiplier = np.where(
            cond1 | cond2, multiplier_first_or_last_author, multiplier_middle_author
        )
        multiplier = multiplier * paa_subset["pagerank"].values
        paa_subset["score"] = multiplier
        return (
            paa_subset.groupby("PaperId")["score"].mean().sort_values(ascending=False)
        )

    def to_card(self):
        """Overrides PaperCollectionHelper.to_card()"""
        logger.debug("getting dygie terms")
        # papers_s = pd.Series(self.paper_weights, index=self.paper_ids)
        # dygie_terms = {lbl: x.tolist() for lbl, x in self.df_terms.groupby('label')['term_display']}
        dygie_terms = {
            label: self.strategy_textrank(label)
            for label in self.df_terms["label"].unique()
        }
        for label in dygie_terms:
            dygie_terms[label] = self.check_for_terms_to_drop(dygie_terms[label])

        similarity_threshold = 0.6
        N = 20
        # dygie_terms_trunc = {lbl: terms[:10] for lbl, terms in dygie_terms.items()}
        dygie_terms_trunc = {
            label: self.get_top_terms_with_similarity_threshold(
                dygie_terms[label],
                self.similarity_graphs[label],
                similarity_threshold,
                N,
            )
            for label in self.df_terms["label"].unique()
        }

        dygie_terms = self.terms_final_cleaning(dygie_terms)
        dygie_terms_trunc = self.terms_final_cleaning(dygie_terms_trunc)
        authors = self.authors.DisplayName.head(N).tolist()
        detailed_data = self.get_details(
            dygie_terms=dygie_terms,
            affiliations=self.affiliations,
            authors=self.authors.DisplayName.tolist(),
        )
        return self._to_card(
            id=self.id,
            type="group",
            papers=self,
            authors=authors,
            affiliations=self.affiliations,
            numPapers=len(self.paper_ids),
            numAuthors=len(self.author_ids),
            topics=None,
            score=None,
            dygie_terms=dygie_terms_trunc,
            author_ids=self.author_ids,
            details=detailed_data,
        )


# partition distance method from https://arxiv.org/pdf/1905.11230.pdf
def jaccard_distance(community1, community2):
    community1 = set(community1)
    community2 = set(community2)
    return 1 - (
        len(set.intersection(community1, community2))
        / len(set.union(community1, community2))
    )


def min_jaccard_distance(community1, partition2):
    dists = [
        jaccard_distance(community1, community2)
        for name, community2 in partition2.items()
    ]
    return min(dists)


def avg_weighted_jaccard_dist(
    community1, partition2, num_community_assignments_parition1
):
    return (
        min_jaccard_distance(community1, partition2)
        * len(community1)
        / num_community_assignments_parition1
    )


def partition_distance(partition1, partition2):
    num_community_assignments_p1 = sum(map(len, partition1.values()))
    num_community_assignments_p2 = sum(map(len, partition2.values()))
    dist1 = sum(
        [
            avg_weighted_jaccard_dist(c, partition2, num_community_assignments_p1)
            for c in partition1.values()
        ]
    )
    return dist1


def symmetric_partition_distance(partition1, partition2):
    dist1 = partition_distance(partition1, partition2)
    dist2 = partition_distance(partition2, partition1)
    return (dist1 / 2) + (dist2 / 2)

