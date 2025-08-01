# -*- coding: utf-8 -*-

DESCRIPTION = """classes for paper collections"""

import sys, os, time, json, tarfile
from pathlib import Path
from typing import Optional, Iterable, Sequence, Union, List, Collection, Dict
import dataclasses
from itertools import combinations
from datetime import datetime
from timeit import default_timer as timer

from dacite import data

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

# from mag_data import MagData

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    cosine_distances,
)
from sklearn.preprocessing import MinMaxScaler

from .util import (
    _tfidf_apply,
    drop_duplicate_titles,
    get_score_column,
    dataclass_from_dict,
)
from .cards import (
    BridgerCard,
    DygieTerms,
    MagTopic,
    MagAuthor,
    BridgerCardDetails,
    Paper,
)


class TermRanker:
    def __init__(
        self,
        data: "DataHelper",
        df_terms: pd.DataFrame,
        df_terms_embeddings: pd.DataFrame,
    ) -> None:
        self.data = data
        self.df_terms = df_terms
        self.df_terms_embeddings = df_terms_embeddings

        self._similarity_graphs = None  # lazy loading, see property below

    @property
    def similarity_graphs(self) -> Dict[str, nx.Graph]:
        if self._similarity_graphs is None:
            self._similarity_graphs = {}
            for lbl, gdf in self.df_terms_embeddings.groupby("label"):
                try:
                    self._similarity_graphs[lbl] = self.get_sim_graph(
                        gdf, distance="euclidean"
                    )
                except ValueError:
                    logger.exception(
                        "error getting similarity graph for label {}".format(lbl)
                    )
                    # create a graph with no edges
                    nodes = gdf["embedding_term"].drop_duplicates().tolist()
                    G = nx.Graph()
                    G.add_nodes_from(nodes)
                    self._similarity_graphs[lbl] = G

        # make sure the necessary labels do exist
        for lbl in ["Method", "Task", "Metric", "Material"]:
            if lbl not in self._similarity_graphs:
                # just add empty graph
                self._similarity_graphs[lbl] = nx.Graph()

        return self._similarity_graphs

    def get_sim_graph(
        self, df_terms_embeddings: pd.DataFrame, distance="euclidean"
    ) -> nx.Graph:
        graph = nx.Graph()
        if distance.lower() == "euclidean":
            f_distance = euclidean_distances
        elif distance.lower().startswith("cosine"):
            f_distance = cosine_distances
        else:
            raise ValueError("`distance` must be one of ['euclidean', 'cosine']")
        embs = df_terms_embeddings.term_idx.map(lambda x: self.data.embeddings[x])
        embs = np.array(embs.tolist())
        dist = f_distance(embs)
        # dist = MinMaxScaler().fit_transform(dist)
        # # subtract from 1 to get similarity
        # dist = 1 - dist
        graph = nx.from_numpy_array(dist, create_using=graph)
        # rename_nodes = {i: name for i, name in enumerate(df_terms_embeddings.embedding_term)}
        rename_nodes = {
            i: name for i, name in enumerate(df_terms_embeddings.term_display)
        }
        graph = nx.relabel_nodes(graph, rename_nodes)
        # rescale weights and subtract from 1 to get similarity
        weights = np.array([w for u, v, w in graph.edges.data("weight")])
        weights = MinMaxScaler().fit_transform(weights.reshape(-1, 1)).flatten()
        weights = 1 - weights
        for i, (u, v) in enumerate(graph.edges()):
            graph[u][v]["weight"] = weights[i]

        return graph

    def get_textrank(self, G: nx.Graph) -> Dict[str, float]:
        return nx.pagerank_numpy(G, weight="weight")

    def _strategy_sorted_col(
        self, df: pd.DataFrame, colname: str, N: int = 20, ascending: bool = False
    ) -> List[str]:
        return (
            df.sort_values(colname, ascending=ascending)
            .head(N)["term_display"]
            .tolist()
        )

    def strategy_freq(self, label: str, N: int = 20) -> List[str]:
        df = self.df_terms[self.df_terms["label"] == label]
        return self._strategy_sorted_col(df, "freq", N, ascending=False)

    def strategy_relevance_score(self, label: str, N: int = 20) -> List[str]:
        df = self.df_terms[self.df_terms["label"] == label]
        return self._strategy_sorted_col(df, "relevance_score", N, ascending=False)

    def strategy_textrank(self, label: str, N: int = 20) -> List[str]:
        G = self.similarity_graphs[label]
        if G is None or G.number_of_nodes() == 0:
            return []
        if G.number_of_edges() == 0:
            return list(G.nodes())
        textrank = self.get_textrank(G)
        textrank_sorted = pd.Series(textrank).sort_values(ascending=False)
        return textrank_sorted.index.tolist()[:N]

    def strategy_random(self, label: str, N: int = 20) -> List[str]:
        df = self.df_terms[self.df_terms["label"] == label]
        if len(df) > N:
            return df.sample(N, random_state=1)["term_display"].tolist()
        else:
            return df.sample(frac=1, random_state=1)["term_display"].tolist()

    def check_for_terms_to_drop(self, term_list: List[str]) -> List[str]:
        if self.data.terms_to_drop:
            return [term for term in term_list if term not in self.data.terms_to_drop]
        return term_list

    def get_dygie_terms(self) -> DygieTerms:
        """Uses default ranking to get dygie terms

        Returns:
            DygieTerms: dygie terms ranked by default strategy
        """
        terms_dict = {
            # label: self.strategy_textrank(label, N=1000)
            label: self.strategy_relevance_score(label, N=1000)
            for label in self.df_terms["label"].unique()
        }
        for label in terms_dict:
            terms_dict[label] = self.check_for_terms_to_drop(terms_dict[label])
        return DygieTerms(
            Method=terms_dict.get("Method", []),
            Task=terms_dict.get("Task", []),
            Material=terms_dict.get("Material", []),
            Metric=terms_dict.get("Metric", []),
        )

    def get_dygie_terms_trunc(
        self, dygie_terms: DygieTerms, similarity_threshold: float = 0.6, N: int = 20
    ) -> DygieTerms:
        dygie_terms_trunc = {}
        labels = ["Method", "Task", "Material", "Metric"]
        for label in labels:
            terms = getattr(dygie_terms, label, [])
            terms = self.get_top_terms_with_similarity_threshold(
                terms, self.similarity_graphs[label], similarity_threshold, N
            )
            dygie_terms_trunc[label] = terms
        # return dygie_terms_trunc
        return DygieTerms(
            Method=dygie_terms_trunc.get("Method", []),
            Task=dygie_terms_trunc.get("Task", []),
            Material=dygie_terms_trunc.get("Material", []),
            Metric=dygie_terms_trunc.get("Metric", []),
        )

    def get_top_terms_with_similarity_threshold(
        self,
        terms_sorted_by_relevance: List[str],
        G: nx.Graph,
        similarity_threshold: float,
        threshold_full_list: Optional[float] = 0.75,
        N: int = 20,
    ) -> List[str]:
        """
        :similarity_threshold: each term will be compared against the last term added. if the new term has a similarity above this threshold, it will not be added.
        :threshold_full_list: this is a more aggressive threshold, to compare each term against *all* the other terms already in the list
        """
        if len(terms_sorted_by_relevance) < 5:
            # not enough terms. forget about thresholding
            return terms_sorted_by_relevance
        top_terms = []
        # logger.debug(
        #     "getting top terms with similarity threshold: using similarity threshold {}".format(
        #         similarity_threshold
        #     )
        # )
        for term in terms_sorted_by_relevance:
            term = term.strip()
            if term in top_terms:
                continue
            try:
                if len(top_terms) == 0:
                    top_terms.append(term)
                    continue
                last_term = top_terms[-1]
                sim = G[term][last_term]["weight"] if G.has_edge(term, last_term) else 0
                if sim > similarity_threshold:
                    # checks if this new term is too similar to the previous term in the top_terms list
                    pass
                elif threshold_full_list is not None and any(
                    G[term][other_term]["weight"] > threshold_full_list
                    for other_term in top_terms
                    if G.has_edge(term, other_term)
                ):
                    # checks if this new term is too similar to *any* of the terms in the top_terms list
                    pass
                else:
                    top_terms.append(term)
            except Exception as e:
                logger.exception(
                    "Exception encountered for term {}: {}".format(term, e)
                )
                top_terms.append(term)
            if len(top_terms) == N:
                break
        return top_terms


class PaperCollectionHelper:

    """collection of MAG papers"""

    def __init__(
        self,
        paper_ids: Iterable[Union[str, int]],
        paper_weights: Optional[Iterable[float]] = None,
        data: Optional["DataHelper"] = None,
        id: Optional[Union[str, int]] = None,
        description: Optional[str] = None,
        collection_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """

        :paper_ids: (MAG) paper ids for this collection
        :paper_weights: list of weights for each paper, in the same order as paper_ids
        :data: DataHelper object

        """
        self.paper_ids = paper_ids
        self.paper_weights = paper_weights
        if self.paper_weights is None:
            self.paper_weights = [1] * len(self.paper_ids)
        self.data = data
        self.id = id
        self.description = description
        self.collection_type = collection_type
        self.name = name

        self._df_papers = None  # lazy loading, see property below
        self._df_paper_authors = None  # lazy loading, see property below
        self._dedup_title_paper_ids = None  # lazy loading, see property below
        self._df_terms = None  # lazy loading, see property below
        self._df_terms_embeddings = None  # lazy loading, see property below
        self._similarity_graphs = None  # lazy loading, see property below
        self._dygie_terms = None  # lazy loading, see property below
        self._term_ranker = None  # lazy loading, see property below
        self._mag_topics = None  # lazy loading, see property below
        self._papers_for_card = None  # lazy loading, see property below
        self._s2_ids_to_specter = None  # lazy loading, see property below

    @classmethod
    def from_defaults(cls, paper_ids, paper_weights=None, **kwargs):
        """load object from defaults (see source code)"""
        from .data_helper import DataHelper

        data = DataHelper.from_defaults()
        if paper_ids is None or len(paper_ids) < 1:
            raise ValueError("must specify a set of paper IDs")
        obj = cls(paper_ids, paper_weights=paper_weights, data=data, **kwargs)
        return obj

    @property
    def df_papers(self):
        if self._df_papers is None:
            df = self.data.mag_data.papers
            papers = pd.Series(
                self.paper_weights, index=self.paper_ids, name="relevance_score"
            )
            df = df.merge(papers, how="inner", left_on="PaperId", right_index=True)
            df = df.merge(self.data.df_s2_id, how="left", on="PaperId")
            self._df_papers = df
        return self._df_papers

    @df_papers.setter
    def df_papers(self, val):
        self._df_papers = val

    @property
    def df_paper_authors(self):
        if self._df_paper_authors is None:
            df = self.data.mag_data.paper_authors
            df = df[df["PaperId"].isin(self.paper_ids)]
            self._df_paper_authors = df
        return self._df_paper_authors

    @property
    def dedup_title_paper_ids(self):
        if self._dedup_title_paper_ids is None:
            from .util import drop_duplicate_titles

            self._dedup_title_paper_ids = drop_duplicate_titles(self.df_papers)[
                "PaperId"
            ]
        return self._dedup_title_paper_ids

    @property
    def df_terms(self):
        if self._df_terms is None:
            logger.debug("getting df_terms")
            groupby_cols = ["label", "term_normalized", "term_display", "term_id"]
            df = self._merge_terms_and_papers(self.data.df_ner, groupby_cols)
            self._df_terms = df
        return self._df_terms

    @df_terms.setter
    def df_terms(self, val):
        self._df_terms = val

    @property
    def df_terms_embeddings(self):
        if self._df_terms_embeddings is None:
            logger.debug("getting df_terms_embeddings")
            df = self.get_terms_embeddings_mapping()
            # df = self.get_tfidf_data(df)
            self._df_terms_embeddings = df
        return self._df_terms_embeddings

    @df_terms_embeddings.setter
    def df_terms_embeddings(self, val):
        self._df_terms_embeddings = val

    @property
    def dygie_terms(self) -> DygieTerms:
        if self._dygie_terms is None:
            logger.debug("getting dygie terms")
            self._dygie_terms = self.term_ranker.get_dygie_terms()
        return self._dygie_terms

    @property
    def term_ranker(self) -> TermRanker:
        if self._term_ranker is None:
            self._term_ranker = TermRanker(
                self.data, self.df_terms, self.df_terms_embeddings
            )
        return self._term_ranker

    @property
    def mag_topics(self) -> pd.DataFrame:
        if self._mag_topics is None:
            self._mag_topics = self.get_mag_topics()
        return self._mag_topics

    @property
    def papers_for_card(self) -> List[Paper]:
        if self._papers_for_card is None:
            self._papers_for_card = self.get_papers()
        return self._papers_for_card

    @property
    def coauthors(self):
        # Not implemented in this class. Implement it in subclasses for which it makes sense (like AuthorHelper)
        return None

    @property
    def s2_ids_to_specter(self) -> pd.Series:
        """returns a series with index s2_id and values specter embedding (as np.ndarray)

        Returns:
            pd.Series
        """
        if self._s2_ids_to_specter is None:
            df_papers = self.df_papers[
                self.df_papers.PaperId.isin(self.dedup_title_paper_ids)
            ].sort_values("Rank")
            specter_embeddings = self.data.specter_embeddings
            author_s2_ids: pd.Series = df_papers["s2_id"].dropna().drop_duplicates()
            # idxs = specter_embeddings_paper_ids[
            #     specter_embeddings_paper_ids.isin(author_s2_ids)
            # ]
            # author_specter = specter_embeddings[idxs.index.tolist()]
            author_s2_ids = author_s2_ids[author_s2_ids.isin(specter_embeddings.index)]
            self._s2_ids_to_specter = specter_embeddings.loc[author_s2_ids]
        return self._s2_ids_to_specter

    # def load_mag_data(self,
    #         path_to_data: str,
    #         tablenames: Optional[List[str]] = None
    #         ) -> None:
    #     from mag_data import MagData
    #     logger.debug("loading MagData from {}".format(path_to_data))
    #     self.mag_data = MagData(path_to_data, tablenames=tablenames)

    # def load_s2_mapping(self,
    #         path_to_data: str
    #         ) -> None:
    #     logger.debug("loading s2 mapping from file {}".format(path_to_data))
    #     self.df_s2_id = pd.read_parquet(path_to_data)

    # def load_ner_data(self,
    #         path_to_terms: str,
    #         path_to_counts: str
    #         ) -> None:
    #     logger.debug("loading ner terms from {}".format(path_to_terms))
    #     self.df_ner = pd.read_parquet(path_to_terms)
    #     logger.debug("dataframe shape: {}".format(self.df_ner.shape))
    #     logger.debug("loading ner term counts from {}".format(path_to_counts))
    #     self.grp_counts = pd.read_parquet(path_to_counts)
    #     logger.debug("counts dataframe shape: {}".format(self.grp_counts.shape))

    # def load_embeddings(self,
    #         path_to_embeddings: str,
    #         path_to_terms: str
    #         ) -> None:
    #     logger.debug("loading embeddings and corresponding terms")
    #     self.embeddings = np.load(path_to_embeddings)
    #     self.embeddings_terms = np.load(path_to_terms, allow_pickle=True)
    #     self.embeddings_terms_to_idx = {val: idx for idx, val in np.ndenumerate(self.embeddings_terms)}

    # def prepare_ner_tfidf(self) -> None:
    #     logger.debug("preparing ner data for tfidf...")
    #     self.df_tfidf = self.df_s2_id.merge(self.df_ner, how='inner', on='s2_id')
    #     self.N_tfidf = self.df_tfidf['PaperId'].nunique()
    #     logger.debug("done. {} unique terms overall".format(self.N_tfidf))

    def df_papers_dedup_titles(self) -> pd.DataFrame:
        return self.df_papers[
            self.df_papers["PaperId"].isin(self.dedup_title_paper_ids)
        ]

    def get_mag_topics(self) -> pd.DataFrame:
        logger.debug("getting MAG topics")
        papers = pd.Series(
            self.paper_weights, index=self.paper_ids, name="relevance_score"
        )
        df = self.data.mag_data.paper_fos.merge(
            papers, how="inner", left_on="PaperId", right_index=True
        )
        df = df[df["Score"] > 0]
        df = df.merge(
            self.data.mag_data.fos[["FieldOfStudyId", "Level", "DisplayName"]],
            on="FieldOfStudyId",
        )
        # filter out Level 0 topics
        levels_to_drop = [0]
        df = df[~(df["Level"].isin(levels_to_drop))]
        df_magtopics = (
            df.groupby(["FieldOfStudyId", "Level", "DisplayName"])["Score"]
            .sum()
            .reset_index()
        )
        return df_magtopics.sort_values("Score", ascending=False)

    def get_papers(self) -> List[Paper]:
        # get paper details
        columns_map = {
            "PaperId": "mag_id",
            "OriginalTitle": "title",
            "Year": "year",
            "venue_name": "venue",
            "Rank": "Rank",
            "Doi": "doi",
            "s2_id": "s2Id",
        }
        # df = self.df_papers[columns_map.keys()]
        # df = df[df["PaperId"].isin(self.dedup_title_paper_ids)]
        df = self.df_papers_dedup_titles()
        df = df[columns_map.keys()]
        df = df.rename(columns=columns_map)

        df_authors = self.data.mag_data.paper_authors
        df_authors = df_authors[df_authors["PaperId"].isin(df["mag_id"])]
        df_authors = df_authors[
            ["PaperId", "AuthorId", "AuthorSequenceNumber", "OriginalAuthor"]
        ].drop_duplicates()
        df_authors = df_authors.sort_values(["PaperId", "AuthorSequenceNumber"])
        # authors_list = authors_list.groupby("PaperId")["OriginalAuthor"].agg(
        #     lambda x: list(x)
        # )
        # df["authors"] = df["mag_id"].map(authors_list)

        # papers = df.to_dict(orient="records")
        # papers = [Paper(**p) for p in papers]
        # papers = [dataclass_from_dict(Paper, p) for p in papers]
        papers = []
        for _, row in df.iterrows():
            p = row.to_dict()
            authors = []
            for _, author_row in df_authors[df_authors["PaperId"] == p["mag_id"]].iterrows():
                authors.append(
                    MagAuthor(
                        AuthorId=author_row["AuthorId"],
                        DisplayName=author_row["OriginalAuthor"],
                    )
                )
            p["authors"] = authors
            papers.append(Paper(**p))

        return papers

    # deprecated
    def get_tfidf_data(self, terms: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        terms = terms.merge(
            # self.data.grp_counts, how="inner", on=["label", "term_display"]
            self.data.grp_counts,
            how="inner",
            on=["label", "term_cleaned"],
        )
        terms.rename(columns={"term_count": "all_count"}, inplace=True)
        terms["term_tfidf"] = terms.apply(
            _tfidf_apply, N=self.data.N_tfidf, tf_colname="relevance_score", axis=1
        )
        # out = {}
        # for lbl, gdf in terms.groupby('label'):
        #     # out[lbl] = gdf.sort_values('term_tfidf', ascending=False)['term_display'].head(10).tolist()
        #     out[lbl] = gdf.sort_values('term_tfidf', ascending=False)
        # return out
        return terms.sort_values("term_tfidf", ascending=False)

    def _merge_terms_and_papers(
        self, df, groupby_cols, papers: Optional[pd.Series] = None
    ):
        if papers is None:
            papers = pd.Series(self.paper_weights, index=self.paper_ids)
        to_keep = self.dedup_title_paper_ids
        papers = papers.reindex(to_keep)
        papers.name = "relevance_score"
        # TODO this takes too long
        x = df.merge(papers, left_on="PaperId", right_index=True)
        # cl_terms = x.groupby(['label', 'term_display'])['cl_score'].sum().reset_index()
        logger.debug("calculating term scores")
        # ! freq is number of papers that have the term at least once. it ignores the "freq" in pre-groupby data, which represents the frequence per paper. revisit this.
        df_terms = (
            x.groupby(groupby_cols)["relevance_score"]
            .agg(relevance_score="sum", freq="count")
            .reset_index()
        )
        return df_terms

    def get_terms_embeddings_mapping(
        self, papers: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        groupby_cols = ["label", "embedding_term", "term_display", "term_idx"]
        df_terms = self._merge_terms_and_papers(
            self.data.df_paper_term_embeddings, groupby_cols, papers
        )
        n_terms = len(df_terms)
        df_terms = df_terms.dropna(subset=["term_idx"])
        df_terms["term_idx"] = df_terms["term_idx"].astype(int)
        if len(df_terms) != n_terms:
            logger.debug(
                f"could not find embeddings for {n_terms-len(df_terms)} of {n_terms} terms. dropping these..."
            )
        return df_terms

    def get_details(
        self,
        dygie_terms=None,
        affiliations: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        coauthors: Optional[List[MagAuthor]] = None,
    ) -> BridgerCardDetails:
        if dygie_terms is None:
            dygie_terms = self.final_cleaning_dygie_terms(self.dygie_terms)

        if authors is None:
            authors = getattr(self, "authors", [])

        if affiliations is None:
            affiliations = getattr(self, "affiliations", [])

        mag_topics = [
            MagTopic(**topic_dict)
            # dataclass_from_dict(MagTopic, topic_dict)
            for topic_dict in self.mag_topics.to_dict(orient="records")
        ]

        out = {
            "id": self.id,
            "authors": authors,
            "affiliations": affiliations,
            "type": self.collection_type,
            "dygie_terms": dygie_terms,
            "topics": mag_topics,
        }
        if coauthors is not None:
            out["coauthors"] = coauthors
        papers = self.papers_for_card
        out["papers"] = papers
        return BridgerCardDetails(**out)
        # return dataclass_from_dict(BridgerCardDetails, out)

    def check_for_terms_to_drop(self, term_list: List[str]) -> List[str]:
        if self.data.terms_to_drop:
            return [term for term in term_list if term not in self.data.terms_to_drop]
        return term_list

    def _to_card(self, cls=BridgerCard, **kwargs) -> BridgerCard:
        """Generic method to be used in general when creating a card"""
        # if 'papers' not in kwargs:
        #     kwargs['papers'] = self
        if "score" not in kwargs:
            kwargs["score"] = 1.0
        return cls(**kwargs)
        # return dataclass_from_dict(cls, kwargs)

    def terms_final_cleaning(self, terms: List[str]) -> List[str]:
        df = self.df_terms.drop_duplicates(subset=["term_display"])
        is_substring = df["term_display"].apply(
            lambda x: df["term_display"].str.contains(x, regex=False).sum() > 1
        )
        df = df[~(is_substring)]
        df = df.set_index("term_display")
        df = df.reindex(terms)
        # ct = df["term_display"].dropna()
        # ct = ct.apply(lambda x: x.strip()).drop_duplicates()
        ct = df.index.to_series().apply(lambda x: x.strip()).drop_duplicates()

        return ct.dropna().tolist()

    def final_cleaning_dygie_terms(
        self, terms: Union[DygieTerms, Dict[str, List[str]]]
    ) -> DygieTerms:
        if dataclasses.is_dataclass(terms):
            terms = dataclasses.asdict(terms)
        cleaned_terms = {}
        for label in terms:
            this_terms = terms[label]
            cleaned_terms[label] = self.terms_final_cleaning(this_terms)
        cleaned_terms = DygieTerms(**cleaned_terms)
        # cleaned_terms = dataclass_from_dict(DygieTerms, cleaned_terms)
        return cleaned_terms

    def to_card(self, cls=BridgerCard, papers: bool = True):
        """Get a BridgerCard object representing this paper collection
        This method may be overridden by sub-classes (e.g., AuthorHelper)
        """
        dygie_terms = self.dygie_terms
        dygie_terms_trunc = self.term_ranker.get_dygie_terms_trunc(dygie_terms)

        # final cleaning
        dygie_terms_trunc = self.final_cleaning_dygie_terms(dygie_terms_trunc)

        mag_topics = [
            MagTopic(**topic_dict)
            # dataclass_from_dict(MagTopic, topic_dict)
            for topic_dict in self.mag_topics.to_dict(orient="records")
        ]
        papers = self.papers_for_card
        papers.sort(key=lambda x: x.Rank)
        papers = papers[:5]
        return self._to_card(
            cls=cls,
            id=self.id,
            type=self.collection_type,
            # papers=self,
            affiliations=None,
            numPapers=len(self.paper_ids),
            topics=mag_topics[:15],
            score=None,
            dygie_terms=dygie_terms_trunc,
            displayName=self.name,
            # details=detailed_data,
            papers=papers,
        )


class PaperHelper:

    """MAG paper"""

    def __init__(
        self,
        paper_id: int,
        data: Optional["DataHelper"] = None,
        df_forecite: Optional[pd.DataFrame] = None,
    ) -> None:
        """

        :paper_id: MAG PaperId
        :data: DataHelper object
        :df_forecite: dataframe with ForeCite data

        """
        self.paper_id = paper_id
        self.data = data
        self.df_forecite = df_forecite

        self._ranked_fos = None

        self._title = None

    @property
    def title(self):
        if self._title is None:
            if (
                self.data is not None
                and self.data.mag_data is not None
                and self.data.mag_data.papers is not None
            ):
                _df = self.data.mag_data.papers
                self._title = _df[_df["PaperId"] == self.paper_id]["PaperTitle"].iloc[0]
        return self._title

    @title.setter
    def title(self, val):
        self._title = val

    @property
    def ranked_fos(self):
        if self._ranked_fos is None:
            self._ranked_fos = self.get_fos_tfidf()
        return self._ranked_fos

    def get_fos_tfidf(self, include_names=True) -> pd.DataFrame:
        """Get FieldsOfStudy for this paper, with TF-IDF weights
        :returns: dataframe

        """
        df = self.data.mag_data.paper_fos
        df = df[df["PaperId"] == self.paper_id]
        term_count = df.groupby("FieldOfStudyId")["Score"].sum()
        term_count = term_count[term_count > 0]
        if term_count.empty:
            return None
        term_count.name = "term_count"
        term_count = term_count.reset_index()
        term_count["all_count"] = term_count["FieldOfStudyId"].map(
            self.data.mag_data.fos_counts
        )
        N = self.data.mag_data.num_papers
        term_count["tfidf"] = term_count.apply(_tfidf_apply, N=N, axis=1)
        if include_names is True:
            to_join = self.data.mag_data.fos[
                ["FieldOfStudyId", "NormalizedName", "Level"]
            ]
            term_count = term_count.merge(to_join, how="inner", on="FieldOfStudyId")
        return term_count.sort_values("tfidf", ascending=False)


class AuthorHelper(PaperCollectionHelper):
    """Class for an author (from 1 or more MAG author IDs)"""

    def __init__(
        self,
        author_id: Union[str, Collection[str]],
        data: Optional["DataHelper"] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        name: Optional[str] = None,
        s2_id: Optional[int] = None,
        collection_type: str = "author",
        coauthor_graph: Optional[nx.Graph] = None,
    ) -> None:
        """
        :author_id: MAG AuthorId, or list
        :data: DataHelper object
        """
        self.author_id = author_id
        self.data = data
        self.min_year = min_year
        self.max_year = max_year
        self.name = name
        self.s2_id = s2_id
        self.collection_type = collection_type
        self.coauthor_graph = coauthor_graph

        self._affiliations = None  # lazy loading, see property below
        self._paa_subset = None  # lazy loading, see property below
        self._coauthors = None  # lazy loading, see property below

        # check if author_id is a collection (e.g., list) of ids, or if it is a single id
        if isinstance(self.author_id, str) or not isinstance(
            self.author_id, Collection
        ):
            self.author_id = [self.author_id]

        # assign the first AuthorId as the `id` for this author
        self.id = self.author_id[0]

        if self.name is None:
            self.name = self.data.mag_data.author_lookup(self.id)["DisplayName"]

        self.paper_ids = None
        self.paper_weights = None
        if self.data is not None:
            self.paper_ids, self.paper_weights = self.get_paper_ids()
            # self.papers = PaperCollectionHelper(self.paper_ids, self.paper_weights, data=self.data, id=self.name)
        super().__init__(
            self.paper_ids,
            self.paper_weights,
            data=self.data,
            id=self.id,
            collection_type=self.collection_type,
            name=self.name,
        )

        if self.name is not None:
            self.authors = [self.name]
        else:
            self.authors = []

    @property
    def affiliations(self) -> List[str]:
        # list of (DisplayName) affiliations according to the papers, sorted descending by frequency
        if self._affiliations is None:
            df = self.df_paper_authors
            df = df[df["AuthorId"].isin(self.author_id)]
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
    def paa_subset(self) -> pd.DataFrame:
        # subset of PaperAuthorAffiliations table for this author
        if self._paa_subset is None:
            logger.debug("getting paper IDs from PaperAuthorAffiliations table")
            paa = self.data.mag_data.paper_authors
            if "num_authors" not in paa.columns:
                paa["num_authors"] = paa.groupby("PaperId")[
                    "AuthorSequenceNumber"
                ].transform("max")
            if "is_last_author" not in paa.columns:
                paa["is_last_author"] = np.where(
                    paa["num_authors"] == paa["AuthorSequenceNumber"], True, False
                )
            paa_subset = paa[paa.AuthorId.isin(self.author_id)].copy()
            paa_subset = paa_subset[
                ["PaperId", "AuthorId", "AuthorSequenceNumber", "is_last_author"]
            ].drop_duplicates()

            # Filter by time period
            if self.min_year is not None or self.max_year is not None:
                paa_subset = paa_subset.merge(
                    self.data.mag_data.papers[["PaperId", "Year"]],
                    on="PaperId",
                    how="inner",
                )
                if self.min_year is not None:
                    logger.debug("removing papers before year {}".format(self.min_year))
                    paa_subset = paa_subset[paa_subset["Year"] >= self.min_year]
                if self.max_year is not None:
                    logger.debug(
                        "removing papers on or after year {}".format(self.max_year)
                    )
                    paa_subset = paa_subset[paa_subset["Year"] < self.max_year]
            self._paa_subset = paa_subset
        return self._paa_subset

    @property
    def coauthors(self) -> List[MagAuthor]:
        if self._coauthors is None:
            self._coauthors = self.get_strong_coauthors()
        return self._coauthors

    def get_paper_ids(self):
        logger.debug("getting paper weights")
        paa_subset = self.paa_subset
        papers_subset = self.data.mag_data.papers
        papers_subset = papers_subset.loc[
            papers_subset["PaperId"].isin(paa_subset["PaperId"]), :
        ]
        paa_subset["score"] = get_score_column(paa_subset, papers_subset)
        paper_weights = (
            paa_subset.groupby("PaperId")["score"].mean().sort_values(ascending=False)
        )
        paper_ids = paper_weights.index.tolist()
        return paper_ids, paper_weights.tolist()

    def get_strong_coauthors(self, threshold: float = 0.005):
        logger.debug("getting strong co-authors")
        if self.coauthor_graph is None:
            raise RuntimeError(
                "cannot get strong coauthors because coauthor_graph is not loaded"
            )
        G = self.coauthor_graph
        if not G.has_node(self.id):
            raise RuntimeError(
                f"cannot get strong coauthors because node {self.id} does not exist in coauthor_graph"
            )
        d = []
        for nbr in G[self.id]:
            weighted_collab_score = (
                G[self.id][nbr]["weight"] / G.nodes[self.id]["n_pubs"]
            )
            if weighted_collab_score >= threshold:
                d.append({"AuthorId": nbr, "score": weighted_collab_score})
        if len(d) == 0:
            strong_coauthors = []
        else:
            df_coauthors = self.data.mag_data.authors[["AuthorId", "DisplayName"]]
            # df_coauthors = df_coauthors[df_coauthors["AuthorId"].isin(coauthor_ids)]
            df_coauthors = df_coauthors.merge(
                pd.DataFrame(d), how="inner", on="AuthorId"
            )
            df_coauthors = df_coauthors.sort_values("score", ascending=False).drop(
                columns="score"
            )
            strong_coauthors = [
                MagAuthor(**coauthor_dict)
                # dataclass_from_dict(MagAuthor, coauthor_dict)
                for coauthor_dict in df_coauthors.to_dict(orient="records")
            ]
        logger.debug(f"found {len(strong_coauthors)} co-authors")
        return strong_coauthors

    def to_card(self, cls=BridgerCard):
        """Overrides PaperCollectionHelper.to_card()"""
        dygie_terms = self.dygie_terms
        dygie_terms_trunc = self.term_ranker.get_dygie_terms_trunc(dygie_terms)

        # final cleaning
        dygie_terms = self.final_cleaning_dygie_terms(dygie_terms)
        dygie_terms_trunc = self.final_cleaning_dygie_terms(dygie_terms_trunc)

        if self.name is not None:
            authors = [self.name]
        else:
            authors = None

        mag_topics = [
            MagTopic(**topic_dict)
            # dataclass_from_dict(MagTopic, topic_dict)
            for topic_dict in self.mag_topics.to_dict(orient="records")
        ]
        papers = self.papers_for_card
        papers.sort(key=lambda x: x.Rank)
        papers = papers[:5]

        coauthors = self.coauthors

        logger.debug("outputting card")
        return self._to_card(
            cls=cls,
            id=self.id,
            type=self.collection_type,
            authors=authors,
            # papers=self,
            affiliations=self.affiliations,
            numPapers=len(self.paper_ids),
            topics=mag_topics,
            score=None,
            dygie_terms=dygie_terms_trunc,
            s2Id=self.s2_id,
            author_ids=self.author_id,
            displayName=self.name,
            papers=papers,
            coauthors=coauthors,
        )
