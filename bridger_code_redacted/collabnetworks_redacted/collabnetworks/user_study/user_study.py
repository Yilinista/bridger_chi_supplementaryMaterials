# -*- coding: utf-8 -*-

DESCRIPTION = """Collect data for user study"""

from dataclasses import dataclass
import sys, os, time, json, re
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    List,
    Union,
    Dict,
    Sequence,
    Tuple,
    MutableSet,
)
from string import ascii_uppercase, ascii_lowercase
from collections import defaultdict
from datetime import datetime
from timeit import default_timer as timer
import joblib
from ..util import dataclass_from_dict, config_dataclass_from_dict, sort_distance_df

from sklearn.feature_extraction.text import TfidfVectorizer

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
from networkx.readwrite import json_graph
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

from ..util import id_to_list, drop_duplicate_titles
from ..data_helper import DataHelper
from ..collection_helper import AuthorHelper, TermRanker, PaperCollectionHelper
from ..average_embeddings import AverageEmbeddings
from ..cards import (
    BridgerCard,
    BridgerCardDistance,
    BridgerCardSimilar,
    DygieTermsSimilar,
    BridgerCardDetails,
    PaperDistance,
)

# from .. import DATADIR
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DATADIR = Path(os.environ["DATADIR"])

EMBEDDINGS_DIRPATH = DATADIR.joinpath(
    "computer_science_papers_20210301/final_processed/embeddings/author_average_embeddings_2015-2022"
)
SPECTER_DIRPATH = DATADIR.joinpath(
    "computer_science_papers_20210301/specter_embeddings/"
)
COAUTHOR_GRAPH_FPATH = DATADIR.joinpath(
    "computer_science_papers_20210301/coauthor_2015-2022_minpubs3_collabweighted/coauthor_graph.pickle"
)


def _get_embeddings_files(embeddings_dirpath, specter_dirpath=SPECTER_DIRPATH):
    return {
        "Task": {
            "fname": embeddings_dirpath.joinpath(
                "average_author_embeddings_task_pandas.pickle"
            ),
            "fname_ids": embeddings_dirpath.joinpath("mat_author_task_row_labels.npy"),
        },
        "Method": {
            "fname": embeddings_dirpath.joinpath(
                "average_author_embeddings_method_pandas.pickle"
            ),
            "fname_ids": embeddings_dirpath.joinpath(
                "mat_author_method_row_labels.npy"
            ),
        },
        "specter": {
            "fname": specter_dirpath.joinpath(
                "average_author_specter_embeddings_2015-2022_pandas.pickle"
            ),
            "fname_ids": None,
        },
    }


EMBEDDINGS_FILES = _get_embeddings_files(EMBEDDINGS_DIRPATH)

EMBEDDINGS_ABBREVIATIONS_EXPANDED_DIRPATH = DATADIR.joinpath(
    "abbreviation_expansion_20210502/final_processed/embeddings/author_average_embeddings_2015-2022"
)

EMBEDDINGS_ABBREVIATIONS_EXPANDED_FILES = _get_embeddings_files(
    EMBEDDINGS_ABBREVIATIONS_EXPANDED_DIRPATH
)

EMBEDDINGS_ABBREVIATIONS_EXPANDED_DIRPATH_V7 = DATADIR.joinpath(
    "abbreviation_expansion_20210502/final_processed/embeddings/author_average_embeddings_2015-2022_v7"
)

EMBEDDINGS_ABBREVIATIONS_EXPANDED_FILES_V7 = _get_embeddings_files(
    EMBEDDINGS_ABBREVIATIONS_EXPANDED_DIRPATH_V7
)


# TODO: update unweighted average embeddings
EMBEDDINGS_UNWEIGHTED_DIRPATH = DATADIR.joinpath(
    "computer_science_papers_20201002/dygie_predictions/predictions_20201005_2/predicted_terms_20201011T161726/author_average_embeddings_2015-2021_unweighted_20201208T0616"
)

EMBEDDINGS_UNWEIGHTED_FILES = {
    "Task": {
        "fname": EMBEDDINGS_UNWEIGHTED_DIRPATH.joinpath(
            "average_author_embeddings_task_pandas.pickle"
        ),
        "fname_ids": EMBEDDINGS_UNWEIGHTED_DIRPATH.joinpath(
            "mat_author_task_row_labels.npy"
        ),
    },
    "Method": {
        "fname": EMBEDDINGS_UNWEIGHTED_DIRPATH.joinpath(
            "average_author_embeddings_method_pandas.pickle"
        ),
        "fname_ids": EMBEDDINGS_UNWEIGHTED_DIRPATH.joinpath(
            "mat_author_method_row_labels.npy"
        ),
    },
    "specter": {
        "fname": SPECTER_DIRPATH.joinpath(
            "average_author_specter_embeddings_2015-2021_unweighted_pandas.pickle"
        ),
        "fname_ids": None,
    },
}

TFIDF_FILES = {
    "Task": {
        "fname": DATADIR.joinpath(
            "computer_science_papers_20210301/author_dygie_tfidf_dropDuplicateTitles_2015-2022_Task.parquet"
        )
    },
    "Method": {
        "fname": DATADIR.joinpath(
            "computer_science_papers_20210301/author_dygie_tfidf_dropDuplicateTitles_2015-2022_Method.parquet"
        )
    },
    "Material": {
        "fname": DATADIR.joinpath(
            "computer_science_papers_20210301/author_dygie_tfidf_dropDuplicateTitles_2015-2022_Material.parquet"
        )
    },
}

TFIDF_VECTORIZERS_FILES = {
    "Task": {
        "fname": DATADIR.joinpath(
            "computer_science_papers_20210301/author_dygie_tfidf_dropDuplicateTitles_2015-2022_vectorizer_Task.joblib"
        )
    },
    "Method": {
        "fname": DATADIR.joinpath(
            "computer_science_papers_20210301/author_dygie_tfidf_dropDuplicateTitles_2015-2022_vectorizer_Method.joblib"
        )
    },
    "Material": {
        "fname": DATADIR.joinpath(
            "computer_science_papers_20210301/author_dygie_tfidf_dropDuplicateTitles_2015-2022_vectorizer_Material.joblib"
        )
    },
}

TFIDF_ABBREVIATIONS_EXPANDED_FILES = {
    "Task": {
        "fname": DATADIR.joinpath(
            "abbreviation_expansion_20210502/author_dygie_tfidf_abbreviations_expanded_2015-2022_Task.parquet"
        )
    },
    "Method": {
        "fname": DATADIR.joinpath(
            "abbreviation_expansion_20210502/author_dygie_tfidf_abbreviations_expanded_2015-2022_Method.parquet"
        )
    },
    "Material": {
        "fname": DATADIR.joinpath(
            "abbreviation_expansion_20210502/author_dygie_tfidf_abbreviations_expanded_2015-2022_Material.parquet"
        )
    },
}

TFIDF_VECTORIZERS_ABBREVIATIONS_EXPANDED_FILES = {
    "Task": {
        "fname": DATADIR.joinpath(
            "abbreviation_expansion_20210502/author_dygie_tfidf_abbreviations_expanded_2015-2022_vectorizer_Task.joblib"
        )
    },
    "Method": {
        "fname": DATADIR.joinpath(
            "abbreviation_expansion_20210502/author_dygie_tfidf_abbreviations_expanded_2015-2022_vectorizer_Method.joblib"
        )
    },
    "Material": {
        "fname": DATADIR.joinpath(
            "abbreviation_expansion_20210502/author_dygie_tfidf_abbreviations_expanded_2015-2022_vectorizer_Material.joblib"
        )
    },
}

# see https://numpy.org/doc/stable/reference/random/generator.html
DEFAULT_RNG_TYPES = Union[
    None,
    int,
    Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]


class UserStudyData:
    """Get the data for a user study, with a focal PaperCollection (e.g., an author, or an author persona).
    Note that this was originally designed to start with an author_id, rather than a PaperCollection.
    I am now rewriting it to use a PaperCollection instead

    TODO: update docstring once finished

    """

    def __init__(
        self,
        paper_collection: PaperCollectionHelper,
        data: DataHelper,
        outdir: Optional[Union[Path, str]] = None,
        author_avg_embeddings: Optional[Dict[str, AverageEmbeddings]] = None,
        # specter_embeddings: Optional[pd.Series] = None,
        tfidf_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        tfidf_vectorizers: Optional[Dict[str, TfidfVectorizer]] = None,
        coauthor_graph: Optional[nx.Graph] = None,
        ego_partition: Optional[Dict] = None,
        exclude_authors: Optional[Iterable[int]] = None,
        min_year: int = 2015,
        max_year: int = 2021,
        min_papers: Optional[int] = None,
        weighted: bool = True,
        abbreviations_expanded: Union[bool, str] = True,
        subdir: Optional[Union[str, Path]] = None,
        random_seed: DEFAULT_RNG_TYPES = None,
    ) -> None:
        self.paper_collection = paper_collection
        self.outdir = outdir
        self.data = data
        self.author_avg_embeddings = author_avg_embeddings
        self.tfidf_dfs = tfidf_dfs
        self.tfidf_vectorizers = tfidf_vectorizers
        self.coauthor_graph = coauthor_graph
        self.ego_partition = ego_partition
        self.exclude_authors = exclude_authors
        if not self.exclude_authors:
            self.exclude_authors = set()
        self.min_year = min_year
        self.max_year = max_year
        self.min_papers = min_papers
        self.weighted = weighted
        self.abbreviations_expanded = abbreviations_expanded
        self.set_rng(random_seed)

        self.specter_embeddings = self.data.specter_embeddings

        if self.outdir is not None:
            self.outdir = Path(self.outdir).resolve()
            if self.outdir.exists():
                # raise FileExistsError(f"output directory {self.outdir} already exists!")
                logger.debug(f"using existing directory: {self.outdir}")
            else:
                logger.debug(f"creating directory {self.outdir}")
                self.outdir.mkdir()

            if subdir is None:
                self.set_subdir()
            else:
                self.subdir = self.outdir.joinpath(subdir)
            if not self.subdir.exists():
                self.subdir.mkdir()

        self.df_distances = None

    def set_subdir(self):
        self.subdir = self.outdir.joinpath(f"author{self.paper_collection.id}_sim")
        if not self.subdir.exists():
            self.subdir.mkdir()

    def set_rng(self, random_seed: DEFAULT_RNG_TYPES = None) -> "UserStudyData":
        # if a np.random.RandomGenerator object is passed to default_rng(), it is returned unaltered
        self.rng: np.random.Generator = np.random.default_rng(random_seed)
        return self

    def load_coauthor_graph(self, force: bool = False) -> "UserStudyData":
        # TODO: move this to data_helper
        if self.coauthor_graph is not None and force is False:
            logger.warn(
                "coauthor_graph is already loaded. run load_coauthor_graph() with force=True to force reload"
            )
        else:
            logger.debug(f"loading coauthor_graph from {COAUTHOR_GRAPH_FPATH}")
            self.coauthor_graph = nx.read_gpickle(str(COAUTHOR_GRAPH_FPATH))
            if (
                hasattr(self.paper_collection, "coauthor_graph")
                and self.paper_collection.coauthor_graph is None
            ):
                self.paper_collection.coauthor_graph = self.coauthor_graph
        return self

    def load_embeddings(self, force: bool = False) -> "UserStudyData":
        if (
            self.author_avg_embeddings is not None
            and len(self.author_avg_embeddings) != 0
            and force is False
        ):
            logger.warn(
                "embeddings are already loaded. run load_embeddings() with force=True to force reload"
            )
        else:
            self.author_avg_embeddings = {}
            if self.weighted is True:
                if self.abbreviations_expanded == 'v7':
                    logger.debug(f"using {self.abbreviations_expanded} files")
                    files = EMBEDDINGS_ABBREVIATIONS_EXPANDED_FILES_V7
                elif self.abbreviations_expanded is True:
                    files = EMBEDDINGS_ABBREVIATIONS_EXPANDED_FILES
                else:
                    files = EMBEDDINGS_FILES
            else:
                files = EMBEDDINGS_UNWEIGHTED_FILES
            for k in files.keys():
                logger.debug(f"loading average embeddings: {k}")
                fname = files[k]["fname"]
                fname_ids = files[k]["fname_ids"]
                self.author_avg_embeddings[k] = AverageEmbeddings.load(fname, fname_ids)
        return self

    def load_specter_embeddings(self, force: bool = False):
        # ! DEPRECATED
        if (
            self.specter_embeddings is not None
            and len(self.specter_embeddings) != 0
            and force is False
        ):
            logger.warn(
                "specter embeddings are already loaded. run load_specter_embeddings() with force=True to force reload"
            )
        else:
            logger.debug("loading specter embeddings")
            specter_embeddings = np.load(
                SPECTER_DIRPATH.joinpath("specter_embeddings.npy")
            )
            specter_embeddings_paper_ids = np.load(
                SPECTER_DIRPATH.joinpath("specter_embeddings_paper_ids.npy")
            )
            self.specter_embeddings = {
                paper_id: embedding
                for paper_id, embedding in zip(
                    specter_embeddings_paper_ids, specter_embeddings
                )
            }
            self.specter_embeddings = pd.Series(self.specter_embeddings)
        return self

    def load_tfidf(self, force: bool = False) -> "UserStudyData":
        if self.tfidf_dfs is not None and len(self.tfidf_dfs) != 0 and force is False:
            logger.warn(
                "tfidf dataframes are already loaded. run load_tfidf() with force=True to force reload"
            )
        else:
            self.tfidf_dfs = {}
            if self.abbreviations_expanded is True:
                files = TFIDF_ABBREVIATIONS_EXPANDED_FILES
            else:
                files = TFIDF_FILES
            for k in files.keys():
                logger.debug(f"loading tfidf data: {k}")
                fname = files[k]["fname"]
                self.tfidf_dfs[k] = pd.read_parquet(fname)
        return self

    def load_tfidf_vectorizers(self, force: bool = False) -> "UserStudyData":
        if (
            self.tfidf_vectorizers is not None
            and len(self.tfidf_vectorizers) != 0
            and force is False
        ):
            logger.warn(
                "tfidf vectorizers are already loaded. run load_tfidf() with force=True to force reload"
            )
        else:
            self.tfidf_vectorizers = {}
            if self.abbreviations_expanded is True:
                files = TFIDF_VECTORIZERS_ABBREVIATIONS_EXPANDED_FILES
            else:
                files = TFIDF_VECTORIZERS_FILES
            for k in files.keys():
                fname = files[k]["fname"]
                logger.debug(f"loading tfidf vectorizer from file: {fname}")
                self.tfidf_vectorizers[k] = joblib.load(fname)
        return self

    def get_focal_embeddings(self):
        """Get embeddings for the focal PaperCollection as a dict of {embedding_type: embedding}"""
        # TODO: move this to PaperCollection?
        self.focal_embeddings = {}
        # first get specter embedding
        df_papers = self.paper_collection.df_papers_dedup_titles()
        # if self.weighted is True:
        #     df_papers["score"] = self.paper_collection.paper_weights
        # else:
        #     df_papers["score"] = 1
        logger.debug("getting focal embedding: specter")
        df_papers = df_papers.dropna(subset=["s2_id"])
        embs = df_papers["s2_id"].map(self.specter_embeddings)
        if self.weighted is True:
            embs = embs * df_papers["relevance_score"]
        embs = embs.dropna()
        embs_avg = np.mean(embs.values)
        self.focal_embeddings["specter"] = embs_avg

        # now get dygie term embeddings
        logger.debug("preparing to get dygie term embeddings")
        df_ner = self.data.df_paper_term_embeddings
        df_ner = df_ner.merge(df_papers, on="PaperId", how="inner")
        for label in ["Task", "Method"]:
            logger.debug(f"getting focal embedding: {label}")
            _df = df_ner[df_ner.label == label]
            _df = _df.dropna(subset=["term_idx"])
            embs = _df["term_idx"].astype(int).apply(lambda x: self.data.embeddings[x])
            # if self.weighted is True:
            #     embs = embs * _df["score"] * _df["softmax_score"]
            embs_avg = np.mean(embs.values)
            self.focal_embeddings[label] = embs_avg
        return self

    def get_sim_df(self, cdist_row, all_author_ids) -> pd.DataFrame:
        df = pd.DataFrame(
            {"cosine_distance": cdist_row, "author_idx": range(len(cdist_row))}
        ).sort_values("cosine_distance")
        df["AuthorId"] = df["author_idx"].map(lambda x: all_author_ids[x])
        return df

    def _get_avg_embedding_cosine_distance(
        self,
        embeddings_type,
        paper_collection: PaperCollectionHelper,
        weighted: bool = True,
    ):
        focal_embedding = self.focal_embeddings[embeddings_type]
        avg_embeddings = self.author_avg_embeddings[embeddings_type].avg_embeddings
        arr = np.array(avg_embeddings.tolist())
        cdist = cosine_distances(focal_embedding.reshape(1, -1), arr)
        return cdist

    def get_all_distances(self) -> "UserStudyData":
        df_sims = dict()
        # TODO: handle multiple author_ids
        # author_id = self.author_ids[0]
        for k in self.author_avg_embeddings:
            logger.debug(f"getting distances for embedding type: {k}")
            cdist = self._get_avg_embedding_cosine_distance(
                embeddings_type=k,
                paper_collection=self.paper_collection,
                weighted=self.weighted,
            )
            df_sims[k] = self.get_sim_df(cdist[0], self.author_avg_embeddings[k].ids)
        all_author_ids = pd.concat(
            [x["AuthorId"] for x in df_sims.values()]
        ).drop_duplicates()
        df_dists = pd.DataFrame(index=all_author_ids)
        for k, v in df_sims.items():
            df_dists[f"{k}_dist"] = v.set_index("AuthorId")["cosine_distance"]

        # get number of papers per author
        paa = self.data.mag_data.paper_authors
        paa = paa[paa.AuthorId.isin(all_author_ids)]
        paa = paa[["AuthorId", "PaperId"]].dropna().drop_duplicates()
        df_dists["paper_count"] = paa.groupby("AuthorId").size()
        logger.debug(f"df_dists shape: {df_dists.shape}")

        # df_dists.drop(index=self.author_ids, inplace=True)
        if self.min_papers:
            logger.debug(
                f"filtering only authors with at least {self.min_papers} papers"
            )
            df_dists = df_dists[df_dists["paper_count"] >= self.min_papers]
            logger.debug(f"df_dists shape: {df_dists.shape}")

        logger.debug("removing authors that do not appear in coauthor_graph")
        df_dists = df_dists.loc[
            df_dists.index.map(lambda x: self.coauthor_graph.has_node(x))
        ]
        logger.debug(f"df_dists shape: {df_dists.shape}")

        if self.paper_collection.coauthors is not None:
            logger.debug("removing strong co-authors")
            df_dists = df_dists.loc[
                ~df_dists.index.isin(
                    [a.AuthorId for a in self.paper_collection.coauthors]
                )
            ]
            logger.debug(f"df_dists shape: {df_dists.shape}")

        self.df_distances = df_dists
        return self

    def _get_sorted_author_sim_df(
        self,
        c: str,
        df_distances: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if df_distances is None:
            df_distances: pd.DataFrame = self.df_distances
            if self.exclude_authors is not None:
                df_distances = df_distances.loc[
                    ~df_distances.index.isin(self.exclude_authors)
                ]
        return sort_distance_df(c, df_distances)

    def get_sim_authors(
        self,
        N: Union[int, Mapping[str, int]] = 10,
        random_seed: int = 1241,
    ) -> "UserStudyData":
        if self.outdir is None:
            raise RuntimeError("outdir must be set")
        df_distances: pd.DataFrame = self.df_distances
        if self.exclude_authors is not None:
            df_distances = df_distances.loc[
                ~df_distances.index.isin(self.exclude_authors)
            ]

        conditions = [
            "simTask",
            "simMethod",
            "simspecter",
            "simTask_distMethod",
            "simMethod_distTask",
        ]
        self.rng.shuffle(conditions)
        if isinstance(N, int):
            N = {c: N for c in conditions}
        else:
            N = {c: N.get(c, 0) for c in conditions}

        sim_authors = {}
        sim_authors_unique = set()
        for c in conditions:
            if N[c]:
                logger.debug(f"getting {N[c]} similar authors for condition: {c}")
                _df = self._get_sorted_author_sim_df(c, df_distances=df_distances)
                # commenting out below to see how often this matters
                _df = _df.loc[~_df.index.isin(sim_authors_unique)]
                sim_authors[c] = _df.head(N[c]).index.tolist()
                sim_authors_unique.update(sim_authors[c])

        # _df = df_distances.sample(n=N, random_state=random_seed)
        # sim_authors["random"] = _df.head(N).index.tolist()

        self.sim_authors = sim_authors
        self.sim_authors_unique = sim_authors_unique
        # self.exclude_authors.update(sim_authors_unique)
        return self

    def get_author_helper(self, author_id: int) -> AuthorHelper:
        author_name = self.data.mag_data.author_lookup(author_id)["DisplayName"]
        return AuthorHelper(
            author_id,
            data=self.data,
            name=author_name,
            min_year=self.min_year,
            max_year=self.max_year,
            coauthor_graph=self.coauthor_graph,
        )

    def get_sim_terms(
        self, collection_1: PaperCollectionHelper, collection_2: PaperCollectionHelper
    ) -> Dict[str, List[str]]:
        sim_terms_dict = {}
        for label in collection_1.df_terms_embeddings.label.unique():
            df_1 = collection_1.df_terms_embeddings[
                collection_1.df_terms_embeddings.label == label
            ]
            embs_1 = df_1.term_idx.map(lambda x: self.data.embeddings[x])
            embs_1 = np.array(embs_1.tolist())
            df_2 = collection_2.df_terms_embeddings[
                collection_2.df_terms_embeddings.label == label
            ]
            embs_2 = df_2.term_idx.map(lambda x: self.data.embeddings[x])
            embs_2 = np.array(embs_2.tolist())
            try:
                cdists = cosine_distances(embs_1, embs_2)
                sim_terms = [
                    df_2.iloc[idx]["term_display"]
                    for idx in np.argsort(cdists.min(axis=0))
                ]
                term_ranker = TermRanker(
                    collection_2.data,
                    collection_2.df_terms,
                    collection_2.df_terms_embeddings,
                )
                sim_terms_top = term_ranker.get_top_terms_with_similarity_threshold(
                    sim_terms, term_ranker.similarity_graphs[label], 0.6
                )
                sim_terms_dict[label] = sim_terms_top
            except ValueError:
                logger.debug(
                    f"ValueError encountered when calculating cosine distances for {label}. skipping..."
                )
        return sim_terms_dict

    def get_paper_distances(self, s2_ids: Iterable[int]) -> pd.Series:
        s2_ids = [
            s2_id for s2_id in s2_ids if s2_id in self.data.specter_embeddings.index
        ]
        s2_ids = np.unique(s2_ids)
        embs = np.vstack([self.data.specter_embeddings.loc[s2_id] for s2_id in s2_ids])
        cdists = cosine_distances(self.focal_embeddings["specter"].reshape(1, -1), embs)
        return pd.Series(cdists[0], index=s2_ids, name="cosine_distance")

    def _get_distance_and_sim_terms(
        self, author
    ) -> Tuple[BridgerCardDistance, DygieTermsSimilar]:
        dist_row = self.df_distances.loc[author.id]
        card_distance = BridgerCardDistance(
            focalId=self.paper_collection.id,
            Method=dist_row.Method_dist,
            Task=dist_row.Task_dist,
            specter=dist_row.specter_dist,
        )
        sim_terms = self.get_sim_terms(self.paper_collection, author)
        card_simTerms = DygieTermsSimilar(
            focalId=self.paper_collection.id,
            Method=sim_terms.get("Method", []),
            Task=sim_terms.get("Task", []),
            Material=sim_terms.get("Material", []),
        )
        return card_distance, card_simTerms

    def _get_single_card(
        self,
        paper_collection: PaperCollectionHelper,
        cls,
        distance=False,
    ) -> BridgerCard:
        card = paper_collection.to_card(cls=cls)

        if distance is True:
            card_distance, card_simTerms = self._get_distance_and_sim_terms(
                paper_collection
            )
            card.distance = [card_distance]
            card.simTerms = [card_simTerms]

        return card

    def get_single_card(
        self,
        paper_collection: PaperCollectionHelper,
        cls,
        distance=False,
    ) -> BridgerCard:
        card = self._get_single_card(paper_collection, cls, distance)
        return card

    def get_single_card_and_details(
        self,
        paper_collection: PaperCollectionHelper,
        cls,
        distance=False,
    ) -> Tuple[BridgerCard, BridgerCardDetails]:
        card = self.get_single_card(paper_collection, cls, distance)
        try:
            coauthors = paper_collection.coauthors
        except AttributeError:
            coauthors = None
        card_details = paper_collection.get_details(coauthors=coauthors)
        try:
            paper_s2_ids = [p.s2Id for p in card_details.papers if not np.isnan(p.s2Id)]
            paper_distances = self.get_paper_distances(paper_s2_ids)
            for paper in card_details.papers:
                if (
                    not np.isnan(paper.s2Id)
                    and paper.s2Id
                    and paper.s2Id in paper_distances.index
                ):
                    pdist = PaperDistance(
                        focalId=self.paper_collection.id,
                        distance=paper_distances.loc[paper.s2Id],
                    )
                    paper.specter_distance = [pdist]
        except AttributeError:
            logger.debug(
                f"could not get paper distances for this card (id: {paper_collection.id}). proceeding with no paper distance information"
            )
        return card, card_details

    def _save_focal_card(self, cls=BridgerCard):
        if self.outdir is None:
            raise RuntimeError("outdir must be set")
        # save focal author
        card, card_details = self.get_single_card_and_details(
            self.paper_collection, cls=cls
        )

        outfp = self.outdir.joinpath(f"author{self.paper_collection.id}_card.json")
        if not outfp.exists():
            logger.debug(f"saving to {outfp}")
            card.to_json(outfp)

        outfp = self.outdir.joinpath(f"author{self.paper_collection.id}_details.json")
        if not outfp.exists():
            logger.debug(f"saving to {outfp}")
            card_details.to_json(outfp)

    def save_focal_card(self) -> None:
        self._save_focal_card()

    def _save_cards(
        self,
        cards_classes: Dict[str, Any] = {
            "focal": BridgerCard,
            "sim": BridgerCardSimilar,
        },
    ) -> None:
        if self.outdir is None:
            raise RuntimeError("outdir must be set")
        # save focal author
        self._save_focal_card(cls=cards_classes["focal"])

        # save other authors
        for sim_id in self.sim_authors_unique:
            outfp_card = self.outdir.joinpath(f"author{sim_id}_card.json")
            outfp_details = self.outdir.joinpath(f"author{sim_id}_details.json")

            if sim_id == self.paper_collection.id:
                logger.debug(
                    "skipping this similar author because it is the same as the focal author"
                )
                continue

            this_author = self.get_author_helper(sim_id)

            if not outfp_card.exists() or not outfp_details.exists():
                card, card_details = self.get_single_card_and_details(
                    this_author, cls=cards_classes["sim"], distance=True
                )

                if not has_enough_data(card_details):
                    logger.warning(f"author {sim_id} failed check: has_enough_data")

                card.to_json(outfp_card)

                card_details.to_json(outfp_details)

            else:
                j = json.loads(outfp_card.read_text())
                # card: BridgerCardSimilar = cards_classes["sim"](**j)
                card: BridgerCardSimilar = dataclass_from_dict(
                    cards_classes["sim"], j, config=config_dataclass_from_dict
                )
                if (
                    self.paper_collection.id in card.distance_focalIds()
                    and self.paper_collection.id in card.simTerms_focalIds()
                ):
                    logger.debug(
                        f"file {outfp_card} already exists and up to date. skipping."
                    )
                    continue

                # add distance and similar terms data to this card
                logger.debug(
                    f"updating card with distance and similar terms data: {outfp_card}"
                )
                if card.distance is None:
                    card.distance = []
                if card.simTerms is None:
                    card.simTerms = []
                card_distance, card_simTerms = self._get_distance_and_sim_terms(
                    this_author
                )
                card.distance.append(card_distance)
                card.simTerms.append(card_simTerms)
                card.to_json(outfp_card)

                j = json.loads(outfp_details.read_text())
                details: BridgerCardDetails = dataclass_from_dict(
                    BridgerCardDetails, j, config=config_dataclass_from_dict
                )
                s2_ids = [p.s2Id for p in details.papers if p.s2Id is not None]
                paper_distances = self.get_paper_distances(s2_ids)
                for i in range(len(details.papers)):
                    paper = details.papers[i]
                    if paper.s2Id is not None and paper.s2Id in paper_distances.index:
                        if paper.specter_distance is None:
                            details.papers[i].specter_distance = []
                        pdist = PaperDistance(
                            focalId=self.paper_collection.id,
                            distance=paper_distances.loc[paper.s2Id],
                        )
                        details.papers[i].specter_distance.append(pdist)
                details.to_json(outfp_details)

    def save_cards(self) -> None:
        self._save_cards()

    def save_sim_author_ids(self) -> None:
        if self.outdir is None:
            raise RuntimeError("outdir must be set")
        # for name, sim_ids in self.sim_authors.items():
        #     dirpath = self.outdir.joinpath(name)
        #     dirpath.mkdir()
        #     G = nx.MultiGraph()
        #     for sim_id in sim_ids:
        #         fp = self.outdir.joinpath(f"author{sim_id}_card.json")
        #         if fp.exists():
        #             node = json.loads(fp.read_text())
        #             G.add_node(node["id"], **node)
        #     out = json_graph.node_link_data(G)
        #     outfp = dirpath.joinpath("graph.json")
        #     logger.debug(f"saving to {outfp}")
        #     outfp.write_text(json.dumps(out))
        outfp = self.subdir.joinpath("sim_authors.json")
        logger.debug(f"saving to {outfp}")
        outfp.write_text(json.dumps(self.sim_authors))

    def persona_subset_factory(
        self, persona_name: str, author_ids: Iterable[Union[int, str]]
    ):
        """Return a new UserStudyData object for a persona (subset of
        other author_ids) of this object. This should be called on a
        UserStudyData object representing an author.

        Args:
            persona_name (str): Name of the persona (e.g., "1")
            author_ids (List): List of other author ids representing the persona

        Returns:
            UserStudyData object
        """
        from ..clustering.persona import PersonaHelper

        persona = PersonaHelper(
            self.paper_collection.id,
            persona_name,
            author_ids,
            data=self.data,
            min_year=self.min_year,
            max_year=self.max_year,
            name=f"{self.paper_collection.name}, Persona {persona_name}",
        )
        return self.__class__(
            persona,
            data=self.data,
            outdir=self.outdir,
            author_avg_embeddings=self.author_avg_embeddings,
            # specter_embeddings=self.specter_embeddings,
            coauthor_graph=self.coauthor_graph,
            exclude_authors=self.exclude_authors,
            tfidf_dfs=self.tfidf_dfs,
            tfidf_vectorizers=self.tfidf_vectorizers,
            min_year=self.min_year,
            max_year=self.max_year,
            min_papers=self.min_papers,
        )

    def paper_subset_factory(
        self,
        id: Union[str, int],
        name: str,
        paper_ids: Iterable[Union[int, str]],
        paper_weights: Optional[Iterable[float]] = None,
        collection_type: str = "author_subset",
    ) -> "UserStudyData":
        """Return a new UserStudyData object for a subset of
        paper ids of this object. This should be called on a
        UserStudyData object representing an author.

        Args:
            name (str): name of the new object
            paper_ids (Iterable): Subset of paper ids
            paper_weights (Iterable): List of weights for the paper ids

        Returns:
            UserStudyData object
        """
        num_paper_ids = len(paper_ids)
        if paper_weights is None:
            paper_weights = pd.Series(
                self.paper_collection.paper_weights,
                index=self.paper_collection.paper_ids,
            )
            paper_ids = [pid for pid in paper_ids if pid in paper_weights.index]
            if len(paper_ids) != num_paper_ids:
                logger.warning(
                    f"Had to remove {num_paper_ids-len(paper_ids)} because they are not part of the parents set of paper_ids"
                )
            paper_weights = paper_weights.loc[paper_ids]
            paper_ids = paper_weights.index.tolist()
            paper_weights = paper_weights.tolist()
        paper_collection = PaperCollectionHelper(
            paper_ids=paper_ids,
            paper_weights=paper_weights,
            data=self.data,
            id=id,
            name=name,
            collection_type=collection_type,
        )
        return self.__class__(
            paper_collection,
            data=self.data,
            outdir=self.outdir,
            author_avg_embeddings=self.author_avg_embeddings,
            # specter_embeddings=self.specter_embeddings,
            coauthor_graph=self.coauthor_graph,
            exclude_authors=self.exclude_authors,
            tfidf_dfs=self.tfidf_dfs,
            tfidf_vectorizers=self.tfidf_vectorizers,
            min_year=self.min_year,
            max_year=self.max_year,
            min_papers=self.min_papers,
        )

    def yield_specter_cluster_personas(
        self, min_papers: int = 4
    ) -> Iterator["UserStudyData"]:
        author = self.paper_collection
        author_specter = author.s2_ids_to_specter
        # clusterer = AgglomerativeClustering(linkage='average', affinity='cosine', n_clusters=None, distance_threshold=.4)
        clusterer = AgglomerativeClustering(
            linkage="ward", affinity="euclidean", n_clusters=None, distance_threshold=88
        )

        embeddings = np.vstack(author_specter.values)
        clusterer.fit(embeddings)
        vc = pd.Series(clusterer.labels_).value_counts()
        # discard personas with not enough papers
        vc = vc[vc >= min_papers]
        score_map = self.paper_collection.df_papers.set_index("PaperId")[
            "relevance_score"
        ]

        personas_papers_s2_ids = [
            author_specter.iloc[clusterer.labels_ == cl].index for cl in vc.index
        ]  # list of lists of s2 paper ids
        paperidsubsets_by_rank = []
        df_papers = author.df_papers[
            author.df_papers.PaperId.isin(author.dedup_title_paper_ids)
        ].sort_values("Rank")
        for s2subset in personas_papers_s2_ids:
            # paperidsubset = (
            #     author.data.df_s2_id[author.data.df_s2_id.s2_id.isin(s2subset)][
            #         "PaperId"
            #     ]
            #     .drop_duplicates()
            #     .tolist()
            # )
            paperidsubset = (
                df_papers[df_papers.s2_id.isin(s2subset)]["PaperId"]
                .drop_duplicates()
                .tolist()
            )
            paperidsubset = [pid for pid in paperidsubset if pid in score_map.index]
            # lower Rank value means more highly ranked (i.e. the top paper is ranked #1)
            min_rank = (
                author.df_papers.set_index("PaperId").loc[paperidsubset]["Rank"].min()
            )
            paperidsubsets_by_rank.append((min_rank, paperidsubset))
        paperidsubsets_by_rank.sort()
        i = 0
        for min_rank, paperidsubset in paperidsubsets_by_rank:
            this_persona_letter = ascii_uppercase[i]
            u_sub = self.paper_subset_factory(
                id=f"{self.paper_collection.id}-{this_persona_letter}",
                name=f"{self.paper_collection.name}, Persona {this_persona_letter}",
                paper_ids=paperidsubset,
                paper_weights=score_map[paperidsubset],
            )
            i += 1
            yield u_sub


def has_enough_data(details: BridgerCardDetails, thresh=10) -> bool:
    num_topics = len(details.topics)
    if num_topics <= thresh:
        return False
    num_tasks = len(details.dygie_terms.Task)
    if num_tasks <= thresh:
        return False
    num_methods = len(details.dygie_terms.Method)
    if num_methods <= thresh:
        return False
    # num_materials = len(details.dygie_terms.Material)
    # if num_materials <= thresh:
    #     return False
    return True


class Initializer:
    def __init__(
        self,
        outdir: Union[str, Path],
        data_helper: DataHelper,
        author_avg_embeddings: Optional[Dict[str, AverageEmbeddings]] = None,
        # specter_embeddings: Optional[pd.Series] = None,
        tfidf_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        tfidf_vectorizers: Optional[Dict[str, TfidfVectorizer]] = None,
        coauthor_graph: Optional[nx.Graph] = None,
        ego_partition: Optional[Dict] = None,
        abbreviations: Optional[pd.DataFrame] = None,
        random_seed: int = 141792,
        abbreviations_expanded: Union[bool, str] = True,
        weighted: bool = True,
        min_year: int = 2015,
        max_year: int = 2022,
    ):
        self.outdir = Path(outdir)
        self.data_helper = data_helper
        self.author_avg_embeddings = author_avg_embeddings
        # self.specter_embeddings = specter_embeddings
        self.tfidf_dfs = tfidf_dfs
        self.tfidf_vectorizers = tfidf_vectorizers
        self.coauthor_graph = coauthor_graph
        self.ego_partition = ego_partition
        self.abbreviations = abbreviations
        self.random_seed = random_seed
        self.abbreviations_expanded = abbreviations_expanded
        self.weighted = weighted
        self.min_year = min_year
        self.max_year = max_year

        self.specter_embeddings = self.data_helper.specter_embeddings

    def increment_random_seed(self):
        self.random_seed += 1

    def initialize(
        self,
        author_id: int,
        cls=UserStudyData,
    ) -> UserStudyData:
        rng = np.random.default_rng(self.random_seed)
        try:
            author = AuthorHelper(
                author_id,
                data=self.data_helper,
                min_year=self.min_year,
                max_year=self.max_year,
                coauthor_graph=self.coauthor_graph,
            )
        except (TypeError, KeyError):
            raise RuntimeError(f"could not lookup author {author_id}")
        logger.debug(
            f"Initializing {cls.__name__} for author {author.id}. author name: {author.name}. Using min_year {self.min_year}, max year {self.max_year}. Random seed: {self.random_seed}"
        )
        u = cls(
            author,
            data=author.data,
            outdir=self.outdir,
            min_year=author.min_year,
            max_year=author.max_year,
            min_papers=6,
            author_avg_embeddings=self.author_avg_embeddings,
            # specter_embeddings=self.specter_embeddings,
            tfidf_dfs=self.tfidf_dfs,
            tfidf_vectorizers=self.tfidf_vectorizers,
            coauthor_graph=author.coauthor_graph,
            ego_partition=self.ego_partition,
            random_seed=rng,
            weighted=self.weighted,
            abbreviations_expanded=self.abbreviations_expanded,
        )
        return u

    def get_authors_with_not_enough_terms(
        self,
        cutoff: int = 10,
        labels: Sequence[str] = ["Task", "Method", "Material"],
    ) -> MutableSet:
        df_ner = self.data_helper.df_ner
        dedup = drop_duplicate_titles(self.data_helper.mag_data.papers)
        df_ner = df_ner[df_ner["PaperId"].isin(dedup["PaperId"])]
        paa = self.data_helper.mag_data.paper_authors
        paa.drop_duplicates(subset=["PaperId", "AuthorId"], inplace=True)
        df_ner.drop_duplicates(subset=["PaperId", "term_id"], inplace=True)
        df_ner = df_ner[["PaperId", "term_id", "label"]]
        author_terms = df_ner.merge(
            paa[["PaperId", "AuthorId"]], how="inner", on="PaperId"
        )
        author_term_counts = (
            author_terms.groupby(["AuthorId", "label"])
            .size()
            .reset_index(name="num_terms")
        )
        not_enough_terms = []
        for label in labels:
            x = author_term_counts[author_term_counts.label == label]
            x = x[x["num_terms"] < cutoff]
            not_enough_terms.extend(x["AuthorId"].tolist())
        return set(not_enough_terms)

    def setup_logging(
        self, log_fpath: Union[str, Path], level=logging.DEBUG
    ) -> logging.FileHandler:
        log_fpath = Path(log_fpath)
        logger.debug(f"outputting to log file: {log_fpath}")
        fhandler = logging.FileHandler(log_fpath)
        fhandler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        fhandler.setLevel(level)
        root_logger.addHandler(fhandler)
        return fhandler

    def load_coauthor_graph(self, force: bool = False):
        # TODO: move this to data_helper
        if self.coauthor_graph is not None and force is False:
            logger.warn(
                "coauthor_graph is already loaded. run load_coauthor_graph() with force=True to force reload"
            )
        else:
            logger.debug(f"loading coauthor_graph from {COAUTHOR_GRAPH_FPATH}")
            self.coauthor_graph = nx.read_gpickle(str(COAUTHOR_GRAPH_FPATH))
        return self

    def load_embeddings(self, force: bool = False):
        if (
            self.author_avg_embeddings is not None
            and len(self.author_avg_embeddings) != 0
            and force is False
        ):
            logger.warn(
                "embeddings are already loaded. run load_embeddings() with force=True to force reload"
            )
        else:
            self.author_avg_embeddings = {}
            if self.weighted is True:
                if self.abbreviations_expanded == 'v7':
                    logger.debug(f"using {self.abbreviations_expanded} files")
                    files = EMBEDDINGS_ABBREVIATIONS_EXPANDED_FILES_V7
                elif self.abbreviations_expanded is True:
                    files = EMBEDDINGS_ABBREVIATIONS_EXPANDED_FILES
                else:
                    files = EMBEDDINGS_FILES
            else:
                files = EMBEDDINGS_UNWEIGHTED_FILES
            for k in files.keys():
                logger.debug(f"loading average embeddings: {k}")
                fname = files[k]["fname"]
                fname_ids = files[k]["fname_ids"]
                self.author_avg_embeddings[k] = AverageEmbeddings.load(fname, fname_ids)
        return self

    def load_specter_embeddings(self, force: bool = False):
        # ! DEPRECATED
        if (
            self.specter_embeddings is not None
            and len(self.specter_embeddings) != 0
            and force is False
        ):
            logger.warn(
                "specter embeddings are already loaded. run load_specter_embeddings() with force=True to force reload"
            )
        else:
            logger.debug("loading specter embeddings")
            specter_embeddings = np.load(
                SPECTER_DIRPATH.joinpath("specter_embeddings.npy")
            )
            specter_embeddings_paper_ids = np.load(
                SPECTER_DIRPATH.joinpath("specter_embeddings_paper_ids.npy")
            )
            self.specter_embeddings = {
                paper_id: embedding
                for paper_id, embedding in zip(
                    specter_embeddings_paper_ids, specter_embeddings
                )
            }
            self.specter_embeddings = pd.Series(self.specter_embeddings)
        return self

    def load_tfidf(self, force: bool = False):
        if self.tfidf_dfs is not None and len(self.tfidf_dfs) != 0 and force is False:
            logger.warn(
                "tfidf dataframes are already loaded. run load_tfidf() with force=True to force reload"
            )
        else:
            self.tfidf_dfs = {}
            if self.abbreviations_expanded is True:
                files = TFIDF_ABBREVIATIONS_EXPANDED_FILES
            else:
                files = TFIDF_FILES
            for k in files.keys():
                logger.debug(f"loading tfidf data: {k}")
                fname = files[k]["fname"]
                self.tfidf_dfs[k] = pd.read_parquet(fname)
        return self

    def load_tfidf_vectorizers(self, force: bool = False):
        if (
            self.tfidf_vectorizers is not None
            and len(self.tfidf_vectorizers) != 0
            and force is False
        ):
            logger.warn(
                "tfidf vectorizers are already loaded. run load_tfidf() with force=True to force reload"
            )
        else:
            self.tfidf_vectorizers = {}
            if self.abbreviations_expanded is True:
                files = TFIDF_VECTORIZERS_ABBREVIATIONS_EXPANDED_FILES
            else:
                files = TFIDF_VECTORIZERS_FILES
            for k in files.keys():
                fname = files[k]["fname"]
                logger.debug(f"loading tfidf vectorizer from file: {fname}")
                self.tfidf_vectorizers[k] = joblib.load(fname)
        return self

    def get_data_single_author_id(
        self,
        author_id: int,
        num_similar: Union[int, Mapping[str, int]] = 10,
        term_rank_compare: bool = False,
        personas: Union[bool, int] = False,
        num_similar_for_personas: Optional[Union[int, Mapping[str, int]]] = None,
        min_papers_per_persona: Union[int, Sequence[int]] = 4,
        save_log: Optional[Union[str, Path]] = None,
    ) -> None:
        if save_log is not None:
            fhandler = self.setup_logging(save_log)

        # data_helper = DataHelper.from_defaults(
        #     min_year=args.min_year, max_year=args.max_year
        # )

        this_start = timer()

        all_sim_authors = set()
        all_user_study_objects: Dict[Union[str, int], UserStudyData] = dict()

        need_recs: List[
            UserStudyData
        ] = (
            []
        )  # we will build a list of UserStudyData objects for which we need to get recommended authors
        do_not_need_recs: List[UserStudyData] = []

        if personas and (num_similar_for_personas is None):
            num_similar_for_personas = num_similar

        logger.debug(f"starting data collection for author: {author_id}")

        min_terms_per_label = 10
        logger.debug(
            f"excluding authors that have fewer than {min_terms_per_label} terms (for each: tasks/methods/materials)"
        )
        exclude_authors = self.get_authors_with_not_enough_terms(
            cutoff=min_terms_per_label
        )

        # exclude focal author when getting similar authors
        exclude_authors.add(author_id)

        if term_rank_compare is True:
            from .user_study_term_rank_compare import UserStudyDataTermRankCompare

            _class = UserStudyDataTermRankCompare
        else:
            _class = UserStudyData

        u = self.initialize(
            author_id,
            _class,
        )
        logger.debug(f"Author name: {u.paper_collection.name}")
        u.load_coauthor_graph()

        # exclude focal author's coauthors when getting similar authors
        coauthors = [a.AuthorId for a in u.paper_collection.coauthors]
        exclude_authors.update(coauthors)

        u.exclude_authors = exclude_authors
        u.load_embeddings()
        # u.load_specter_embeddings()
        if term_rank_compare is True:
            u.load_tfidf()
            u.load_tfidf_vectorizers()
        # u.get_focal_embeddings()
        # u.get_all_distances()
        # u.get_sim_authors(N=num_similar)
        # all_sim_authors.update(u.sim_authors_unique)
        # u.save_cards()
        # u.save_sim_author_ids()
        # all_user_study_objects[u.paper_collection.id] = u

        duplicate_authors: List[int] = []

        need_recs.append(u)

        if personas:
            if isinstance(personas, bool):
                personas = 2
            logger.debug(
                f"getting data for this author's personas. will get recommendations for the top {personas} personas"
            )
            if isinstance(min_papers_per_persona, Sequence):
                # Special procedure for adaptive threshold
                logger.debug(
                    f"using min_papers threshold (adaptive threshold): {min_papers_per_persona}"
                )
                u_personas: List[UserStudyData] = []
                for (
                    min_papers
                ) in (
                    min_papers_per_persona
                ):  # min_papers_per_persona should look something like [4,3], meaning we'll start with threshold 4, then try 3
                    this_personas = list(u.yield_specter_cluster_personas(min_papers))
                    for u_sub in this_personas:
                        # add only if it's not already there
                        if not any(
                            [
                                u_sub.paper_collection.paper_ids
                                == x.paper_collection.paper_ids
                                for x in u_personas
                            ]
                        ):
                            u_personas.append(u_sub)
                    # make sure ids and names are in order
                    for i, u_sub in enumerate(u_personas):
                        this_persona_letter = ascii_uppercase[i]
                        u_sub.paper_collection.id = re.sub(
                            r"-[A-Z]",
                            f"-{this_persona_letter}",
                            u_sub.paper_collection.id,
                        )
                        u_sub.paper_collection.name = re.sub(
                            r"Persona [A-Z]",
                            f"Persona {this_persona_letter}",
                            u_sub.paper_collection.name,
                        )
                        u_sub.set_subdir()

                    if len(u_personas) >= personas:
                        break
                    if len(u_personas) == 0:
                        # if the first threshold doesn't yield any personas, give up
                        break
            else:
                logger.debug(f"using min_papers threshold: {min_papers_per_persona}")
                u_personas = list(
                    u.yield_specter_cluster_personas(min_papers=min_papers_per_persona)
                )

            logger.debug(f"found {len(u_personas)} total personas")

            for u_sub in u_personas:
                if len(need_recs) < personas + 1:
                    need_recs.append(u_sub)
                else:
                    do_not_need_recs.append(u_sub)

        logger.debug(f"shuffling authors and top {len(need_recs)-1} personas")
        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(need_recs)
        for u in need_recs:
            logger.debug(
                f"getting data for: {u.paper_collection.name} ({len(u.paper_collection.paper_ids)} papers)"
            )
            try:
                u.exclude_authors.update(all_sim_authors)
                u.get_focal_embeddings()
                u.get_all_distances()
                if "persona" in u.paper_collection.name.lower():
                    N = num_similar_for_personas
                else:
                    N = num_similar
                u.get_sim_authors(N=N)
                for sim_id in u.sim_authors_unique:
                    if sim_id in all_sim_authors:
                        # This should not happen
                        logger.error(
                            f"found an author recommendation that was a duplicate: {sim_id}"
                        )
                        duplicate_authors.append(sim_id)
                all_sim_authors.update(u.sim_authors_unique)
                u.save_cards()
                u.save_sim_author_ids()
                all_user_study_objects[u.paper_collection.id] = u
            except ZeroDivisionError:
                logger.warning(
                    f"ZeroDivisionError encountered when processing: {u.paper_collection.name}. skipping"
                )
            except ValueError:
                logger.warning(
                    f"ValueError encountered when processing: {u.paper_collection.name}. skipping"
                )

        # testing something, commenting out for now
        # for u in do_not_need_recs:
        #     logger.debug(
        #         f"getting single card for persona: {u.paper_collection.name} ({len(u.paper_collection.paper_ids)} papers)"
        #     )
        #     u.save_focal_card()

        # i = 0
        # for u_sub in u_personas:
        #     try:
        #         if i < personas:
        #             logger.debug(
        #                 f"getting data for persona: {u_sub.paper_collection.name} ({len(u_sub.paper_collection.paper_ids)} papers)"
        #             )
        #             u_sub.get_focal_embeddings()
        #             u_sub.get_all_distances()
        #             u_sub.get_sim_authors(N=num_similar_for_personas)
        #             for sim_id in u_sub.sim_authors_unique:
        #                 if sim_id in all_sim_authors:
        #                     logger.error(
        #                         f"persona {u_sub.paper_collection.id} had an author recommendation that was a duplicate: {sim_id}"
        #                     )
        #                     duplicate_authors.append(sim_id)
        #                     # if save_log is not None:
        #                     #     root_logger.removeHandler(fhandler)
        #                     # return all_user_study_objects, all_sim_authors
        #             all_sim_authors.update(u_sub.sim_authors_unique)
        #             u_sub.save_cards()
        #             u_sub.save_sim_author_ids()
        #         else:
        #             logger.debug(
        #                 f"getting single card for persona: {u_sub.paper_collection.name} ({len(u_sub.paper_collection.paper_ids)} papers)"
        #             )
        #             u_sub.save_focal_card()

        #         i += 1
        #         all_user_study_objects[u_sub.paper_collection.id] = u_sub
        #     except ZeroDivisionError:
        #         logger.warning(
        #             f"ZeroDivisionError encountered when processing persona: {u_sub.paper_collection.name}. skipping"
        #         )
        #     except ValueError:
        #         logger.warning(
        #             f"ValueError encountered when processing persona: {u_sub.paper_collection.name}. skipping"
        #         )
        self.cleanup(u)
        logger.debug(
            f"done getting data for this author ({author_id}). took {format_timespan(timer()-this_start)}"
        )
        if save_log is not None:
            root_logger.removeHandler(fhandler)
        # if len(duplicate_authors) > 0:
        #     self.deal_with_duplicate_authors(duplicate_authors)
        return all_user_study_objects, all_sim_authors

    def save_df_distances(
        self,
        author_id: int,
        outdir: Union[str, Path],
        num_similar: Union[int, Mapping[str, int]] = 10,
        term_rank_compare: bool = False,
        personas: Union[bool, int] = False,
        num_similar_for_personas: Optional[Union[int, Mapping[str, int]]] = None,
        min_papers_per_persona: Union[int, Sequence[int]] = 4,
        save_log: Optional[Union[str, Path]] = None,
    ) -> None:
        outdir = Path(outdir)
        if save_log is not None:
            fhandler = self.setup_logging(save_log)

        # data_helper = DataHelper.from_defaults(
        #     min_year=args.min_year, max_year=args.max_year
        # )

        this_start = timer()

        all_sim_authors = set()
        all_user_study_objects: Dict[Union[str, int], UserStudyData] = dict()

        # we will build a list of UserStudyData objects for which we need to get recommended authors
        need_recs: List[UserStudyData] = []
        do_not_need_recs: List[UserStudyData] = []

        if personas and (num_similar_for_personas is None):
            num_similar_for_personas = num_similar

        logger.debug(f"starting data collection for author: {author_id}")

        min_terms_per_label = 10
        logger.debug(
            f"excluding authors that have fewer than {min_terms_per_label} terms (for each: tasks/methods/materials)"
        )
        exclude_authors = self.get_authors_with_not_enough_terms(
            cutoff=min_terms_per_label
        )

        # exclude focal author when getting similar authors
        exclude_authors.add(author_id)

        if term_rank_compare is True:
            from .user_study_term_rank_compare import UserStudyDataTermRankCompare

            _class = UserStudyDataTermRankCompare
        else:
            _class = UserStudyData

        u = self.initialize(
            author_id,
            _class,
        )
        logger.debug(f"Author name: {u.paper_collection.name}")
        u.load_coauthor_graph()

        # exclude focal author's coauthors when getting similar authors
        coauthors = [a.AuthorId for a in u.paper_collection.coauthors]
        exclude_authors.update(coauthors)

        u.exclude_authors = exclude_authors
        u.load_embeddings()
        if term_rank_compare is True:
            u.load_tfidf()
            u.load_tfidf_vectorizers()

        need_recs.append(u)

        if personas:
            if isinstance(personas, bool):
                personas = 2
            logger.debug(
                f"getting data for this author's personas. will get recommendations for the top {personas} personas"
            )
            if isinstance(min_papers_per_persona, Sequence):
                # Special procedure for adaptive threshold
                logger.debug(
                    f"using min_papers threshold (adaptive threshold): {min_papers_per_persona}"
                )
                u_personas: List[UserStudyData] = []
                for (
                    min_papers
                ) in (
                    min_papers_per_persona
                ):  # min_papers_per_persona should look something like [4,3], meaning we'll start with threshold 4, then try 3
                    this_personas = list(u.yield_specter_cluster_personas(min_papers))
                    for u_sub in this_personas:
                        # add only if it's not already there
                        if not any(
                            [
                                u_sub.paper_collection.paper_ids
                                == x.paper_collection.paper_ids
                                for x in u_personas
                            ]
                        ):
                            u_personas.append(u_sub)
                    # make sure ids and names are in order
                    for i, u_sub in enumerate(u_personas):
                        this_persona_letter = ascii_uppercase[i]
                        u_sub.paper_collection.id = re.sub(
                            r"-[A-Z]",
                            f"-{this_persona_letter}",
                            u_sub.paper_collection.id,
                        )
                        u_sub.paper_collection.name = re.sub(
                            r"Persona [A-Z]",
                            f"Persona {this_persona_letter}",
                            u_sub.paper_collection.name,
                        )
                        u_sub.set_subdir()

                    if len(u_personas) >= personas:
                        break
                    if len(u_personas) == 0:
                        # if the first threshold doesn't yield any personas, give up
                        break
            else:
                logger.debug(f"using min_papers threshold: {min_papers_per_persona}")
                u_personas = list(
                    u.yield_specter_cluster_personas(min_papers=min_papers_per_persona)
                )

            logger.debug(f"found {len(u_personas)} total personas")

            for u_sub in u_personas:
                if len(need_recs) < personas + 1:
                    need_recs.append(u_sub)
                else:
                    do_not_need_recs.append(u_sub)

        # logger.debug(f"shuffling authors and top {len(need_recs)-1} personas")
        # rng = np.random.default_rng(self.random_seed)
        # rng.shuffle(need_recs)
        for u in need_recs:
            logger.debug(
                f"getting data for: {u.paper_collection.name} ({len(u.paper_collection.paper_ids)} papers)"
            )
            try:
                # DON'T exclude previously seen authors
                u.get_focal_embeddings()
                u.get_all_distances()
                if "persona" in u.paper_collection.name.lower():
                    N = num_similar_for_personas
                else:
                    N = num_similar
                u.get_sim_authors(N=N)
                all_sim_authors.update(u.sim_authors_unique)
                all_user_study_objects[u.paper_collection.id] = u
                df_distances: pd.DataFrame = u.df_distances
                if u.exclude_authors is not None:
                    df_distances = df_distances.loc[
                        ~df_distances.index.isin(u.exclude_authors)
                    ]
                outfp = outdir.joinpath(f"df_distances_{u.paper_collection.id}.csv")
                logger.debug(f"writing to {outfp}")
                df_distances.to_csv(outfp, index=True, header=True)
            except ZeroDivisionError:
                logger.warning(
                    f"ZeroDivisionError encountered when processing: {u.paper_collection.name}. skipping"
                )
            except ValueError:
                logger.warning(
                    f"ValueError encountered when processing: {u.paper_collection.name}. skipping"
                )

        self.cleanup(u)
        logger.debug(
            f"done saving df_distances for this author ({author_id}). took {format_timespan(timer()-this_start)}"
        )
        if save_log is not None:
            root_logger.removeHandler(fhandler)
        return all_user_study_objects

    def load_all_data(self, force: bool = False):
        self.load_coauthor_graph(force=force)
        self.load_embeddings(force=force)
        self.load_tfidf(force=force)
        self.load_tfidf_vectorizers(force=force)

    def cleanup(self, u: UserStudyData) -> None:
        if self.coauthor_graph is None:
            self.coauthor_graph = u.coauthor_graph
        if self.author_avg_embeddings is None:
            self.author_avg_embeddings = u.author_avg_embeddings
        if self.specter_embeddings is None:
            self.specter_embeddings = u.specter_embeddings
        if self.tfidf_dfs is None:
            self.tfidf_dfs = u.tfidf_dfs
        if self.tfidf_vectorizers is None:
            self.tfidf_vectorizers = u.tfidf_vectorizers


DEFAULT_NUM_SIM_FOR_USER_STUDY = {
    "author": {
        "simTask": 4,
        "simTask_distMethod": 4,
        "simspecter": 4,
    },
    "persona": {"simTask": 2, "simTask_distMethod": 2},
}