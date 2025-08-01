# -*- coding: utf-8 -*-

DESCRIPTION = """Class for Data"""

import sys, os, time
from pathlib import Path
from typing import Optional, Iterable, Union, List, Collection, Dict
from itertools import combinations
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

# from mag_data import MagData

import pandas as pd
import numpy as np

# import networkx as nx

from . import PACKAGE_ROOT

ROOT_DIR = PACKAGE_ROOT.parent

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DATADIR = os.environ["DATADIR"]

from .mag_data import MagData

# DEFAULTS = {
#     "data_fnames": {
#         "mag_data": os.path.join(ROOT_DIR, "data/interim/mag-2020-07-02_CSsubset/"),
#         "s2_mapping": os.path.join(
#             ROOT_DIR,
#             "data/interim/mag-2020-07-02_CSsubset/mag_PaperId_to_s2_id.parquet",
#         ),
#         "ner_terms": os.path.join(
#             ROOT_DIR,
#             "data/dygie_predictions/predicted_terms_20200816T131632/df_ner_cleaned.parquet",
#         ),
#         "ner_grp_counts": os.path.join(
#             ROOT_DIR,
#             "data/dygie_predictions/predicted_terms_20200816T131632/df_term_display_counts.parquet",
#         ),
#         "embeddings": os.path.join(
#             ROOT_DIR,
#             "data/dygie_predictions/predicted_terms_20200816T131632/embeddings_cs_roberta_finetuneSTS_20200911T201433/embeddings_dedup.npy",
#         ),
#         "embeddings_terms": os.path.join(
#             ROOT_DIR,
#             "data/dygie_predictions/predicted_terms_20200816T131632/embeddings_cs_roberta_finetuneSTS_20200911T201433/terms_dedup.npy",
#         ),
#         "terms_to_drop": os.path.join(ROOT_DIR, "data/terms_to_drop.csv"),
#     },
# }

DEFAULTS_V2 = {
    "data_fnames": {
        "mag_data": os.path.join(
            DATADIR, "computer_science_papers_20201002/mag-2020-09-25_CSsubset"
        ),
        "s2_mapping": os.path.join(
            DATADIR,
            "computer_science_papers_20201002/computer_science_papers.parquet",
        ),
        "ner_terms": os.path.join(
            DATADIR,
            "computer_science_papers_20201002/dygie_predictions/predictions_20201005_2/predicted_terms_20201011T161726/terms_lemmatized_cleaned_202010120800.parquet",
        ),
        "embeddings": os.path.join(
            DATADIR,
            "computer_science_papers_20201002/dygie_predictions/predictions_20201005_2/predicted_terms_20201011T161726/embeddings_cs_roberta_finetuneSTS_20201012T131943/embeddings_dedup.npy",
        ),
        "embeddings_terms": os.path.join(
            DATADIR,
            "computer_science_papers_20201002/dygie_predictions/predictions_20201005_2/predicted_terms_20201011T161726/embeddings_cs_roberta_finetuneSTS_20201012T131943/terms_dedup.npy",
        ),
        "terms_to_drop": os.path.join(ROOT_DIR, "data/terms_to_drop.csv"),
    },
}

DEFAULTS_V2_5 = {
    "data_fnames": {
        "mag_data": os.path.join(
            DATADIR, "computer_science_papers_20201002/mag-2020-09-25_CSsubset"
        ),
        "s2_mapping": os.path.join(
            DATADIR,
            "computer_science_papers_20201002/computer_science_papers.parquet",
        ),
        "ner_terms": os.path.join(
            DATADIR,
            "computer_science_papers_20201002/final_processed/dygie_terms_to_s2_id_softmaxThreshold0.90.parquet",
        ),
        "papers_to_term_embeddings": os.path.join(
            DATADIR,
            "computer_science_papers_20201002/final_processed/embeddings/dygie_embedding_term_ids_to_s2_id_softmaxThreshold0.90.parquet",
        ),
        "embeddings": os.path.join(
            DATADIR,
            "computer_science_papers_20201002/final_processed/embeddings/embeddings.npy",
        ),
        "embeddings_terms": os.path.join(
            DATADIR,
            "computer_science_papers_20201002/final_processed/embeddings/embedding_term_to_id.parquet",
        ),
        "terms_to_drop": os.path.join(ROOT_DIR, "data/terms_to_drop.csv"),
    },
}

DEFAULTS_V3 = {
    "data_fnames": {
        "mag_data": os.path.join(
            DATADIR, "computer_science_papers_20210301/mag-2021-03-01_CSsubset"
        ),
        "s2_mapping": os.path.join(
            DATADIR,
            "computer_science_papers_20210301/computer_science_papers.parquet",
        ),
        "specter_dirpath": os.path.join(
            DATADIR, "computer_science_papers_20210301/specter_embeddings/"
        ),
        "ner_terms": os.path.join(
            DATADIR,
            "computer_science_papers_20210301/final_processed/dygie_terms_to_s2_id_softmaxThreshold0.90.parquet",
        ),
        "papers_to_term_embeddings": os.path.join(
            DATADIR,
            "computer_science_papers_20210301/final_processed/embeddings/dygie_embedding_term_ids_to_s2_id_softmaxThreshold0.90.parquet",
        ),
        "embeddings": os.path.join(
            DATADIR,
            "computer_science_papers_20210301/final_processed/embeddings/embeddings.npy",
        ),
        "embeddings_terms": os.path.join(
            DATADIR,
            "computer_science_papers_20210301/final_processed/embeddings/embedding_term_to_id.parquet",
        ),
        "terms_to_drop": os.path.join(ROOT_DIR, "data/terms_to_drop.csv"),
    },
}

# DEFAULTS_V3_ABBREVIATIONS_EXPANDED = {
#     "data_fnames": {
#         "mag_data": os.path.join(
#             DATADIR, "computer_science_papers_20210301/mag-2021-03-01_CSsubset"
#         ),
#         "s2_mapping": os.path.join(
#             DATADIR,
#             "computer_science_papers_20210301/computer_science_papers.parquet",
#         ),
#         "specter_dirpath": os.path.join(
#             DATADIR, "computer_science_papers_20210301/specter_embeddings/"
#         ),
#         "ner_terms": os.path.join(
#             DATADIR,
#             "computer_science_papers_20210301/final_processed_abbreviations_expanded/dygie_terms_to_s2_id_softmaxThreshold0.90.parquet",
#         ),
#         "papers_to_term_embeddings": os.path.join(
#             DATADIR,
#             "computer_science_papers_20210301/final_processed_abbreviations_expanded/embeddings/dygie_embedding_term_ids_to_s2_id_softmaxThreshold0.90.parquet",
#         ),
#         "embeddings": os.path.join(
#             DATADIR,
#             "computer_science_papers_20210301/final_processed_abbreviations_expanded/embeddings/embeddings.npy",
#         ),
#         "embeddings_terms": os.path.join(
#             DATADIR,
#             "computer_science_papers_20210301/final_processed_abbreviations_expanded/embeddings/embedding_term_to_id.parquet",
#         ),
#         "terms_to_drop": os.path.join(ROOT_DIR, "data/terms_to_drop.csv"),
#     },
# }

DEFAULTS_ABBREVIATIONS_EXPANDED = {
    "data_fnames": {
        "mag_data": os.path.join(
            DATADIR, "computer_science_papers_20210301/mag-2021-03-01_CSsubset"
        ),
        "s2_mapping": os.path.join(
            DATADIR,
            "computer_science_papers_20210301/computer_science_papers.parquet",
        ),
        "specter_dirpath": os.path.join(
            DATADIR, "computer_science_papers_20210301/specter_embeddings/"
        ),
        "ner_terms": os.path.join(
            DATADIR,
            "abbreviation_expansion_20210502/final_processed/dygie_terms_to_s2_id_softmaxThreshold0.90.parquet",
        ),
        "papers_to_term_embeddings": os.path.join(
            DATADIR,
            "abbreviation_expansion_20210502/final_processed/embeddings/dygie_embedding_term_ids_to_s2_id_softmaxThreshold0.90.parquet",
        ),
        "embeddings": os.path.join(
            DATADIR,
            "abbreviation_expansion_20210502/final_processed/embeddings/embeddings.npy",
        ),
        "embeddings_terms": os.path.join(
            DATADIR,
            "abbreviation_expansion_20210502/final_processed/embeddings/embedding_term_to_id.parquet",
        ),
        "terms_to_drop": os.path.join(ROOT_DIR, "data/terms_to_drop.csv"),
    },
}


class DataHelper:

    """Collect all the data we might need for processing"""

    def __init__(
        self,
        mag_data: Optional[MagData] = None,
        df_s2_id: Optional[pd.DataFrame] = None,
        # G: Optional[nx.Graph] = None,
        # memberships: Optional[OverlappingPartition] = None,
        df_ner: Optional[pd.DataFrame] = None,
        grp_counts: Optional[pd.DataFrame] = None,
        specter_embeddings: Optional[pd.Series] = None,
        embeddings: Optional[np.ndarray] = None,
        embeddings_terms: Optional[np.ndarray] = None,
        df_paper_term_embeddings: Optional[pd.DataFrame] = None,
        terms_to_drop: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> None:
        """TODO: to be defined.
        if min_year is specified, all papers before this year will be dropped.
        If max_year is specified, all papers this year and later will be dropped.
        """
        self.mag_data = mag_data
        self.df_s2_id = df_s2_id
        # self.G = G
        # self.memberships = memberships
        self.df_ner = df_ner
        self.grp_counts = grp_counts
        self.specter_embeddings = specter_embeddings
        self.embeddings = embeddings
        self.embeddings_terms = embeddings_terms
        self.df_paper_term_embeddings = df_paper_term_embeddings
        self.terms_to_drop = terms_to_drop
        self.min_year = min_year
        self.max_year = max_year

        self.cl_to_authors = None
        self.embeddings_terms_to_idx = None

    @classmethod
    def from_defaults(
        cls, defaults=DEFAULTS_V3, exclude: Optional[List[str]] = None, **kwargs
    ):
        """load object from defaults (see source code)"""
        if exclude is None:
            exclude = []
        obj = cls(**kwargs)
        fnames = defaults["data_fnames"]
        if obj.mag_data is None and "mag_data" not in exclude:
            obj.load_mag_data(fnames["mag_data"])
        else:
            logger.debug("excluded")
        if obj.df_s2_id is None and "s2_mapping" not in exclude:
            obj.load_s2_mapping(fnames["s2_mapping"])
        # if obj.G is None:
        #     obj.load_coauthorship_graph(fnames['coauthorship_edgelist'])
        # if obj.memberships is None:
        #     obj.load_cluster_memberships(fnames['memberships'])
        if obj.specter_embeddings is None and "specter" not in exclude and "specter_embeddings" not in exclude:
            obj.load_specter_embeddings(fnames["specter_dirpath"])
        if obj.df_ner is None and "ner_terms" not in exclude:
            obj.load_ner_data(fnames["ner_terms"], fnames.get("ner_grp_counts"))
        if obj.embeddings is None and "embeddings" not in exclude:
            obj.load_embeddings(
                fnames["embeddings"],
                fnames["embeddings_terms"],
                fnames["papers_to_term_embeddings"],
            )
        if obj.terms_to_drop is None and fnames.get("terms_to_drop") is not None:
            obj.load_terms_to_drop(fnames["terms_to_drop"])

        # DEPRECATED
        # if (
        #     obj.df_ner is not None
        #     and obj.embeddings_terms_to_idx is not None
        #     and "map_term_idx" not in exclude
        # ):
        #     obj.map_term_idx_to_df_ner()

        return obj

    def load_mag_data(
        self, path_to_data: str, tablenames: Optional[List[str]] = None
    ) -> None:
        logger.debug("loading MagData from {}".format(path_to_data))
        self.mag_data = MagData(
            path_to_data,
            tablenames=tablenames,
            min_year=self.min_year,
            max_year=self.max_year,
        )

    def load_s2_mapping(self, path_to_data: str) -> None:
        logger.debug("loading s2 mapping from file {}".format(path_to_data))
        self.df_s2_id = pd.read_parquet(path_to_data)
        if (
            "s2_id" not in self.df_s2_id.columns
            and "corpus_paper_id" in self.df_s2_id.columns
        ):
            self.df_s2_id = self.df_s2_id.rename(columns={"corpus_paper_id": "s2_id"})
        self.df_s2_id = self.df_s2_id[["PaperId", "s2_id"]]
        logger.debug("dataframe shape: {}".format(self.df_s2_id.shape))
        logger.debug("dropping na and converting s2_id column to integer type")
        self.df_s2_id.dropna(inplace=True)
        self.df_s2_id["s2_id"] = self.df_s2_id["s2_id"].astype(int)
        logger.debug("dataframe shape: {}".format(self.df_s2_id.shape))

    # def load_coauthorship_graph(self,
    #         path_to_edgelist_csv: str):
    #     logger.debug("loading coauthorship graph from file {}".format(path_to_edgelist_csv))
    #     df_edgelist = pd.read_csv(path_to_edgelist_csv)
    #     G = nx.from_pandas_edgelist(df_edgelist, source='AuthorId_1', target='AuthorId_2', edge_attr=True)
    #     # With edge weight thresholding
    #     weight_thresh = 0.02
    #     logger.debug("removing edges with weight <= {}".format(weight_thresh))
    #     edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w <= weight_thresh]
    #     G.remove_edges_from(edges_to_remove)
    #     logger.debug("coauthorship graph has {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    #     self.G = G

    # def load_cluster_memberships(self,
    #         path_to_pickle: str
    #         ) -> None:
    #     logger.debug("loading cluster memberships from file {}".format(path_to_pickle))
    #     fpath = Path(path_to_pickle)
    #     self.memberships = pickle.loads(fpath.read_bytes())
    #     self.cl_to_authors = get_cl_to_authors(self.memberships)
    #     logger.debug("there are {} clusters".format(len(self.cl_to_authors)))

    def load_specter_embeddings(self, dirpath_specter: Union[Path, str]):
        dirpath_specter = Path(dirpath_specter)
        logger.debug(f"loading specter embeddings from directory: {dirpath_specter}")
        specter_embeddings = np.load(dirpath_specter.joinpath("specter_embeddings.npy"))
        specter_embeddings_paper_ids = np.load(
            dirpath_specter.joinpath("specter_embeddings_paper_ids.npy")
        )
        self.specter_embeddings = {
            paper_id: embedding
            for paper_id, embedding in zip(
                specter_embeddings_paper_ids, specter_embeddings
            )
        }
        self.specter_embeddings = pd.Series(self.specter_embeddings, name="specter_embedding")

    def load_ner_data(
        self, path_to_terms: str, path_to_counts: Optional[str] = None
    ) -> None:
        logger.debug("loading ner terms from {}".format(path_to_terms))
        self.df_ner = pd.read_parquet(path_to_terms)
        self.df_ner["s2_id"] = self.df_ner["s2_id"].astype(int)
        logger.debug("dataframe shape: {}".format(self.df_ner.shape))

        logger.debug("merging in MAG paper ids")
        self.df_ner = self.df_s2_id.merge(self.df_ner, how="inner", on="s2_id")
        logger.debug("dataframe shape: {}".format(self.df_ner.shape))
        if getattr(self.mag_data, "paper_ids", None) is not None:
            logger.debug(f"filtering ner dataframe by mag_data paper_ids")
            self.df_ner = self.df_ner.loc[
                self.df_ner["PaperId"].isin(self.mag_data.paper_ids), :
            ]
            logger.debug("dataframe shape: {}".format(self.df_ner.shape))

        if path_to_counts is not None:
            logger.debug("loading ner term counts from {}".format(path_to_counts))
            self.grp_counts = pd.read_parquet(path_to_counts)
            logger.debug("counts dataframe shape: {}".format(self.grp_counts.shape))

    def load_embeddings(
        self,
        path_to_embeddings: str,
        path_to_terms: str,
        path_to_paper_term_embeddings: str,
    ) -> None:
        logger.debug("loading embeddings and corresponding terms")
        logger.debug("loading embeddings from {}".format(path_to_embeddings))
        self.embeddings = np.load(path_to_embeddings)
        logger.debug("loading terms from {}".format(path_to_terms))
        self.embeddings_terms = pd.read_parquet(path_to_terms)
        logger.debug(
            f"len(embeddings): {len(self.embeddings)}; len(embeddings_terms): {len(self.embeddings_terms)}"
        )

        logger.debug(
            "loading paper-to-term-embeddings from {}".format(
                path_to_paper_term_embeddings
            )
        )
        self.df_paper_term_embeddings = pd.read_parquet(path_to_paper_term_embeddings)
        logger.debug("dataframe shape: {}".format(self.df_paper_term_embeddings.shape))

        self.df_paper_term_embeddings["s2_id"] = self.df_paper_term_embeddings[
            "s2_id"
        ].astype(int)

        logger.debug("merging in MAG paper ids")
        self.df_paper_term_embeddings = self.df_s2_id.merge(
            self.df_paper_term_embeddings, how="inner", on="s2_id"
        )
        logger.debug("dataframe shape: {}".format(self.df_paper_term_embeddings.shape))
        if getattr(self.mag_data, "paper_ids", None) is not None:
            logger.debug(f"filtering ner dataframe by mag_data paper_ids")
            self.df_paper_term_embeddings = self.df_paper_term_embeddings.loc[
                self.df_paper_term_embeddings["PaperId"].isin(self.mag_data.paper_ids),
                :,
            ]
            logger.debug(
                "dataframe shape: {}".format(self.df_paper_term_embeddings.shape)
            )

        # TODO: not ideal
        self.embeddings_terms = self.embeddings_terms.index.values
        logger.debug("dropping unused embeddings...")
        terms_set = set(self.df_paper_term_embeddings["embedding_term"].values)
        keep_idx = []
        for i, term in enumerate(self.embeddings_terms):
            if term in terms_set:
                keep_idx.append(i)
        self.embeddings_terms = self.embeddings_terms[keep_idx]
        self.embeddings = self.embeddings[keep_idx]
        logger.debug(
            f"len(embeddings): {len(self.embeddings)}; len(embeddings_terms): {len(self.embeddings_terms)}"
        )
        logger.debug("getting embeddings_terms_to_idx")
        self.embeddings_terms_to_idx = {
            val: idx[0] for idx, val in np.ndenumerate(self.embeddings_terms)
        }
        logger.debug(
            "mapping embeddings_terms_to_idx to paper-to-term-embeddings dataframe"
        )
        self.df_paper_term_embeddings["term_idx"] = self.df_paper_term_embeddings[
            "embedding_term"
        ].map(self.embeddings_terms_to_idx)

    def load_terms_to_drop(
        self,
        path_to_terms: str,
    ) -> None:
        self.terms_to_drop = Path(path_to_terms).read_text().split("\n")

    def map_term_idx_to_df_ner(self) -> None:
        # DEPRECATED
        logger.debug("mapping embeddings_terms_to_idx to terms dataframe")
        self.df_ner["term_idx"] = self.df_ner["term_cleaned"].map(
            self.embeddings_terms_to_idx
        )
        self.N_tfidf = self.df_ner["PaperId"].nunique()
        # logger.debug("done. {} unique papers overall".format(self.N_tfidf))
