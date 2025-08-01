# -*- coding: utf-8 -*-

DESCRIPTION = """work with author personas"""

import sys, os, time
from typing import Optional, Iterable, Union, List, Collection, Dict
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

from .clustering import load_ego_partition
from ..util import get_score_column
from ..collection_helper import PaperCollectionHelper

def get_default_personas_dirname():
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    DATADIR = os.environ['DATADIR']
    dirname = os.path.join(DATADIR, "computer_science_papers_20201002/coauthor_2015-2021_minpubs3_collabweighted/components_local/")
    return dirname

class PersonaHelper(PaperCollectionHelper):
    def __init__(
        self,
        author_id: Union[str, int],
        persona_id: Union[str, int],
        other_authors_ids: Collection[Union[str, int]],
        data: Optional["DataHelper"] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        name: Optional[str] = None,
        collection_type: str = "author_persona",
        paa_subset_focal: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        :author_id: MAG AuthorId, or list
        :data: DataHelper object
        """
        self.author_id = author_id
        self.persona_id = persona_id
        self.other_authors_ids = other_authors_ids
        self.data = data
        self.min_year = min_year
        self.max_year = max_year
        self.name = name
        self.collection_type = collection_type
        self.paa_subset_focal = paa_subset_focal

        # check if author_id is a collection (e.g., list) of ids, or if it is a single id
        if isinstance(self.author_id, str) or not isinstance(
            self.author_id, Collection
        ):
            self.author_id = [self.author_id]

        # assign the first AuthorId as the `id` for this author
        self.id = f"{self.author_id[0]}-{self.persona_id}"

        self.paper_ids = None
        self.paper_weights = None
        if self.data is not None:
            self.paper_ids, self.paper_weights = self.get_paper_ids(
                self.paa_subset_focal
            )
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

    @classmethod
    def from_author_helper(
        cls, author_helper, persona_id, other_authors_ids, name: Optional[str] = None
    ):
        return cls(
            author_helper.author_id,
            persona_id,
            other_authors_ids,
            author_helper.data,
            author_helper.min_year,
            author_helper.max_year,
            name,
            paa_subset_focal=author_helper.paa_subset,
        )

    #     self._affiliations = None  # lazy loading, see property below

    # @property
    # def affiliations(self) -> List[str]:
    #     # list of (DisplayName) affiliations according to the papers, sorted descending by frequency
    #     if self._affiliations is None:
    #         df = self.df_paper_authors
    #         df = df[df["AuthorId"].isin(self.author_id)]
    #         df = df.merge(
    #             self.data.mag_data.affiliations, how="inner", on="AffiliationId"
    #         )
    #         affil = (
    #             df["AffiliationId"]
    #             .dropna()
    #             .map(
    #                 self.data.mag_data.affiliations.set_index("AffiliationId")[
    #                     "DisplayName"
    #                 ]
    #             )
    #         )
    #         self._affiliations = affil.value_counts().index.tolist()
    #     return self._affiliations

    def get_paper_ids(self, paa_subset_focal: Optional[pd.DataFrame] = None):
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

        if paa_subset_focal is None:
            paa_subset_focal = paa[paa.AuthorId.isin(self.author_id)].copy()
            paa_subset_focal = paa_subset_focal[
                ["PaperId", "AuthorId", "AuthorSequenceNumber", "is_last_author"]
            ].drop_duplicates()

        paa_subset_other = paa[paa.AuthorId.isin(self.other_authors_ids)].copy()
        paa_subset = paa_subset_focal[
            paa_subset_focal.PaperId.isin(paa_subset_other.PaperId)
        ]
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

        logger.debug("getting paper weights")
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
