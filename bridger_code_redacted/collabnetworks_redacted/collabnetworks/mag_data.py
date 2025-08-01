# -*- coding: utf-8 -*-

DESCRIPTION = """Helper class for MAG data"""

import sys, os, time
from pathlib import Path
from typing import Optional, Union, Iterable, Dict
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

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .util import get_parquet_filepath


class MagData:

    """Helper class for MAG data"""

    def __init__(
        self,
        datadir: Union[Path, str],
        tablenames: Optional[Iterable] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> None:
        """
        :datadir: directory containing the data
        :tablenames: load these tables (defaults will be used if this is not specified)
        :min_year: if specified, will only include papers from this year and later
        :max_year: if specified, will only include papers before this year
        """
        # self.datadir = self.resolve_datadir(datadir)
        self.datadir = datadir
        self.tablenames = tablenames
        self.min_year = min_year
        self.max_year = max_year
        if self.tablenames is None:
            self.tablenames = [
                "Papers",
                "PaperAuthorAffiliations",
                "PaperReferences",
                "Authors",
                "PaperFieldsOfStudy",
                "FieldsOfStudy",
                "Affiliations",
                "Journals",
                "ConferenceSeries",
                "ConferenceInstances",
            ]
        self.fpaths = self.get_fpaths(self.datadir, self.tablenames)

        # load data
        self.data = {}
        for tablename in self.tablenames:
            fpath = self.fpaths[tablename]
            logger.debug("Loading data from {}".format(fpath))
            self.data[tablename] = self.load_dataframe(fpath)

        self.papers = self.data.get("Papers")
        if self.papers is not None and "Rank" in self.papers.columns:
            self.papers["rank_scaled"] = (
                1 - MinMaxScaler().fit_transform(self.papers[["Rank"]]).flatten()
            )

        self.paper_authors = self.data.get("PaperAuthorAffiliations")
        self.paper_refs = self.data.get("PaperReferences")
        self.authors = self.data.get("Authors")
        self.fos = self.data.get("FieldsOfStudy")
        self.paper_fos = self.data.get("PaperFieldsOfStudy")
        self.affiliations = self.data.get("Affiliations")
        self.journals = self.data.get("Journals")
        self.conference_series = self.data.get("ConferenceSeries")
        self.conference_instance = self.data.get("ConferenceInstance")

        if self.papers is not None:
            if self.min_year is not None or self.max_year is not None:
                logger.debug(
                    f"filtering Papers table by year: >= {self.min_year}; < {self.max_year}. dataframe shape before: {self.papers.shape}"
                )
                self.papers = self.year_filter_papers(
                    self.papers, self.min_year, self.max_year
                )
                logger.debug(f"dataframe shape after: {self.papers.shape}")
            self.paper_ids = self.papers["PaperId"]
            logger.debug(f"there are {len(self.paper_ids)} paper ids")
            if self.paper_authors is not None:
                logger.debug(
                    f"filtering PaperAuthorAffiliations table by paper_id. dataframe shape before: {self.paper_authors.shape}"
                )
                self.paper_authors = self.paper_authors.loc[
                    self.paper_authors["PaperId"].isin(self.paper_ids), :
                ]
                logger.debug(f"dataframe shape after: {self.paper_authors.shape}")
            if self.paper_refs is not None:
                logger.debug(
                    f"filtering PaperReferences table by paper_id. dataframe shape before: {self.paper_refs.shape}"
                )
                self.paper_refs = self.paper_refs.loc[
                    (
                        self.paper_refs["PaperId"].isin(self.paper_ids)
                        | (self.paper_refs["PaperReferenceId"].isin(self.paper_ids))
                    ),
                    :,
                ]
                logger.debug(f"dataframe shape after: {self.paper_refs.shape}")
            if self.paper_fos is not None:
                logger.debug(
                    f"filtering PaperFieldsOfStudy table by paper_id. dataframe shape before: {self.paper_fos.shape}"
                )
                self.paper_fos = self.paper_fos.loc[
                    self.paper_fos["PaperId"].isin(self.paper_ids), :
                ]
                logger.debug(f"dataframe shape after: {self.paper_fos.shape}")

            if self.journals is not None and self.conference_series is not None:
                # get "venue" column in papers dataframe
                self.get_venue_column()

        # self._num_papers = None
        self._fos_counts = None  # pandas Series

    @property
    def num_papers(self):
        return len(self.papers)

    @property
    def fos_counts(self):
        if self._fos_counts is None:
            self._fos_counts = self.get_fos_counts(self.paper_fos)
        return self._fos_counts

    @fos_counts.setter
    def fos_counts(self, _fos_counts):
        self._fos_counts = _fos_counts

    # def resolve_datadir(self, datadir: Union[Path, str]) -> Path:
    #     datadir = Path(datadir)
    #     project_root = os.environ.get("PROJECT_ROOT")
    #     if project_root:
    #         return datadir.resolve().relative_to(project_root)
    #     return datadir

    def get_fpaths(self, datadir: Path, tablenames: Iterable) -> Dict[str, Path]:
        """Get file paths for the data

        :datadir: base datadir containing the parquet data
        :returns: dict of tablename -> Path to parquet data

        """
        fpaths = {}
        for tablename in tablenames:
            fpaths[tablename] = get_parquet_filepath(datadir, tablename)
        return fpaths

    def load_dataframe(self, fpath: Path) -> pd.DataFrame:
        """Load pandas dataframe

        :fpath: Path (parquet file)
        :returns: dataframe

        """
        return pd.read_parquet(fpath)

    def get_fos_counts(
        self, df: pd.DataFrame, id_colname: str = "FieldOfStudyId"
    ) -> pd.Series:
        """Get sum of scores for Fields of Study across all papers in corpus

        :df: PaperFieldsOfStudy dataframe
        :returns: Series of floats with index FieldOfStudyId

        """
        fos_counts = df.groupby(id_colname)["Score"].sum()
        fos_counts = fos_counts[fos_counts > 0]
        return fos_counts

    def year_filter_papers(
        self,
        df: pd.DataFrame,
        min_year: Optional[int],
        max_year: Optional[int],
        colname: str = "Year",
    ) -> pd.DataFrame:
        if min_year is not None:
            df = df.loc[df[colname] >= min_year, :]
        if max_year is not None:
            df = df.loc[df[colname] < max_year, :]
        return df

    def get_venue_column(self):
        logger.debug("getting venue_name column for papers dataframe")
        dfs_to_merge = [self.journals, self.conference_series]
        names = ["Journal", "ConferenceSeries"]
        for df_v, s in zip(dfs_to_merge, names):
            id_colname = "{}Id".format(s)
            rename = {"DisplayName": "{}DisplayName".format(s)}
            self.papers = self.papers.merge(
                df_v[[id_colname, "DisplayName"]].rename(columns=rename),
                how="left",
                on=id_colname,
            )
        self.papers["venue_name"] = self.papers["JournalDisplayName"].fillna(
            self.papers["ConferenceSeriesDisplayName"]
        )

    def author_lookup(self, author_id) -> Union[pd.Series, None]:
        """get the author row for an author id"""
        r = self.authors[self.authors["AuthorId"] == author_id]
        if r.empty:
            return None
        return r.iloc[0]
