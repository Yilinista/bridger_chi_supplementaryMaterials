# -*- coding: utf-8 -*-

DESCRIPTION = """Get subsets of relevant MAG tables based on a list of MAG paper IDs"""

import sys, os, time
from typing import List
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
from ..util import filter_and_get_df, get_parquet_filepath


class MagSubsetGetter:

    """Class to get the data subsets"""

    def __init__(
        self, outdir: str, subset_ids: List[int], parquet_datadir: str, config=None
    ):
        self.outdir = Path(outdir)
        self.subset_ids = subset_ids
        self.parquet_datadir = Path(parquet_datadir)
        self._config = config

        self.mag_author_ids = (
            None  # need to get these after pulling the PaperAuthorAffiliations data
        )
        self.mag_affil_ids = (
            None  # need to get these after pulling the PaperAuthorAffiliations data
        )
        self.mag_journal_ids = None  # get these after pulling the Papers data
        self.mag_conference_series_ids = None  # get these after pulling the Papers data
        self.mag_conference_instance_ids = (
            None  # get these after pulling the Papers data
        )
        self.outfpath_template = "mag_{}_subset.parquet"

    def filter_and_save(self, tablename: str, filter_ids: list, filter_col="PaperId"):
        """load parquet file as a pandas dataframe, filter, and save subset

        :tablename: e.g., "Papers"
        :filter_col: default: 'PaperId'

        """
        fpath = get_parquet_filepath(self.parquet_datadir, tablename)
        logger.debug("loading {} data from {}".format(tablename, fpath))
        df = pd.read_parquet(fpath)
        outfpath = self.outdir.joinpath(self.outfpath_template.format(tablename))
        logger.debug("getting {} subset and saving to {}".format(tablename, outfpath))
        df_subset = filter_and_get_df(df, filter_ids, filter_col=filter_col)
        logger.debug("dataframe has {} rows".format(len(df_subset)))
        df_subset.to_parquet(outfpath)
        logger.debug("done saving to {}".format(outfpath))
        return df_subset

    def get_references_and_citations(self, paper_ids):
        """Get both in-citations and out-citations

        :paper_ids: list or Series of paper ids
        :returns: edgelist subset as pandas DataFrame

        """
        if hasattr(paper_ids, "values"):
            paper_ids = paper_ids.values
        elif hasattr(paper_ids, "tolist"):
            paper_ids = paper_ids.tolist()
        fpath = get_parquet_filepath(self.parquet_datadir, "PaperReferences")
        logger.debug("loading PaperReferences data from {}".format(fpath))
        df_citations = pd.read_parquet(fpath)
        logger.debug("collecting in- and out-citations from pandas DataFrame")
        refs = df_citations[df_citations["PaperId"].isin(paper_ids)]
        cites = df_citations[df_citations["PaperReferenceId"].isin(paper_ids)]
        both = pd.concat([refs, cites]).drop_duplicates()
        return both

    def main(self):
        if self.outdir.is_dir():
            logger.debug("using output directory {}".format(self.outdir))
        else:
            logger.debug("creating output directory {}".format(self.outdir))
            self.outdir.mkdir()

        # Get Papers
        df_papers_subset = self.filter_and_save("Papers", self.subset_ids, "PaperId")

        self.mag_journal_ids = df_papers_subset["JournalId"].dropna().drop_duplicates()
        self.mag_conference_series_ids = (
            df_papers_subset["ConferenceSeriesId"].dropna().drop_duplicates()
        )
        self.mag_conference_instance_ids = (
            df_papers_subset["ConferenceInstanceId"].dropna().drop_duplicates()
        )

        # Get PaperAuthorAffiliations
        df_paa_subset = self.filter_and_save(
            "PaperAuthorAffiliations", self.subset_ids, "PaperId"
        )

        # Optionally get expanded PaperAuthorAffiliations
        # TODO

        # Get PaperFieldsOfStudy
        df_paperfos_subset = self.filter_and_save(
            "PaperFieldsOfStudy", self.subset_ids, "PaperId"
        )
        df_fos_subset = self.filter_and_save(
            "FieldsOfStudy",
            df_paperfos_subset["FieldOfStudyId"].unique(),
            filter_col="FieldOfStudyId",
        )

        self.mag_author_ids = df_paa_subset["AuthorId"].dropna().drop_duplicates()
        self.mag_affil_ids = df_paa_subset["AffiliationId"].dropna().drop_duplicates()
        logger.debug(
            "There are {} MAG Author IDs and {} MAG Affiliation IDs".format(
                len(self.mag_author_ids), len(self.mag_affil_ids)
            )
        )

        # Get Authors
        df_authors_subset = self.filter_and_save(
            "Authors", self.mag_author_ids, "AuthorId"
        )

        # Get Affiliations
        df_affil_subset = self.filter_and_save(
            "Affiliations", self.mag_affil_ids, "AffiliationId"
        )

        # Get Journals and Conferences
        df_journals_subset = self.filter_and_save(
            "Journals", self.mag_journal_ids, "JournalId"
        )
        df_conference_series_subset = self.filter_and_save(
            "ConferenceSeries", self.mag_conference_series_ids, "ConferenceSeriesId"
        )
        df_conference_instance_subset = self.filter_and_save(
            "ConferenceInstances",
            self.mag_conference_instance_ids,
            "ConferenceInstanceId",
        )

        # Get PaperReferences
        outfpath = self.outdir.joinpath(
            self.outfpath_template.format("PaperReferences")
        )
        df_refs_subset = self.get_references_and_citations(self.subset_ids)
        logger.debug("dataframe has {} rows".format(len(df_refs_subset)))
        df_refs_subset.to_parquet(outfpath)
        logger.debug("done saving to {}".format(outfpath))
