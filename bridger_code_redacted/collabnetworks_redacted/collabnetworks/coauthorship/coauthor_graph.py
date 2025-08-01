# -*- coding: utf-8 -*-

DESCRIPTION = """class to construct co-authorship graph"""

import sys, os, time
from pathlib import Path
from typing import Optional, Union
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
from ..mag_data import MagData

class GraphConstructor:

    """Do data cleaning and construct co-authorship graph"""

    def __init__(self,
                # papers_data: pd.DataFrame,
                # paper_author_data: pd.DataFrame,
                mag_data: MagData,
                min_year:  Optional[int] = None,
                max_year: Optional[int] = None,
                min_pubs: int = 0,
                max_authors_per_paper: Optional[int] = None,
                paper_id_colname: str = 'PaperId',
                author_id_colname: str = 'AuthorId'
                ) -> None:
        """
        :mag_data: MagData object with Papers and PaperAuthorAffiliations data loaded
        :min_year: filter out papers published before this year (will include this year)
        :max_year: filter out papers published this year and forward (will not include this year)
        :min_pubs: only keep authors with at least this many papers
        :max_authors_per_paper: filter out papers with more than this many authors
        """
        # self.papers_data = papers_data
        # self.paper_author_data = paper_author_data
        self.mag_data = mag_data
        self.papers_data = mag_data.papers
        self.paper_author_data = mag_data.paper_authors
        self.min_year = min_year
        self.max_year = max_year
        self.min_pubs = min_pubs
        self.max_authors_per_paper = max_authors_per_paper
        self.paper_id_colname = paper_id_colname
        self.author_id_colname = author_id_colname

    def clean_num_refs(self, 
                      papers_data: pd.DataFrame,
                      min_refs: Optional[int] = None,
                      max_refs: Optional[int] = None,
                      colname: str = "ReferenceCount"
                      ) -> pd.DataFrame:
        """

        :returns: filtered MAG paper data

        """
        if min_refs is not None:
            papers_data = papers_data[papers_data[colname]>=min_refs]
        if max_refs is not None:
            papers_data = papers_data[papers_data[colname]<max_refs]
        return papers_data

    def clean_papers(self) -> pd.DataFrame:
        """Do data cleaning

        :returns: filtered MAG paper data

        """
        papers = self.papers_data

        # filter based on number of references
        papers = self.clean_num_refs(papers, min_refs=1)

        # data cleaning done
        return papers

    def filter_time_interval(self, 
                             min_year:  Optional[int] = None,
                             max_year: Optional[int] = None,
                             colname: str = "Year"
                             ) -> None:
        """Restrict data to within a given time interval (in years)

        :min_year: filter out papers published before this year (will include this year)
        :max_year: filter out papers published this year and forward (will not include this year)
        :returns: None (modifies the instance's data)

        """
        if min_year is not None:
            self.papers_data = self.papers_data[self.papers_data[colname]>=min_year]
        if max_year is not None:
            self.papers_data = self.papers_data[self.papers_data[colname]<max_year]

    def clean_authors(self, 
            min_pubs: int = 0, 
            ) -> pd.DataFrame:
        """Filter out records from the paper-authors data for authors with less than min_pubs

        :min_pubs: only keep authors with at least this many papers
        :returns: filtered MAG PaperAuthorAffiliations data

        """
        paa = self.paper_author_data
        if min_pubs > 0:
            author_num_pubs = paa.groupby(self.author_id_colname)[self.paper_id_colname].nunique()
            paa = paa[paa[self.author_id_colname].map(author_num_pubs>=min_pubs)]
        return paa

    def filter_out_papers_with_many_authors(self, 
            edgelist: pd.DataFrame, 
            max_authors_per_paper: int
            ) -> pd.DataFrame:
        """Filter out papers from the paper-author edgelist that have more than a certain number of authors

        :edgelist: edgelist of papers to authors, as a pandas dataframe
        :max_authors_per_paper: int
        :returns: filtered edgelist (dataframe)

        """
        num_authors_per_paper = edgelist[self.paper_id_colname].value_counts()
        include = num_authors_per_paper[num_authors_per_paper<=max_authors_per_paper]
        return edgelist[edgelist[self.paper_id_colname].isin(include.index)]

    def get_edgelist(self) -> pd.DataFrame:
        """
        :returns: edgelist as 2-column pandas dataframe

        """
        paa = self.paper_author_data
        edgelist = paa[[self.paper_id_colname, self.author_id_colname]].drop_duplicates()
        return edgelist

    def clean_filter_and_get_edgelist(self) -> pd.DataFrame:
        """
        :returns: edgelist as 2-column pandas dataframe

        """
        # clean papers data
        logger.debug("Papers data has shape: {}".format(self.papers_data.shape))
        logger.debug("Cleaning papers...")
        self.papers_data = self.clean_papers()
        logger.debug("Done cleaning. Papers data has shape: {}".format(self.papers_data.shape))

        # filter by time period
        min_year = self.min_year
        max_year = self.max_year
        logger.debug("Filtering by time period: min_year {}; max_year: {}".format(min_year, max_year))
        self.filter_time_interval(min_year=min_year, max_year=max_year)
        self.paper_author_data = self.paper_author_data.merge(self.papers_data[[self.paper_id_colname]], how='inner', on=self.paper_id_colname)
        logger.debug("Done filtering by time period. Papers data has shape: {}".format(self.papers_data.shape))
        logger.debug("PaperAuthorAffiliations data has shape: {}".format(self.paper_author_data.shape))
        logger.debug("{} unique papers".format(self.paper_author_data[self.paper_id_colname].nunique()))
        logger.debug("{} unique authors".format(self.paper_author_data[self.author_id_colname].nunique()))

        # filter out some authors
        logger.debug("Cleaning authors...")
        self.paper_author_data = self.clean_authors(min_pubs=self.min_pubs)
        logger.debug("Done cleaning authors. PaperAuthorAffiliations data has shape: {}".format(self.paper_author_data.shape))
        logger.debug("{} unique papers".format(self.paper_author_data[self.paper_id_colname].nunique()))
        logger.debug("{} unique authors".format(self.paper_author_data[self.author_id_colname].nunique()))

        # get edgelist
        logger.debug("Getting edgelist")
        self.edgelist = self.get_edgelist()

        if self.max_authors_per_paper is not None:
            logger.debug("filtering out papers that have more than {} authors".format(self.max_authors_per_paper))
            self.edgelist = self.filter_out_papers_with_many_authors(self.edgelist, self.max_authors_per_paper)
            logger.debug("Edgelist has shape: {}".format(self.edgelist.shape))

        return self.edgelist

    def clean_filter_and_save_edgelist(self, outfpath: Union[Path, str]) -> None:
        """Do all cleaning and filtering by time period, get edgelist, and save to file

        """
        edgelist = self.clean_filter_and_get_edgelist()
        logger.debug("saving edgelist to CSV file: {}".format(outfpath))
        edgelist.to_csv(outfpath, index=False)

    def construct_coauthorship_graph(self, edgelist: pd.DataFrame) -> nx.Graph:
        """Construct the co-authorship graph, with nodes as authors and edges as co-authorship, weighted by number of co-authorships (normalized by number of authors)

        :edgelist: edgelist of papers to authors, as a pandas dataframe
        :returns: coauthorship graph (networkx Graph)

        """
        # create bipartite graph
        B = nx.from_edgelist(edgelist.values)
        # self.G = nx.bipartite.weighted_projected_graph(B, edgelist[self.author_id_colname].unique())
        self.G = nx.bipartite.collaboration_weighted_projected_graph(B, edgelist[self.author_id_colname].unique())

        num_pubs = edgelist[self.author_id_colname].value_counts()
        for author_id, num in num_pubs.iteritems():
            self.G.nodes[author_id]['n_pubs'] = num
        return self.G

    def coauthorship_graph_to_pandas_edgelist(self, G: nx.Graph) -> pd.DataFrame:
        """TODO: Docstring for coauthorship_graph_to_pandas_edgelist.

        :G: co-authorship graph
        :returns: weighted edgelist as pandas dataframe with columns: ["AuthorId_1", "AuthorId_2", "weight"]

        """
        cn = self.author_id_colname
        return nx.to_pandas_edgelist(G, f"{cn}_1", f"{cn}_2")

