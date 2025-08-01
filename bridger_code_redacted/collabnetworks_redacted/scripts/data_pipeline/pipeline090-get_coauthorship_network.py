# -*- coding: utf-8 -*-

DESCRIPTION = """Given a subset of MAG papers and a time interval (in years), do data cleaning and construct the co-authorship graph """

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

from collabnetworks.mag_data import MagData
from collabnetworks.coauthorship import GraphConstructor

        
def check_file_exists(fpath):
    if not fpath.exists():
        raise FileNotFoundError("required file does not exist: {}".format(fpath))

def main(args):
    if not args.min_year:
        min_year = None
    else:
        min_year = args.min_year
    if not args.max_year:
        max_year = None
    else:
        max_year = args.max_year
    min_pubs = args.min_pubs
    max_authors = args.max_authors
    datadir = Path(args.mag_subset)

    # fpath_papers = get_parquet_filepath(datadir, "Papers")
    # check_file_exists(fpath_papers)
    # fpath_paa = get_parquet_filepath(datadir, "PaperAuthorAffiliations")
    # check_file_exists(fpath_paa)

    outfpath = Path(args.output)

    # logger.debug("Loading Papers data from file: {}".format(fpath_papers))
    # papers_data = pd.read_parquet(fpath_papers)
    # logger.debug("Papers data shape: {}".format(papers_data.shape))
    #
    # logger.debug("Loading PaperAuthorAffiliations data from file: {}".format(fpath_paa))
    # paper_author_data = pd.read_parquet(fpath_paa)
    # logger.debug("PaperAuthorAffiliations data shape: {}".format(paper_author_data.shape))

    mag_data = MagData(datadir, tablenames=["Papers", "PaperAuthorAffiliations"])

    graph_constructor = GraphConstructor(
            # papers_data=papers_data,
            # paper_author_data=paper_author_data,
            mag_data=mag_data,
            min_year=min_year,
            max_year=max_year,
            min_pubs=min_pubs,
            max_authors_per_paper=max_authors)
    # graph_constructor.clean_filter_and_save_edgelist(outfpath)
    edgelist = graph_constructor.clean_filter_and_get_edgelist()
    if args.save_papers is not None:
        logger.debug("saving paper IDs to {}".format(args.save_papers))
        edgelist[graph_constructor.paper_id_colname].drop_duplicates().to_csv(args.save_papers, header=False, index=False)
    logger.debug("Constructing co-authorship graph...")
    G = graph_constructor.construct_coauthorship_graph(edgelist)
    logger.debug("Saving co-authorship graph edgelist CSV to {}".format(outfpath))
    df = graph_constructor.coauthorship_graph_to_pandas_edgelist(G)
    df.to_csv(outfpath, index=False)

    if args.pickle is not None:
        logger.debug("saving pickled co-authorship graph to {}".format(args.pickle))
        nx.write_gpickle(G, args.pickle)

if __name__ == "__main__":
    total_start = timer()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    logger.info("pid: {}".format(os.getpid()))
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("mag_subset", help="directory with subset of MAG data (parquet files)")
    parser.add_argument("output", help="output filename (CSV)")
    parser.add_argument("--min-year", type=int, help="filter out papers published before this year (will include this year)")
    parser.add_argument("--max-year", type=int, help="filter out papers published this year and forward (will not include this year)")
    parser.add_argument("--min-pubs", type=int, default=0, help="only keep authors with at least this many papers (default: keep all authors)")
    parser.add_argument("--max-authors", type=int, default=None, help="filter out papers with more than this many authors")
    parser.add_argument("--save-papers", default=None, help="save paper IDs to specified file (default behavior is to not save paper IDs)")
    parser.add_argument("--pickle", default=None, help="save graph (and attributes) as a pickle to specified file path, in addition to the edgelist csv")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
