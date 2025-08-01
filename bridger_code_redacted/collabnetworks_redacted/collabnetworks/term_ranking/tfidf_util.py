# -*- coding: utf-8 -*-

DESCRIPTION = """tfidf methods"""

import sys, os, time
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
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data_helper import DEFAULTS_V2
DATADIR_TFIDF = Path(DEFAULTS_V2['data_fnames']['ner_terms']).parent.joinpath('tfidf_experiments')


def apply_cutoff(df_ner: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    vc = df_ner["term_id"].value_counts()
    to_keep = vc[vc >= cutoff]
    return df_ner[df_ner["term_id"].isin(to_keep.index)]


def aggregate_author_term_data(df_author_term: pd.DataFrame) -> pd.Series:
    """aggregate each author's terms into a space-separated list

    Args:
        df_author_term (pd.DataFrame): pandas dataframe with columns ['AuthorId', 'term_id']

    Returns:
        terms_concat (pd.Series): pandas Series with index of AuthorIds
        and values of space-separated list of term_ids (as string)
    """
    logger.debug("aggregating author term data")
    df_author_term["term_id"] = df_author_term.term_id.astype(str)
    gb = df_author_term.groupby("AuthorId")
    terms_concat = gb["term_id"].agg(lambda term_id: " ".join(term_id))
    return terms_concat


def get_author_term_tfidf(terms_concat: pd.Series) -> pd.DataFrame:
    """Takes a list of term_ids per author
    and returns a dataframe of term tfidf scores per author.

    Args:
        terms_concat (pd.Series): pandas Series with index of AuthorIds
        and values of space-separated list of term_ids (as string)

    Returns:
        pd.DataFrame: dataframe with columns ['AuthorId', 'term_id', 'tfidf_score']
    """
    logger.debug("fitting TfidfVectorizer")
    vectorizer = TfidfVectorizer()
    sparse_mat = vectorizer.fit_transform(terms_concat)
    logger.debug("converting to dataframe")
    row_idx, col_idx = sparse_mat.nonzero()
    author_term_tfidf = pd.DataFrame(
        {"row_idx": row_idx, "col_idx": col_idx, "tfidf_score": sparse_mat.data}
    )

    logger.debug("mapping row_idx to AuthorId")
    author_map = terms_concat.reset_index()["AuthorId"]
    author_term_tfidf["AuthorId"] = author_term_tfidf["row_idx"].map(author_map)

    logger.debug("mapping col_idx to term_id")
    term_id_map = pd.Series(vectorizer.get_feature_names())
    author_term_tfidf["term_id"] = author_term_tfidf["col_idx"].map(term_id_map)

    author_term_tfidf.drop(columns=["row_idx", "col_idx"], inplace=True)
    return author_term_tfidf


def main(args):
    pass


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
