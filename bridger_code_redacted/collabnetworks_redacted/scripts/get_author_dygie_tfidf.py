# -*- coding: utf-8 -*-

DESCRIPTION = """Get tfidf scores for dygie terms, based on occurrence by author"""

import sys, os, time
from typing import Union, Optional
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

from collabnetworks import DataHelper
from collabnetworks.data_helper import DEFAULTS_V3, DEFAULTS_ABBREVIATIONS_EXPANDED

# from collabnetworks.util import _tfidf_apply

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

defaults = {
    "DEFAULTS_V3": DEFAULTS_V3,
    "DEFAULTS_ABBREVIATIONS_EXPANDED": DEFAULTS_ABBREVIATIONS_EXPANDED,
}


def apply_cutoff(df_ner: pd.DataFrame, cutoff: int):
    vc = df_ner["term_id"].value_counts()
    to_keep = vc[vc >= cutoff]
    return df_ner[df_ner["term_id"].isin(to_keep.index)]


def get_tfidf_scores_for_one_label(
    data_helper: DataHelper,
    label: str,
    cutoff: int = 1,
    dedup_titles: bool = False,
    save_model: Optional[Union[str, Path]] = None,
):
    logger.debug(f"Getting dataframe of dygie terms with label == {label}")
    df_ner = data_helper.df_ner[data_helper.df_ner.label == label]
    logger.debug(f"dataframe shape: {df_ner.shape}")

    df_ner["term_id"] = df_ner["term_id"].astype(int)

    if dedup_titles is True:
        from collabnetworks.util import drop_duplicate_titles

        logger.debug("dropping duplicate paper titles")
        dedup = drop_duplicate_titles(data_helper.mag_data.papers)
        df_ner = df_ner[df_ner["PaperId"].isin(dedup["PaperId"])]
        logger.debug(f"dataframe shape: {df_ner.shape}")

    logger.debug("only keeping one occurrence of each term in each paper")
    df_ner.drop_duplicates(subset=["PaperId", "term_id"], inplace=True)
    logger.debug(f"dataframe shape: {df_ner.shape}")

    if cutoff > 1:
        logger.debug(f"only keeping terms that occur in at least {cutoff} papers")
        df_ner = apply_cutoff(df_ner, cutoff)
        logger.debug(f"dataframe shape: {df_ner.shape}")

    df_paa = data_helper.mag_data.paper_authors
    df_paa = df_paa[df_paa.PaperId.isin(df_ner.PaperId)]
    left = df_paa[["PaperId", "AuthorId"]].drop_duplicates()
    logger.debug("merging paper-term data with paper-author data")
    right = df_ner[
        [
            "PaperId",
            "label",
            "term_display",
            "term_id",
        ]
    ]
    df_author_term = left.merge(right, on="PaperId", how="inner")
    logger.debug(f"dataframe shape: {df_author_term.shape}")

    # logger.debug("calculating term frequencies")
    # df_author_term_count = df_author_term.groupby(["AuthorId", "term_idx"]).size()
    # df_author_term_count = df_author_term_count.reset_index(name="term_count")

    # logger.debug("calculating document frequencies")
    # author_freq = df_author_term[["AuthorId", "term_idx"]].drop_duplicates()
    # vc = author_freq.term_idx.value_counts()
    # df_author_term_count["all_count"] = df_author_term_count.term_idx.map(vc)
    # N = df_author_term_count.AuthorId.nunique()
    # logger.debug(f"there are {N} unique AuthorIds")

    # logger.debug("calculating tfidf scores...")
    # df_author_term_count["tfidf_score"] = df_author_term_count.apply(
    #     _tfidf_apply, N=N, axis=1
    # )

    # outfname = f"{args.output}_{label}.parquet"
    # logger.debug(f"saving to {outfname}")
    # df_author_term_count.to_parquet(outfname)

    logger.debug("aggregating author term data")
    df_author_term["term_id"] = df_author_term.term_id.astype(str)
    gb = df_author_term.groupby("AuthorId")
    terms_concat = gb["term_id"].agg(lambda term_id: " ".join(term_id))

    logger.debug(f"there are {len(terms_concat)} authors")

    logger.debug("fitting TfidfVectorizer")
    vectorizer = TfidfVectorizer()
    sparse_mat = vectorizer.fit_transform(terms_concat)
    if save_model is not None:
        logger.debug(f"saving fitted model to {save_model}")
        joblib.dump(vectorizer, save_model)
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
    author_term_tfidf["term_id"] = author_term_tfidf["term_id"].astype(int)
    logger.debug("merging in term names data")
    terms = df_ner[["term_display", "term_id"]].drop_duplicates()
    author_term_tfidf = author_term_tfidf.merge(terms, on="term_id", how="left")
    return author_term_tfidf


def main(args):
    data_helper = DataHelper.from_defaults(
        defaults[args.defaults],
        min_year=args.min_year,
        max_year=args.max_year,
        exclude=["embeddings"],
    )
    labels = ["Method", "Task", "Material", "Metric"]
    # labels = ["Method", "Task"]

    for label in labels:
        if args.save_model is True:
            save_model = f"{args.output}_vectorizer_{label}.joblib"
        else:
            save_model = None
        author_term_tfidf = get_tfidf_scores_for_one_label(
            data_helper,
            label,
            cutoff=args.cutoff,
            dedup_titles=args.drop_duplicate_titles,
            save_model=save_model,
        )
        outfname = f"{args.output}_{label}.parquet"
        logger.debug(f"saving to {outfname}")
        author_term_tfidf.to_parquet(outfname)


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
    parser.add_argument(
        "output", help="output file basename (will create files for each label)"
    )
    parser.add_argument(
        "--min-year", type=int, help="minimum year for papers (inclusive)"
    )
    parser.add_argument(
        "--max-year", type=int, help="maximum year for papers (exclusive)"
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=1,
        help="minimum paper-term frequency-- only keep terms that occur in at least this many papers (default: 1)",
    )
    parser.add_argument(
        "--drop-duplicate-titles",
        action="store_true",
        help="deduplicate papers by title",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="save the models using joblib (will use the output file basename",
    )
    parser.add_argument(
        "--defaults",
        default="EMBEDDINGS_V3",
        help=f"specify defaults for data helper (options: {list(defaults.keys())}",
    )
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
