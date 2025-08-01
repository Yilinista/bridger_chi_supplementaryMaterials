# -*- coding: utf-8 -*-

DESCRIPTION = """Final processing for the dygie terms, after cleaning and normalization and embedding."""

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


def main(args):
    outdir = Path(args.outdir)
    logger.debug(f"reading input file: {args.input}")
    df_ner = pd.read_parquet(args.input)
    logger.debug(f"dataframe shape: {df_ner.shape}")
    labels = ["Task", "Method", "Material", "Metric"]
    logger.debug(f"filtering by labels: {labels}")
    df_ner = df_ner[df_ner["label"].isin(labels)]
    logger.debug(f"dataframe shape: {df_ner.shape}")
    logger.debug(f"filtering by softmax >= {args.softmax_threshold}")
    df_ner = df_ner[df_ner["softmax_score"] >= args.softmax_threshold]
    logger.debug(f"dataframe shape: {df_ner.shape}")

    columns_to_keep = ["s2_id", "label", "term_normalized", "term_display"]
    df_ner = df_ner[columns_to_keep]
    logger.debug(
        "groupby and count: keeping only one row per normalized term per s2_id"
    )
    df_ner = df_ner.groupby(list(df_ner.columns)).size().rename("freq").reset_index()
    logger.debug(f"dataframe shape: {df_ner.shape}")

    df_ner["s2_id"] = df_ner["s2_id"].astype(int)

    if args.old_data:
        logger.debug(f"loading old data from {args.old_data}")
        df_ner_old = pd.read_parquet(args.old_data).drop(columns=["term_id"]).dropna(subset=['s2_id'])
        df_ner_old["s2_id"] = df_ner_old["s2_id"].astype(int)
        logger.debug(f"old data dataframe shape: {df_ner_old.shape}")
        logger.debug("removing papers from old data that we have new data for")
        df_ner_old = df_ner_old[~(df_ner_old["s2_id"].isin(df_ner["s2_id"]))]
        logger.debug(f"old data dataframe shape: {df_ner_old.shape}")
        logger.debug("merging old and new data")
        df_ner = pd.concat([df_ner_old, df_ner], ignore_index=True)
        logger.debug(f"dataframe shape: {df_ner.shape}")

        logger.debug(
            "for terms that already existed in the old data, keeping old 'term_display' data"
        )
        map_term_normalized_to_term_display = (
            df_ner[["term_normalized", "term_display"]]
            .drop_duplicates(subset=["term_normalized"])
            .set_index("term_normalized")["term_display"]
        )
        df_ner["term_display"] = df_ner["term_normalized"].map(
            map_term_normalized_to_term_display
        )

    logger.debug(
        "checking to make sure the combination of ['term_normalized' and 'term_display'] columns is unique"
    )
    num_unique = len(df_ner.drop_duplicates(subset=["term_normalized", "term_display"]))
    if len(df_ner.drop_duplicates(subset=["term_normalized"])) != num_unique:
        raise RuntimeError
    else:
        logger.debug(f"check passed. there are {num_unique} terms")

    logger.debug("getting unique term IDs")
    term_ids = (
        df_ner[["term_normalized", "term_display"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename_axis(index="term_id")
    )

    map_term_normalized_to_term_id = (
        term_ids["term_normalized"]
        .reset_index()
        .set_index("term_normalized")["term_id"]
    )

    df_ner["term_id"] = df_ner["term_normalized"].map(map_term_normalized_to_term_id)


    if not outdir.exists():
        logger.debug(f"creating output directory: {outdir}")
        outdir.mkdir()
    else:
        logger.debug(f"using output directory: {outdir}")

    outf = outdir.joinpath(
        f"dygie_terms_to_s2_id_softmaxThreshold{args.softmax_threshold:.2f}.parquet"
    )
    logger.debug(f"saving to {outf} (dataframe shape: {df_ner.shape}")
    df_ner.to_parquet(outf)

    outf = outdir.joinpath(
        f"dygie_term_ids_softmaxThreshold{args.softmax_threshold:.2f}.parquet"
    )
    logger.debug(f"saving to {outf} (dataframe shape: {term_ids.shape}")
    term_ids.to_parquet(outf)


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
    parser.add_argument("input", help="input file (parquet dataframe)")
    parser.add_argument(
        "outdir", help="output directory (will be created if it doesn't exist)"
    )
    parser.add_argument(
        "--softmax-threshold",
        type=float,
        default=0.9,
        help="remove rows with softmax threshold below this value (default: 0.9)",
    )
    parser.add_argument(
        "--old-data",
        help="path to existing 'dygie_terms_to_s2_id' parquet file. The new data will be merged in with the old data.",
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
