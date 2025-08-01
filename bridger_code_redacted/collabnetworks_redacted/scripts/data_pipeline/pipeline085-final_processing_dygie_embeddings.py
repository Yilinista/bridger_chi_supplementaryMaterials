# -*- coding: utf-8 -*-

DESCRIPTION = """Final processing for the dygie term embeddings, optionally merging in previous data."""

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
    embeddings_dirpath = Path(args.embeddings_dirpath)
    fpath_embeddings = embeddings_dirpath.joinpath("embeddings_dedup.npy")
    fpath_terms = embeddings_dirpath.joinpath("terms_dedup.npy")

    logger.debug(f"loading terms from {fpath_terms}")
    terms = np.load(fpath_terms, allow_pickle=True)
    logger.debug(f"there are {len(terms)} terms")

    logger.debug(f"loading embeddings from {fpath_embeddings}")
    embeddings = np.load(fpath_embeddings)
    logger.debug(f"there are {len(embeddings)} embeddings")

    logger.debug("dropping duplicates and sorting")
    terms, terms_unique_indices = np.unique(terms, return_index=True)
    embeddings = embeddings[terms_unique_indices]
    logger.debug(f"there are now {len(terms)} terms and {len(embeddings)} embeddings")

    if args.existing:
        dirpath_old = Path(args.existing)
        fpath = dirpath_old.joinpath("embedding_term_to_id.parquet")
        logger.debug(f"loading old embedding terms from {fpath}")
        embedding_term_to_id_old = pd.read_parquet(fpath)
        logger.debug(f"array shape: {embedding_term_to_id_old.shape}")

        fpath = dirpath_old.joinpath("embeddings.npy")
        logger.debug(f"loading old embeddings data from {fpath}")
        embeddings_old = np.load(fpath)
        logger.debug(f"array shape: {embeddings_old.shape}")

        logger.debug("merging old and new terms, and deduplicating")
        terms = np.hstack((embedding_term_to_id_old.index.values, terms))
        terms, terms_dedup_unique_indices = np.unique(terms, return_index=True)
        logger.debug(f"deduplicated array shape: {terms.shape}")
        logger.debug("merging old and new embeddings, and deduplicating")
        embeddings = np.vstack((embeddings_old, embeddings))
        embeddings = embeddings[terms_dedup_unique_indices]
        logger.debug(f"deduplicated array shape: {embeddings.shape}")

    logger.debug(f"reading input file: {args.input}")
    df = pd.read_parquet(args.input)
    logger.debug(f"dataframe shape: {df.shape}")
    labels = ["Task", "Method", "Material", "Metric"]
    logger.debug(f"filtering by labels: {labels}")
    df = df[df["label"].isin(labels)]
    logger.debug(f"dataframe shape: {df.shape}")
    logger.debug(f"filtering by softmax >= {args.softmax_threshold}")
    df = df[df["softmax_score"] >= args.softmax_threshold]
    logger.debug(f"dataframe shape: {df.shape}")
    df["s2_id"] = df["s2_id"].astype(int)

    columns_to_keep = ["s2_id", "label", "term_cleaned", "term_display"]
    df = df[columns_to_keep].rename(columns={"term_cleaned": "embedding_term"})
    logger.debug("groupby and count: keeping only one row per embedding term per s2_id")
    df = df.groupby(list(df.columns)).size().rename("freq").reset_index()
    logger.debug(f"dataframe shape: {df.shape}")

    if args.existing:
        dirpath_old = Path(args.existing)
        fpath = dirpath_old.joinpath(
            f"dygie_embedding_term_ids_to_s2_id_softmaxThreshold{args.softmax_threshold:.2f}.parquet"
        )
        logger.debug(f"reading old paper-to-terms data from {fpath}")
        # ignore old term IDs
        df_papers_embterms_old = pd.read_parquet(
            fpath, columns=["s2_id", "label", "embedding_term", "term_display", "freq"]
        )
        df_papers_embterms_old["s2_id"] = df_papers_embterms_old["s2_id"].astype(int)
        logger.debug("removing papers from old data that we have new data for")
        df_papers_embterms_old = df_papers_embterms_old[~(df_papers_embterms_old["s2_id"].isin(df["s2_id"]))]
        logger.debug("concatenating old and new paper-to-terms data")
        df = pd.concat([df, df_papers_embterms_old])
        logger.debug(f"dataframe shape: {df.shape}")
        logger.debug("dropping duplicates")
        df.drop_duplicates(inplace=True)
        logger.debug(f"dataframe shape: {df.shape}")

    logger.debug("dropping blank rows")
    df = df[df["embedding_term"] != ""]
    logger.debug(f"dataframe shape: {df.shape}")

    # logger.debug("Only keeping embeddings that actually occur in the dataframe")
    # indices = np.isin(terms, df["embedding_term"].unique(), assume_unique=True)
    # terms = terms[indices]
    # logger.debug(f"terms array shape: {terms.shape}")
    # embeddings = embeddings[indices]
    # logger.debug(f"embeddings array shape: {embeddings.shape}")

    logger.debug("Only keeping embeddings that actually occur in the dataframe")
    _idxmap = pd.Series(range(len(terms)), index=terms).rename("embedding_term_idx")
    _idxmap.index.name = "embedding_term"
    embedding_term_idx_occurrences = df["embedding_term"].map(_idxmap)
    embedding_term_idx_occurrences = (
        embedding_term_idx_occurrences.sort_values().drop_duplicates()
    )
    terms = terms[embedding_term_idx_occurrences.values]
    logger.debug(f"terms array shape: {terms.shape}")
    embeddings = embeddings[embedding_term_idx_occurrences.values]
    logger.debug(f"embeddings array shape: {embeddings.shape}")

    map_embedding_term_to_embedding_term_id = pd.Series(
        range(len(terms)), index=terms
    ).rename("embedding_term_id")
    map_embedding_term_to_embedding_term_id.index.name = "embedding_term"
    logger.debug("mapping embedding_term_id to embedding_term")
    df["embedding_term_id"] = df["embedding_term"].map(
        map_embedding_term_to_embedding_term_id
    )

    df["s2_id"] = df["s2_id"].astype(int)

    if not outdir.exists():
        logger.debug(f"creating output directory: {outdir}")
        outdir.mkdir()
    else:
        logger.debug(f"using output directory: {outdir}")

    outf = outdir.joinpath(
        f"dygie_embedding_term_ids_to_s2_id_softmaxThreshold{args.softmax_threshold:.2f}.parquet"
    )
    logger.debug(f"saving to {outf} (dataframe shape: {df.shape}")
    df.to_parquet(outf)

    outf = outdir.joinpath("embedding_term_to_id.parquet")
    out_frame = map_embedding_term_to_embedding_term_id.to_frame()
    logger.debug(f"saving to {outf} (dataframe shape: {out_frame.shape}")
    out_frame.to_parquet(outf)

    outf = outdir.joinpath("embeddings.npy")
    logger.debug(f"saving to {outf} ({len(embeddings)} embeddings)")
    np.save(outf, embeddings)


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
        "embeddings_dirpath",
        help="directory with files 'terms_dedup.npy' and 'embeddings_dedup.npy'",
    )
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
        "--existing",
        help="path to existing processed embeddings data, to merge in with the new data.",
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
