# -*- coding: utf-8 -*-

DESCRIPTION = """get sentence transformer embeddings for dygie terms"""

import sys, os, time
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import Iterable

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
from sentence_transformers import SentenceTransformer
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def process_parquet_input(fname, colname="term_cleaned"):
    logger.debug("loading input data from parquet file: {}".format(args.input))
    df = pd.read_parquet(fname)
    logger.debug("dataframe shape: {}".format(df.shape))
    labels = ["Method", "Task", "Material", "Metric"]
    logger.debug(f"filtering only rows with label: {labels}")
    df = df.loc[df["label"].isin(labels), :]
    logger.debug("dataframe shape: {}".format(df.shape))

    logger.debug(f"using column: {colname}")
    df = df.loc[:, [colname]]
    logger.debug("dropping duplicates...")
    df.drop_duplicates(inplace=True)
    logger.debug("dataframe shape: {}".format(df.shape))
    logger.debug("dropping na...")
    df.dropna(inplace=True)
    logger.debug("dataframe shape: {}".format(df.shape))
    df.rename(columns={colname: "terms"}, inplace=True)
    return df


def drop_terms(
    df: pd.DataFrame, terms_to_drop: Iterable[str], colname: str = "terms"
) -> pd.DataFrame:
    return df[~df[colname].isin(terms_to_drop)]


def main(args):
    outdir = Path(args.outdir)
    logger.debug("creating output directory: {}".format(outdir))
    outdir.mkdir()
    logger.debug("loading model: {}".format(args.model))
    transformer_model = SentenceTransformer(args.model)
    logger.debug("loading input data from file: {}".format(args.input))
    if args.input.endswith("parquet"):
        df = process_parquet_input(args.input)
    else:
        df = pd.read_csv(args.input, names=["terms"])
        logger.debug("dataframe shape: {}".format(df.shape))
        logger.debug("dropping na...")
        df.dropna(inplace=True)
        logger.debug("dataframe shape: {}".format(df.shape))
        logger.debug("dropping duplicates...")
        df.drop_duplicates(inplace=True)
        logger.debug("dataframe shape: {}".format(df.shape))
    if args.existing is not None:
        logger.debug(f"reading existing terms from file: {args.existing}")
        df_terms_existing = pd.read_parquet(args.existing)
        terms_to_drop = df_terms_existing.index.values
        logger.debug(f"found {len(terms_to_drop)} terms. Dropping these from the dataframe...")
        df = drop_terms(df, terms_to_drop)
        logger.debug("dataframe shape: {}".format(df.shape))

    start_idx = 0
    step = args.chunksize
    logger.debug("getting embeddings for terms, {} at a time...".format(args.chunksize))
    i = 0
    while True:
        this_start = timer()
        terms_fname = outdir.joinpath("terms_{:06d}.npy".format(i))
        embeddings_fname = outdir.joinpath("embeddings_{:06d}.npy".format(i))
        end_idx = start_idx + step
        terms = df["terms"].values[start_idx:end_idx]
        if len(terms) == 0:
            break
        logger.debug(
            "Chunk {}. Getting embeddings for {} terms...".format(i, len(terms))
        )
        embeddings = transformer_model.encode(terms, batch_size=8)
        logger.debug(
            "Done getting embeddings for chunk {}. Took {}".format(
                i, format_timespan(timer() - this_start)
            )
        )
        logger.debug("saving terms to output file: {}".format(terms_fname))
        np.save(terms_fname, terms)
        logger.debug("saving embeddings to output file: {}".format(embeddings_fname))
        np.save(embeddings_fname, embeddings)
        start_idx += step
        i += 1
    logger.debug("done getting embeddings for all chunks")


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
        "input",
        help="input dataframe (newline separated text file, or parquet dataframe which will be processed)",
    )
    parser.add_argument("outdir", help="output directory (will be created)")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000000,
        help="process this many terms at a time, writing output after each chunk",
    )
    # parser.add_argument("--colname", default='term_display', help="column in the input to use (default: 'term_display')")
    parser.add_argument(
        "--model",
        default="roberta-large-nli-stsb-mean-tokens",
        help="model name or path to use for sentence transformer",
    )
    parser.add_argument("--existing", help="path to parquet file with index being existing terms to skip")
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
