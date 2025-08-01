# -*- coding: utf-8 -*-

DESCRIPTION = """Collect all of the s2 paper ids"""

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

def main(args):
    id_colname = args.id_colname
    logger.debug(f"loading data from {args.input}")
    df = pd.read_parquet(args.input, columns=[id_colname])
    logger.debug(f"dataframe shape: {df.shape}")
    logger.debug("dropping na...")
    df.dropna(inplace=True)
    df[id_colname] = df[id_colname].astype(int)
    df.sort_values(id_colname, inplace=True)
    logger.debug(f"dataframe shape: {df.shape}")
    logger.debug("dropping duplicates...")
    df.drop_duplicates(inplace=True)
    logger.debug(f"dataframe shape: {df.shape}")

    logger.debug(f"writing to file: {args.output}")
    df.to_csv(args.output, index=False, header=False)


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
    parser.add_argument("input", help="input filename (parquet)")
    parser.add_argument("output", help="output filename (csv)")
    parser.add_argument(
        "--id-colname",
        default="corpus_paper_id",
        help="column name in the input file for paper ID (default: corpus_paper_id)",
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

