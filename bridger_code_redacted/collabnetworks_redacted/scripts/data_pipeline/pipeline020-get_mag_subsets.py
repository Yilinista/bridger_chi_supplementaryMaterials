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
from collabnetworks.data_preprocessing import MagSubsetGetter


def main(args):
    df_input = pd.read_csv(args.input, header=None, names=["magId"])
    subset_ids = df_input["magId"]
    logger.debug("{} paper IDs to get subset for".format(len(df_input)))
    data_getter = MagSubsetGetter(args.outdir, subset_ids, args.datadir)
    data_getter.main()


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
        "input", help="newline separated text file with the IDs to filter"
    )
    parser.add_argument("outdir", help="output directory")
    parser.add_argument("--datadir", help="directory with MAG parquet data")
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

