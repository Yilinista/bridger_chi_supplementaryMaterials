# -*- coding: utf-8 -*-

DESCRIPTION = """Use multiprocessing to collect user study data for multiple authors"""

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

from multiprocessing import Pool, cpu_count

from collabnetworks import DataHelper
from collabnetworks.data_helper import DEFAULTS_ABBREVIATIONS_EXPANDED
from collabnetworks.user_study import Initializer, DEFAULT_NUM_SIM_FOR_USER_STUDY


def process_single(author_id):
    this_start = timer()
    logger.debug(f"starting process for author_id: {author_id}")
    num_similar = DEFAULT_NUM_SIM_FOR_USER_STUDY["author"]
    num_similar_for_personas = DEFAULT_NUM_SIM_FOR_USER_STUDY["persona"]
    initializer.get_data_single_author_id(
        author_id,
        num_similar=num_similar,
        term_rank_compare=True,
        personas=2,
        num_similar_for_personas=num_similar_for_personas,
        # save_log=log_fpath,
    )
    logger.debug(f"done processing author {author_id}. took {timer() - this_start}")


def main(args):
    data_helper = DataHelper.from_defaults(
        defaults=DEFAULTS_ABBREVIATIONS_EXPANDED, min_year=2015, max_year=2022
    )
    global initializer
    initializer = Initializer(
        outdir=args.outdir, data_helper=data_helper, abbreviations_expanded=True
    )
    initializer.load_all_data()
    authors_to_get = [
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
        ("REDACTED", 0000000000),
    ]
    authors_to_get = [author_id for _, author_id in authors_to_get]
    n_proc = len(authors_to_get)
    logger.debug(f"spawning {len(authors_to_get)} processes, {n_proc} at a time")
    with Pool(processes=n_proc) as pool:
        pool.map(process_single, authors_to_get)


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
    parser.add_argument("outdir", help="output directory (will be created)")
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
