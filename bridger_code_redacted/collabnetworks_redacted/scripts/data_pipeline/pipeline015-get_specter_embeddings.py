# -*- coding: utf-8 -*-

DESCRIPTION = """Collect the SPECTER embeddings for relevant papers from dynamo DB and output them as a numpy array, and a corresponding array of S2 paper IDs"""

import sys, os, time, json
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

import pys2
# Set logging level for libraries with overly aggressive logging:
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


def main(args):
    outdir = Path(args.output)
    if not outdir.exists():
        logger.debug("creating output directory: {}".format(outdir))
        outdir.mkdir()
    else:
        logger.debug("using output directotry: {}".format(outdir))

    logger.debug(f"loading list of s2 IDs from file: {args.input}")
    s2_ids = pd.read_csv(args.input, names=["s2_id"], squeeze=True)
    logger.debug(f"loaded {len(s2_ids)} S2 IDs")

    if args.existing:
        dirpath_existing = Path(args.existing)
        fpath = dirpath_existing.joinpath("specter_embeddings_paper_ids.npy")
        logger.debug(f"loading list of existing s2 IDs from file: {fpath}")
        s2_ids_existing = np.load(fpath)
        logger.debug(f"loaded {len(s2_ids_existing)} existing S2 IDs")
        logger.debug("removing these from the list to use for DB query")
        s2_ids = s2_ids[~(s2_ids.isin(s2_ids_existing))].drop_duplicates()

    logger.debug(f"querying DB for embeddings using a list of {len(s2_ids)} IDs")
    specter_embeddings = pys2.get_embeddings(s2_ids.values)
    logger.debug(f"done with query. got {len(specter_embeddings)} embeddings")

    paper_ids = []
    embeddings = []
    for k, v in specter_embeddings.items():
        paper_ids.append(k)
        embeddings.append(v.embedding.data)
    paper_ids = np.array(paper_ids)
    embeddings = np.array(embeddings)

    if args.existing:
        fpath = dirpath_existing.joinpath("specter_embeddings.npy")
        logger.debug(f"loading existing embeddings from file: {fpath}")
        embeddings_existing = np.load(fpath)
        logger.debug(f"loaded embeddings array with shape: {embeddings_existing.shape}")
        logger.debug("appending new embeddings to existing")
        embeddings = np.vstack((embeddings_existing, embeddings))
        paper_ids = np.hstack((s2_ids_existing, paper_ids))

    outfpath = outdir.joinpath("specter_embeddings.npy")
    logger.debug(f"saving array of shape {embeddings.shape}) to {outfpath}")
    np.save(outfpath, embeddings)

    outfpath = outdir.joinpath("specter_embeddings_paper_ids.npy")
    logger.debug(f"saving array of shape {paper_ids.shape} to {outfpath}")
    np.save(outfpath, paper_ids)


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
        "input", help="input file with s2 IDs (no header, newline delimited)"
    )
    parser.add_argument("output", help="output directory")
    parser.add_argument(
        "--existing",
        help="path to directory with existing embeddings data, to merge in with the new data.",
    )
    parser.add_argument("--debug", action="store_true", help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug("debug mode is on")
    logger.debug("Description of this script:\n{}".format(DESCRIPTION))
    main(args)
    total_end = timer()
    logger.info(
        "all finished. total time: {}".format(format_timespan(total_end - total_start))
    )
