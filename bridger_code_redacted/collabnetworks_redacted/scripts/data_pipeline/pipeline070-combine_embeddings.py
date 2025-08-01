# -*- coding: utf-8 -*-

DESCRIPTION = """Combine embedding chunk files into one file (and do the same for corresponding terms).
This will save the combined files
`embeddings_dedup.npy` and `terms_dedup.npy` into the directory"""

import sys, os, time
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from typing import Tuple

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


def verify_existing_files_exist(dirpath_existing: Path) -> Tuple[Path, Path]:
    fpath_existing_terms = dirpath_existing.joinpath("terms_dedup.npy")
    fpath_existing_embeddings = dirpath_existing.joinpath("embeddings_dedup.npy")
    if (not fpath_existing_terms.exists()) or (not fpath_existing_embeddings.exists()):
        raise FileNotFoundError(
            f"make sure {fpath_existing_terms.name} and {fpath_existing_embeddings.name} files exist in specified existing directory: {dirpath_existing}"
        )
    return fpath_existing_terms, fpath_existing_embeddings


def main(args):
    dirpath = Path(args.dirpath)

    if args.existing is not None:
        dirpath_existing = Path(args.existing)
        fpath_existing_terms, fpath_existing_embeddings = verify_existing_files_exist(
            dirpath_existing
        )

    fpaths = list(dirpath.glob("terms*.npy"))
    fpaths.sort()
    arrs = []
    logger.debug(f"loading {len(fpaths)} terms files from {dirpath}")
    for fpath in fpaths:
        arr = np.load(fpath, allow_pickle=True)
        arrs.append(arr)
    terms = np.concatenate(arrs)
    logger.debug(f"loaded {len(terms)} terms")

    fpaths = list(dirpath.glob("embeddings*.npy"))
    fpaths.sort()
    arrs = []
    logger.debug(f"loading {len(fpaths)} embeddings files from {dirpath}")
    for fpath in fpaths:
        arr = np.load(fpath)
        arrs.append(arr)
    embeddings = np.concatenate(arrs)
    logger.debug(f"loaded {len(embeddings)} embeddings")

    if args.existing is not None:
        logger.debug(f"loading old existing terms from {fpath_existing_terms}")
        terms_existing = np.load(fpath_existing_terms, allow_pickle=True)
        logger.debug(f"loaded {len(terms_existing)} terms")
        logger.debug(
            f"loading old existing embeddings from {fpath_existing_embeddings}"
        )
        embeddings_existing = np.load(fpath_existing_embeddings)
        logger.debug(f"loaded {len(embeddings_existing)} embeddings")
        logger.debug("combining old existing and new embeddings and terms")
        terms = np.hstack((terms_existing, terms))
        logger.debug(f"new terms array shape: {terms.shape}")
        embeddings = np.vstack((embeddings_existing, embeddings))
        logger.debug(f"new embeddings array shape: {embeddings.shape}")

    logger.debug("dropping duplicate terms...")
    terms, terms_dedup_unique_indices = np.unique(terms, return_index=True)
    logger.debug(f"deduplicated array shape: {terms.shape}")
    # s_terms = pd.Series(terms)
    # s_terms_dedup = s_terms.drop_duplicates()
    # logger.debug(f"there are {len(s_terms_dedup)} terms")

    logger.debug("aligning embeddings with terms...")
    embeddings = embeddings[terms_dedup_unique_indices]
    logger.debug(f"deduplicated array shape: {embeddings.shape}")
    # df_embeddings = pd.DataFrame(embeddings)
    # df_embeddings_dedup = df_embeddings.reindex(s_terms_dedup.index)

    # s_terms_dedup.reset_index(drop=True, inplace=True)
    # df_embeddings_dedup.reset_index(drop=True, inplace=True)
    # logger.debug(f"shape of embeddings dataframe: {df_embeddings_dedup.shape}")

    outfname = dirpath.joinpath("terms_dedup.npy")
    logger.debug(f"saving {outfname}")
    np.save(outfname, terms)

    outfname = dirpath.joinpath("embeddings_dedup.npy")
    logger.debug(f"saving {outfname}")
    np.save(outfname, embeddings)


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
        "dirpath",
        help="directory containing embeddings*.npy files and terms*.npy files",
    )
    parser.add_argument(
        "--existing",
        help="path to directory with existing embeddings_dedup.npy and terms_dedup.npy files",
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
