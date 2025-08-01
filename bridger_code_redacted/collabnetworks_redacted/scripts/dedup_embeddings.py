# -*- coding: utf-8 -*-

DESCRIPTION = """When embeddings are creted, they are stored as multiple .npy files, alongside aligned .npy files for corresponding terms. This script collects and deduplicates these embeddings, and saves as a single .npy file for embeddings and a single .npy file for terms"""

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

from util import curr_mem_usage

def load_npy_files(dirpath: Path, glob_pattern: str, allow_pickle: bool = False) -> np.ndarray:
    fpaths = list(dirpath.glob(glob_pattern))
    fpaths.sort()
    arrs = []
    for fpath in fpaths:
        logger.debug(f"loading {fpath}...")
        arr = np.load(fpath, allow_pickle=allow_pickle)
        arrs.append(arr)
    return np.concatenate(arrs)

def main(args):
    dirpath = Path(args.dirpath)
    outfpath_terms = dirpath.joinpath('terms_dedup.npy')
    outfpath_embeddings = dirpath.joinpath('embeddings_dedup.npy')
    if outfpath_embeddings.exists():
        raise FileExistsError(f"{outfpath_embeddings} already exists")
    if outfpath_terms.exists():
        raise FileExistsError(f"{outfpath_terms} already exists")
    logger.debug("loading terms...")
    terms = load_npy_files(dirpath, 'terms*.npy', allow_pickle=True)
    logger.debug(f"there are {len(terms)} terms")
    logger.debug(f"current memory usage: {curr_mem_usage()}")
    logger.debug("loading embeddings...")
    embeddings = load_npy_files(dirpath, 'embeddings*.npy')
    logger.debug(f"there are {len(embeddings)} embeddings")
    logger.debug(f"current memory usage: {curr_mem_usage()}")
    s_terms = pd.Series(terms)
    logger.debug("dropping duplicate terms...")
    s_terms_dedup = s_terms.drop_duplicates()
    logger.debug(f"there are {len(s_terms_dedup)} terms")

    df_embeddings = pd.DataFrame(embeddings)
    
    logger.debug("dropping duplicate embeddings (by reindexing to terms index)...")
    df_embeddings_dedup = df_embeddings.reindex(s_terms_dedup.index)
    logger.debug(f"dataframe shape: {df_embeddings_dedup.shape}")
    
    s_terms_dedup.reset_index(drop=True, inplace=True)
    df_embeddings_dedup.reset_index(drop=True, inplace=True)
    outfname = dirpath.joinpath('terms_dedup.npy')
    logger.debug(f"saving to {outfname}")
    np.save(outfname, s_terms_dedup.values)
    outfname = dirpath.joinpath('embeddings_dedup.npy')
    logger.debug(f"saving to {outfname}")
    np.save(outfname, df_embeddings_dedup.values)

if __name__ == "__main__":
    total_start = timer()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    logger.info("pid: {}".format(os.getpid()))
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("dirpath", help="path to data directory")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))