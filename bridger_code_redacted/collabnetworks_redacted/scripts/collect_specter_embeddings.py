# -*- coding: utf-8 -*-

DESCRIPTION = """Collect the SPECTER embeddings for relevant papers and output them as a numpy array, and a corresponding array of S2 paper IDs"""

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

import numpy as np

def main(args):
    outdir = Path(args.output)
    if not outdir.exists():
        logger.debug("creating output directory: {}".format(outdir))
        outdir.mkdir()
    else:
        logger.debug("using output directotry: {}".format(outdir))

    dirpath = Path(args.input)
    logger.debug(f"loading embeddings from directory: {dirpath}")
    embeddings = []
    paper_ids = []
    for fpath in dirpath.glob("*.json*"):
        with fpath.open("r") as f:
            for line in f:
                d = json.loads(line)
                embeddings.append(d['embedding'])
                paper_ids.append(d['id'])
    embeddings = np.array(embeddings)
    logger.debug("done loading embeddings")
    logger.debug(f"embeddings is an array of shape {embeddings.shape}")
    paper_ids = np.array(paper_ids)
    logger.debug(f"paper_ids is an array of shape {paper_ids.shape}")

    outfpath = outdir.joinpath("specter_embeddings.npy")
    logger.debug(f"saving to {outfpath}")
    np.save(outfpath, embeddings)

    outfpath = outdir.joinpath("specter_embeddings_paper_ids.npy")
    logger.debug(f"saving to {outfpath}")
    np.save(outfpath, paper_ids)

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
    parser.add_argument("input", help="directory containing JSONL files with input data")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    logger.debug("Description of this script:\n{}".format(DESCRIPTION))
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))