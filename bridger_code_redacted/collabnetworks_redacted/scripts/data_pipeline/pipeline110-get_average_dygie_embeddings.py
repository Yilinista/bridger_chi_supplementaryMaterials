# -*- coding: utf-8 -*-

DESCRIPTION = """average embeddings for authors"""

import sys, os, time
from pathlib import Path
from typing import Optional, List, Dict, Union
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

# from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, save_npz

from collabnetworks.data_helper import DataHelper, DEFAULTS_V3, DEFAULTS_ABBREVIATIONS_EXPANDED
from collabnetworks.average_embeddings import get_author_term_matrix, get_avg_embeddings

defaults = {
    'DEFAULTS_V3': DEFAULTS_V3,
    # 'DEFAULTS_V3_ABBREVIATIONS_EXPANDED': DEFAULTS_V3_ABBREVIATIONS_EXPANDED,
    'DEFAULTS_ABBREVIATIONS_EXPANDED': DEFAULTS_ABBREVIATIONS_EXPANDED
}


def main(args):
    outdir = Path(args.outdir)
    if not outdir.exists():
        logger.debug(f"creating directory: {outdir}")
        outdir.mkdir()
    else:
        logger.debug(f"using directory {outdir}")
    data_helper = DataHelper.from_defaults(
        defaults[args.defaults], min_year=args.min_year, max_year=args.max_year
    )
    weighted = not args.ignore_weights

    for label in ["Task", "Method", "Material", "Metric"]:
        outf_avg_embs = outdir.joinpath(
            f"average_author_embeddings_{label.lower()}_pandas.pickle"
        )
        if outf_avg_embs.exists():
            logger.debug(f"skipping label {label} because file already exists: {outf_avg_embs}")
            continue

        ssmat = get_author_term_matrix(
            data_helper, label, weighted=weighted, dedup_titles=True
        )

        outf = outdir.joinpath(f"mat_author_{label.lower()}.npz")
        logger.debug(f"saving matrix to {outf}")
        save_npz(str(outf), ssmat.mat)
        outf = outdir.joinpath(f"mat_author_{label.lower()}_row_labels.npy")
        logger.debug(f"saving row labels to {outf}")
        np.save(outf, ssmat.row_labels)
        outf = outdir.joinpath(f"mat_author_{label.lower()}_col_labels.npy")
        logger.debug(f"saving column labels to {outf}")
        np.save(outf, ssmat.col_labels)

        logger.debug("getting average embeddings")
        avg_embeddings = get_avg_embeddings(
            ssmat.mat, data_helper.embeddings, weighted=weighted
        )

        logger.debug(f"saving average embeddings to {outf_avg_embs}")
        avg_embeddings.to_pickle(outf_avg_embs)


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
    parser.add_argument("outdir", help="output directory")
    parser.add_argument(
        "--min-year",
        type=int,
        default=2015,
        help="filter out papers published before this year (will include this year)",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2022,
        help="filter out papers published this year and forward (will not include this year)",
    )
    parser.add_argument(
        "--ignore-weights",
        action="store_true",
        help="ignore paper weights per author",
    )
    parser.add_argument(
        "--defaults",
        default="EMBEDDINGS_V3",
        help=f"specify defaults for data helper (options: {list(defaults.keys())}",
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
