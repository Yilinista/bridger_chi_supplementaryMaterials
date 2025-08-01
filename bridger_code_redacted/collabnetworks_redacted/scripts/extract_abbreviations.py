# -*- coding: utf-8 -*-

DESCRIPTION = """Use Scispacy abbreviation detector to get the long forms for abbreviations
"""

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

import spacy
from scispacy.abbreviation import AbbreviationDetector
import pandas as pd
import numpy as np

from multiprocessing import cpu_count


def main(args):
    spacy_model = "en_core_sci_sm"
    nlp = spacy.load(spacy_model, disable=["parser", "ner"])

    # Add the abbreviation pipe to the spacy pipeline.
    nlp.add_pipe("abbreviation_detector")

    outfpath = Path(args.output)
    logger.debug("loading input file: {}".format(args.input))
    df = pd.read_parquet(args.input)
    logger.debug("input file has shape: {}".format(df.shape))
    df = df.loc[:, [args.paperid_col, "title", "abstract"]].dropna()
    logger.debug(
        "after dropping NA and unnecessary columns, shape is {}".format(df.shape)
    )

    df = df.reset_index()

    df = df.set_index(args.paperid_col)
    data = df["title"] + ". " + df["abstract"]
    # data cleaning
    data = data.str.replace("\n", " ")
    # data = data.str.replace('\u2008', '')
    logger.debug("processing {} papers (titles and abstracts)".format(len(data)))
    logger.debug("starting with paper {}".format(data.index[0]))

    with outfpath.open("w") as outf:
        i = 0
        for doc in nlp.pipe(data.values, n_process=1):
            paper_id = data.index[i]
            abbreviations = []
            for abrv in doc._.abbreviations:
                abbreviations.append({str(abrv): str(abrv._.long_form)})
            if abbreviations:
                outline = {
                    "s2_id": int(paper_id),
                    "abbrv": abbreviations,
                }
                print(json.dumps(outline), file=outf)
            i += 1


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
    parser.add_argument("input", help="path to input file (parquet)")
    parser.add_argument("output", help="path to output file (.jsonl)")
    # parser.add_argument("--batch-size", type=int, default=None, help="output multiple files with a maximum of this many lines")
    # parser.add_argument("--start", type=int, default=None, help="if specified, start with this index")
    # parser.add_argument("--end", type=int, default=None, help="if specified, end with this index (not included)")
    parser.add_argument(
        "--paperid-col",
        default="corpus_paper_id",
        help="column name in the input file for paper ID",
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
