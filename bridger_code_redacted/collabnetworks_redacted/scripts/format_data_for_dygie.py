# -*- coding: utf-8 -*-

DESCRIPTION = """Format dataset for use with dygie model (consistent with SciERC dataset)

using as guide:
https://github.com/dwadden/dygiepp/blob/allennlp-v1/scripts/new-dataset/format_new_dataset.py
"""

import sys, os, time, json
from typing import List, TypedDict, Union
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
import pandas as pd
import numpy as np


def format_doc(
    doc_id: Union[int, str], text: str, nlp: spacy.Language
) -> TypedDict(
    "doc_dygie_format",
    {"doc_key": Union[int, str], "sentences": List[str], "dataset": str},
):
    spacy_doc = nlp(text)
    sentences = []
    # max_tokens_per_sentence = 300  # need to split long sentences or BERT will fail
    # Edit: don't need to do the above anymore, now that the dygie model incorporates a max_length parameter
    for sent in spacy_doc.sents:
        toks = [tok.text for tok in sent if not tok.is_space]
        # Split into chunks (will only apply to long sentences. For most sentences the loop will only run once.)
        # Actually, don't need to do this anymore
        # for i in range(0, len(toks), max_tokens_per_sentence):
        #     sentences.append(toks[i:i+max_tokens_per_sentence])
        if len(toks) != 0:
            sentences.append(toks)
    dataset = "scierc"  # this is needed, for now at least
    return {"doc_key": doc_id, "sentences": sentences, "dataset": dataset}


def format_doc_expand_abbreviations(
    doc_id: Union[int, str], text: str, nlp: spacy.Language
) -> TypedDict(
    "doc_dygie_format",
    {"doc_key": Union[int, str], "sentences": List[str], "dataset": str},
):
    spacy_doc = nlp(text)
    sentences = []
    abrv_map = {abrv.start_char: abrv for abrv in spacy_doc._.abbreviations}
    for sent in spacy_doc.sents:
        toks = []
        for sent in spacy_doc.sents:
            for tok in sent:
                if tok.is_space:
                    continue
                if tok.idx in abrv_map:
                    toks.extend([x.text for x in abrv_map[tok.idx]._.long_form])
                else:
                    toks.append(tok.text)
        if len(toks) != 0:
            sentences.append(toks)
    dataset = "scierc"  # this is needed, for now at least
    return {"doc_key": doc_id, "sentences": sentences, "dataset": dataset}


def get_outfile_name(p, i: int) -> Path:
    """Append filename with a zero-padded number

    :p: file path
    :i: number to append
    :returns: file path

    """
    p = Path(p)
    n = f"{p.stem}_{i:04d}{p.suffix}"
    return p.with_name(n)


def main(args):
    nlp = spacy.load("en_core_sci_sm")
    if args.expand_abbreviations is True:
        from scispacy.abbreviation import AbbreviationDetector

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

    idx_start = args.start
    if idx_start is None:
        idx_start = 0
    idx_end = args.end
    if idx_end is None:
        idx_end = len(df)
    if args.batch_size is not None:
        file_num = idx_start
        outfpath = get_outfile_name(args.output, file_num)
    # df = df.loc[idx_start:idx_end]  # dataframe should have a sequential integer index
    df = df.reset_index()
    df = df.loc[idx_start:idx_end]

    df = df.set_index(args.paperid_col)
    data = df["title"] + ". " + df["abstract"]
    # data cleaning
    data = data.str.replace("\n", " ")
    # data = data.str.replace('\u2008', '')
    logger.debug("processing {} papers (titles and abstracts)".format(len(data)))

    logger.debug("writing to file: {}".format(outfpath))
    outf = outfpath.open("w")

    line_num = idx_start
    logger.debug("starting with paper {}".format(data.index[0]))
    try:
        for paper_id, text in data.iteritems():
            if args.expand_abbreviations is True:
                outline = format_doc_expand_abbreviations(paper_id, text, nlp)
            else:
                outline = format_doc(paper_id, text, nlp)
            if outline["sentences"]:
                print(json.dumps(outline), file=outf)
                line_num += 1
                if args.batch_size is not None and line_num % args.batch_size == 0:
                    outf.close()
                    # file_num += line_num
                    file_num += 1
                    outfpath = get_outfile_name(args.output, file_num)
                    # line_num = 0
                    logger.debug("writing to file: {}".format(outfpath))
                    outf = outfpath.open("w")
    finally:
        outf.close()


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="output multiple files with a maximum of this many lines",
    )
    parser.add_argument(
        "--start", type=int, default=None, help="if specified, start with this index"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="if specified, end with this index (not included)",
    )
    parser.add_argument(
        "--paperid-col",
        default="paper_id",
        help="column name in the input file for paper ID",
    )
    parser.add_argument(
        "--expand-abbreviations",
        action="store_true",
        help="use scispacy to expand all abbreviations to their long forms",
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
