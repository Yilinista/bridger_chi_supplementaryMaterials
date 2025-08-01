# -*- coding: utf-8 -*-

DESCRIPTION = """Parse dygie predictions from JSONL files"""

import sys, os, time
from pathlib import Path
from typing import Dict, Iterator, Iterable, List, Union, Optional
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

import pandas as pd
import numpy as np

from collabnetworks.util_dygie import DygieDoc, yield_dygie_docs

def yield_ner_row(doc: DygieDoc) -> Iterator[Dict]:
    for ent in doc.ner:
        yield {
            's2_id': int(ent.doc.id),
            'label': ent.label,
            'term': ent.term,
            'softmax_score': ent.softmax_score
        }

def yield_relation_row(doc: DygieDoc) -> Iterator[Dict]:
    for rel in doc.relations:
        src_label = rel.src.label if rel.src else None
        dst_label = rel.dst.label if rel.dst else None
        yield {
            's2_id': int(rel.doc.id),
            'label': rel.label,
            'src_term': rel.src_term,
            'dst_term': rel.dst_term,
            'src_label': src_label,
            'dst_label': dst_label,
            'softmax_score': rel.softmax_score
        }

def get_data(files: Iterable[Union[str, Path]], verbose=False):
    data_ner = []
    data_relations = []
    i = 0
    for doc in yield_dygie_docs(files):
        for row in yield_ner_row(doc):
            data_ner.append(row)
        for row in yield_relation_row(doc):
            data_relations.append(row)
        i += 1
        if verbose is True and i % 200000 == 0:
            logger.debug("{} docs collected".format(i))
    return data_ner, data_relations

def output(outdir, df_ner, df_relations):
    df_ner['s2_id'] = df_ner['s2_id'].astype(str)
    df_relations['s2_id'] = df_relations['s2_id'].astype(str)

    outfpath = outdir.joinpath('df_ner.parquet')
    logger.debug("saving file: {}".format(outfpath))
    logger.debug("dataframe shape: {}".format(df_ner.shape))
    df_ner.to_parquet(outfpath)

    outfpath = outdir.joinpath('df_relations.parquet')
    logger.debug("saving file: {}".format(outfpath))
    logger.debug("dataframe shape: {}".format(df_relations.shape))
    df_relations.to_parquet(outfpath)

def chunk_args(list_of_args, num_chunks):
    # just calling np.array() on a complicated structure seems to screw things up. This might get around that
    arr = np.empty(len(list_of_args), dtype=object)
    for i, item in enumerate(list_of_args):
        arr[i] = item
    return np.array_split(arr, num_chunks)

def main_multiprocessing(args, dirpath, outdir):
    n_procs = args.processes
    if args.recursive is True:
        files = list(dirpath.rglob('predictions*.jsonl'))
    else:
        files = list(dirpath.glob('predictions*.jsonl'))
    logger.debug(f"{len(files)} files found")
    files_chunks = np.array_split(files, n_procs)

    
    logger.debug(f"running {len(files_chunks)} processes, {n_procs} at a time")
    with Pool(processes=n_procs) as p:
        data = p.map(get_data, files_chunks)

    data_ner = []
    data_relations = []
    for data_ner_chunk, data_relations_chunk in data:
        data_ner.extend(data_ner_chunk)
        data_relations.extend(data_relations_chunk)
    df_ner = pd.DataFrame(data_ner)
    df_relations = pd.DataFrame(data_relations)
    output(outdir, df_ner, df_relations)

def main(args):
    dirpath = Path(args.dirpath)
    if args.outdir is None:
        # now = datetime.strftime(datetime.now(), '%Y%m%dT%H%M%S')
        # outdir = dirpath.joinpath('predicted_terms_{}'.format(now))
        outdir = dirpath.joinpath('predicted_terms')
    else:
        outdir = Path(args.outdir)
    logger.debug("creating directory: {}".format(outdir))
    outdir.mkdir()

    if args.processes != 1:
        main_multiprocessing(args, dirpath, outdir)
        return

    if args.recursive is True:
        files = list(dirpath.rglob('predictions*.jsonl'))
    else:
        files = list(dirpath.glob('predictions*.jsonl'))
    logger.debug("collecting data from {}: {} JSONL files".format(dirpath, len(files)))
    data_ner, data_relations = get_data(files, verbose=True)
    df_ner = pd.DataFrame(data_ner)
    df_relations = pd.DataFrame(data_relations)
    output(outdir, df_ner, df_relations)

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
    parser.add_argument("dirpath", help="path to directory with input JSONL files")
    parser.add_argument("-o", "--outdir", help="output directory (will be created)")
    parser.add_argument("--processes", type=int, default=1, help="number of processes, if using multiprocessing (default: 1)")
    parser.add_argument("--recursive", action='store_true', help="look for JSONL files recursively in the dirpath")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
