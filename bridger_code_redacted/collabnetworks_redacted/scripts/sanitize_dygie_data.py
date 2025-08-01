# -*- coding: utf-8 -*-

DESCRIPTION = """Despite best efforts, some of the tokenized sentences in the dygie input JSONL files are too long. That is, they have too many (more than 512) BERT tokens, AKA wordpieces. This script will remove these longs sentences, based on a CSV file that identifies where they are."""

import sys, os, time, re, json
from pathlib import Path
from shutil import copyfile
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

PATTERN_JSONL = re.compile(r"check_length_(.*\.jsonl)\.log")

def sanitize(fpath: Path, backup_dir: Path, gdf: pd.DataFrame) -> None:
    """Sanitize one JSONL file

    :fpath: JSONL file

    """
    fpath = Path(fpath)
    # back up file
    copyfile(fpath, backup_dir.joinpath(fpath.name))

    doc_ids = gdf['doc_id'].unique()
    with fpath.open('r') as f:
        jlines = [json.loads(line) for line in f]
    with fpath.open('w') as outf:
        for doc in jlines:
            if doc['doc_key'] in doc_ids:
                # need to sanitize this doc
                new_sents = []
                sent_idxs = gdf[gdf['doc_id']==doc['doc_key']]['sent_idx'].values
                for i, sent in enumerate(doc['sentences']):
                    if i in sent_idxs:
                        # don't put it in new_sents
                        logger.debug("deleting doc_id {} sentence {}".format(doc['doc_key'], i))
                    else:
                        new_sents.append(sent)
                # replace the doc's sentences with new_sents
                doc['sentences'] = new_sents
            # rewrite the doc to new file
            print(json.dumps(doc), file=outf)


def main(args):
    longsents = pd.read_csv(args.longsents)
    dirpath = Path(args.dirpath)
    backup_dir = dirpath.joinpath('backup_dirty')
    logger.debug("there are {} sentences to remove".format(len(longsents)))
    for fname, gdf in longsents.groupby('logfile'):
        m = PATTERN_JSONL.search(fname)
        jsonl_fpath = dirpath.joinpath(m.group(1))
        logger.debug("sanitizing file {}".format(jsonl_fpath))
        sanitize(jsonl_fpath, backup_dir, gdf)

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
    parser.add_argument("longsents", help="path to CSV file identifying the sentences with too many tokens")
    parser.add_argument("dirpath", help="directory with JSONL data (data to sanitize)")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
