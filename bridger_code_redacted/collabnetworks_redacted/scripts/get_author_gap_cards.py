# -*- coding: utf-8 -*-

DESCRIPTION = """Save author cards to JSON files, for authors with tasks/methods gaps to a seed author"""

import sys, os, time, json, tarfile
from pathlib import Path
from typing import Optional, Union, List, Dict, Collection
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

from config import Config
from util import AuthorHelper, get_root_dir, upload_to_google, google_save_directory
from data_helper import DataHelper

import pandas as pd
import numpy as np

from slugify import slugify

ROOT_DIR = get_root_dir()

config = Config()

def tar_output(dirpath: Path, tarfpath: Union[Path, str]):
    with tarfile.open(tarfpath, 'w:gz') as tar:
        tar.add(dirpath, arcname=dirpath.name)

def save_author_card(outdir, data, name, author_id, s2_id=None, min_year=None, max_year=None, overwrite=True, basedir_name='author_gap'):
        logger.debug(f"Getting data for {name}")
        # check if author_id is a collection (e.g., list) of ids, or if it is a single id
        if isinstance(author_id, str) or not isinstance(author_id, Collection):
            first_author_id = author_id
        else:
            # it's a collection of ids
            first_author_id = author_id[0]
        outfname = outdir.joinpath(f"{slugify(name)}_{first_author_id}_{min_year}-{max_year}.json")
        if overwrite is False and outfname.exists():
            logger.info(f"{outfname} already exists. skipping...")
            return
        try:
            author = AuthorHelper(author_id, data=data, min_year=min_year, max_year=max_year, name=name, s2_id=s2_id)
            card = author.to_card()
            card.to_json(outfname)
            logger.debug("uploading to google cloud storage")
            upload_to_google(outfname, config.gcloud_storage, subdir=f'authors/visgraphdata/{basedir_name}/{outdir.name}/')
        except Exception as e:
            logger.exception(f"Exception encountered for author {name} (magId: {author_id}). skipping...\n{e}")

def main(args):
    if args.outdir is None:
        outdir_base = Path(ROOT_DIR).joinpath('data/visgraph_data/author_gap')
    else:
        outdir_base = Path(args.outdir)
    basedir_name = outdir_base.name
    logger.debug(f"Using output base directory: {outdir_base}")
    if not outdir_base.exists():
        logger.debug(f"{outdir_base} does not exist. creating it.")
        outdir_base.mkdir()
    logger.debug(f"will write to google subdir: authors/visgraphdata/{basedir_name}/")
    data_helper = DataHelper.from_defaults()
    overwrite = not args.no_overwrite
    for fp in Path(args.dirpath).glob('*.csv'):
        name = fp.stem.split('_')[0]
        label = fp.stem.split('_')[-1]
        outdir = outdir_base.joinpath(f"{name}_{args.min_year}-{args.max_year}_gap_sim_{label}")
        if outdir.exists():
            logger.debug("using output directory {}".format(outdir))
        else:
            logger.debug("creating output directory {}".format(outdir))
            outdir.mkdir()
        logger.debug(f"loading {fp}")
        df = pd.read_csv(fp)
        df = df.drop_duplicates(subset=['author_mag_id'])
        sort_col = f"{label}_sim_norm"
        N = 10
        logger.debug(f"sorting by column {sort_col} (descending), then taking the top {N}")
        df = df.sort_values(sort_col, ascending=False)
        for _, row in df.head(N).iterrows():
            author_id = row['author_mag_id']
            this_name = row['DisplayName']
            # s2_id = check_for_s2_id(author_id)
            s2_id = None
            save_author_card(outdir, 
                            data_helper, 
                            this_name, 
                            author_id, 
                            s2_id, 
                            min_year=args.min_year, 
                            max_year=args.max_year, 
                            overwrite=overwrite,
                            basedir_name=basedir_name)
        google_save_directory(config.gcloud_storage, sub_subdir=f'{basedir_name}/{outdir.name}/')

    if args.tar is True:
        tarfpath = outdir.parent.with_suffix('.tar.gz')
        logger.debug(f"saving tar archive to {tarfpath}")
        tar_output(outdir.parent, tarfpath)

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
    parser.add_argument("dirpath", help="input directory with CSV files")
    parser.add_argument("-o", "--outdir", default=None, help="base output directory (default: data/visgraph_data/author_gap")
    parser.add_argument("--min-year", type=int, default=2015, help="filter out papers published before this year (will include this year)")
    parser.add_argument("--max-year", type=int, default=2020, help="filter out papers published this year and forward (will not include this year)")
    parser.add_argument("--no-overwrite", action='store_true', help="don't overwrite existing files (default behavior is to overwrite)")
    parser.add_argument("--tar", action='store_true', help="create a tar archive of the parent directory for 'outdir' (e.g., visgraph_data)")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))


