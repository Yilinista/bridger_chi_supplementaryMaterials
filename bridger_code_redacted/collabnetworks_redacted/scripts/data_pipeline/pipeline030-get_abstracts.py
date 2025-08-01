# -*- coding: utf-8 -*-

DESCRIPTION = """Get abstracts from corpus dev database"""

import sys, os, time
from typing import List, Union
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

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_CORPUS = {
    "dbname": "REDACTED",
    "host": "REDACTED",
    "user": "REDACTED",
    "password": os.environ.get("REDACTED", None),
    "port": "REDACTED",
}
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**DB_CORPUS)
)

import pandas as pd
import numpy as np


def get_exclude_ids(path_to_parquet: Union[str, Path]) -> List[int]:
    # existing data uses columns name 's2_id' (not 'corpus_paper_id')
    colname = "s2_id"
    df_exclude = pd.read_parquet(path_to_parquet, columns=[colname]).dropna()
    df_exclude[colname] = df_exclude[colname].astype(int)
    return df_exclude[colname].drop_duplicates().tolist()


def main(args):
    id_colname = args.id_colname
    logger.debug(f"loading data from {args.input}")
    df_new = pd.read_parquet(args.input, columns=[id_colname])
    logger.debug(f"dataframe shape: {df_new.shape}")
    logger.debug("dropping na...")
    df_new.dropna(inplace=True)
    df_new[id_colname] = df_new[id_colname].astype(int)
    df_new.sort_values(id_colname, inplace=True)
    logger.debug(f"dataframe shape: {df_new.shape}")
    logger.debug("dropping duplicates...")
    df_new.drop_duplicates(inplace=True)
    logger.debug(f"dataframe shape: {df_new.shape}")

    if args.existing is not None:
        logger.debug(f"excluding existing IDs from file {args.existing}")
        exclude_ids = get_exclude_ids(args.existing)
        logger.debug(f"{len(exclude_ids)} existing ids. excluding...")
        df_new = df_new[~(df_new[id_colname].isin(exclude_ids))]
        logger.debug(f"dataframe shape: {df_new.shape}")

    tmp_tablename = "tmp_paperids_get_abstracts"
    logger.debug(f"creating temporary table: {tmp_tablename}")
    df_new.to_sql(
        tmp_tablename,
        engine,
        method="multi",
        chunksize=10000,
        index=False,
        if_exists="replace",
    )
    # engine.execute(f"ALTER TABLE {tmp_tablename} add primary key({id_colname})")

    sq = """
    select a.id as {id_colname}, a.doi, a.title, a.abstract
    from papers as a
    inner join tmp_paperids_get_abstracts as b
    on a.id = b.{id_colname};
    """.format(
        id_colname=id_colname
    )
    logger.debug(f"getting data: sql: {sq}")
    this_start = timer()
    df_out = pd.read_sql(sq, engine)
    logger.debug(f"done. took {format_timespan(timer()-this_start)}")
    logger.debug(f"output dataframe shape: {df_out.shape}")

    logger.debug(f"writing to file: {args.output}")
    df_out.to_parquet(args.output)

    logger.debug(f"dropping temporary table: {tmp_tablename}")
    engine.execute(f"DROP TABLE {tmp_tablename}")


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
    parser.add_argument("input", help="input filename (parquet)")
    parser.add_argument("output", help="output filename (parquet)")
    parser.add_argument(
        "--existing",
        help="parquet with IDs for papers for which we already have abstracts. should have a column 's2_id' whose values will be excluded",
    )
    parser.add_argument(
        "--id-colname",
        default="corpus_paper_id",
        help="column name in the input file for paper ID (default: corpus_paper_id)",
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
