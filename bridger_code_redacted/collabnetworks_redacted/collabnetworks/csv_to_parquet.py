# -*- coding: utf-8 -*-

DESCRIPTION = """Convert a MAG table (in TSV format) to Parquet using spark"""

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
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

from config import Config
from mag_schema import getSchema, schema_data

def main(args):
    config = Config()
    spark = config.spark
    dirpath = Path(args.input)
    output_basedir = Path(args.output)
    try:
        for fpath in dirpath.glob('*.txt*'):
            schema_key = fpath.parts[-1].split('.')[0]
            if schema_key not in schema_data:
                logger.debug("schema not found for {}. skipping".format(fpath))
                continue
            outdir = output_basedir.joinpath('{}_parquet'.format(schema_key))
            if outdir.exists():
                logger.debug("{} already exists. skipping".format(outdir))
                continue
            logger.debug("reading file {}".format(fpath.resolve()))
            sdf = spark.read.csv(str(fpath), sep=args.sep, schema=getSchema(schema_key))
            logger.debug("dataframe schema: {}".format(sdf.schema))
            logger.debug("saving to parquet: {}".format(outdir))
            sdf.write.parquet(str(outdir))
            logger.debug("done saving to {}".format(outdir))
            logger.debug("number of columns: {}".format(len(sdf.columns)))
            logger.debug("number of rows: {}".format(sdf.count()))
            logger.debug("-----------------------------")
    finally:
        config.teardown()

if __name__ == "__main__":
    total_start = timer()
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input", help="path to input base directory with TSV files")
    parser.add_argument("output", help="path to output base directory for parquet directories")
    parser.add_argument("--sep", default='\t', help="separator for the input file (default: tab)")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    else:
        logger.setLevel(logging.INFO)
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))

