# -*- coding: utf-8 -*-

DESCRIPTION = (
    """Use multiprocessing and subprocessing to run abbreviation extraction in parallel"""
)

import sys, os, time, re
from typing import List
from pathlib import Path
import shutil
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

from multiprocessing import Pool, cpu_count
import subprocess

SCRIPT = "./scripts/extract_abbreviations.py"

# CMD = "python {path_to_script} {path_to_input} {path_to_output} --batch-size 5000 --start {idx_start} --end {idx_end} --paperid-col s2_id --debug"
CMD = "python {path_to_script} {path_to_input} {path_to_output} --paperid-col corpus_paper_id --debug"


def run_format_subprocess(
    path_to_input: Path,
    path_to_output: Path,
    logfpath: Path,
):
    with logfpath.open("w") as logf:
        # cmd = CMD.format(path_to_script=SCRIPT, path_to_input=path_to_input, path_to_output=path_to_output, idx_start=idx_start, idx_end=idx_end)
        cmd = CMD.format(
            path_to_script=SCRIPT,
            path_to_input=path_to_input,
            path_to_output=path_to_output,
        )
        logger.debug(cmd)
        logger.debug("logfile: {}".format(logfpath))
        subprocess.run(cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT)


def split_input_data(df: pd.DataFrame, outdir: Path, chunksize: int) -> List[Path]:
    i = 0
    start_idx = 0
    outfpaths = []
    while True:
        end_idx = start_idx + chunksize
        outfpath = outdir.joinpath("titles_abstracts_raw_{:06d}.parquet".format(i))
        df.iloc[start_idx:end_idx].to_parquet(outfpath)
        outfpaths.append(outfpath)
        if end_idx >= len(df):
            break
        start_idx += chunksize
        i += 1
    return outfpaths


def main(args):
    outdir = Path(args.outdir)
    if outdir.exists():
        raise FileExistsError(f"output directory {outdir} already exists!")
    logger.debug(f"creating output directory: {outdir}")
    outdir.mkdir()

    tmp_dir = outdir.joinpath("tmp_abstract_chunks")
    logger.debug(f"creating temporary directory: {tmp_dir}")
    tmp_dir.mkdir()

    logger.debug(f"reading input file: {args.input}")
    df = pd.read_parquet(args.input, columns=["corpus_paper_id", "title", "abstract"])
    logger.debug(f"dataframe shape: {df.shape}")
    logger.debug(f"dropping rows with missing titles or abstracts")
    df.dropna(inplace=True)
    logger.debug(f"dataframe shape: {df.shape}")

    n_proc = args.processes
    chunksize = args.chunksize
    if not chunksize or chunksize < 1:
        chunksize = (len(df) // n_proc) + 1

    logger.debug(f"splitting input data (chunksize: {chunksize})")
    fpaths = split_input_data(df, tmp_dir, chunksize)

    # input_basedir = Path(args.input)
    # path_to_output = Path(args.output)
    # if args.logdirpath is None:
    #     logdirpath = Path('.')
    # else:
    #     logdirpath = Path(args.logdirpath)
    logdirpath = outdir.joinpath("logs")
    logger.debug(f"creating directory: {logdirpath}")
    logdirpath.mkdir()

    # subprocess_args = []
    # idx_start = 0
    # step = 100000
    # log_idx = 1
    # while True:
    #     idx_end = idx_start + step
    #     today = datetime.strftime(datetime.now(), format="%Y%m%d")
    #     log_file = logdirpath.joinpath("format_data_for_dygie_{:06d}_{}.log".format(log_idx, today))
    #     path_to_input = input_basedir.joinpath('titles_abstracts_raw_{:09d}.parquet'.format(idx_start))
    #     subprocess_args.append( (path_to_input, path_to_output, log_file, idx_start, idx_end) )
    #     if idx_end >= 9523170:  # number of rows in the input file
    #         break
    #     idx_start += step
    #     log_idx += 1

    subprocess_args = []
    for i, fpath in enumerate(fpaths):
        today = datetime.strftime(datetime.now(), format="%Y%m%d")
        log_file = logdirpath.joinpath(
            "extract_abbreviations_{:06d}_{}.log".format(i, today)
        )
        outfpath = outdir.joinpath(
            "abbreviations_{:06d}.jsonl".format(i)
        )
        subprocess_args.append((fpath, outfpath, log_file))

    logger.debug(
        "running {} processes, {} at a time".format(len(subprocess_args), n_proc)
    )
    with Pool(processes=n_proc) as pool:
        pool.starmap(run_format_subprocess, subprocess_args)

    logger.debug(f"removing temporary directory: {tmp_dir}")
    shutil.rmtree(tmp_dir)


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
    parser.add_argument(
        "input", help="path to input parquet file with paper IDs, titles, and abstracts"
    )
    parser.add_argument("outdir", help="path to output directory")
    # parser.add_argument("--logdirpath", help="directory for log files")
    parser.add_argument(
        "--processes", type=int, default=30, help="number of parallel processes to run"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="number of chunks to split the input data into (default is number of rows divided by number of processes)",
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

