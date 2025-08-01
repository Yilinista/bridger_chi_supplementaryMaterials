# -*- coding: utf-8 -*-

DESCRIPTION = """Get papers identified as Computer Science papers by either MAG or S2"""

import sys, os, time
from typing import Union, Optional, List
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

import pandas as pd
import numpy as np


def load_mag_data(
    dirpath: Union[str, Path], fos_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    """Load MAG data

    Args:
        dirpath (Union[str, Path]): path to directory with MAG data in parquet format
        fos_ids (Optional[List[int]], optional): filter by these fos_ids. Defaults to None.

    Returns:
        pd.DataFrame: dataframe with columns ['PaperId', 'FieldOfStudyId', 'Year']
    """

    dirpath = Path(dirpath)
    # path_to_papers = dirpath.joinpath("Papers_parquet")
    # logger.debug(f"loading papers from {path_to_papers}")
    # papers = pd.read_parquet(path_to_papers, columns=["PaperId", "Year"])
    # logger.debug(f"dataframe shape: {papers.shape}")

    path_to_pfos = dirpath.joinpath("PaperFieldsOfStudy_parquet")
    logger.debug(f"loading data from {path_to_pfos}")
    pfos = pd.read_parquet(path_to_pfos, columns=["PaperId", "FieldOfStudyId"])
    logger.debug(f"dataframe shape: {pfos.shape}")

    if fos_ids is not None:
        logger.debug(f"filtering only FieldOfStudyId: {fos_ids}")
        pfos = pfos.loc[pfos["FieldOfStudyId"].isin(fos_ids), :]
        logger.debug(f"dataframe shape: {pfos.shape}")

    # logger.debug("merging dataframes")
    # out = pfos.drop_duplicates(subset=["PaperId"]).merge(
    #     papers, how="inner", on="PaperId"
    # )
    # logger.debug(f"dataframe shape: {out.shape}")

    # return out

    return pfos


def load_s2_data_and_merge_with_mag(
    fpath: Union[str, Path],
    df_mag: pd.DataFrame,
    fields_of_study: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load S2 data

    Args:
        fpath (Union[str, Path]): path to data (parquet)
        fields_of_study (Optional[List[str]]): filter to fields of study containing these terms

    Returns:
        pd.DataFrame: [description]
    """
    logger.debug(f"reading data from {fpath}")
    df_s2 = pd.read_parquet(fpath).rename(columns={"mag_id": "PaperId"})
    df_s2['PaperId'] = df_s2['PaperId'].astype(int)
    logger.debug(f"dataframe shape: {df_s2.shape}")
    if fields_of_study is not None:
        logger.debug(f"filtering by field_of_study: {fields_of_study}")
        df_s2_tomerge = df_s2.dropna(subset=["fields_of_study"])
        for fos in fields_of_study:
            df_s2_tomerge = df_s2_tomerge.loc[
                df_s2_tomerge["fields_of_study"].str.contains(fos), :
            ]
    else:
        df_s2_tomerge = df_s2
    df_s2_tomerge = df_s2_tomerge[["PaperId"]]
    logger.debug(f"number of rows: {len(df_s2_tomerge)}")
    logger.debug("merging in MAG data")
    outermerge = df_mag.merge(df_s2_tomerge, how="outer", on="PaperId", indicator=True)
    logger.debug(f"dataframe shape: {outermerge.shape}")

    # merge in additional data from S2
    logger.debug("merging in additional data from S2")
    outermerge = outermerge.merge(df_s2, how="left", on="PaperId")
    logger.debug(f"dataframe shape: {outermerge.shape}")
    return outermerge

def merge_in_year_data(df: pd.DataFrame, magdir: Union[Path, str]):
    """merge in year data from MAG

    Args:
        df (pd.DataFrame): dataframe with PaperId column
        magdir (Union[str, Path]): path to directory with MAG data in parquet format
    """
    dirpath = Path(magdir)
    path_to_papers = dirpath.joinpath("Papers_parquet")
    logger.debug(f"loading papers from {path_to_papers}")
    df_year = pd.read_parquet(path_to_papers, columns=["PaperId", "Year"])

    logger.debug("merging in Year data")
    df = df.merge(df_year, how='left', on='PaperId')
    logger.debug(f"dataframe shape: {df.shape}")
    return df




def main(args):
    outfpath = Path(args.output)
    if not outfpath.parent.exists():
        raise NotADirectoryError(
            f"output file parent directory {outfpath.parent} does not exist!"
        )
    if args.out_ids is not None:
        outfpath_ids = Path(args.out_ids)
        if not outfpath_ids.parent.exists():
            raise NotADirectoryError(
                f"output for IDs file: parent directory {outfpath_ids.parent} does not exist!"
            )

    cs_fos_ids = [41008148]
    df_mag = load_mag_data(args.magdir, fos_ids=cs_fos_ids)

    s2_fos = ["Computer Science"]
    cs_papers = load_s2_data_and_merge_with_mag(
        args.s2papers, df_mag, fields_of_study=s2_fos
    )

    cs_papers = merge_in_year_data(cs_papers, args.magdir)

    logger.debug(f"writing dataframe (parquet, shape: {cs_papers.shape}) to {outfpath}")
    cs_papers.to_parquet(outfpath)

    if args.out_ids is not None:
        paper_ids = cs_papers["PaperId"].drop_duplicates()
        logger.debug(f"writing {len(paper_ids)} paper IDs to file: {args.out_ids}")
        paper_ids.to_csv(args.out_ids, index=False, header=False)


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
    parser.add_argument("magdir", help="path to MAG data directory (with parquet data)")
    parser.add_argument(
        "s2papers",
        help="path to parquet file with S2 papers, including MAG IDs and field of study",
    )
    parser.add_argument("output", help="path to output file (parquet)")
    parser.add_argument(
        "--out-ids",
        type=str,
        default=None,
        help="additionally, output txt file of the MAG paper IDs",
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
