# -*- coding: utf-8 -*-

DESCRIPTION = """Get average specter embeddings for authors"""

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

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

import pandas as pd
import numpy as np

from collabnetworks.util import get_score_column
from collabnetworks.data_helper import DataHelper, DEFAULTS_V3


def get_avg_specter(
    df_paper_authors, specter_embeddings, weighted: bool = True
) -> pd.Series:
    aembs = df_paper_authors["s2_idx"].apply(lambda x: specter_embeddings[x])
    if weighted is True:
        aembs = aembs * df_paper_authors["score"]
    df_paper_authors["embs"] = aembs
    gb = df_paper_authors.groupby("AuthorId")
    avg_specter = gb["embs"].apply(lambda x: np.mean(x, axis=0))
    return avg_specter


def get_df_paper_authors(
    data_helper,
    specter_embeddings_paper_ids,
    weighted: bool = True,
    dedup_titles: bool = True,
) -> pd.DataFrame:
    _df = data_helper.df_s2_id
    _df = _df[_df.s2_id.isin(specter_embeddings_paper_ids)]
    paa = data_helper.mag_data.paper_authors
    paa = paa.loc[paa["PaperId"].isin(_df.PaperId)]
    paa.drop_duplicates(subset=["PaperId", "AuthorId"], inplace=True)
    if weighted is True:
        paa["score"] = get_score_column(paa, data_helper.mag_data.papers)
    paa = paa.merge(_df, on="PaperId", how="inner")
    s2_to_idx = {
        val: idx[0] for idx, val in np.ndenumerate(specter_embeddings_paper_ids)
    }
    paa["s2_idx"] = paa["s2_id"].map(s2_to_idx)
    if dedup_titles is True:
        from collabnetworks.util import drop_duplicate_titles

        logger.debug("dropping duplicate paper titles")
        dedup = drop_duplicate_titles(data_helper.mag_data.papers)
        paa = paa[paa["PaperId"].isin(dedup["PaperId"])]
        logger.debug(f"dataframe shape: {paa.shape}")
    return paa


def load_slim_data_helper(defaults, min_year=None, max_year=None) -> DataHelper:
    fnames = defaults["data_fnames"]
    data_helper = DataHelper(min_year=min_year, max_year=max_year)
    data_helper.load_mag_data(fnames["mag_data"])
    data_helper.load_s2_mapping(fnames["s2_mapping"])
    return data_helper


def main(args):
    outfpath = Path(args.output)
    data_helper = load_slim_data_helper(
        DEFAULTS_V3, min_year=args.min_year, max_year=args.max_year
    )
    specter_dir = Path(args.specter_dir)
    weighted = not args.ignore_weights

    fname = specter_dir.joinpath("specter_embeddings.npy")
    logger.debug(f"loading specter embeddings from {fname}")
    specter_embeddings = np.load(fname)
    logger.debug(f"specter_embeddings shape: {specter_embeddings.shape}")

    fname = specter_dir.joinpath("specter_embeddings_paper_ids.npy")
    logger.debug(f"loading specter embeddings paper IDs from {fname}")
    specter_embeddings_paper_ids = np.load(fname)
    logger.debug(
        f"specter_embeddings_paper_ids shape: {specter_embeddings_paper_ids.shape}"
    )

    df_paper_authors = get_df_paper_authors(
        data_helper, specter_embeddings_paper_ids, weighted=weighted
    )
    logger.debug(f"df_paper_authors shape: {df_paper_authors.shape}")
    logger.debug(f"{df_paper_authors.AuthorId.nunique()} unique authors")

    logger.debug("getting average specter embeddings for authors...")
    avg_specter = get_avg_specter(
        df_paper_authors, specter_embeddings, weighted=weighted
    )
    logger.debug(f"done. calculated {len(avg_specter)} average embeddings")

    logger.debug(f"writing to {outfpath}")
    avg_specter.to_pickle(outfpath)


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
        "specter_dir",
        help="directory containing `specter_embeddings.npy` and `specter_embeddings_paper_ids.npy`",
    )
    parser.add_argument("output", help="output filename (pandas pickle file)")
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
        "--ignore-weights", action="store_true", help="ignore paper weights per author"
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
