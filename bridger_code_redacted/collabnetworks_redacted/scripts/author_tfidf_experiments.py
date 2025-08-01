# -*- coding: utf-8 -*-

DESCRIPTION = """Try out a few strategies to get different tfidf scores per author per dygie term"""

import sys, os, time
from pathlib import Path
from typing import Optional, List, Union, Dict
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

from collabnetworks.mag_data import MagData
from collabnetworks.data_helper import DataHelper, DEFAULTS_V2
from collabnetworks.term_ranking.tfidf_util import (
    DATADIR_TFIDF,
    aggregate_author_term_data,
    get_author_term_tfidf,
    apply_cutoff,
)

MAG_DATADIR = DEFAULTS_V2["data_fnames"]["mag_data"]
GENERIC_WORDS = [
    "component",
    "block",
    "module",
    "task",
    "methodology",
    "component",
    "technology",
    "mechanism",
    "approach",
    "method",
    "technique",
    "framework",
    "system",
    "model",
    "algorithm",
    "procedure",
]


def load_slim_data_helper(
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
):
    mag_data = MagData(
        MAG_DATADIR,
        tablenames=["Papers", "PaperAuthorAffiliations"],
        min_year=min_year,
        max_year=max_year,
    )
    data_helper = DataHelper.from_defaults(
        mag_data=mag_data,
        exclude=["ner_terms", "embeddings"],
        min_year=min_year,
        max_year=max_year,
    )
    return data_helper


class AuthorTfidfExperiments:
    def __init__(
        self,
        data: DataHelper,
        term_normalized_ids: pd.Series,
        df_paper_term: pd.DataFrame,
        outdir: Union[Path, str],
    ):
        self.data = data
        self.term_normalized_ids = term_normalized_ids
        self.df_paper_term = df_paper_term
        self.outdir = Path(outdir)

        # logger.debug("getting term_normalized_map")
        # self.term_normalized_map = (
        #     term_normalized_ids.drop_duplicates()
        #     .reset_index()
        #     .set_index("term_normalized")["term_id"]
        # )

        # logger.debug("mapping term_id to df_paper_term")
        # self.df_paper_term["term_id"] = self.df_paper_term.term_normalized.map(
        #     self.term_normalized_map
        # )
        self.df_paper_term.dropna(subset=["term_id"], inplace=True)
        self.df_paper_term["term_id"] = self.df_paper_term["term_id"].astype(int)

        self.df_paper_author = data.mag_data.paper_authors
        # self.df_paper_author = self.df_paper_author[
        #     self.df_paper_author["PaperId"].isin(self.df_paper_term["PaperId"])
        # ]
        self.df_paper_author = self.df_paper_author[
            ["PaperId", "AuthorId"]
        ].drop_duplicates()

        self.id_mappings = {}

    def get_mapping_generic_truncate(self, generic_words: List[str] = GENERIC_WORDS):
        # merge terms that end in generic words
        logger.debug("getting mapping for terms that ends in generic words")
        truncate = self.term_normalized_ids
        for i, word in enumerate(generic_words):
            logger.debug(f'processing word "{word}" -- {i+1} of {len(generic_words)}')
            truncate = truncate.str.replace(f" {word}$", "", regex=True)
        logger.debug("mapping terms after truncating...")
        truncate.index.name = "term_id"
        truncate = truncate.reset_index()
        truncate = truncate.merge(
            self.term_normalized_ids.reset_index().rename(
                columns={"term_id": "term_id_new"}
            ),
            how="inner",
            on="term_normalized",
        )
        return truncate.set_index("term_id")["term_id_new"]

    def strategy_01(self, df_ner: pd.DataFrame, label: str):
        logger.debug("---strategy 01---")
        outfname = self.outdir.joinpath(f"author_term_tfidf_{label}_strategy01.parquet")
        if outfname.exists():
            logger.info(f"{outfname} already exists. skipping.")
            return

        logger.debug(f"starting with paper-term data with shape {df_ner.shape}")
        logger.debug("merging paper-term data with paper-author data")
        df_author_term = df_ner.merge(self.df_paper_author, on="PaperId", how="inner")
        logger.debug(f"dataframe shape: {df_author_term.shape}")

        terms_concat = aggregate_author_term_data(df_author_term)
        author_term_tfidf = get_author_term_tfidf(terms_concat)

        logger.debug(f"saving to {outfname}")
        author_term_tfidf.to_parquet(outfname)

    def limit_one_term_per_paper(self, df_ner: pd.DataFrame):
        logger.debug("only keeping one occurrence of each term in each paper")
        df_ner = df_ner.drop_duplicates(subset=["PaperId", "term_id"])
        logger.debug(f"dataframe shape: {df_ner.shape}")
        return df_ner

    def strategy_02(self, df_ner: pd.DataFrame, label: str):
        logger.debug("---strategy 02---")
        outfname = self.outdir.joinpath(f"author_term_tfidf_{label}_strategy02.parquet")
        if outfname.exists():
            logger.info(f"{outfname} already exists. skipping.")
            return

        logger.debug(f"starting with paper-term data with shape {df_ner.shape}")

        df_ner = self.limit_one_term_per_paper(df_ner)

        logger.debug("merging paper-term data with paper-author data")
        df_author_term = df_ner.merge(self.df_paper_author, on="PaperId", how="inner")
        logger.debug(f"dataframe shape: {df_author_term.shape}")

        terms_concat = aggregate_author_term_data(df_author_term)
        author_term_tfidf = get_author_term_tfidf(terms_concat)

        logger.debug(f"saving to {outfname}")
        author_term_tfidf.to_parquet(outfname)

    def strategy_03(self, df_ner: pd.DataFrame, label: str):
        logger.debug("---strategy 03---")
        outfname = self.outdir.joinpath(f"author_term_tfidf_{label}_strategy03.parquet")
        if outfname.exists():
            logger.info(f"{outfname} already exists. skipping.")
            return

        logger.debug(f"starting with paper-term data with shape {df_ner.shape}")

        df_ner = self.limit_one_term_per_paper(df_ner)

        cutoff = 2
        logger.debug(f"only keeping terms that occur in at least {cutoff} papers")
        df_ner = apply_cutoff(df_ner, cutoff)
        logger.debug(f"dataframe shape: {df_ner.shape}")

        logger.debug("merging paper-term data with paper-author data")
        df_author_term = df_ner.merge(self.df_paper_author, on="PaperId", how="inner")
        logger.debug(f"dataframe shape: {df_author_term.shape}")

        terms_concat = aggregate_author_term_data(df_author_term)
        author_term_tfidf = get_author_term_tfidf(terms_concat)

        logger.debug(f"saving to {outfname}")
        author_term_tfidf.to_parquet(outfname)

    def strategy_04(self, df_ner: pd.DataFrame, label: str):
        logger.debug("---strategy 04---")
        outfname = self.outdir.joinpath(f"author_term_tfidf_{label}_strategy04.parquet")
        if outfname.exists():
            logger.info(f"{outfname} already exists. skipping.")
            return

        logger.debug(f"starting with paper-term data with shape {df_ner.shape}")

        df_ner = self.limit_one_term_per_paper(df_ner)

        mapping_name = "endswith_generic"
        logger.debug(f"applying mapping: {mapping_name}")
        mapping = self.id_mappings[mapping_name]
        df_ner["term_id"] = df_ner["term_id"].map(mapping)
        logger.debug(f"dataframe shape: {df_ner.shape}")

        logger.debug("merging paper-term data with paper-author data")
        df_author_term = df_ner.merge(self.df_paper_author, on="PaperId", how="inner")
        logger.debug(f"dataframe shape: {df_author_term.shape}")

        terms_concat = aggregate_author_term_data(df_author_term)
        author_term_tfidf = get_author_term_tfidf(terms_concat)

        logger.debug(f"saving to {outfname}")
        author_term_tfidf.to_parquet(outfname)

    def strategy_05(self, df_ner: pd.DataFrame, label: str):
        logger.debug("---strategy 05---")
        outfname = self.outdir.joinpath(f"author_term_tfidf_{label}_strategy05.parquet")
        if outfname.exists():
            logger.info(f"{outfname} already exists. skipping.")
            return

        logger.debug(f"starting with paper-term data with shape {df_ner.shape}")

        df_ner = self.limit_one_term_per_paper(df_ner)

        cutoff = 2
        logger.debug(f"only keeping terms that occur in at least {cutoff} papers")
        df_ner = apply_cutoff(df_ner, cutoff)
        logger.debug(f"dataframe shape: {df_ner.shape}")

        mapping_name = "endswith_generic"
        logger.debug(f"applying mapping: {mapping_name}")
        mapping = self.id_mappings[mapping_name]
        df_ner["term_id"] = df_ner["term_id"].map(mapping)
        logger.debug(f"dataframe shape: {df_ner.shape}")

        logger.debug("merging paper-term data with paper-author data")
        df_author_term = df_ner.merge(self.df_paper_author, on="PaperId", how="inner")
        logger.debug(f"dataframe shape: {df_author_term.shape}")

        terms_concat = aggregate_author_term_data(df_author_term)
        author_term_tfidf = get_author_term_tfidf(terms_concat)

        logger.debug(f"saving to {outfname}")
        author_term_tfidf.to_parquet(outfname)

    def run_one_label(self, label: str):
        logger.debug(f"Getting dataframe of dygie terms with label == {label}")
        df_ner = self.df_paper_term[self.df_paper_term.label == label]
        logger.debug(f"dataframe shape: {df_ner.shape}")

        softmax_threshold = 0.9
        logger.debug(f"only keeping terms with softmax score >{softmax_threshold}")
        df_ner = df_ner[df_ner.softmax_score > softmax_threshold]
        logger.debug(f"dataframe shape: {df_ner.shape}")

        self.strategy_01(df_ner, label)
        self.strategy_02(df_ner, label)
        self.strategy_03(df_ner, label)
        self.strategy_04(df_ner, label)
        self.strategy_05(df_ner, label)

    def run(self):
        self.id_mappings["endswith_generic"] = self.get_mapping_generic_truncate(
            generic_words=GENERIC_WORDS
        )
        labels = ["Method", "Task"]
        for label in labels:
            self.run_one_label(label)


def main(args):
    # load data
    logger.debug("loading slim DataHelper")
    data_helper = load_slim_data_helper(min_year=args.min_year, max_year=args.max_year)

    fname = DATADIR_TFIDF.joinpath("term_normalized_ids.tsv.gz")
    logger.debug(f"loading term ids from {fname}")
    term_normalized_ids = pd.read_csv(
        fname,
        sep="\t",
        index_col=0,
        squeeze=True,
    )
    term_normalized_ids.index.name = "term_id"
    logger.debug(f"term_normalized_ids.shape={term_normalized_ids.shape}")
    logger.debug("dropping na")
    term_normalized_ids.dropna(inplace=True)
    logger.debug(f"term_normalized_ids.shape={term_normalized_ids.shape}")

    fname = DATADIR_TFIDF.joinpath("paper_to_term_normalized_id.parquet")
    logger.debug(f"loading df_paper_term from {fname}")
    df_paper_term = pd.read_parquet(fname)
    logger.debug(f"dataframe shape: {df_paper_term.shape}")

    tfidf_experiments = AuthorTfidfExperiments(
        data_helper, term_normalized_ids, df_paper_term, args.outdir
    )
    tfidf_experiments.run()


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
    parser.add_argument("outdir", help="path to output directory")
    parser.add_argument(
        "--min-year", type=int, help="minimum year for papers (inclusive)"
    )
    parser.add_argument(
        "--max-year", type=int, help="maximum year for papers (exclusive)"
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
