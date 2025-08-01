# -*- coding: utf-8 -*-

DESCRIPTION = """store data as sparse matrices"""

import sys, os, time
from typing import Optional, Union, List, Dict
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
from scipy.sparse import coo_matrix, csr_matrix, save_npz


def map_label_to_idx(labels: np.ndarray):
    return {val: idx[0] for idx, val in np.ndenumerate(labels)}


class BridgerMatrix:
    """Keeps track of a matrix and the labels for rows and columns"""

    def __init__(
        self,
        mat: Optional[csr_matrix] = None,
        row_labels: Optional[np.ndarray] = None,
        col_labels: Optional[np.ndarray] = None,
        description: Optional[str] = None,
    ):
        self.mat = mat
        self.row_labels = row_labels
        self.row_label_to_row_idx = None
        self.col_labels = col_labels
        self.col_label_to_col_idx = None
        self.description = description

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        colname_for_rowdata: str,
        colname_for_coldata: str,
        colname_for_vals: Optional[str] = None,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        **kwargs
    ):
        obj = cls(**kwargs)
        if obj.row_labels is None:
            obj.row_labels = (
                df[colname_for_rowdata]
                .sort_values()
                .drop_duplicates()
                .reset_index()[colname_for_rowdata]
            )
        obj.row_label_to_row_idx = map_label_to_idx(obj.row_labels)
        if obj.col_labels is None:
            obj.col_labels = (
                df[colname_for_coldata]
                .sort_values()
                .drop_duplicates()
                .reset_index()[colname_for_coldata]
            )
        obj.col_label_to_col_idx = map_label_to_idx(obj.col_labels)

        row_data = df[colname_for_rowdata].map(obj.row_label_to_row_idx)
        col_data = df[colname_for_coldata].map(obj.col_label_to_col_idx)
        if colname_for_vals is None:
            vals = None
        else:
            vals = df[colname_for_vals].values
        if nrows is None:
            nrows = len(obj.row_labels)
        if ncols is None:
            ncols = len(obj.col_labels)
        obj.load_matrix(row_data, col_data, nrows, ncols, vals)
        return obj

    @classmethod
    def from_matrix(
        cls,
        mat: csr_matrix,
        row_labels: np.ndarray,
        col_labels: np.ndarray,
        row_label_to_row_idx: Optional[Dict] = None,
        col_label_to_col_idx: Optional[Dict] = None,
        description=None,
    ):
        obj = cls(description=description)
        obj.mat = mat
        obj.row_labels = row_labels
        obj.col_labels = col_labels

        obj.row_label_to_row_idx = row_label_to_row_idx
        if obj.row_label_to_row_idx is None:
            obj.row_label_to_row_idx = map_label_to_idx(obj.row_labels)

        obj.col_label_to_col_idx = col_label_to_col_idx
        if obj.col_label_to_col_idx is None:
            obj.col_label_to_col_idx = map_label_to_idx(obj.col_labels)

        return obj

    def load_matrix(self, row_data, col_data, nrows, ncols, vals=None):
        if vals is None:
            vals = np.ones_like(
                row_data
            )  # this could be row_data or col_data. both should be the same length
        self.mat = coo_matrix((vals, (row_data, col_data)), shape=(nrows, ncols))
        self.mat = self.mat.tocsr()

    def reindex(self, new_labels, axis=0):
        if axis not in [0, 1]:
            raise ValueError("axis must be either 0 (rows) or 1 (columns)")
        mat = self.mat.tocoo()
        row_data = mat.row
        col_data = mat.col
        vals = mat.data

        logger.debug("df_new_labels")
        df_new_labels = pd.Series(new_labels).reset_index(name="lab")
        if axis == 0:
            logger.debug("df_old_labels")
            df_old_labels = pd.Series(self.row_labels).reset_index(name="lab")
            # new_row_labels = pd.Series(new_labels).reset_index(name='lab')
            logger.debug("merging")
            old_to_new = df_old_labels.merge(
                df_new_labels, on="lab", how="inner"
            ).set_index("index_x")["index_y"]
            self.row_labels = new_labels
            self.row_label_to_row_idx = map_label_to_idx(self.row_labels)
            logger.debug("mapping data")
            # row_data = [old_to_new[x] for x in row_data]
            row_data = np.vectorize(old_to_new.get)(row_data)
        elif axis == 1:
            # old_col_labels = self.col_labels.reset_index(name='lab')
            logger.debug("df_old_labels")
            df_old_labels = pd.Series(self.col_labels).reset_index(name="lab")
            # new_col_labels = pd.Series(new_labels).reset_index(name='lab')
            logger.debug("merging")
            old_to_new = df_old_labels.merge(
                df_new_labels, on="lab", how="inner"
            ).set_index("index_x")["index_y"]
            self.col_labels = new_labels
            self.col_label_to_col_idx = map_label_to_idx(self.col_labels)
            logger.debug("mapping data")
            # col_data = [old_to_new[x] for x in col_data]
            col_data = np.vectorize(old_to_new.get)(col_data)
        nrows = len(self.row_labels)
        ncols = len(self.col_labels)
        logger.debug("loading matrix")
        self.load_matrix(row_data, col_data, nrows, ncols, vals)


def main(args):
    pass


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
