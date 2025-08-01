# -*- coding: utf-8 -*-

DESCRIPTION = """utils"""

import sys, os, time, json, tarfile
from pathlib import Path
from typing import Optional, Iterable, Union, List, Collection, Dict, Any
import subprocess, shlex
from itertools import combinations
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

# from mag_data import MagData

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from .config import Config

# handle loading nested dataclasses using this function
from dacite import from_dict as dataclass_from_dict
from dacite.config import Config as ConfigDataclassFromDict

config_dataclass_from_dict = ConfigDataclassFromDict(cast=[int])


def get_timestamp() -> str:
    """Get the current date and time

    Returns:
        str: in the form '%Y%m%dT%H%M%S' (e.g.: "20201031T000001")
    """
    return datetime.strftime(datetime.now(), "%Y%m%dT%H%M%S")


# def save_to_skiff_files(
#         f: Union[str, Path],
#         recurse: bool = True,
#         app_name: str = 'co-mention-viz',
#         subfolder: Union[None, str, Path] = None
#         ) -> None:
#     remote_fpath = Path(f'/skiff_files/apps/{app_name}')
#     if subfolder:
#         remote_fpath = remote_fpath.joinpath(subfolder)
#     cmd = f"gcloud compute scp {'--recurse' if recurse is True else ''} {f} skiff-files-writer:{remote_fpath}"
#     subprocess_args = shlex.split(cmd)
#     # with subprocess.Popen(subprocess_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
#     #     logger.debug(proc.stdout.read())
#     return subprocess.run(subprocess_args, capture_output=True)


def get_root_dir():
    # return os.path.dirname(os.path.abspath(__file__))
    from . import PACKAGE_ROOT

    return PACKAGE_ROOT.parent


def map_label_to_idx(labels: np.ndarray):
    return {val: idx[0] for idx, val in np.ndenumerate(labels)}


def id_to_list(x: Any):
    """Coerces input to list if it is a single number or string"""
    # check if x is a collection (e.g., list) of ids, or if it is a single id
    if isinstance(x, str) or not isinstance(x, Collection):
        x = [x]
    return x


def tar_visdata() -> None:
    dirpath = Path(get_root_dir()).joinpath("data/visgraph_data")
    outfname = dirpath.parent.joinpath("visgraph_data.tar.gz")
    with tarfile.open(outfname, "w:gz") as tar:
        tar.add(dirpath, arcname=dirpath.name)


def upload_to_google(
    file_or_str,
    client: "google.cloud.storage.Client",
    bucket_name: str = "REDACTED",
    subdir: str = "authors/visgraphdata/",
    sub_subdir: Optional[str] = None,
    method: str = "filename",
    target_fname: Optional[str] = None,
) -> None:
    """Upload to a google bucket. will use the method named 'upload_from_<method>'.
    So method should be 'filename' (upload_from_filename()) or 'string' (upload_from_string()) or 'file' (upload_from_file())
    """
    method = method.lower()
    if method == "str":
        method = "string"
    bucket = client.bucket(bucket_name)
    if target_fname is None:
        try:
            target_fname = Path(file_or_str).name
            if len(target_fname) > 1000:
                # this is too long and something went wrong
                raise ValueError("must supply a 'target_fname'")
        except TypeError:
            raise ValueError("must supply a 'target_fname'")
    subdir = Path(subdir)
    if sub_subdir is not None:
        subdir = subdir.joinpath(sub_subdir)
    target_fname = str(Path(subdir).joinpath(target_fname))
    blob = bucket.blob(target_fname)
    logger.debug("uploading file to blob: {}".format(blob.name))
    f = getattr(blob, "upload_from_" + method)
    f(file_or_str)


def google_get_directory(
    client: "google.cloud.storage.Client",
    bucket_name: str = "REDACTED",
    base_subdir: str = "authors/visgraphdata/",
    sub_subdir: Optional[str] = None,
    ext: Optional[str] = None,
):
    """save a 'directory.txt' file with a list of the files within the directory"""
    bucket = client.bucket(bucket_name)
    subdir = Path(base_subdir).joinpath(sub_subdir)
    directory = []
    for b in bucket.list_blobs(prefix=str(subdir)):
        p = Path(b.name)
        # we don't want subdirectories
        if p.parent != subdir:
            continue
        if ext is not None and p.suffix != ext:
            continue
        fname = p.name
        if fname == "directory.txt":
            continue
        directory.append(fname)
    return directory


def google_save_directory(
    client: "google.cloud.storage.Client",
    bucket_name: str = "comention",
    base_subdir: str = "authors/visgraphdata/",
    sub_subdir: Optional[str] = None,
    ext: Optional[str] = None,
):
    """save a 'directory.txt' file with a list of the files within the directory"""
    subdir = Path(base_subdir).joinpath(sub_subdir)
    directory = google_get_directory(client, bucket_name, base_subdir, sub_subdir, ext)
    upload_to_google(
        "\n".join(directory),
        client,
        subdir=str(subdir),
        method="string",
        target_fname="directory.txt",
    )


def filter_and_get_df(df, filter_vals, filter_col="PaperId"):
    """
    Given either a pandas dataframe or a spark dataframe, filter a certain column on a list of values, and return a pandas dataframe
    """
    if hasattr(filter_vals, "values"):
        filter_vals = filter_vals.values
    elif hasattr(filter_vals, "tolist"):
        filter_vals = filter_vals.tolist()

    if hasattr(df, "toPandas"):
        # df is a spark dataframe
        return df.filter(df[filter_col].isin(filter_vals)).toPandas()
    else:
        # assume df is a pandas dataframe
        return df[df[filter_col].isin(filter_vals)]


def get_parquet_filepath(dirname: str, tablename: str):
    """Get the file path for a parquet file/directory within dirname

    :returns: Path

    """
    dirname = Path(dirname)
    # tablename = Path(tablename)
    g = list(dirname.glob("{}_*parquet*".format(tablename)))
    if len(g) == 1:
        return g[0]
    g = list(dirname.glob("*_{}_*parquet*".format(tablename)))
    if len(g) == 1:
        return g[0]
    else:
        raise ValueError(
            "Could not find parquet file path for tablename: {}".format(tablename)
        )


def tfidf(tf: float, df: float, n: int) -> float:
    """Return TF-IDF score for a given term

    :tf: term count (or weighted term count) within document or cluster
    :df: term count in entire corpus
    :n: number of documents in corpus
    :returns: TF-IDF score

    """
    if not tf:
        return 0
    tf = np.log(tf + 1)
    idf = np.log(n / df)
    return tf * idf


def _tfidf_apply(row, N, tf_colname="term_count", df_colname="all_count"):
    """helper function to apply tfidf row-wise to a pandas dataframe
    :N: number of papers in corpus

    usage:
    # calculate N before using
    df.apply(_tfidf_apply, N=N, axis=1)
    """
    return tfidf(row[tf_colname], row[df_colname], N)


def JSD(P, Q):
    """
    Jensen-Shannon divergence

    from:
    https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    """
    from scipy.stats import entropy
    from numpy.linalg import norm

    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def calc_js_div(s1: pd.Series, s2: pd.Series) -> float:
    """Given two distributions, calculate the Jensen-Shannon divergence

    :s1, s2: distributions represented as pandas Series with index being IDs and values being frequencies
    :returns: Jensen-Shannon divergence (float)

    """
    total_set = set(s1.index.tolist() + s2.index.tolist())
    P = s1.reindex(total_set, fill_value=0)
    Q = s2.reindex(total_set, fill_value=0)
    return JSD(P, Q)


def curr_mem_usage() -> str:
    """Get the current memory usage of the running process
    Use inside a jupyter notebook to learn the current memory usage of the notebook
    :returns: human readable string

    """
    import psutil
    from humanfriendly import format_size

    proc = psutil.Process()
    return format_size(proc.memory_info()[0])


def get_score_column(df: pd.DataFrame, df_papers: pd.DataFrame) -> np.ndarray:
    """Assign a score to each paper for each author based on paper metadata.
    Score is based on author position (first or last author, or middle author)
    and the rank of the paper (paper importance).

    Example usage:
    scores = get_score_columns(df_paper_authors, df_papers)
    df_paper_authors["score"] = scores

    Args:
        df (pd.DataFrame): PaperAuthor dataframe, with one row per paper per author. Should have columns "PaperId", "AuthorId", "AuthorSequenceNumber"
        df_papers (pd.DataFrame): Papers dataframe, with columns "PaperId", "rank_scaled"

    Returns:
        np.ndarray: array of scores corresponding to the rows in `df`
    """
    from sklearn.preprocessing import MinMaxScaler

    if "num_authors" not in df.columns:
        df["num_authors"] = df.groupby("PaperId")["AuthorSequenceNumber"].transform(
            "max"
        )
    if "is_last_author" not in df.columns:
        df["is_last_author"] = np.where(
            df["num_authors"] == df["AuthorSequenceNumber"], True, False
        )

    # df = df[["PaperId", "AuthorId", "AuthorSequenceNumber", "is_last_author"]]

    multiplier_first_or_last_author = 1.0
    cond1 = df["AuthorSequenceNumber"] == 1
    multiplier_middle_author = 0.75
    cond2 = df["is_last_author"] == True
    multiplier = np.where(
        cond1 | cond2, multiplier_first_or_last_author, multiplier_middle_author
    )
    df["rank_scaled"] = df.PaperId.map(df_papers.set_index("PaperId")["rank_scaled"])
    rank_scaled = (
        MinMaxScaler(feature_range=(0.5, 1))
        .fit_transform(df[["rank_scaled"]])
        .flatten()
    )
    # rank_scaled = 1 - rank_scaled
    multiplier = multiplier * rank_scaled
    return multiplier


def drop_duplicate_titles(df_papers: pd.DataFrame) -> pd.DataFrame:
    # remove duplicate papers by titles
    # first sort by DocType to prefer Journal and Conference papers
    sorter = [
        "Journal",
        "Conference",
        "Book",
        "BookChapter",
        "Repository",
        "Dataset",
    ]
    df_papers["DocType"] = df_papers["DocType"].astype("category")
    df_papers["DocType"].cat.set_categories(sorter, inplace=True)
    df_papers["PaperTitleNoSpace"] = df_papers["PaperTitle"].str.replace(" ", "")
    return (
        df_papers.sort_values("DocType")
        .drop_duplicates(subset=["PaperTitleNoSpace"])
        .drop(columns=["PaperTitleNoSpace"])
    )


def sort_distance_df(
    c: str,
    df_distances: pd.DataFrame,
) -> pd.DataFrame:
    """Given c: a condition in ['simTask', 'simMethod', 'simspecter', 'simTask_distMethod', 'simMethod_distTask']
    sort the df_distances dataframe appropriately for the condition, and return the sorted dataframe.

    Args:
        c (str): condition
        df_distances pd.DataFrame: distances dataframe

    Returns:
        pd.DataFrame: sorted distances dataframe
    """
    if c == "simTask_distMethod":
        _df = (
            df_distances.sort_values("Task_dist")
            .head(1000)
            .sort_values("Method_dist", ascending=False)
        )
    elif c == "simMethod_distTask":
        _df = (
            df_distances.sort_values("Method_dist")
            .head(1000)
            .sort_values("Task_dist", ascending=False)
        )
    else:
        k = c.replace("sim", "")  # will be "Task", "Method", or "specter"
        _df = df_distances.sort_values(f"{k}_dist")
    return _df

def overlap_ratio(a: Iterable, b: Iterable) -> float:
    a = set(a)
    b = set(b)
    if len(a) == 0 or len(b) == 0:
        return 0
    intrsct = set.intersection(a, b)
    # ratio = len(intrsct) / len(b)
    # trying jaccard similarity instead
    union = set.union(a, b)
    ratio = len(intrsct) / len(union)
    return ratio
