# -*- coding: utf-8 -*-

DESCRIPTION = """deal with abbreviations extractions data"""

import sys, os, time, json
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


def get_abbreviations_dataframe(datadir):
    abbrevs = []
    for fpath in datadir.glob("abbreviations*.jsonl"):
        with fpath.open() as f:
            for line in f:
                abbrevs.append(json.loads(line))
    data = []

    for paper in abbrevs:
        s2_id = paper["s2_id"]
        for abbrv in paper["abbrv"]:
            for k, v in abbrv.items():
                data.append({"s2_id": s2_id, "abbrv": k, "long_form": v})
    df_abbrvs = pd.DataFrame(data)
    df_abbrvs = (
        df_abbrvs.groupby(df_abbrvs.columns.tolist()).size().reset_index(name="freq")
    )
    return df_abbrvs