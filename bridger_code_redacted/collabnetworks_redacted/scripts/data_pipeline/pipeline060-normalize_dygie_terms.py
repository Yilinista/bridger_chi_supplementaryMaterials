# -*- coding: utf-8 -*-

DESCRIPTION = (
    """Use spacy to lemmatize and normalize dygie terms (tasks, methods, etc.)"""
)

import sys, os, time, re
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
from string import punctuation, ascii_uppercase
from typing import Mapping, Optional, Tuple

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
import spacy
import scispacy

from collabnetworks.util import get_timestamp


def any_tok_mostly_upper(x: str, thresh: float = 0.75) -> bool:
    toks = x.split()
    for tok in toks:
        num_upper = 0
        for char in tok:
            if char in ascii_uppercase:
                num_upper += 1
        if num_upper / len(tok) >= thresh:
            return True
    return False


# def expand_abbreviations(terms: pd.Series, abbreviations: pd.Series) -> pd.Series:
#     """expand any abbreviations found in terms

#     Args:
#         terms (pd.Series): series of terms, with index of s2_ids
#         abbreviations (pd.Series): series of long-form expansions, with multi-index of (s2_id, abbreviation)

#     Returns:
#         pd.Series: series of terms with abbreviations expanded, with index of s2_ids
#     """
#     new_terms = []
#     for s2_id, term in terms.iteritems():
#         try:
#             expanded = term
#             for abbrv, long_form in abbreviations.loc[s2_id].iteritems():
#                 expanded = re.sub(r"{}".format(abbrv), long_form, expanded)
#             new_terms.append(expanded)
#         except KeyError:
#             new_terms.append(term)
#     return pd.Series(new_terms, index=terms.index)


def expand_abbreviations_single_term(
    term: str, abbrv_to_long_form: Mapping[str, str]
) -> str:
    ret = term
    try:
        for abbrv, long_form in abbrv_to_long_form.items():
            if abbrv in term:
                pattern = r"\b{}\b".format(re.escape(abbrv))
                ret = re.sub(pattern, long_form.replace("\\", r"\\"), ret)
    except:
        logger.error(f"error found for term: {ret}")
        pass
    return ret


def _apply_expansion_byterm(gdf: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe grouped by paper ID, with fields 'term', 'abbrv', and 'long_form'
    Usage: df_expanded = df.groupby('s2_id').apply(_apply_expansion_byterm).reset_index(level=1, drop=True).reset_index()
    This will return a DataFrame with fields 's2_id', 'term', and 'expanded'

    Args:
        gdf (pd.DataFrame)

    Returns:
        pd.DataFrame:
    """
    ret = []
    s2_id = gdf.iloc[0]["s2_id"]
    for term, group in gdf.groupby("term"):
        abbrv_to_long_form = (
            group[["abbrv", "long_form"]]
            .drop_duplicates()
            .set_index("abbrv")["long_form"]
            .to_dict()
        )
        ret.append(
            {
                "s2_id": s2_id,
                "term": term,
                "expanded": expand_abbreviations_single_term(term, abbrv_to_long_form),
            }
        )
    return pd.DataFrame(ret)


def expand_abbreviations_spark(
    df_terms: pd.DataFrame, df_abbreviations: pd.DataFrame, spark
) -> pd.Series:
    sdf_terms = spark.createDataFrame(df_terms)
    sdf_abbreviations = spark.createDataFrame(df_abbreviations)
    sdf_merged = sdf_abbreviations.join(sdf_terms, how="inner", on="s2_id")
    df_expanded = (
        sdf_merged.groupby("s2_id")
        .applyInPandas(
            _apply_expansion_byterm, schema="s2_id int, term string, expanded string"
        )
        .toPandas()
    )
    df_expanded = df_terms.merge(df_expanded, how="left", on=["s2_id", "term"])
    df_expanded["expanded"] = df_expanded["expanded"].fillna(df_expanded["term"])
    expanded_terms = df_expanded["expanded"]
    return expanded_terms


def expand_abbreviations(
    df_terms: pd.DataFrame, df_abbreviations: pd.DataFrame, use_spark: bool = False
) -> pd.Series:
    if use_spark is True:
        logger.debug("using spark to expand abbreviations...")
        from collabnetworks.config import Config

        config = Config()
        spark = config.spark
        return expand_abbreviations_spark(df_terms, df_abbreviations, spark)
    merged = df_abbreviations.merge(df_terms, how="inner", on="s2_id")
    df_expanded = (
        merged.groupby("s2_id")
        .apply(_apply_expansion_byterm)
        .reset_index(level=1, drop=True)
        .reset_index()
    )
    df_expanded = df_terms.merge(df_expanded, how="left", on=["s2_id", "term"])
    df_expanded["expanded"] = df_expanded["expanded"].fillna(df_expanded["term"])
    expanded_terms = df_expanded["expanded"]
    return expanded_terms


def _normalize_term(doc: spacy.tokens.Doc) -> str:
    t = " ".join([tok.lemma_ for tok in doc if not tok.is_punct])
    return t.lower()


def clean_terms(
    df_terms: pd.DataFrame, abbreviations: Optional[pd.DataFrame] = None
) -> pd.Series:
    terms_cleaned = df_terms.term
    logger.debug("removing parentheticals and unclosed parentheticals")
    terms_cleaned = terms_cleaned.str.replace(
        r"\s\(.*?\)", "", regex=True
    )  # parentheticals
    terms_cleaned = terms_cleaned.str.replace(
        r"\s\(.*", "", regex=True
    )  # unclosed parentheticals
    logger.debug(f"terms_cleaned shape: {terms_cleaned.shape}")
    if abbreviations is not None:
        logger.debug("expanding abbreviations...")
        if len(terms_cleaned) > 10000000:
            use_spark = True
        else:
            use_spark = False
        terms_cleaned = expand_abbreviations(
            pd.concat([terms_cleaned, df_terms["s2_id"]], axis=1),
            abbreviations,
            use_spark=use_spark,
        )
        # df_expanded = expand_abbreviations(pd.concat([terms_cleaned, df_terms['s2_id']], axis=1), abbreviations)
        # df_expanded = df_expanded[['s2_id', 'expanded']].drop_duplicates().rename(columns={'expanded': 'term'})
        # terms_cleaned = df_expanded.set_index('s2_id')['term']
        # logger.debug(f"terms_cleaned shape: {terms_cleaned.shape}")
    # logger.debug("removing punctuation")
    # # terms_cleaned = terms_cleaned.apply(lambda s: re.sub(r"[^\w\s]", " ", s))
    # terms_cleaned = terms_cleaned.str.replace(
    #     r"[{}]".format(punctuation), " ", regex=True
    # )
    logger.debug("removing multiple consecutive spaces")
    terms_cleaned = terms_cleaned.apply(lambda s: re.sub(r" +", " ", s))
    terms_cleaned = terms_cleaned.str.strip()
    logger.debug("lower-casing")
    terms_cleaned = terms_cleaned.str.lower()
    return terms_cleaned


# def _map_display_term(term_normalized, term_map):
#     try:
#         return term_map[term_normalized]
#     except KeyError:
#         return term_normalized


def get_term_map(df_norm: pd.DataFrame) -> pd.Series:
    """Given a dataframe that has 'term_normalized' and 'term_cleaned' data,
    map any normalized term that appears more than once to its most common 'term_cleaned'.

    The idea here is that sometimes, the normalized (lemmatized) term isn't the best representation,
    such as 'datum' instead of 'data'. So it is better to take the most frequently occurring
    representation of a normalized term.

    Args:
        df_norm (pd.Dataframe): dataframe with 'term_normalized' and 'term_cleaned' data

    Returns:
        pd.Series: series with index 'term_normalized' and values being most common 'term_cleaned'
    """
    # _df = df_ner.drop(columns=['term']).merge(df_norm, how='inner', on='term_cleaned')
    # x = _df.groupby(['term_normalized', 'term_cleaned', 'term']).size()
    x = df_norm.groupby(["term_normalized", "term_cleaned", "term"]).size()
    xcount = x.groupby(level=0).size()
    xdf = x.reset_index(name="freq")
    xdf["num_terms_mapped_to_norm"] = xdf["term_normalized"].map(xcount)
    xdf = xdf.loc[xdf["num_terms_mapped_to_norm"] > 1, :]
    xdf = xdf.sort_values(
        ["term_normalized", "freq", "term_cleaned"], ascending=[True, False, True]
    )
    xdf = xdf.drop_duplicates(subset=["term_normalized"], keep="first")
    xmap = xdf.set_index("term_normalized")["term_cleaned"]
    return xmap


def load_abbreviations(parquet_fpath) -> pd.DataFrame:
    df = pd.read_parquet(parquet_fpath)
    if "freq" in df.columns:
        df = df.sort_values(["s2_id", "abbrv", "freq"], ascending=[True, True, False])
    else:
        df = df.sort_values(["s2_id", "abbrv"])
    df = df.drop_duplicates(subset=["s2_id", "abbrv"])
    df = df.dropna(subset=["s2_id", "abbrv"])
    df = df[df["abbrv"].apply(any_tok_mostly_upper)]
    df["s2_id"] = df["s2_id"].astype(int)
    # remove any rows with parentheses
    df = df[~(df["long_form"].str.contains(r"[()<>{}]", regex=True))]
    df = df[~(df["abbrv"].str.contains(r"[()<>{}]", regex=True))]
    return df


def main(args):
    nlp = spacy.load(args.spacy_model, disable=["parser", "ner"])
    logger.debug("reading input data: {}".format(args.input))
    df_ner = (
        pd.read_parquet(args.input)
        .dropna(subset=["s2_id", "term"])
        .drop_duplicates(subset=["s2_id", "term"])
    )
    df_ner["s2_id"] = df_ner["s2_id"].astype(int)
    logger.debug("dataframe shape: {}".format(df_ner.shape))
    labels = ["Task", "Method", "Material", "Metric"]
    logger.debug(f"keeping only labels: {labels}")
    df_ner = df_ner[df_ner["label"].isin(labels)].reset_index(drop=True)
    logger.debug("dataframe shape: {}".format(df_ner.shape))
    # raw_terms = df_ner.drop_duplicates(subset=["s2_id", "term"]).set_index("s2_id")[
    #     "term"
    # ]
    logger.debug(f"there are {df_ner.term.nunique()} unique terms")

    if args.abbreviations is not None:
        df_abbreviations = load_abbreviations(args.abbreviations)
    #     raw_terms = expand_abbreviations(raw_terms, df_abbreviations)
    else:
        df_abbreviations = None

    logger.debug("cleaning terms before lemmatizing...")
    # df_ner['term_cleaned'] = df_ner['term'].apply(clean_terms)
    df_ner["term_cleaned"] = clean_terms(df_ner, df_abbreviations)

    if args.debug:
        ckpt_outfp = Path(args.output)
        ckpt_outfp = ckpt_outfp.with_name(
            f"{ckpt_outfp.stem}_CHECKPOINT_{get_timestamp()}{ckpt_outfp.suffix}"
        )
        logger.debug(f"CHECKPOINTING: saving to file: {ckpt_outfp}")
        df_ner.to_parquet(ckpt_outfp)

    num_na = df_ner["term_cleaned"].isna().sum()
    logger.debug(f"term_cleaned column has {num_na} NA values")
    terms = df_ner["term_cleaned"].dropna().unique()
    # df_terms = df_ner.drop_duplicates(subset=['term_cleaned'])
    # terms = df_terms['term_cleaned'].values
    logger.debug("there are {} unique terms".format(len(terms)))
    logger.debug("lemmatizing and normalizing...")
    docs = list(nlp.pipe(terms, n_process=args.processes))
    logger.debug("done. {} spacy docs".format(len(docs)))
    normterms = [_normalize_term(doc) for doc in docs]
    logger.debug("{} normalized terms".format(len(normterms)))
    # df_out = pd.DataFrame({'term_cleaned': terms, 'term_normalized': normterms, 'term': df_terms['term'].values})
    to_join = pd.DataFrame({"term_cleaned": terms, "term_normalized": normterms})
    df_out = df_ner.merge(to_join, how="left", on="term_cleaned")
    logger.debug("getting display_terms")
    display_term_map = get_term_map(df_out)
    # df_out['term_display'] = df_out['term_normalized'].apply(_map_display_term, term_map=display_term_map)
    df_out["term_display"] = df_out["term_normalized"].map(display_term_map)
    # df_out.term_display.fillna(df_out.term, inplace=True)
    df_out.term_display.fillna(df_out.term_cleaned, inplace=True)
    logger.debug(
        "writing dataframe (shape: {}) to file: {}".format(df_out.shape, args.output)
    )
    df_out.to_parquet(args.output)


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
    parser.add_argument("input", help="input NER data (parquet format)")
    parser.add_argument("output", help="output file (parquet)")
    parser.add_argument(
        "--abbreviations",
        help="path to parquet file containing abbreviations and their long-forms. only specify this if you want to expand abbreviations",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_sci_sm",
        help="spacy model to use. default is `en_core_sci_sm` from `scispacy`",
    )
    parser.add_argument(
        "--processes", type=int, default=30, help="number of processes for spacy to use"
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
