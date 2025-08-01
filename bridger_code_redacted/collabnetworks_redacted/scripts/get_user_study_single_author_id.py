# -*- coding: utf-8 -*-

DESCRIPTION = """Get the user study data for a single author ID"""

from typing import MutableSet, Sequence
from collabnetworks.util import drop_duplicate_titles
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

from collabnetworks import DataHelper, AuthorHelper
from collabnetworks.user_study import UserStudyData
from collabnetworks.clustering import load_graph_and_partitions

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DATADIR = Path(os.environ["DATADIR"])
path_to_coauthor_data = DATADIR.joinpath(
    "computer_science_papers_20201002/coauthor_2015-2021_minpubs3_collabweighted"
)
fname_edgelist = path_to_coauthor_data.joinpath("coauthor_edgelist.csv")
dirname_local_partition = path_to_coauthor_data.joinpath("components_local/")


def get_authors_with_not_enough_terms(
    data_helper: DataHelper,
    cutoff: int = 10,
    labels: Sequence[str] = ["Task", "Method", "Material"],
) -> MutableSet:
    df_ner = data_helper.df_ner
    dedup = drop_duplicate_titles(data_helper.mag_data.papers)
    df_ner = df_ner[df_ner["PaperId"].isin(dedup["PaperId"])]
    paa = data_helper.mag_data.paper_authors
    paa.drop_duplicates(subset=["PaperId", "AuthorId"], inplace=True)
    df_ner.drop_duplicates(subset=["PaperId", "term_id"], inplace=True)
    df_ner = df_ner[["PaperId", "term_id", "label"]]
    author_terms = df_ner.merge(paa[["PaperId", "AuthorId"]], how="inner", on="PaperId")
    author_term_counts = (
        author_terms.groupby(["AuthorId", "label"]).size().reset_index(name="num_terms")
    )
    not_enough_terms = []
    for label in labels:
        x = author_term_counts[author_term_counts.label == label]
        x = x[x["num_terms"] < cutoff]
        not_enough_terms.extend(x["AuthorId"].tolist())
    return set(not_enough_terms)


def main(args):
    weighted = not args.unweighted
    outdir = Path(args.outdir)
    # assert not outdir.exists()
    data_helper = DataHelper.from_defaults(
        min_year=args.min_year, max_year=args.max_year
    )

    # if args.personas is True:
    #     logger.debug("loading data for personas")
    #     min_weight = 0.02
    #     _, ego_partition = load_graph_and_partitions(
    #         fname_edgelist, dirname_local_partition, min_weight
    #     )

    min_terms_per_label = 10
    logger.debug(
        f"excluding authors that have fewer than {min_terms_per_label} terms (for each: tasks/methods/materials)"
    )
    exclude_authors = get_authors_with_not_enough_terms(
        data_helper, cutoff=min_terms_per_label
    )

    logger.debug(f"getting AuthorHelper for author_id: {args.author_id}")
    author = AuthorHelper(
        args.author_id, data=data_helper, min_year=args.min_year, max_year=args.max_year
    )
    # exclude focal author when getting similar authors
    exclude_authors.add(author.id)

    if args.term_rank_compare is True:
        from collabnetworks.user_study.user_study_term_rank_compare import (
            UserStudyDataTermRankCompare,
        )

        _class = UserStudyDataTermRankCompare
    else:
        _class = UserStudyData

    u = _class(
        author,
        data=data_helper,
        outdir=outdir,
        exclude_authors=exclude_authors,
        min_year=args.min_year,
        max_year=args.max_year,
        min_papers=6,
        weighted=weighted,
    )
    u.load_coauthor_graph()
    # u.paper_collection.coauthor_graph = u.coauthor_graph

    # exclude focal author's coauthors when getting similar authors
    coauthors = [a.AuthorId for a in author.coauthors]
    exclude_authors.update(coauthors)

    u.exclude_authors = exclude_authors
    u.load_embeddings()
    u.load_specter_embeddings()
    u.load_tfidf()
    u.load_tfidf_vectorizers()
    u.get_focal_embeddings()
    u.get_all_distances()
    u.get_sim_authors(N=args.num_similar)
    u.save_cards()
    u.save_sim_author_ids()

    if args.personas is True:
        logger.debug("getting data for this author's personas")
        i = 0
        for u_sub in u.yield_specter_cluster_personas():
            try:
                if i < 3:
                    logger.debug(
                        f"getting data for persona: {u_sub.paper_collection.name} ({len(u_sub.paper_collection.paper_ids)} papers)"
                    )
                    u_sub.get_focal_embeddings()
                    u_sub.get_all_distances()
                    u_sub.get_sim_authors(N=args.num_similar)
                    u_sub.save_cards()
                    u_sub.save_sim_author_ids()
                else:
                    logger.debug(
                        f"getting single card for persona: {u_sub.paper_collection.name} ({len(u_sub.paper_collection.paper_ids)} papers)"
                    )
                    u_sub.save_focal_card()

                i += 1
            except ZeroDivisionError:
                logger.warning(
                    f"ZeroDivisionError encountered when processing persona: {u_sub.paper_collection.name}. skipping"
                )
            except ValueError:
                logger.warning(
                    f"ValueError encountered when processing persona: {u_sub.paper_collection.name}. skipping"
                )


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
    parser.add_argument("author_id", type=int, help="MAG Author ID")
    parser.add_argument("outdir", help="output directory (will be created)")
    parser.add_argument(
        "-n",
        "--num-similar",
        type=int,
        default=10,
        help="number of similar authors to collect for each category",
    )
    parser.add_argument(
        "--unweighted", action="store_true", help="use unweighted embeddings"
    )
    parser.add_argument(
        "--min-year", type=int, default=2015, help="minimum year to consider for papers"
    )
    parser.add_argument(
        "--max-year", type=int, default=2022, help="maximum year to consider for papers"
    )
    parser.add_argument(
        "--personas", action="store_true", help="also get data for author personas"
    )
    parser.add_argument(
        "--term-rank-compare", action="store_true", help="get term rank compare data"
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
