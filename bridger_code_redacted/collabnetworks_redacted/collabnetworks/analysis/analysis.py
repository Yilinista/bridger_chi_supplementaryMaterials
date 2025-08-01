# -*- coding: utf-8 -*-

DESCRIPTION = """functions to help with analysis"""

import sys, os, time, json, math
from pathlib import Path
from datetime import datetime
from collections import Counter
from timeit import default_timer as timer
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    Iterable,
)

try:
    from humanfriendly import format_timespan
except ImportError:

    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)


import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

from collections import defaultdict

import pandas as pd
import numpy as np
import networkx as nx

from .. import PACKAGE_ROOT
from ..util import sort_distance_df, overlap_ratio

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DATADIR = Path(os.environ["DATADIR"])

# UI_DATA_BASEDIR = PACKAGE_ROOT.parent.joinpath("data/user_study_data/")
UI_DATA_BASEDIR = DATADIR.joinpath("user_study_data/")
from ..user_study import COAUTHOR_GRAPH_FPATH

LOGS_BASEDIR = DATADIR.joinpath("user_study_data/user_logs")


def filter_highpass(df):
    _df = df
    return _df[_df.num_papers_overall > 5]


def filter_lowhigh(df):
    _df = df
    _df = filter_highpass(_df)
    return _df[_df.authorCitationCount < 2000]


def novelty_wins(
    gdf,
    a_conds=["simTask", "simTask_distMethod"],
    b_conds=["simspecter", "simspecter_hideTerms"],
):
    # this is calculated per user, per feedbackItemType
    checked_ratio = gdf["num_checked"] / gdf["num_observed"]
    a = checked_ratio.loc[gdf.condition.isin(a_conds)].mean()
    b = checked_ratio.loc[gdf.condition.isin(b_conds)].mean()
    if a == b:
        return np.nan
    # return a >= b
    return a > b


def preference_delta(
    gdf,
    a_conds=["simTask", "simTask_distMethod"],
    b_conds=["simspecter", "simspecter_hideTerms"],
):
    # this is calculated per user, per feedbackItemType
    checked_ratio = gdf["num_checked"] / gdf["num_observed"]
    a = checked_ratio.loc[gdf.condition.isin(a_conds)].mean()
    b = checked_ratio.loc[gdf.condition.isin(b_conds)].mean()
    return b - a


not_found_return_value = np.nan


def get_shortest_path_length(G, u, v) -> Union[int, not_found_return_value.__class__]:
    try:
        return nx.shortest_path_length(G, int(u), int(v))
    except nx.NetworkXNoPath:
        return not_found_return_value


def limit_each_term_to_only_one_category(card_or_details):
    running_list_terms = []
    for term in card_or_details["topics"][:10]:
        running_list_terms.append(term["DisplayName"].lower())
    for label in ["Task", "Method", "Material", "Metric"]:
        terms = card_or_details["dygie_terms"].get(label, [])
        if terms:
            # drop any terms that have been seen before
            terms = [t for t in terms if t.lower() not in running_list_terms]
            # terms_dict[label] = terms
            card_or_details["dygie_terms"][label] = terms
            running_list_terms.extend(terms)
    return card_or_details


def get_author_details(
    author_id: Union[int, str],
    dirpath: Optional[str] = None,
    basedir: Union[Path, str] = UI_DATA_BASEDIR,
    limit_one_term_per_category: bool = False,
):
    basedir = Path(basedir)
    if dirpath is not None:
        basedir = basedir.joinpath(dirpath)
    g = basedir.rglob(f"author{author_id}_details.json")
    fp = list(g)[0]
    details = json.loads(fp.read_text())
    if limit_one_term_per_category is True:
        details = limit_each_term_to_only_one_category(details)
    return details


def match_focal_id(arr, focal_id: str):
    if arr:
        for i in range(len(arr)):
            this_item = arr[i]
            this_item_focal_id = str(this_item["focalId"])
            if focal_id == this_item_focal_id:
                return this_item
    return {}


##############################################


def remove_from_details(details: Dict, exclude_paper_ids: List[int]) -> Dict:
    exclude_idx = []
    for i, p in enumerate(details["papers"]):
        if p["mag_id"] in exclude_paper_ids:
            exclude_idx.append(i)
    for idx in exclude_idx:
        del details["papers"][i]
    return details


class UserStudyCard:
    def __init__(
        self,
        card_id="",
        hideTerms: bool = False,
        feedback=None,
        tracked=None,
        ui_details=None,
        card_time=None,
        isThinkAloud=None,
    ) -> None:
        self.card_id = str(card_id)
        self.hideTerms: bool = hideTerms
        self.feedback: List[Dict] = feedback
        if self.feedback is None:
            self.feedback = []
        self.tracked: List[Dict] = tracked
        if self.tracked is None:
            self.tracked = []
        self.ui_details: Dict = ui_details
        if self.ui_details is None:
            self.ui_details = {}

        self._card_time: float = card_time  # time spent on card in seconds

        self.isThinkAloud: Union[bool, None] = isThinkAloud

    @property
    def card_time(self):
        if self._card_time is None:
            if self.tracked:
                for events in self.tracked:
                    this_card_times = []
                    try:
                        this_time = events[-1]["timestamp"] - events[0]["timestamp"]
                    except KeyError:
                        # occasionally the last event doesn't have a timestamp
                        # take the second to last instead
                        this_time = events[-2]["timestamp"] - events[0]["timestamp"]
                    this_card_times.append(this_time)
                self._card_time = sum(this_card_times) / 1000
        return self._card_time

    def get_main_tracked_items(self):
        # get the longest set of tracked items by time
        # the tracked items are already segmented by card click
        events_by_time = []
        for events in self.tracked:
            try:
                this_time = events[-1]["timestamp"] - events[0]["timestamp"]
            except KeyError:
                # occasionally the last event doesn't have a timestamp
                # take the second to last instead
                this_time = events[-2]["timestamp"] - events[0]["timestamp"]
            events_by_time.append((this_time, events))
        events_by_time.sort(reverse=True, key=lambda x: x[0])
        return events_by_time[0][1]

    def num_papers_observed(self, focalId: str, N=5) -> int:
        paper_ids_observed = set()
        papers: List = self.ui_details["papers"]
        papers.sort(
            key=lambda x: match_focal_id(x["specter_distance"], focalId).get(
                "distance", 1
            )
        )
        for paper in papers[:N]:
            paper_ids_observed.add(paper["mag_id"])

        papers.sort(key=lambda paper: paper["title"].lower())
        papers.sort(key=lambda paper: paper["year"], reverse=True)
        for paper in papers[:N]:
            paper_ids_observed.add(paper["mag_id"])
        return len(paper_ids_observed)


class UserStudyCondition:
    def __init__(self, condition="", cards=None, parent=None) -> None:
        self.condition = str(condition)
        self.cards: Dict[int, UserStudyCard] = cards
        if self.cards is None:
            self.cards = {}
        self.parent: "UserStudySession" = parent

    def get_top_cards_from_all(self, N: int):
        df_distances: pd.DataFrame = self.parent.df_distances
        if not isinstance(df_distances, pd.DataFrame) or df_distances.empty:
            raise RuntimeError("df_distances not found")
        condition = self.condition.replace("_hideTerms", "")
        df = sort_distance_df(condition, df_distances)
        other_card_ids = df.head(N).AuthorId
        return other_card_ids


class UserStudySession:
    def __init__(
        self,
        focalId="",
        profile=None,
        conditions=None,
        log_fpath=None,
        df_distances=None,
        ui_details=None,
        session_distances=None,
    ) -> None:
        self.focalId = str(focalId)
        self.profile: str = profile
        self.conditions: Dict[str, UserStudyCondition] = conditions
        if self.conditions is None:
            self.conditions = {}
        self.log_fpath: Path = log_fpath
        self.df_distances: pd.DataFrame = df_distances

        self.ui_details: Dict = ui_details
        if self.ui_details is None:
            self.ui_details = {}

        self.session_distances = session_distances

    def update_card_items(self, card_id, items_list, attr_name="feedback"):
        """update a card attribute that is a list of items with some more items"""
        for cond_obj in self.conditions.values():
            if card_id in cond_obj.cards:
                card_obj: UserStudyCard = cond_obj.cards[card_id]
                to_update = getattr(card_obj, attr_name)
                to_update.extend(items_list)
                break  # I think that if the card appears in multiple conditions, it's a shared instance so we should only update once

    def update_feedback_items(self, feedback):
        for card in feedback["feedbackList"]:
            feedback_items_list = []
            card_id = str(card["cardId"])
            for item in card["feedback"]:
                if item["feedbackItemType"] == "card":
                    for x in item["feedback"]["checked"]:
                        x: str
                        if x.lower().startswith("other"):
                            if "other" in item:
                                x = f"OTHER: {item['other']}"
                        feedback_items_list.append(
                            {
                                "feedbackItemType": item["feedbackItemType"],
                                "checkedItem": x,
                            }
                        )
                else:
                    if item["feedback"]["checked"] is True:
                        feedback_items_list.append(
                            {
                                "feedbackItemType": item["feedbackItemType"],
                                "checkedItem": item["feedbackItemId"],
                            }
                        )
            self.update_card_items(card_id, feedback_items_list, attr_name="feedback")

    def update_tracked_items(self, feedback):
        events = feedback["tracked"]
        events_by_card = defaultdict(list)
        this_card_id = None
        for e in events:
            if e["action"] == "card click":
                this_card_id = str(e["nodeId"])
                events_by_card[this_card_id].append([])
            if this_card_id:
                events_by_card[this_card_id][-1].append(e)
        for card_id, tracked_items_list in events_by_card.items():
            self.update_card_items(card_id, tracked_items_list, attr_name="tracked")

    @classmethod
    def from_dict(cls, feedback: Dict):
        focalId: str = feedback["sessionInfo"]["focalId"]
        profile = (
            "persona A"
            if "A" in focalId
            else "persona B"
            if "B" in focalId
            else "overall"
        )
        obj = cls(focalId=focalId, profile=profile)
        conditions: Dict[int, str] = feedback["sessionInfo"]["conditions"]
        obj.session_distances = feedback["sessionInfo"]["distances"]
        for AuthorId, condition in conditions.items():
            if "_hideTerms" in condition:
                hideTerms = True
                condition = condition.replace("_hideTerms", "")
            else:
                hideTerms = False
            if condition not in obj.conditions:
                obj.conditions[condition] = UserStudyCondition(
                    condition=condition, parent=obj
                )
            cond_obj: UserStudyCondition = obj.conditions[condition]
            if AuthorId not in cond_obj.cards:
                cond_obj.cards[AuthorId] = UserStudyCard(
                    card_id=AuthorId, hideTerms=hideTerms
                )

        obj.update_feedback_items(feedback)
        obj.update_tracked_items(feedback)

        return obj

    @classmethod
    def from_file(
        cls, fpath: Union[Union[Path, str], Iterable[Union[Path, str]]]
    ) -> "UserStudySession":
        """Can supply a single file or a list of files (JSON)"""
        if isinstance(fpath, str) or isinstance(fpath, Path):
            fpath = [fpath]
        fpath = [Path(x) for x in fpath]
        feedback = {}
        for p in fpath:
            j = json.loads(p.read_text())
            if feedback:
                feedback["tracked"].extend(j["tracked"])
                feedback["feedbackList"].extend(j["feedbackList"])
            else:
                feedback.update(j)
        obj = cls.from_dict(feedback)
        obj.log_fpath = fpath

        return obj


class UserStudySessionGroup:
    """Keep overall and personas together"""

    def __init__(
        self,
        overall_id="",
        session_ids=None,
        logs_basedir=None,
        log_fpaths=None,
        sessions=None,
        bridger_data_version: str = "v8",
        dfs_distances_basedir=None,
        name="",
    ) -> None:
        self.overall_id = str(overall_id)
        self.session_ids: Iterable[
            Iterable[str]
        ] = session_ids  # must be a list of lists of IDs (most will be a list of one ID, but this allows for multiple session IDs)
        if self.session_ids is None:
            self.session_ids = []
        self.logs_basedir = logs_basedir
        self.log_fpaths: Iterable[Iterable[Path]] = log_fpaths
        if self.log_fpaths is None:
            self.log_fpaths = []
        self.sessions: Dict[str, UserStudySession] = sessions
        if self.sessions is None:
            self.sessions = {}
        self.bridger_data_version = bridger_data_version
        self.dfs_distances_basedir = dfs_distances_basedir
        self.name = name

    def get_card_details(self) -> None:
        for session_obj in self.sessions.values():
            ui_details = get_author_details(
                session_obj.focalId,
                dirpath=self.bridger_data_version,
                basedir=UI_DATA_BASEDIR,
            )
            # data quality issues that need to be manually corrected
            exclude_paper_ids = [
                # REDACTED
                0000000000,  # (not a proper paper)
                0000000000,  # (duplicate)
            ]
            ui_details = remove_from_details(ui_details, exclude_paper_ids)
            session_obj.ui_details = ui_details

            for cond_obj in session_obj.conditions.values():
                for card_id, card_obj in cond_obj.cards.items():
                    card_obj.ui_details = get_author_details(
                        card_id,
                        dirpath=self.bridger_data_version,
                        basedir=UI_DATA_BASEDIR,
                    )

    def load_sessions(self, get_ui_data: bool = True) -> None:
        if not self.session_ids and not self.log_fpaths:
            raise RuntimeError("can't load sessions without session_ids or log_fpaths")

        if not self.log_fpaths:
            if not self.logs_basedir:
                raise RuntimeError("can't load sessions without log_basedir")
            self.log_fpaths = self.get_log_fpaths_from_session_ids()

        for files in self.log_fpaths:
            session_obj = UserStudySession.from_file(files)
            if self.dfs_distances_basedir:
                fp = Path(self.dfs_distances_basedir).joinpath(
                    f"df_distances_{session_obj.focalId}.csv"
                )
                if fp.exists():
                    session_obj.df_distances = pd.read_csv(fp)
                else:
                    logger.warning(f"could not find file: {fp}")
            self.sessions[session_obj.focalId] = session_obj

        if get_ui_data is True:
            self.get_card_details()

        self.identify_thinkalouds()

    def get_log_fpaths_from_session_ids(self) -> Iterable[Iterable[Path]]:
        basedir = Path(self.logs_basedir)
        files_lists = []
        for session_id_list in self.session_ids:
            this_session_files = []
            for session_id in session_id_list:
                this_glob = list(basedir.glob(f"*{self.overall_id}*{session_id}.json"))
                if len(this_glob) != 1:
                    logger.warning(
                        f"Found {len(this_glob)} for session_id {session_id}"
                    )
                if this_glob:
                    this_session_files.append(this_glob[0])
            files_lists.append(this_session_files)
        return files_lists

    def find_card(self, card_id):
        for focal_id, session_obj in self.sessions.items():
            for condition, cond_obj in session_obj.conditions.items():
                for id_, card_obj in cond_obj.cards.items():
                    if card_id == id_:
                        return card_obj, focal_id, condition
        return None

    def identify_thinkalouds(self) -> None:
        cards_by_timestamp = []
        for focalId, session_obj in self.sessions.items():
            for condition, cond_obj in session_obj.conditions.items():
                for card_id, card_obj in cond_obj.cards.items():
                    events = card_obj.get_main_tracked_items()
                    timestamp = events[0]["timestamp"]
                    cards_by_timestamp.append((timestamp, card_obj))
        cards_by_timestamp.sort()
        for i in range(len(cards_by_timestamp)):
            card_obj = cards_by_timestamp[i][1]
            if i == 0 or i == len(cards_by_timestamp) - 1:  # first or last:
                card_obj.isThinkAloud = True
            else:
                card_obj.isThinkAloud = False

    def identify_post_hoc_sim_author_duplicates(self) -> None:
        self.post_hoc_duplicates = []
        for session_obj in self.sessions.values():
            for cond_obj in session_obj.conditions.values():
                # N = 4 if session_obj.profile == "overall" else 2
                N = 10
                topcardids = cond_obj.get_top_cards_from_all(N).astype(str)
                # assert len(topcardids) == N
                actualcardids = list(cond_obj.cards.keys())
                # assert len(actualcardids) == N
                topcardids = topcardids[~topcardids.isin(actualcardids)]
                if len(topcardids):
                    for card_id in topcardids.tolist():
                        findcard = self.find_card(card_id)
                        self.post_hoc_duplicates.append(
                            (cond_obj.condition, findcard, card_id, session_obj.focalId)
                        )
                        if findcard is not None:
                            card_obj = findcard[0]
                            if not card_obj.hideTerms:
                                cond_obj.cards[card_obj.card_id] = card_obj


class AnalysisDataGetter:
    def __init__(self, logs_basedir=LOGS_BASEDIR, exclude_ids=None) -> None:
        default_exclude_ids = [
            # REDACTED
            0000000000,
            0000000000,  # exclude first two sessions (for now) because there were some changes after this
            0000000000,  # A.S. bad data
            0000000000,  # T.N. did not understand the task well
            0000000000,  # R.S. went through the task too quickly/not carefully enough
        ]
        self.exclude_ids = exclude_ids or default_exclude_ids
        self.logs_basedir = logs_basedir

        self.df_authors: pd.DataFrame = None
        self.min_year: pd.Series = None
        self.df_incitations: pd.DataFrame = None
        self.df_outcitations: pd.DataFrame = None
        self.coauthor_graph: nx.Graph = None

        self.df_results: pd.DataFrame = None
        self.session_groups: Dict = None

    def yield_row(
        self,
        session_group,
        session_obj,
        cond_obj,
        card_obj,
        author_row,
        focal_overall_venues,
        focal_overall_paper_ids,
        focal_venues,
        focal_paper_ids,
        focal_incitations,
        focal_outcitations,
    ):
        self.df_authors: pd.DataFrame
        self.min_year: pd.Series
        self.df_incitations: pd.DataFrame
        self.df_outcitations: pd.DataFrame
        self.coauthor_graph: nx.Graph

        counters = Counter()
        for item in card_obj.feedback:
            counters[item["feedbackItemType"]] += 1
        if card_obj.hideTerms is True:
            item_types = [
                "card",
                "paper",
                "topic",
            ]
        else:
            item_types = [
                "card",
                "paper",
                "topic",
                "Task",
                "Method",
                "Material",
            ]

        for item_type in item_types:
            overall_id = session_group.overall_id
            num_checked = counters.get(item_type, 0)
            if item_type == "paper":
                num_observed = card_obj.num_papers_observed(session_obj.focalId)
            elif item_type == "card":
                num_observed = 5
            else:
                num_observed = 10
            this_venues = [
                p["venue"] for p in card_obj.ui_details["papers"] if p["venue"]
            ]
            this_paper_ids = [p["mag_id"] for p in card_obj.ui_details["papers"]]
            this_incitations = self.df_incitations[
                self.df_incitations["PaperId"].isin(this_paper_ids)
            ]["PaperReferenceId"]
            this_outcitations = self.df_outcitations[
                self.df_outcitations["PaperReferenceId"].isin(this_paper_ids)
            ]["PaperId"]
            shortest_path_length = get_shortest_path_length(
                self.coauthor_graph,
                session_group.overall_id,
                card_obj.card_id,
            )
            row = {
                "overall_id": overall_id,
                "name": session_group.name,
                "focalId": session_obj.focalId,
                "num_papers": len(session_obj.ui_details["papers"]),
                "authorRank": author_row["Rank"],
                "authorPaperCount": author_row["PaperCount"],
                "authorCitationCount": author_row["CitationCount"],
                "authorInitialPublishYear": self.min_year.loc[int(overall_id)],
                "condition": cond_obj.condition,
                "hideTerms": card_obj.hideTerms,
                "card_id": card_obj.card_id,
                "profile": session_obj.profile,
                "feedbackItemType": item_type,
                "num_checked": num_checked,
                "num_observed": num_observed,
                "venue_overlap": overlap_ratio(this_venues, focal_venues),
                "incitation_overlap": overlap_ratio(
                    this_incitations, focal_incitations
                ),
                "outcitation_overlap": overlap_ratio(
                    this_outcitations, focal_outcitations
                ),
                "coauthor_shortest_path": shortest_path_length,
                "card_time": card_obj.card_time,
                "venue_diversity_focal": len(set(focal_overall_venues))
                / len(focal_overall_paper_ids),
                "venue_diversity_recommended": len(set(this_venues))
                / len(this_paper_ids),
                "isThinkAloud": card_obj.isThinkAloud,
            }
            for label in ["Task", "Method", "specter"]:
                try:
                    row[f"distance_{label}"] = session_obj.session_distances[
                        card_obj.card_id
                    ][label]
                except KeyError:
                    pass
            yield row

    def get_num_papers_categorical(self, df_results):
        num_papers_overall = df_results.drop_duplicates(
            subset=["overall_id"]
        ).set_index("overall_id")["num_papers_overall"]
        quantiles = num_papers_overall.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).apply(
            math.ceil
        )
        labels = [
            f"<={quantiles.iloc[1]-1}",
            f"{quantiles.iloc[1]}-{quantiles.iloc[2]-1}",
            f"{quantiles.iloc[2]}-{quantiles.iloc[3]-1}",
            f"{quantiles.iloc[3]}-{quantiles.iloc[4]-1}",
            f">={quantiles.iloc[4]}",
        ]
        cats_map = pd.qcut(num_papers_overall, quantiles.index, labels=labels)
        num_papers_categorical = df_results.overall_id.map(cats_map)
        return num_papers_categorical

    def load_df_results(self):
        _data = []
        session_groups = self.session_groups

        for overall_id, session_group in session_groups.items():
            focal_overall_venues = [
                p["venue"]
                for p in session_group.sessions[overall_id].ui_details["papers"]
                if p["venue"]
            ]
            focal_overall_paper_ids = [
                p["mag_id"]
                for p in session_group.sessions[overall_id].ui_details["papers"]
            ]
            author_row = self.df_authors.loc[int(overall_id)]
            for session_obj in session_group.sessions.values():
                focal_venues = [
                    p["venue"] for p in session_obj.ui_details["papers"] if p["venue"]
                ]
                focal_paper_ids = [
                    p["mag_id"] for p in session_obj.ui_details["papers"]
                ]
                focal_incitations = self.df_incitations[
                    self.df_incitations["PaperId"].isin(focal_paper_ids)
                ]["PaperReferenceId"]
                focal_outcitations = self.df_outcitations[
                    self.df_outcitations["PaperReferenceId"].isin(focal_paper_ids)
                ]["PaperId"]

                for cond_obj in session_obj.conditions.values():
                    for card_obj in cond_obj.cards.values():
                        for row in self.yield_row(
                            session_group,
                            session_obj,
                            cond_obj,
                            card_obj,
                            author_row,
                            focal_overall_venues,
                            focal_overall_paper_ids,
                            focal_venues,
                            focal_paper_ids,
                            focal_incitations,
                            focal_outcitations,
                        ):
                            _data.append(row)

        df_results = pd.DataFrame(_data).sort_values(
            ["name", "condition", "hideTerms"], ascending=[False, False, True]
        )
        df_results["num_papers_overall"] = df_results.groupby("overall_id")[
            "num_papers"
        ].transform("max")
        df_results["condition"] = df_results.condition.astype("category")
        df_results["profile"] = df_results.profile.astype("category")
        df_results["feedbackItemType"] = df_results.feedbackItemType.astype("category")
        df_results["isSpecterBaseline"] = df_results.condition.str.contains("specter")

        # REMOVE cards FROM ALL ANALYSIS
        df_results = df_results[df_results.feedbackItemType != "card"]

        df_results["novelty_score_overall"] = self.get_novelty_score_overall(df_results)
        df_results["novelty_score_overall_sT"] = self.get_novelty_score_overall(
            df_results, a_conds=["simTask"]
        )
        df_results["novelty_score_overall_sTdM"] = self.get_novelty_score_overall(
            df_results, a_conds=["simTask_distMethod"]
        )
        df_results["novelty_score_overall_onlypaper"] = self.get_novelty_score_overall(
            df_results, exclude_types=["Task", "Method", "Material", "topic"]
        )
        df_results[
            "novelty_score_overall_onlypaper_sT"
        ] = self.get_novelty_score_overall(
            df_results,
            a_conds=["simTask"],
            exclude_types=["Task", "Method", "Material", "topic"],
        )
        df_results[
            "novelty_score_overall_onlypaper_sTdM"
        ] = self.get_novelty_score_overall(
            df_results,
            a_conds=["simTask_distMethod"],
            exclude_types=["Task", "Method", "Material", "topic"],
        )
        df_results["novelty_score_overall_onlyfacets"] = self.get_novelty_score_overall(
            df_results, exclude_types=["paper"]
        )
        df_results[
            "novelty_score_overall_onlyfacets_sT"
        ] = self.get_novelty_score_overall(
            df_results, a_conds=["simTask"], exclude_types=["paper"]
        )
        df_results[
            "novelty_score_overall_onlyfacets_sTdM"
        ] = self.get_novelty_score_overall(
            df_results, a_conds=["simTask_distMethod"], exclude_types=["paper"]
        )
        df_results["num_papers_cat"] = self.get_num_papers_categorical(df_results)
        self.df_results = df_results

    def get_novelty_score_overall(
        self, df_results, a_conds=["simTask", "simTask_distMethod"], exclude_types=None
    ):
        _df = df_results.copy()
        if exclude_types is not None:
            _df = _df[~_df.feedbackItemType.isin(exclude_types)]
        novelty_scores = (
            _df.groupby(["overall_id", "feedbackItemType"])
            .apply(
                novelty_wins,
                a_conds=a_conds,
            )
            .astype(float)
        )
        mean_score_per_subject = novelty_scores.dropna().groupby(level=0).mean()
        return df_results.overall_id.map(mean_score_per_subject)

    @classmethod
    def from_log_index(
        cls,
        fname: Union[Path, str],
        logs_basedir=LOGS_BASEDIR,
        merge_sim_author_duplicates: bool = False,
    ) -> "AnalysisDataGetter":
        fname = Path(fname)
        participants = json.loads(fname.read_text())["participants"]
        obj = cls(logs_basedir)
        supp_datadir = DATADIR.joinpath("analysis_supplemental_data/")
        obj.df_authors = pd.read_parquet(supp_datadir.joinpath("df_authors.parquet"))
        obj.min_year = pd.read_parquet(supp_datadir.joinpath("min_year.parquet"))[
            "Year"
        ]
        obj.df_incitations = pd.read_parquet(
            supp_datadir.joinpath("df_incitations.parquet")
        )
        obj.df_outcitations = pd.read_parquet(
            supp_datadir.joinpath("df_outcitations.parquet")
        )
        obj.coauthor_graph = nx.read_gpickle(COAUTHOR_GRAPH_FPATH)
        session_groups = {}
        for p in participants:
            if p["name"].startswith("example") or p["overall_id"] in obj.exclude_ids:
                continue
            bridger_data_version = p["bridger_data_version"]
            dfs_distances_basedir = Path(
                f"../data/user_study_data/{bridger_data_version}_df_distances/"
            )
            session_group = UserStudySessionGroup(
                overall_id=p["overall_id"],
                session_ids=p["session_ids"],
                logs_basedir=obj.logs_basedir,
                bridger_data_version=bridger_data_version,
                dfs_distances_basedir=dfs_distances_basedir,
                name=p.get("name", ""),
            )
            session_group.load_sessions()
            # TESTING below
            if merge_sim_author_duplicates is True:
                session_group.identify_post_hoc_sim_author_duplicates()
            session_groups[session_group.overall_id] = session_group

        obj.session_groups = session_groups

        obj.load_df_results()

        return obj


def collect_all_feedback(
    fp_log_index, logs_basedir=LOGS_BASEDIR, exclude_ids=None
) -> Dict[str, Any]:
    ret = {}
    fp_log_index = Path(fp_log_index)
    participants = json.loads(fp_log_index.read_text())["participants"]
    if not exclude_ids:
        exclude_ids = []
    for p in participants:
        if p["name"].startswith("example") or p["overall_id"] in exclude_ids:
            continue
        bridger_data_version = p["bridger_data_version"]
        session_group = UserStudySessionGroup(
            overall_id=p["overall_id"],
            session_ids=p["session_ids"],
            logs_basedir=logs_basedir,
            bridger_data_version=bridger_data_version,
            name=p.get("name", ""),
        )
        log_fpaths = session_group.get_log_fpaths_from_session_ids()
        for files in log_fpaths:
            feedback = {}
            for p in files:
                j = json.loads(p.read_text())
                if feedback:
                    feedback["feedbackList"].extend(j["feedbackList"])
                else:
                    feedback.update(j)
            focalId: str = feedback["sessionInfo"]["focalId"]
            ret[focalId] = {
                "bridger_data_version": bridger_data_version,
                "overall_id": session_group.overall_id,
                "session_ids": session_group.session_ids,
                "log_fpaths": [x.name for x in files],
                "overall_name": session_group.name,
                "feedbackList": feedback["feedbackList"],
            }
    return ret


def save_all_feedback(fb: Dict, outfp: Union[str, Path]) -> None:
    outfp = Path(outfp)
    out_txt = json.dumps(fb)
    out_txt = out_txt.replace(
        "I can use resources with which they work that I had not considered previously.",
        "I can imagine using resources with which they work that I had not considered previously.",
    )
    outfp.write_text(out_txt)
