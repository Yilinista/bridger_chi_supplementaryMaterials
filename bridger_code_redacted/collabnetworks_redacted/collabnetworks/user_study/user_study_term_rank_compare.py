from collections import defaultdict
from collabnetworks.cards.cards import BridgerCard
import sys, os, time, json
from pathlib import Path
from typing import (
    Callable,
    TypedDict,
    Union,
    List,
    Optional,
    Dict,
    Tuple,
    Any,
    Mapping,
    Sequence,
    Iterable,
)
from string import ascii_uppercase, ascii_lowercase
import dataclasses
from dataclasses import dataclass
import logging

root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

from .user_study import (
    UserStudyData,
    DEFAULT_RNG_TYPES,
)
from ..cards import (
    BridgerCard,
    BridgerCardTermRankCompare,
    DygieTerms,
    DygieTermsSimilar,
    BridgerCardDistance,
    BridgerCardSimilar,
    BridgerCardDetails,
)
from ..collection_helper import PaperCollectionHelper
from ..data_helper import DataHelper
from ..collection_helper import AuthorHelper, TermRanker, PaperCollectionHelper
from ..util import overlap_ratio
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import simplejson


@dataclass
class BridgerCardSimilarTermRankCompare(
    BridgerCardSimilar, BridgerCardTermRankCompare
):
    """This is a class that combines the attributes of BridgerCardSimilar, and BridgerCardTermRankCompare"""

    def __post_init__(self) -> None:
        BridgerCardSimilar.__post_init__(self)


class UserStudyDataTermRankCompare(UserStudyData):
    def strategy_tfidf(
        self, label: str, paper_collection: PaperCollectionHelper, N: int = 20
    ) -> Union[List[str], None]:
        if paper_collection.collection_type == "author":
            return self._strategy_tfidf_author(label, paper_collection.id, N)
        try:
            vectorizer = self.tfidf_vectorizers[label]
        except KeyError:
            return None
        df_terms = paper_collection.df_terms[paper_collection.df_terms.label == label]
        terms_concat = df_terms["term_id"].astype(str).tolist()
        terms_concat = " ".join(terms_concat)
        mat = vectorizer.transform([terms_concat])
        index = pd.Series(vectorizer.get_feature_names()).astype(int)
        tfidf_scores = (
            pd.Series(mat.toarray().flatten(), index=index).replace(0, np.nan).dropna()
        )
        tfidf_map = (
            df_terms["term_id"].map(tfidf_scores).dropna().sort_values(ascending=False)
        )
        return df_terms["term_display"].reindex_like(tfidf_map).head(N).tolist()

    def _strategy_tfidf_author(
        self, label: str, author_id: int, N: int = 20
    ) -> Union[List[str], None]:
        try:
            df = self.tfidf_dfs[label]
        except KeyError:
            return None
        df = df[df["AuthorId"] == author_id]
        return (
            df.sort_values("tfidf_score", ascending=False)
            .head(N)["term_display"]
            .tolist()
        )

    def _get_dygie_terms(
        self,
        paper_collection: PaperCollectionHelper,
        ranking_function: Union[Callable, str],
        N: int = 20,
        labels: Iterable[str] = ["Task", "Method"],
        diversification: bool = True,
    ) -> DygieTerms:
        # terms_dict = {label: paper_collection.term_ranker.strategy_textrank(label) for label in paper_collection.term_ranker.df_terms['label'].unique()}
        if ranking_function == "strategy_tfidf":
            terms_dict = {
                label: self.strategy_tfidf(label, paper_collection, N=N)
                # for label in paper_collection.term_ranker.df_terms["label"].unique()
                for label in labels
            }

        else:
            terms_dict = {
                label: ranking_function(label, N=N)
                # for label in paper_collection.term_ranker.df_terms["label"].unique()
                for label in labels
            }
        terms_dict = {
            k: v for k, v in terms_dict.items() if v is not None and len(v) > 0
        }
        running_list_terms = (
            paper_collection.mag_topics.DisplayName.str.lower().tolist()
        )
        for label in labels:
            terms = terms_dict.get(label, [])
            terms = paper_collection.term_ranker.check_for_terms_to_drop(terms)
            # drop any terms that have been seen before
            terms = [t for t in terms if t.lower() not in running_list_terms]
            terms_dict[label] = terms
            running_list_terms.extend(terms_dict[label])
        dygie_terms = DygieTerms(
            Method=terms_dict.get("Method", []),
            Task=terms_dict.get("Task", []),
            Material=terms_dict.get("Material", []),
            Metric=terms_dict.get("Metric", []),
        )
        if diversification is True:
            # logger.debug("applying diversification to truncate terms list")
            dygie_terms = paper_collection.term_ranker.get_dygie_terms_trunc(
                dygie_terms, N=N
            )
        return dygie_terms

    def get_multiple_term_ranking_strategies(
        self,
        paper_collection: PaperCollectionHelper,
        num_terms: int = 20,
        labels: Iterable[str] = ["Task", "Method"],
    ) -> BridgerCard:
        term_rank_dict = {}
        logger.debug("getting multiple term ranking strategies...")
        strategies = [
            "strategy_textrank",
            "strategy_relevance_score",
            "strategy_random",
        ]
        for strategy in strategies:
            diversification = True
            fn = getattr(paper_collection.term_ranker, strategy)
            if strategy == "strategy_random":
                diversification = False
            term_rank_dict[strategy] = self._get_dygie_terms(
                paper_collection,
                fn,
                N=num_terms,
                labels=labels,
                diversification=diversification,
            )

        if (
            self.tfidf_dfs is not None and paper_collection.collection_type == "author"
        ) or (self.tfidf_vectorizers is not None):
            strategy = "strategy_tfidf"
            diversification = True
            term_rank_dict[strategy] = self._get_dygie_terms(
                paper_collection,
                strategy,
                N=num_terms,
                labels=labels,
                diversification=diversification,
            )

        logger.debug(
            f"got term rankings for {len(term_rank_dict)} strategies: {list(term_rank_dict.keys())}"
        )
        return term_rank_dict

    def get_single_card(
        self,
        paper_collection: PaperCollectionHelper,
        cls,
        distance=False,
        num_terms: int = 20,
        labels: Iterable[str] = ["Task", "Method", "Material"],
    ) -> BridgerCard:
        """Overrides UserStudyData.get_single_card()"""
        card = self._get_single_card(paper_collection, cls, distance)
        card.dygie_terms_rank_compare = self.get_multiple_term_ranking_strategies(
            paper_collection, num_terms, labels
        )
        return card

    def save_focal_card(self) -> None:
        """Overrides UserStudyData.save_focal_card()"""
        self._save_focal_card(cls=BridgerCardSimilarTermRankCompare)

    def save_cards(self) -> None:
        """Overrides UserStudyData.save_cards()"""
        cards_classes = {
            "focal": BridgerCardTermRankCompare,
            "sim": BridgerCardSimilarTermRankCompare,
        }
        self._save_cards(cards_classes=cards_classes)

    def get_terms_ranking_from_card(
        self,
        card: BridgerCardTermRankCompare,
        N: int = 10,
        labels: Iterable[str] = ["Task", "Method"],
    ) -> Dict[str, Any]:
        strategy_dict = card.dygie_terms_rank_compare
        data = {label: {} for label in labels}

        for strategy_name, dygie_terms in strategy_dict.items():
            for label in labels:
                terms = getattr(dygie_terms, label)
                data[label][strategy_name] = pd.Series(terms[:N], name=label + "s")
        # Get MAG topics
        topics = [t.DisplayName for t in card.topics]
        data["Topics"] = pd.Series(topics[:N], name="Topics")
        return data

    def create_spreadsheet_focal_author(
        self,
        card: BridgerCardTermRankCompare,
        N: int = 10,
        labels: Iterable[str] = ["Task", "Method", "Material"],
    ) -> pd.DataFrame:
        data = self.get_terms_ranking_from_card(card, N, labels)
        # randomize order
        strategies_random_order = {}
        if "Task" in data:
            strategies_random_order["Task"] = self.rng.choice(
                list(data["Task"].keys()), size=len(data), replace=False
            )
        if "Method" in data:
            strategies_random_order["Method"] = self.rng.choice(
                list(data["Method"].keys()), size=len(data), replace=False
            )
        if "Material" in data:
            # same order for methods and materials
            strategies_random_order["Material"] = strategies_random_order["Method"]

        # 1st column: Topics
        topics = data["Topics"]
        df = pd.DataFrame(topics)

        # other columns
        for label in labels:
            label_start_column_index = len(df.columns)
            if label in data:
                i = 0
                # for strategy_name, terms_series in data[label].items():
                # strategy_letter = ascii_lowercase[i]
                # df[f"{label}s ({strategy_letter})"] = terms_series
                for strategy_name in strategies_random_order[label]:
                    terms_series = data[label][strategy_name]
                    df[f"{label}s ({strategy_name})"] = terms_series
                    i += 1
                    if label == "Material":
                        # blank column in between each Material
                        df.insert(len(df.columns), "<blank>", "", allow_duplicates=True)

                # insert topics in a random position among the tasks/methods
                random_index = self.rng.integers(
                    label_start_column_index, len(df.columns) + 1
                )
                random_index = int(random_index)
                if label == "Task":
                    df.insert(
                        random_index,
                        "Topics_10-20",
                        topics.iloc[10:20].reset_index(drop=True),
                    )
                elif label == "Method":
                    df.insert(
                        random_index,
                        "Topics_20-30",
                        topics.iloc[20:30].reset_index(drop=True),
                    )

                # blank column
                df.insert(len(df.columns), "<blank>", "", allow_duplicates=True)

        df = df.fillna(value="")
        return df

    def create_spreadsheet_unknown_author(
        self,
        card: BridgerCardTermRankCompare,
        N: int = 10,
        labels: Iterable[str] = ["Task", "Method", "Material"],
        strategy: str = "strategy_tfidf",
    ) -> pd.DataFrame:
        data = self.get_terms_ranking_from_card(card, N, labels)
        # 1st column: Topics
        topics = data["Topics"]
        df = pd.DataFrame(topics)

        # other columns
        for label in labels:
            if label in data:
                df[f"{label}s"] = data[label][strategy]
        return df

    def get_sub_df_persona(
        self, data, strategy_name: str, labels: Iterable[str] = ["Task"]
    ) -> pd.DataFrame:
        # Get a dataframe with column(s) for dygie terms for a given strategy
        # topics = data["Topics"]
        # sub_df = pd.DataFrame(topics)
        sub_df = pd.DataFrame()
        for label in labels:
            if label in data:
                column = data[label][strategy_name]
                sub_df[column.name] = column
        return sub_df

    def create_spreadsheet_personas(
        self,
        card: BridgerCardTermRankCompare,
        persona_list: Sequence[Iterable[int]],
        N: int = 10,
        labels: Iterable[str] = ["Task"],
    ) -> Union[Tuple[str, pd.DataFrame], None]:
        i = 0
        persona_names = []
        persona_name_to_author_ids = {}
        strategies_dfs = defaultdict(list)
        card = self.focal_card()
        data = self.get_terms_ranking_from_card(card, N=N, labels=labels)
        for strategy in card.dygie_terms_rank_compare.keys():
            sub_df = self.get_sub_df_persona(data, strategy, labels=labels)
            strategies_dfs[strategy].append(sub_df)
        topics = data["Topics"]
        strategies_dfs["strategy_topics"].append(pd.DataFrame(topics))
        persona_names.append(self.paper_collection.name)
        for ids in persona_list:
            this_persona_letter = ascii_uppercase[i]
            u_persona = self.persona_subset_factory(this_persona_letter, ids)
            if len(u_persona.paper_collection.dedup_title_paper_ids) > 2:
                # card = u_persona.get_single_card(u_persona.paper_collection, cls=BridgerCardTermRankCompare, num_terms=30)
                persona_name = u_persona.paper_collection.name
                card = u_persona.focal_card(labels=labels)
                data = u_persona.get_terms_ranking_from_card(card, N=N, labels=labels)
                for strategy in card.dygie_terms_rank_compare.keys():
                    sub_df = self.get_sub_df_persona(data, strategy, labels)
                    strategies_dfs[strategy].append(sub_df)
                # terms_df_list.append((persona_name, data))

                topics = data["Topics"]
                strategies_dfs["strategy_topics"].append(pd.DataFrame(topics))
                persona_names.append(persona_name)
                persona_name_to_author_ids[persona_name] = ids
                i += 1
        if len(persona_names) < 3:
            logger.warning(
                f"Found {len(persona_names)-1} (non-negligible) personas, so not saving any personas data for author {self.paper_collection.id}"
            )
            return None
        ret = []
        # strategies_random_order = self.rng.choice(
        #     list(strategies_dfs.keys()),
        #     size=len(card.dygie_terms_rank_compare),
        #     replace=False,
        # )
        strategies_list = [
            "strategy_tfidf",
            "strategy_random",
            "strategy_relevance_score",
            "strategy_textrank",
            "strategy_topics",
        ]
        for strategy in strategies_list:
            sheet_name = strategy.split("_")[-1]
            df = pd.concat(strategies_dfs[strategy], axis=1, keys=persona_names)
            ret.append((sheet_name, df))
        return ret

        # data = self.get_terms_ranking_from_card(card, N, labels)
        # # randomize order
        # strategies_random_order = {}
        # if "Task" in data:
        #     strategies_random_order["Task"] = self.rng.choice(
        #         list(data["Task"].keys()), size=len(data), replace=False
        #     )
        # if "Method" in data:
        #     strategies_random_order["Method"] = self.rng.choice(
        #         list(data["Method"].keys()), size=len(data), replace=False
        #     )
        # if "Material" in data:
        #     # same order for methods and materials
        #     strategies_random_order["Material"] = strategies_random_order["Method"]

        # # 1st column: Topics
        # topics = data["Topics"]
        # df = pd.DataFrame(topics)

        # # other columns
        # for label in labels:
        #     if label in data:
        #         i = 0
        #         for strategy_name, terms_series in data[label].items():
        #             strategy_letter = ascii_lowercase[i]
        #             df[f"{label}s ({strategy_letter})"] = terms_series
        #             i += 1
        #             if label == 'Material':
        #                 # blank column in between Material columns
        #                 df.insert(len(df.columns), "<blank>", "", allow_duplicates=True)
        #         # blank column
        #         df.insert(len(df.columns), "<blank>", "", allow_duplicates=True)
        # return df

    def create_spreadsheet_personas_papers_and_authors(
        self,
        card: BridgerCardTermRankCompare,
        persona_list: Sequence[Iterable[int]],
    ) -> Union[Tuple[str, pd.DataFrame], None]:
        i = 0
        persona_names = []
        persona_name_to_author_ids = {}
        dfs = defaultdict(list)
        for ids in persona_list:
            this_persona_letter = ascii_uppercase[i]
            u_persona = self.persona_subset_factory(this_persona_letter, ids)
            if len(u_persona.paper_collection.dedup_title_paper_ids) > 2:
                # card = u_persona.get_single_card(u_persona.paper_collection, cls=BridgerCardTermRankCompare, num_terms=30)
                persona_name = u_persona.paper_collection.name
                # card = u_persona.focal_card(labels=labels)
                # data = u_persona.get_terms_ranking_from_card(card, N=N, labels=labels)
                df_papers = u_persona.paper_collection.df_papers
                df_papers = df_papers[
                    df_papers["PaperId"].isin(
                        u_persona.paper_collection.dedup_title_paper_ids
                    )
                ]
                dfs["papers"].append(
                    df_papers.sort_values("Year", ascending=False)["OriginalTitle"]
                )
                df_paa = u_persona.paper_collection.df_paper_authors
                df_paa = df_paa[df_paa["PaperId"].isin(df_papers["PaperId"])]
                vc = df_paa.AuthorId.value_counts()
                author_name_map = u_persona.data.mag_data.authors.set_index("AuthorId")[
                    "DisplayName"
                ]
                dfs["authors"].append(pd.Series(vc.index.map(author_name_map).tolist()))
                persona_names.append(persona_name)
                persona_name_to_author_ids[persona_name] = ids
                i += 1
        if len(persona_names) < 3:
            logger.warning(
                f"Found {len(persona_names)-1} (non-negligible) personas, so not saving any personas data for author {self.paper_collection.id}"
            )
            return None
        ret = []
        for k in dfs.keys():
            df = pd.concat(dfs[k], axis=1, keys=persona_names)
            ret.append((k, df))
        return ret

    def create_ranking_df(
        self,
        card: BridgerCardTermRankCompare,
        N: int = 10,
        labels: Iterable[str] = ["Task", "Method"],
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        data = self.get_terms_ranking_from_card(card, N, labels)
        # randomize order
        random_indices = self.rng.choice(
            range(len(data)), size=len(data), replace=False
        )
        data = [data[i] for i in random_indices]

        # anonymize strategy names while keeping track of them
        profile_names = []
        unpacked_data = []
        anon_mapping = {}
        for i, item in enumerate(data):
            strategy_name = item["strategy_name"]
            unpacked_data.append(item["sub_df"])
            profile_name = f"Profile {i+1}"
            profile_names.append(profile_name)
            anon_mapping[profile_name] = strategy_name
        df = pd.concat(unpacked_data, axis=1, keys=profile_names)
        return df, anon_mapping

    def get_spreadsheet_and_metadata(
        self,
        paper_collection: PaperCollectionHelper,
        num_terms: int = 10,
        additional_metadata: Optional[Mapping[str, Any]] = None,
        labels: Iterable[str] = ["Task", "Method"],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.coauthor_graph is None:
            self.load_coauthor_graph()
        if self.tfidf_dfs is None:
            self.load_tfidf()

        if paper_collection.id == self.paper_collection.id:
            card = self.focal_card(labels=labels)
        else:
            card = self.get_single_card(
                paper_collection,
                cls=BridgerCardTermRankCompare,
                num_terms=30,
                labels=labels,
            )
        df, anon_mapping = self.create_ranking_df(card, N=num_terms, labels=labels)

        metadata = {
            "id": card.id,
            "s2_id": card.s2Id,
            "type": card.type,
            "displayName": card.displayName,
            "profile_map": anon_mapping,
            "num_papers": card.numPapers,
            "num_papers_titles_dedup": len(paper_collection.dedup_title_paper_ids),
        }
        coauthors = getattr(card, "coauthors", None)
        if coauthors:
            metadata["strong_coauthors"] = [
                dataclasses.asdict(a) for a in card.coauthors
            ]
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        return df, metadata

    def focal_card(
        self,
        force_reload: bool = False,
        num_terms: int = 30,
        labels: Iterable[str] = ["Task", "Method"],
    ) -> BridgerCardSimilarTermRankCompare:
        # allow caching of focal card
        self._focal_card = getattr(self, "_focal_card", None)
        if self._focal_card is None or force_reload is True:
            self._focal_card = self.get_single_card(
                self.paper_collection,
                cls=BridgerCardTermRankCompare,
                num_terms=num_terms,
                labels=labels,
            )
        return self._focal_card

    def save_spreadsheet_and_metadata(
        self,
        outdir: Union[str, Path],
        other_author_ids: Sequence[int],
        other_author_metadata: Optional[Sequence[Dict[str, Any]]] = None,
        num_terms: int = 15,
        labels: Iterable[str] = ["Task", "Method"],
    ) -> None:
        outdir = Path(outdir)
        if not outdir.exists():
            logger.debug(f"creating output directory: {outdir}")
            outdir.mkdir()
        else:
            logger.debug(f"using existing output directory: {outdir}")
        if other_author_metadata is None:
            other_author_metadata = []
        metadata_lines = []
        author = self.paper_collection
        df, metadata = self.get_spreadsheet_and_metadata(
            author,
            num_terms=num_terms,
            additional_metadata={"isFocal": True},
            labels=labels,
        )
        if other_author_metadata is None:
            other_author_metadata = []
        metadata_lines.append(metadata)
        s = f"compareterms_author{author.id}_{author.name.replace(' ', '')}"
        outfp_spreadsheet = outdir.joinpath(s + ".xlsx")
        outfp_metadata = outdir.joinpath(s + "_metadata.jsonl")
        # pylint: disable=abstract-class-instantiated
        with pd.ExcelWriter(outfp_spreadsheet) as writer:
            df.to_excel(writer, sheet_name=author.name)
            for i in range(len(other_author_ids)):
                other_author_id = other_author_ids[i]
                try:
                    additional_metadata = other_author_metadata[i]
                except IndexError:
                    additional_metadata = {}
                logger.debug(f"getting author {other_author_id}")
                author = self.get_author_helper(other_author_id)
                logger.debug(f"author name: {author.name}")
                additional_metadata.update({"isFocal": False})
                df, metadata = self.get_spreadsheet_and_metadata(
                    author,
                    num_terms=num_terms,
                    additional_metadata=additional_metadata,
                    labels=labels,
                )
                metadata_lines.append(metadata)
                df.to_excel(writer, sheet_name=author.name)
        logger.debug(f"writing {len(metadata_lines)} lines to {outfp_metadata}")
        with outfp_metadata.open("w") as outf:
            for metadata in metadata_lines:
                print(simplejson.dumps(metadata, ignore_nan=True), file=outf)

    def save_spreadsheet_and_metadata_personas(
        self,
        outdir: Union[str, Path],
        persona_list: Sequence[Iterable[int]],
        num_terms: int = 15,
        labels: Iterable[str] = ["Task", "Method"],
    ) -> None:
        if self.tfidf_vectorizers is None:
            self.load_tfidf_vectorizers()
        outdir = Path(outdir)
        if not outdir.exists():
            logger.debug(f"creating output directory: {outdir}")
            outdir.mkdir()
        else:
            logger.debug(f"using existing output directory: {outdir}")

        s = f"compareterms_personas_author{self.paper_collection.id}_{self.paper_collection.name.replace(' ', '')}"
        outfp_spreadsheet = outdir.joinpath(s + ".xlsx")
        outfp_metadata = outdir.joinpath(s + "_metadata.json")
        terms_df_list = []
        card = self.focal_card(labels=labels)
        data = self.get_terms_ranking_from_card(card, N=num_terms, labels=labels)
        terms_df_list.append((self.paper_collection.name, data))
        i = 0
        persona_name_to_author_ids = {}
        for ids in persona_list:
            this_persona_letter = ascii_uppercase[i]
            u_persona = self.persona_subset_factory(this_persona_letter, ids)
            if len(u_persona.paper_collection.dedup_title_paper_ids) > 2:
                # card = u_persona.get_single_card(u_persona.paper_collection, cls=BridgerCardTermRankCompare, num_terms=30)
                persona_name = u_persona.paper_collection.name
                card = u_persona.focal_card(labels=labels)
                data = u_persona.get_terms_ranking_from_card(
                    card, N=num_terms, labels=labels
                )
                terms_df_list.append((persona_name, data))
                persona_name_to_author_ids[persona_name] = ids
                i += 1

        if len(terms_df_list) < 3:
            logger.warning(
                f"Found {len(terms_df_list)-1} (non-negligible) personas, so not writing any personas data for author {self.paper_collection.id}"
            )
            return

        sheets_data = {}

        # this will save one sheet per ranking strategy, with all personas (tasks, methods, and topics) within each sheet
        logger.debug(f"saving data for author and {len(terms_df_list)-1} personas")
        for persona_name, pdata in terms_df_list:
            for item in pdata:
                strategy_name = item["strategy_name"]
                #     if strategy_name == "strategy_mag":
                #         topics = item["sub_df"]
                #     else:
                #         if strategy_name not in sheets_data:
                #             sheets_data[strategy_name] = {"names": [], "dfs": []}
                #         sheets_data[strategy_name]["dfs"].append(item["sub_df"])
                # for strategy_name in sheets_data:
                #     sub_df = sheets_data[strategy_name]["dfs"][-1]
                #     sheets_data[strategy_name]["dfs"][-1] = pd.concat(
                #         [sub_df, topics], axis=1
                #     )
                #     sheets_data[strategy_name]["names"].append(persona_name)
                if strategy_name not in sheets_data:
                    sheets_data[strategy_name] = {"names": [], "dfs": []}
                sheets_data[strategy_name]["dfs"].append(item["sub_df"])
            for strategy_name in sheets_data:
                sheets_data[strategy_name]["names"].append(persona_name)
        sheets = {
            strategy_name: pd.concat(val["dfs"], axis=1, keys=val["names"])
            for strategy_name, val in sheets_data.items()
        }

        logger.debug(f"writing to {outfp_spreadsheet}")
        # pylint: disable=abstract-class-instantiated
        with pd.ExcelWriter(outfp_spreadsheet) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name)

        metadata = {"persona_name_to_author_ids": persona_name_to_author_ids}
        logger.debug(f"writing to {outfp_metadata}")
        outfp_metadata.write_text(simplejson.dumps(metadata, ignore_nan=True))


def has_enough_data_term_rank_compare(card, thresh=10):
    num_topics = len(card.topics)
    if num_topics <= thresh:
        return False
    num_tasks = min(len(v.Task) for v in card.dygie_terms_rank_compare.values())
    if num_tasks <= thresh:
        return False
    num_methods = min(len(v.Method) for v in card.dygie_terms_rank_compare.values())
    if num_methods <= thresh:
        return False
    num_materials = min(len(v.Material) for v in card.dygie_terms_rank_compare.values())
    if num_materials <= thresh:
        return False
    return True


class Initializer:
    def __init__(
        self,
        outdir: Union[str, Path],
        data_helper: DataHelper,
        tfidf_dfs: Dict[str, pd.DataFrame],
        tfidf_vectorizers: Dict[str, TfidfVectorizer],
        coauthor_graph: nx.Graph,
        ego_partition: Dict,
        abbreviations: Optional[pd.DataFrame] = None,
        random_seed: int = 141792,
        min_year: int = 2015,
        max_year: int = 2022,
    ):
        self.outdir = Path(outdir)
        self.data_helper = data_helper
        self.tfidf_dfs = tfidf_dfs
        self.tfidf_vectorizers = tfidf_vectorizers
        self.coauthor_graph = coauthor_graph
        self.ego_partition = ego_partition
        self.abbreviations = abbreviations
        self.random_seed = random_seed
        self.min_year = min_year
        self.max_year = max_year

    def increment_random_seed(self):
        self.random_seed += 1

    def initialize(
        self,
        author_id: int,
    ) -> UserStudyDataTermRankCompare:
        rng = np.random.default_rng(self.random_seed)
        try:
            author = AuthorHelper(
                author_id,
                data=self.data_helper,
                min_year=self.min_year,
                max_year=self.max_year,
                coauthor_graph=self.coauthor_graph,
            )
        except (TypeError, KeyError):
            raise RuntimeError(f"could not lookup author {author_id}")
        logger.debug(
            f"Initializing UserStudyDataTermRankCompare for author {author.id}. author name: {author.name}. Using min_year {self.min_year}, max year {self.max_year}. Random seed: {self.random_seed}"
        )
        u = UserStudyDataTermRankCompare(
            author,
            data=author.data,
            min_year=author.min_year,
            max_year=author.max_year,
            min_papers=6,
            tfidf_dfs=self.tfidf_dfs,
            tfidf_vectorizers=self.tfidf_vectorizers,
            coauthor_graph=author.coauthor_graph,
            ego_partition=self.ego_partition,
            random_seed=rng,
        )
        return u

    def collect_all_data_from_author_id(
        self, author_id: int, num_other_authors: int = 5
    ) -> None:
        u = self.initialize(author_id)
        collect_all_data_for_experiment(
            u, self.outdir, num_other_authors=num_other_authors
        )

    def get_abbreviations(
        self, paper_collection: PaperCollectionHelper
    ) -> Union[pd.DataFrame, None]:
        if self.abbreviations is None:
            return None
        df = self.abbreviations[
            self.abbreviations.s2_id.isin(paper_collection.df_papers.s2_id)
        ].sort_values("abbrv")
        df.drop(
            columns=["embedding_term_id", "long_form_cleaned"],
            errors="ignore",
            inplace=True,
        )
        return df

    def save_full_excel_file(
        self,
        author_id: int,
        increment_random_seed: bool = True,
        labels: Sequence[str] = ["Task", "Method", "Material"],
        N: int = 30,
        max_other_authors: int = 5,
        save_log: bool = True,
    ) -> None:
        if save_log is True:
            outfp_log = self.outdir.joinpath(f"compareterms_author{author_id}.log")
            logger.debug(f"outputting to log file: {outfp_log}")
            fhandler = logging.FileHandler(outfp_log)
            fhandler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            fhandler.setLevel(logging.DEBUG)
            root_logger.addHandler(fhandler)
        if increment_random_seed is True:
            self.increment_random_seed()
        u = self.initialize(author_id)
        author = u.paper_collection
        # outfp_spreadsheet = Path(output)
        s = f"compareterms_author{author.id}_{author.name.replace(' ', '')}"
        outfp_spreadsheet = self.outdir.joinpath(s + ".xlsx")
        outfp_authors = self.outdir.joinpath(s + "_authors.tsv")
        outfp_abbreviations = self.outdir.joinpath(s + "_abbreviations.xlsx")
        authors = []
        author_abbreviations = []

        card = u.focal_card(labels=labels)

        spreadsheets = []
        logger.debug("getting focal author sheet")
        spreadsheets.append(
            (author.name, u.create_spreadsheet_focal_author(card, N=N, labels=labels))
        )
        authors.append(
            {
                "AuthorId": author.id,
                "name": author.name,
            }
        )
        if self.abbreviations is not None:
            author_abbreviations.append((author.name, self.get_abbreviations(author)))

        other_author_ids, other_author_metadata = get_other_author_ids(
            u, num_other_authors=100
        )
        num_authors_added = 0
        logger.debug(
            f"getting other authors sheets (max_other_authors: {max_other_authors})"
        )
        for _id in other_author_ids:
            u_unknown = self.initialize(_id)
            card = u_unknown.focal_card(labels=labels)
            if not has_enough_data_term_rank_compare(card):
                logger.debug(f"author {_id} does not have enough data. skipping.")
                continue
            df_unknown = u_unknown.create_spreadsheet_unknown_author(
                card, N=N, labels=labels
            )
            spreadsheets.append((u_unknown.paper_collection.name, df_unknown))
            authors.append(
                {
                    "AuthorId": u_unknown.paper_collection.id,
                    "name": u_unknown.paper_collection.name,
                }
            )
            if self.abbreviations is not None:
                author_abbreviations.append(
                    (
                        u_unknown.paper_collection.name,
                        self.get_abbreviations(u_unknown.paper_collection),
                    )
                )
            num_authors_added += 1
            if num_authors_added == max_other_authors:
                break

        persona_list = [
            ids
            for p, ids in sorted(
                u.ego_partition[u.paper_collection.id].items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
            if len(ids) > 2
        ]
        logger.debug(
            f"attempting to create personas sheets with a list of {len(persona_list)} personas"
        )
        dfs_personas = u.create_spreadsheet_personas(card, persona_list, N=N)
        if dfs_personas is not None:
            #     for i, df_p in enumerate(dfs_personas):
            #         spreadsheets.append((f"Personas {i+1}", df_p))
            for strategy_name, df_p in dfs_personas:
                spreadsheets.append((f"Personas ({strategy_name})", df_p))

        logger.debug(f"writing to {outfp_spreadsheet}")
        # pylint: disable=abstract-class-instantiated
        with pd.ExcelWriter(outfp_spreadsheet) as writer:
            for sheet_name, df in spreadsheets:
                df.to_excel(writer, sheet_name=sheet_name)

        logger.debug(f"writing to {outfp_authors}")
        pd.DataFrame(authors).to_csv(outfp_authors, sep="\t", index=False)

        if len(author_abbreviations) > 0:
            logger.debug(f"writing to {outfp_abbreviations}")
            # pylint: disable=abstract-class-instantiated
            with pd.ExcelWriter(outfp_abbreviations) as writer:
                for sheet_name, df in author_abbreviations:
                    df.to_excel(writer, sheet_name=sheet_name)

        if save_log is True:
            root_logger.removeHandler(fhandler)

    def save_excel_personas_papers_authors(
        self,
        author_id: int,
        increment_random_seed: bool = True,
        labels: Sequence[str] = ["Task", "Method", "Material"],
        N: int = 30,
    ) -> None:
        if increment_random_seed is True:
            self.increment_random_seed()
        u = self.initialize(author_id)
        author = u.paper_collection
        # outfp_spreadsheet = Path(output)
        s = f"compareterms_author{author.id}_{author.name.replace(' ', '')}_personasPapersAndAuthors"
        outfp_spreadsheet = self.outdir.joinpath(s + ".xlsx")

        card = u.focal_card(labels=labels)

        spreadsheets = []
        logger.debug("getting focal author sheet")
        spreadsheets.append(
            (author.name, u.create_spreadsheet_focal_author(card, N=N, labels=labels))
        )

        persona_list = [
            ids
            for p, ids in sorted(
                u.ego_partition[u.paper_collection.id].items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
            if len(ids) > 2
        ]
        logger.debug(
            f"attempting to create personas sheets with a list of {len(persona_list)} personas"
        )
        # dfs_personas = u.create_spreadsheet_personas(card, persona_list, N=N)
        # if dfs_personas is not None:
        #     #     for i, df_p in enumerate(dfs_personas):
        #     #         spreadsheets.append((f"Personas {i+1}", df_p))
        #     for strategy_name, df_p in dfs_personas:
        #         spreadsheets.append((f"Personas ({strategy_name})", df_p))

        dfs_personas_papers_and_authors = (
            u.create_spreadsheet_personas_papers_and_authors(card, persona_list)
        )
        if dfs_personas_papers_and_authors is not None:
            for k, df in dfs_personas_papers_and_authors:
                spreadsheets.append((f"Personas {k}", df))

        logger.debug(f"writing to {outfp_spreadsheet}")
        # pylint: disable=abstract-class-instantiated
        with pd.ExcelWriter(outfp_spreadsheet) as writer:
            for sheet_name, df in spreadsheets:
                df.to_excel(writer, sheet_name=sheet_name)



def get_author_venue_ratio(
    author: AuthorHelper,
    include_ids: Optional[Sequence[int]] = None,
    min_venues: int = 3,
) -> pd.Series:
    venue_ids = (
        pd.concat([author.df_papers.JournalId, author.df_papers.ConferenceSeriesId])
        .dropna()
        .astype(int)
    )
    vc = venue_ids.value_counts()
    paa = author.data.mag_data.paper_authors[["PaperId", "AuthorId"]].drop_duplicates()
    if include_ids is not None:
        paa = paa[paa["AuthorId"].isin(include_ids)]
    paa = paa.merge(
        author.data.mag_data.papers[["PaperId", "JournalId", "ConferenceSeriesId"]],
        on="PaperId",
    )
    paa.dropna(subset=["JournalId", "ConferenceSeriesId"], how="all", inplace=True)
    paa["venue_id"] = paa["JournalId"].fillna(paa["ConferenceSeriesId"])
    paa["venue_id"] = paa["venue_id"].astype(int)
    paa.sort_values("AuthorId", inplace=True)
    author_venues = (
        paa.sort_values("AuthorId").groupby("AuthorId")["venue_id"].agg(list)
    )
    author_venues = author_venues[author_venues.apply(lambda x: len(x) >= min_venues)]
    author_venue_ratio = author_venues.apply(overlap_ratio, args=(vc.index,))
    return author_venue_ratio.sort_values(ascending=False)


def get_other_author_ids(
    u: UserStudyDataTermRankCompare, num_other_authors: int = 5
) -> Tuple[List[int], List[Dict]]:
    strong_coauthors = [a.AuthorId for a in u.paper_collection.coauthors]
    spls = nx.shortest_path_length(u.coauthor_graph, source=u.paper_collection.id)
    to_keep = [
        other_author_id
        for other_author_id, spl in spls.items()
        if (spl >= 3 and spl < 6)
        and other_author_id != u.paper_collection.id
        and other_author_id not in strong_coauthors
    ]
    author_venue_ratio = get_author_venue_ratio(u.paper_collection, include_ids=to_keep)
    author_venue_ratio = author_venue_ratio[author_venue_ratio >= 0.2]
    n = min(num_other_authors, len(author_venue_ratio))
    other_authors = author_venue_ratio.sample(n=n, random_state=u.rng.bit_generator)
    other_author_ids = []
    other_author_metadata = []
    for other_author_id, venue_ratio in other_authors.iteritems():
        other_author_ids.append(other_author_id)
        other_author_metadata.append(
            {
                "venue_ratio": venue_ratio,
                "shortest_path_length": spls[other_author_id],
            }
        )
    return other_author_ids, other_author_metadata


def collect_all_data_for_experiment(
    u: UserStudyDataTermRankCompare,
    outdir: Union[str, Path],
    num_terms: int = 15,
    num_other_authors: int = 5,
    labels: Iterable[str] = ["Method", "Task", "Material"],
) -> None:
    other_author_ids, other_author_metadata = get_other_author_ids(u, num_other_authors)
    logger.debug(f"other_author_ids: {other_author_ids}")
    u.save_spreadsheet_and_metadata(
        outdir=outdir,
        other_author_ids=other_author_ids,
        other_author_metadata=other_author_metadata,
        num_terms=num_terms,
        labels=labels,
    )

    try:
        persona_list = [
            ids
            for p, ids in sorted(
                u.ego_partition[u.paper_collection.id].items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
            if len(ids) > 2
        ]
    except KeyError:
        raise RuntimeError(
            f"could not find author {u.paper_collection.id} in the persona graph"
        )
    u.save_spreadsheet_and_metadata_personas(
        outdir, persona_list, num_terms=num_terms, labels=labels
    )


def specter_cluster_spreadsheet(
    u: UserStudyDataTermRankCompare,
    specter_embeddings_paper_ids: pd.Series,
    specter_embeddings: np.ndarray,
):
    if u.outdir is None:
        raise RuntimeError("need to specify outdir for UserStudyData object")
    strategies = ["strategy_relevance_score", "strategy_tfidf", "strategy_textrank"]
    labels = ["Task", "Method"]
    author = u.paper_collection
    sub_dfs = defaultdict(list)
    persona_names = []
    # score_map = u.paper_collection.df_papers.set_index("PaperId")["relevance_score"]
    spreadsheets = []
    for u_sub in u.yield_specter_cluster_personas():
        card = u_sub.focal_card(labels=labels)
        data = u_sub.get_terms_ranking_from_card(card, N=30)
        for strategy in strategies:
            sub_df = pd.DataFrame(data["Topics"])
            for label in labels:
                if label in data:
                    sub_df[label] = data[label][strategy]
            sub_dfs[strategy].append(sub_df)
        df_papers = u_sub.paper_collection.df_papers[
            ["PaperId", "Rank", "Year", "OriginalTitle", "OriginalVenue"]
        ].sort_values("Rank")
        df_papers = df_papers[
            df_papers["PaperId"].isin(u_sub.paper_collection.dedup_title_paper_ids)
        ]
        sub_dfs["papers"].append(df_papers)

        persona_names.append(u_sub.paper_collection.name)
    for strategy in strategies:
        df = pd.concat(sub_dfs[strategy], axis=1, keys=persona_names)
        sheet_name = f"ranking_{strategy.split('_')[-1]}"
        spreadsheets.append((sheet_name, df))
    spreadsheets.append(
        ("papers", pd.concat(sub_dfs["papers"], axis=1, keys=persona_names))
    )
    outfp = u.outdir.joinpath(
        f"specterAgglomPersonas_author{author.id}_{author.name.replace(' ', '')}.xlsx"
    )
    # print(f"writing to {outfp}")
    # pylint: disable=abstract-class-instantiated
    with pd.ExcelWriter(outfp) as writer:
        for sheet_name, df in spreadsheets:
            df.to_excel(writer, sheet_name=sheet_name)
