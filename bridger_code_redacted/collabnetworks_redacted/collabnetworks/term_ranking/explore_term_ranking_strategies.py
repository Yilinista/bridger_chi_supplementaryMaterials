# -*- coding: utf-8 -*-

DESCRIPTION = """Try out different strategies for ranking terms"""

import sys, os, time
from typing import List, Union, Optional, Dict
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
import networkx as nx

from ..util import PaperCollectionHelper, AuthorHelper
from ..data_helper import DataHelper

TopTerms = List[str]
TopTermsByStrategy = Dict[str, TopTerms]

class TermRanker(PaperCollectionHelper):

    """Try out different term-ranking strategies"""

    def __init__(self,
            outdir: Union[str, Path],
            min_year: Optional[int] = None,
            max_year: Optional[int] = None,
            *args, **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)
        self.outdir = Path(outdir)
        if self.outdir.exists():
            logger.debug("using output directory: {}".format(self.outdir))
        else:
            logger.debug("creating output directory: {}".format(self.outdir))
            self.outdir.mkdir()
        self.min_year = min_year
        self.max_year = max_year
        logger.debug("getting mapping for terms to embeddings")
        df_terms = self.get_terms_embeddings_mapping()
        logger.debug("getting tfidf data")
        df_terms = self.get_tfidf_data(df_terms)
        self.df_terms = df_terms

        self._similarity_graphs = None  # lazy loading

    @property
    def similarity_graphs(self) -> Dict[str, nx.Graph]:
        if self._similarity_graphs is None:
            logger.debug("getting similarity graphs...")
            self._similarity_graphs = {}
            for lbl, gdf in self.df_terms.groupby('label'):
                self._similarity_graphs[lbl] = self.get_sim_graph(gdf, distance='euclidean')
        return self._similarity_graphs

    @classmethod
    def from_author_obj(cls,
            author_obj: AuthorHelper,
            outdir: Union[str, Path]
            ):
        obj = cls(paper_ids=author_obj.papers.paper_ids,
                    paper_weights=author_obj.papers.paper_weights,
                    data=author_obj.data,
                    outdir=outdir,
                    min_year=author_obj.min_year,
                    max_year=author_obj.max_year)
        return obj

    @classmethod
    def from_author_id(cls,
            author_id: Union[str, List[str]],
            data: DataHelper,
            outdir: Union[str, Path],
            min_year: Optional[int] = None,
            max_year: Optional[int] = None
            ):
        author = AuthorHelper(author_id, data=data, min_year=min_year, max_year=max_year)
        return cls.from_author_obj(author, outdir=outdir)

    def strategy_random(self, label: str, N: int = 20) -> List[str]:
        df = self.df_terms[self.df_terms['label']==label]
        return df.sample(N, random_state=1)['term_display'].tolist()

    def _strategy_sorted_col(self,
            df:pd.DataFrame,
            colname: str,
            N: int=20,
            ascending: bool = False
            ) -> List[str]:
        return df.sort_values(colname, ascending=ascending).head(N)['term_display'].tolist()

    def strategy_freq(self, label: str, N: int = 20) -> List[str]:
        df = self.df_terms[self.df_terms['label']==label]
        return self._strategy_sorted_col(df, 'freq', N, ascending=False)

    def strategy_relevance_score(self, label: str, N: int = 20) -> List[str]:
        df = self.df_terms[self.df_terms['label']==label]
        return self._strategy_sorted_col(df, 'relevance_score', N, ascending=False)

    def strategy_tfidf(self, label: str, N: int = 20) -> List[str]:
        df = self.df_terms[self.df_terms['label']==label]
        return self._strategy_sorted_col(df, 'term_tfidf', N, ascending=False)

    def get_textrank(self, G: nx.Graph) -> Dict[str, float]:
        return nx.pagerank_numpy(G, weight='weight')

    def strategy_textrank(self, label: str, N: int = 20) -> List[str]:
        G = self.similarity_graphs[label]
        textrank = self.get_textrank(G)
        textrank_sorted = pd.Series(textrank).sort_values(ascending=False)
        return textrank_sorted.head(N).index.tolist()

    def get_top_tasks_with_similarity_threshold(self, terms_sorted_by_relevance: List[str], G: nx.Graph, similarity_threshold: float, N: int = 20) -> List[str]:
        top_tasks = []
        logger.debug("getting top tasks with similarity threshold: using similarity threshold {}".format(similarity_threshold))
        for term in terms_sorted_by_relevance:
            if len(top_tasks) == 0:
                top_tasks.append(term)
                continue
            last_term = top_tasks[-1]
            sim = G[term][last_term]['weight']
            if sim > similarity_threshold:
                pass
            else:
                top_tasks.append(term)
            if len(top_tasks) == N:
                break
        return top_tasks

    def strategy_similarity_threshold(self, label: str, similarity_threshold: float, N: int = 20) -> List[str]:
        df = self.df_terms[self.df_terms['label']==label].sort_values('relevance_score', ascending=False)
        terms = df['term_display'].tolist()
        G = self.similarity_graphs[label]
        top_tasks = self.get_top_tasks_with_similarity_threshold(terms, G, similarity_threshold, N)
        return top_tasks

    def all_strategy_methods(self):
        return [getattr(self, _method_name) for _method_name in dir(self) if _method_name.startswith('strategy')]

    def get_top_terms_by_strategy(self, label: str, N: int, similarity_threshold: Optional[float] = None) -> TopTermsByStrategy:
        strategy_methods = self.all_strategy_methods()
        top_terms_by_strategy = {}
        for _f in strategy_methods:
            strategy_name = _f.__name__
            if 'similarity_threshold' in strategy_name and similarity_threshold is not None:
                topterms = _f(label, similarity_threshold, N=N)
            else:
                topterms = _f(label, N=N)
            top_terms_by_strategy[strategy_name] = topterms
        return top_terms_by_strategy

    def get_top_terms_all_labels(self, similarity_threshold: float, N: int = 20) -> Dict[str, TopTermsByStrategy]:
        N_max = N
        top_terms_all_labels = {}
        for label, gdf in self.df_terms.groupby('label'):
            N = min(N_max, len(gdf))
            top_terms_all_labels[label] = self.get_top_terms_by_strategy(label, N=N, similarity_threshold=similarity_threshold)
        return top_terms_all_labels

    def get_top_terms_df(self,
                         top_terms_by_strategy: TopTermsByStrategy,
                         N: int = 20) -> pd.DataFrame:
        """Get dataframe of top terms for a single label"""
        # need to make sure all columns have the same number of rows (even if some are empty)
        _data = {}
        for strategy, topterms in top_terms_by_strategy.items():
            _topterms = []
            for i in range(N):
                try:
                    _topterms.append(topterms[i])
                except IndexError:
                    _topterms.append(np.nan)
            _data[strategy] = _topterms
        return pd.DataFrame(_data)

    def get_all_dfs(self, top_terms_all_labels: Dict[str, TopTermsByStrategy], randomize: bool = False) -> Dict[str, pd.DataFrame]:
        from collections import OrderedDict
        out = OrderedDict()
        # randomized_columns = None
        rename_columns = OrderedDict()
        for label, top_terms_by_strategy in top_terms_all_labels.items():
            N = max(len(x) for x in top_terms_by_strategy.values())
            df_sheet = self.get_top_terms_df(top_terms_by_strategy, N=N)
            if randomize:
                if len(rename_columns) == 0:
                    random_seed = 1
                    logger.debug("randomizing columns for blind evaluation (random seed is {})".format(random_seed))
                    random_state = np.random.RandomState(seed=1)
                    randomized_columns = random_state.permutation(df_sheet.columns)
                    for i, colname in enumerate(randomized_columns):
                        rename_columns[colname] = f"strategy_{i+1}"
                    logger.debug("renaming columns using rename_columns dict: {}".format(rename_columns))
                df_sheet = df_sheet.rename(columns=rename_columns)
                df_sheet = df_sheet[sorted(df_sheet.columns)]
            out[label] = df_sheet
        return out

    def save_excel_multi_sheet(self, outfpath: Union[str, Path], output_dfs: Dict[str, pd.DataFrame], randomize: bool = False) -> None:
        outfpath = Path(outfpath)
        logger.debug("writing to file: {}".format(outfpath))
        with pd.ExcelWriter(outfpath) as writer:
            for label, df_sheet in output_dfs.items():
                df_sheet.to_excel(writer, sheet_name=label, index=False)



def main(args):
    outdir = Path(args.outdir)
    from data.sample_ids import sample_authors, sample_authors_dict
    author_id = sample_authors_dict[args.name]
    data_helper = DataHelper.from_defaults()
    tr = TermRanker.from_author_id(author_id, data=data_helper, outdir=outdir, min_year=args.min_year, max_year=args.max_year)
    if args.min_year or args.max_year:
        outfpath = tr.outdir.joinpath(f"{args.name.replace(' ', '')}_{args.min_year}-{args.max_year}_term_ranking.xlsx")
    else:
        outfpath = tr.outdir.joinpath(f"{args.name.replace(' ', '')}_term_ranking.xlsx")
    top_terms_all_labels = tr.get_top_terms_all_labels(args.similarity_threshold, N=args.N)
    output_dfs = tr.get_all_dfs(top_terms_all_labels, randomize=args.blind)
    tr.save_excel_multi_sheet(outfpath, output_dfs)

if __name__ == "__main__":
    total_start = timer()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    logger.info("pid: {}".format(os.getpid()))
    import argparse
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("name", type=str, help="author name. Must exist within the sample_authors_dict")
    parser.add_argument("outdir", help="output directory")
    parser.add_argument("--min-year", type=int, default=None, help="filter out papers published before this year (will include this year)")
    parser.add_argument("--max-year", type=int, default=None, help="filter out papers published this year and forward (will not include this year)")
    parser.add_argument("--similarity-threshold", type=float, default=0.4, help="filter out papers published this year and forward (will not include this year)")
    parser.add_argument("--blind", action='store_true', help="randomize and blind the column names")
    parser.add_argument("-N", "--N", type=int, default=20, help="maximum number of top terms to get for each strategy")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
