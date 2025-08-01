# -*- coding: utf-8 -*-

DESCRIPTION = """output JSON data for Bridger network visualization"""

import sys, os, time, pickle, json
from typing import Optional, Hashable, Container, Mapping, Dict, List, Union, Collection
from pathlib import Path
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

import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.metrics.pairwise import cosine_similarity

from clustering import get_cl_to_authors, ClusterHelper
from mag_data import MagData

from util import get_root_dir
ROOT_DIR = get_root_dir()

# load mag data
# load s2 mapping
# load coauthorship graph
# do edge thresholding (remove edges <= 0.02)
# load cluster memberships
# load df_ner
# load ner group counts
# process ner data for tfidf
# get output graph data
# save to JSON file

NodeId = Hashable
PartitionId = Hashable
Component = Container[NodeId]
OverlappingPartition = Dict[NodeId, List[PartitionId]]

DEFAULTS = {
    'data_fnames': {
        'mag_data': os.path.join(ROOT_DIR, 'data/interim/mag-2020-07-02_CSsubset/'),
        's2_mapping': os.path.join(ROOT_DIR, 'data/interim/mag-2020-07-02_CSsubset/mag_PaperId_to_s2_id.parquet'),
        'coauthorship_edgelist': os.path.join(ROOT_DIR, 'data/coauthor_CSsubset/coauthor_2015-2020_minpubs3_collabweighted/coauthor_edgelist.csv'),
        'memberships': os.path.join(ROOT_DIR, 'data/coauthor_CSsubset/coauthor_2015-2020_minpubs3_collabweighted/infomap_runs/memberships_minWeight02_seed00030.pickle'),
        'ner_terms': os.path.join(ROOT_DIR, 'data/dygie_predictions/predicted_terms_20200816T131632/df_ner_cleaned.parquet'),
        'ner_grp_counts': os.path.join(ROOT_DIR, 'data/dygie_predictions/predicted_terms_20200816T131632/df_term_display_counts.parquet'),
        'embeddings': os.path.join(ROOT_DIR, 'data/dygie_predictions/predicted_terms_20200816T131632/embeddings_cs_roberta_finetuneSTS_20200824T122808/embeddings_dedup.npy'),
        'embeddings_terms': os.path.join(ROOT_DIR, 'data/dygie_predictions/predicted_terms_20200816T131632/embeddings_cs_roberta_finetuneSTS_20200824T122808/terms_dedup.npy')
    },
    'min_year': 2015,
    'max_year': 2020
}

class AuthorClustersVisDataGetter:

    """Collect data for the clusters that an author belongs to"""

    def __init__(self,
            min_year: int = DEFAULTS['min_year'],
            max_year: int = DEFAULTS['max_year'],
            mag_data: Optional[MagData] = None,
            df_s2_id: Optional[pd.DataFrame] = None,
            G: Optional[nx.Graph] = None,
            memberships: Optional[OverlappingPartition] = None,
            df_ner: Optional[pd.DataFrame] = None,
            grp_counts: Optional[pd.DataFrame] = None,
            embeddings: Optional[np.ndarray] = None,
            embeddings_terms: Optional[np.ndarray] = None
            ) -> None:
        """TODO: to be defined. """
        self.min_year = min_year
        self.max_year = max_year
        self.mag_data = mag_data
        self.df_s2_id = df_s2_id
        self.G = G
        self.memberships = memberships
        self.df_ner = df_ner
        self.grp_counts = grp_counts
        self.embeddings = embeddings
        self.embeddings_terms = embeddings_terms

        self.cl_to_authors = None

    @classmethod
    def from_defaults(cls, defaults=DEFAULTS, **kwargs):
        """load object from defaults (see source code)

        """
        obj = cls(min_year=defaults['min_year'],
                max_year=defaults['max_year'],
                **kwargs)
        fnames = defaults['data_fnames']
        if obj.mag_data is None:
            obj.load_mag_data(fnames['mag_data'])
        if obj.df_s2_id is None:
            obj.load_s2_mapping(fnames['s2_mapping'])
        if obj.G is None:
            obj.load_coauthorship_graph(fnames['coauthorship_edgelist'])
        if obj.memberships is None:
            obj.load_cluster_memberships(fnames['memberships'])
        if obj.df_ner is None:
            obj.load_ner_data(fnames['ner_terms'], fnames['ner_grp_counts'])
        if obj.embeddings is None:
            obj.load_embeddings(fnames['embeddings'], fnames['embeddings_terms'])
        obj.prepare_ner_tfidf()
        return obj

    def load_mag_data(self,
            path_to_data: str,
            tablenames: Optional[List[str]] = None
            ) -> None:
        logger.debug("loading MagData from {}".format(path_to_data))
        self.mag_data = MagData(path_to_data, tablenames=tablenames)

    def load_s2_mapping(self,
            path_to_data: str
            ) -> None:
        logger.debug("loading s2 mapping from file {}".format(path_to_data))
        self.df_s2_id = pd.read_parquet(path_to_data)

    def load_coauthorship_graph(self,
            path_to_edgelist_csv: str):
        logger.debug("loading coauthorship graph from file {}".format(path_to_edgelist_csv))
        df_edgelist = pd.read_csv(path_to_edgelist_csv)
        G = nx.from_pandas_edgelist(df_edgelist, source='AuthorId_1', target='AuthorId_2', edge_attr=True)
        # With edge weight thresholding
        weight_thresh = 0.02
        logger.debug("removing edges with weight <= {}".format(weight_thresh))
        edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w <= weight_thresh]
        G.remove_edges_from(edges_to_remove)
        logger.debug("coauthorship graph has {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
        self.G = G

    def load_cluster_memberships(self,
            path_to_pickle: str
            ) -> None:
        logger.debug("loading cluster memberships from file {}".format(path_to_pickle))
        fpath = Path(path_to_pickle)
        self.memberships = pickle.loads(fpath.read_bytes())
        self.cl_to_authors = get_cl_to_authors(self.memberships)
        logger.debug("there are {} clusters".format(len(self.cl_to_authors)))

    def load_ner_data(self,
            path_to_terms: str,
            path_to_counts: str
            ) -> None:
        logger.debug("loading ner terms from {}".format(path_to_terms))
        self.df_ner = pd.read_parquet(path_to_terms)
        logger.debug("dataframe shape: {}".format(self.df_ner.shape))
        logger.debug("loading ner term counts from {}".format(path_to_counts))
        self.grp_counts = pd.read_parquet(path_to_counts)
        logger.debug("counts dataframe shape: {}".format(self.grp_counts.shape))

    def load_embeddings(self,
            path_to_embeddings: str,
            path_to_terms: str
            ) -> None:
        logger.debug("loading embeddings and corresponding terms")
        self.embeddings = np.load(path_to_embeddings)
        self.embeddings_terms = np.load(path_to_terms, allow_pickle=True)
        self.embeddings_terms_to_idx = {val: idx[0] for idx, val in np.ndenumerate(self.embeddings_terms)}

    def prepare_ner_tfidf(self) -> None:
        # TODO depcrecate
        logger.debug("preparing ner data for tfidf...")
        self.df_tfidf = self.df_s2_id.merge(self.df_ner, how='inner', on='s2_id')
        self.N_tfidf = self.df_tfidf['PaperId'].nunique()
        logger.debug("done. {} unique terms overall".format(self.N_tfidf))

    def _run_textrank(self, df_terms: pd.DataFrame) -> pd.Series:
        textgraph = nx.Graph()
        for (_, row1), (_, row2) in combinations(df_terms.iterrows(), 2):
            a = self.embeddings[row1['term_idx']]
            b = self.embeddings[row2['term_idx']]
            csim = cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))
            cdist = csim.item()
            textgraph.add_edge(row1['term_display'], row2['term_display'], weight=cdist)
        textrank = pd.Series(nx.pagerank_numpy(textgraph, weight='weight')).sort_values(ascending=False)
        return textrank

    def _get_tfidf_data(self, papers: pd.DataFrame) -> Dict[str, List[str]]:
        from util import _tfidf_apply
        papers.name = 'cl_score'
        x = self.df_tfidf.merge(papers, left_on='PaperId', right_index=True)
        cl_terms = x.groupby(['label', 'term_display'])['cl_score'].sum().reset_index()
        cl_terms = cl_terms.merge(self.grp_counts, how='inner', on=['label', 'term_display'])
        cl_terms.rename(columns={'term_count': 'all_count'}, inplace=True)
        cl_terms['term_tfidf'] = cl_terms.apply(_tfidf_apply, N=self.N_tfidf, tf_colname='cl_score', axis=1)
        out = {}
        for lbl, gdf in cl_terms.groupby('label'):
            out[lbl] = gdf.sort_values('term_tfidf', ascending=False)['term_display'].head(10).tolist()
        return out

    def _get_textrank_data(self, papers: pd.DataFrame) -> Dict[str, List[str]]:
        papers.name = 'cl_score'
        x = self.df_tfidf.merge(papers, left_on='PaperId', right_index=True)
        cl_terms = x.groupby(['label', 'term_display'])['cl_score'].sum().reset_index()
        # cl_terms = cl_terms.merge(self.grp_counts, how='inner', on=['label', 'term_display'])
        cl_terms.rename(columns={'term_count': 'all_count'}, inplace=True)
        # cl_terms['term_tfidf'] = cl_terms.apply(_tfidf_apply, N=self.N_tfidf, tf_colname='cl_score', axis=1)
        cl_terms['term_idx'] = cl_terms['term_display'].map(self.embeddings_terms_to_idx)
        out = {}
        for lbl, gdf in cl_terms.groupby('label'):
            textrank = self._run_textrank(gdf)
            out[lbl] = textrank.head(10).index.tolist()
        return out

    def get_author_cluster_data(self, author_id) -> None:
        if self.cl_to_authors is None:
            self.cl_to_authors = get_cl_to_authors(self.memberships)
        # check if author_id is a collection (e.g., list) of ids, or if it is a single id
        if isinstance(author_id, str) or not isinstance(author_id, Collection):
            author_id = [author_id]
        logger.debug("getting data for author ID: {} (NormalizedName: {})".format(author_id, [ self.mag_data.author_lookup(id_).get('NormalizedName') for id_ in author_id if self.mag_data.author_lookup(id_) is not None]))
        cl_ids = []
        for id_ in author_id:
            this_cl_id = self.memberships.get(id_)
            if this_cl_id:
                cl_ids.extend(self.memberships[id_])
        cl_ids = list(set(cl_ids))
        logger.debug("{} clusters with this author".format(len(cl_ids)))
        logger.debug("cluster IDs: {}".format(cl_ids))
        logger.debug("sizes of these clusters: {}".format([len(self.cl_to_authors[cl_id]) for cl_id in cl_ids]))
        clusters = [ClusterHelper(cl_id, self.G.subgraph(self.cl_to_authors[cl_id]), self.mag_data, min_year=self.min_year, max_year=self.max_year) for cl_id in cl_ids]
        G_out = nx.MultiGraph()
        for cl in clusters:
            logger.debug("collecting data for cluster {}".format(cl.id))
            authors = cl.authors
            papers = cl.get_relevance_scores()
            if len(papers) < 2:
                logger.debug("this cluster has too few papers. skipping...")
                continue
            dygie_terms = self._get_textrank_data(papers)

            # get score of this cluster relative to the author of interest
            score = 0
            for rank, id_ in enumerate(authors.AuthorId.tolist()):
                if id_ in author_id:
                    score = ( len(authors) - rank ) / len(authors)
                    break

            node_attrs = {
                    'affiliations': ["Test affiliation"] * 5,
                    'authors': authors.DisplayName.head(5).tolist(),
                    'numAuthors': len(authors),
                    'numPapers': len(papers),
                    'topics': ['Test topic'] * 5,
                    'score': score,
                    'dygie_terms': dygie_terms
                    }
            G_out.add_node(cl.id, **node_attrs)
        self.G_out = G_out

    def write_json(self, outfname: str) -> None:
        try:
            outf = Path(outfname)
            outf.write_text(json.dumps(json_graph.node_link_data(self.G_out)))
        except AttributeError:
            raise RuntimeError("must collect data by running get_author_cluster_data()")




def main(args):
    pass

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
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
