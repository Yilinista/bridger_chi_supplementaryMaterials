# -*- coding: utf-8 -*-

DESCRIPTION = """utils for dygie (methods, tasks, and relations extraction)"""

import sys, os, time, json
from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

from typing import Iterable, Iterator, List, Dict, Tuple, NamedTuple, Optional, Union

import logging
root_logger = logging.getLogger()
logger = root_logger.getChild(__name__)

class DygieEntity:
    """Represents an entity, such as a "Method" or "Task" (as specified by the `label`)"""

    def __init__(self, 
                doc: 'DygieDoc', 
                idx_start: int, 
                idx_end: int, 
                label: str, 
                raw_score: float, 
                softmax_score: float
                ) -> None:
        self.doc = doc
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.label = label
        self.raw_score = raw_score
        self.softmax_score = softmax_score

        self.toks = self.doc.sents[idx_start:idx_end+1]
        self.term = " ".join(self.toks)

class DygieRelation:
    """Represents a relation, such as a "USED-FOR" relation (as specified by the `label`). `src` and `dst` are terms"""

    def __init__(self, 
                doc: 'DygieDoc', 
                idx_start_1: int, 
                idx_end_1: int, 
                idx_start_2: int, 
                idx_end_2: int, 
                label: str, 
                raw_score: float, 
                softmax_score: float
                ) -> None:
        self.doc = doc
        self.idx_start_1 = idx_start_1
        self.idx_end_1 = idx_end_1
        self.idx_start_2 = idx_start_2
        self.idx_end_2 = idx_end_2
        self.label = label
        self.raw_score = raw_score
        self.softmax_score = softmax_score
        self.src = self.doc.get_entity_by_idx(self.idx_start_1, self.idx_end_1)
        self.dst = self.doc.get_entity_by_idx(self.idx_start_2, self.idx_end_2)

        self.src_toks = self.doc.sents[self.idx_start_1:self.idx_end_1+1]
        self.src_term = " ".join(self.src_toks)
        self.dst_toks = self.doc.sents[self.idx_start_2:self.idx_end_2+1]
        self.dst_term = " ".join(self.dst_toks)

class DygieDoc:
    """Represents a document analyzed by dygie, with all of the predicted entities and relations within"""

    def __init__(self, 
                doc: Dict,
                src_file: Optional[Union[Path, str]] = None
                ) -> None:
        self.doc = doc
        if src_file is not None:
            self.src_file = Path(src_file)
        self.id = doc['doc_key']
        self.sents = []
        for sent in doc['sentences']:
            self.sents.extend(sent)

        self.ner = self.load_entities()
        self.relations = self.load_relations(labels=['USED-FOR'])

    def load_entities(self, 
            labels: Optional[Iterable[str]]=None
            ) -> List[DygieEntity]:
        ents = []
        for p_sent in self.doc['predicted_ner']:
            if p_sent:
                for start_tok, end_tok, label, raw_score, softmax_score in p_sent:
                    if labels and (label not in labels):
                        continue
                    ents.append(DygieEntity(self, start_tok, end_tok, label, raw_score, softmax_score))
        return ents

    def load_relations(self, 
            labels: Optional[Iterable[str]]=None
            ) -> List[DygieRelation]:
        rels = []
        for p_sent in self.doc['predicted_relations']:
            if p_sent:
                for start_tok_1, end_tok_1, start_tok_2, end_tok_2, label, raw_score, softmax_score in p_sent:
                    if (labels) and (label not in labels):
                        continue
                    rels.append(DygieRelation(self, start_tok_1, end_tok_1, start_tok_2, end_tok_2, label, raw_score, softmax_score))
        return rels

    def get_entity_by_idx(self, idx_start, idx_end) -> Union[DygieEntity, None]:
        """Search for an entity within this doc by the start/end indices of the tokens"""
        for ent in self.ner:
            if ent.idx_start == idx_start and ent.idx_end == idx_end:
                return ent
        return None
                        


def yield_dygie_docs(files: Iterable[Union[Path, str]]
                    ) -> Iterator[DygieDoc]:
    """yield dygie doc objects

    :files: list of JSONL file paths
    :yields: DygieDoc objects

    """
    for fp in files:
        fp = Path(fp)
        with fp.open() as f:
            predictions = [json.loads(line) for line in f]
            for d in predictions:
                doc = DygieDoc(d, src_file=fp)
                yield doc
