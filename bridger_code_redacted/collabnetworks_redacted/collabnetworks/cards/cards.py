# -*- coding: utf-8 -*-

DESCRIPTION = """Save author cards to JSON files"""

import sys, os, time, json, tarfile
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import dataclasses
from dataclasses import dataclass
import simplejson
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


def tar_output(dirpath: Path, tarfpath: Union[Path, str]):
    with tarfile.open(tarfpath, "w:gz") as tar:
        tar.add(dirpath, arcname=dirpath.name)


def save_author_card(
    outdir: Union[str, Path],
    data: "DataHelper",
    name: str,
    author_id,
    s2_id: Optional[int] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    overwrite: bool = True,
    google_cloud_dir: Optional[str] = None,
    google_client: Optional["google.cloud.storage.Client"] = None,
) -> None:
    """Save an author card, and optionally upload it to google cloud storage

    Args:
        outdir (Union[str, Path]): output directory (local)
        data (DataHelper): DataHelper object
        name (str): name of the author
        author_id ([type]): MAG author ID
        s2_id (int, optional): S2 Author ID. Defaults to None.
        min_year (int, optional): Exclude papers before this year. Default is to include all years.
        max_year (int, optional): Exclude papers this year and after. Default is to include all years.
        overwrite (bool, optional): Overwrite existing files. Defaults to True.
        google_cloud_dir (str, optional): subdirectory under comention/author/visgraphdata in
            google cloud storage. Default (None) is to not upload to google cloud storage
        google_client (optional): Google Cloud client object. Needed if `google_cloud_dir` is not None
    """
    from ..collection_helper import AuthorHelper
    from ..util import get_root_dir, upload_to_google, google_save_directory

    outdir = Path(outdir)
    logger.debug(f"Getting data for {name}")
    outfname = outdir.joinpath(f"{name.replace(' ', '')}_{min_year}-{max_year}.json")
    if overwrite is False and outfname.exists():
        logger.info(f"{outfname} already exists. skipping...")
        return
    try:
        author = AuthorHelper(
            author_id,
            data=data,
            min_year=min_year,
            max_year=max_year,
            name=name,
            s2_id=s2_id,
        )
        card = author.to_card()
        card.to_json(outfname)
        if google_cloud_dir is not None:
            logger.debug("uploading to google cloud storage")
            upload_to_google(
                outfname,
                google_client,
                sub_subdir=google_cloud_dir,
            )
    except ValueError as e:
        logger.exception(
            f"Exception encountered for author {name} (magId: {author_id}). skipping...\n{e}"
        )


@dataclass
class DygieTerms:
    Method: List[str]
    Task: List[str]
    Material: Optional[List[str]] = None
    Metric: Optional[List[str]] = None


@dataclass
class MagTopic:
    FieldOfStudyId: int
    DisplayName: str
    Level: int
    Score: float


@dataclass
class MagAuthor:
    AuthorId: Union[int, List[int]]
    DisplayName: str


@dataclass
class BridgerCardDistance:
    focalId: Union[str, int]
    Method: Optional[float] = None
    Task: Optional[float] = None
    Material: Optional[float] = None
    Metric: Optional[float] = None
    specter: Optional[float] = None

@dataclass
class PaperDistance:
    focalId: Union[str, int]
    distance: float

@dataclass
class Paper:
    title: str
    year: int
    authors: List[MagAuthor]
    venue: Optional[str] = None
    Rank: Optional[int] = None
    url: Optional[str] = None
    dygie_terms: Optional[DygieTerms] = None
    topics: Optional[List[MagTopic]] = None
    mag_id: Optional[int] = None
    s2Id: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    specter_distance: Optional[List[PaperDistance]] = None


@dataclass
class BridgerCardDetails:
    id: Union[str, int]
    papers: List[Paper]
    authors: List[str]
    affiliations: List[str]
    type: Optional[str] = None
    dygie_terms: Optional[DygieTerms] = None
    topics: Optional[List[MagTopic]] = None
    coauthors: Optional[List[MagAuthor]] = None

    def to_json(self, fname: Union[str, Path]):
        fname = Path(fname)
        logger.debug("writing to {}".format(fname))
        out = dataclasses.asdict(self)
        fname.write_text(simplejson.dumps(out, ignore_nan=True))


@dataclass
class BridgerCard:
    """Class for a card (node) in the Bridger network visualization"""

    id: Union[str, int]
    type: Optional[str] = None
    authors: Optional[List[str]] = None
    topics: Optional[List[MagTopic]] = None
    affiliations: Optional[List[str]] = None
    score: Optional[float] = None
    numAuthors: Optional[int] = None
    numPapers: Optional[int] = None
    dygie_terms: Optional[DygieTerms] = None
    s2Id: Optional[int] = None
    author_ids: Optional[Union[List[int], List[str]]] = None  # MAG author IDs
    details: Optional[Any] = None
    displayName: Optional[str] = None
    coauthors: Optional[List[MagAuthor]] = None
    papers: Optional[List[Paper]] = None  # only top 5 papers

    def to_json(self, fname: Union[str, Path]):
        fname = Path(fname)
        logger.debug("writing to {}".format(fname))
        # node = self.to_node_attr_dict()
        node = dataclasses.asdict(self)
        fname.write_text(simplejson.dumps(node, ignore_nan=True))


@dataclass
class DygieTermsSimilar(DygieTerms):
    focalId: Optional[Union[str, int]] = None


@dataclass
class BridgerCardSimilar(BridgerCard):
    """A BridgerCard with extra attributes relating to similarity to a different (focal) card"""

    distance: Optional[List[BridgerCardDistance]] = None
    simTerms: Optional[List[DygieTermsSimilar]] = None

    def __post_init__(self) -> None:
        # if distance and simTerms are actually lists of dicts (as may be the case if created from a json file),
        # need to convert
        #
        # We don't need this if using collabnetworks.util.dataclass_from_dict()

        if self.distance is not None:
            new_distance = []
            for x in self.distance:
                if isinstance(x, dict):
                    new_distance.append(BridgerCardDistance(**x))
                else:
                    new_distance.append(x)
            self.distance = new_distance
        
        if self.simTerms is not None:
            new_simTerms = []
            for x in self.simTerms:
                if isinstance(x, dict):
                    new_simTerms.append(DygieTermsSimilar(**x))
                else:
                    new_simTerms.append(x)
            self.simTerms = new_simTerms

    def distance_focalIds(self) -> List[Union[str, int]]:
        if self.distance is None:
            return []
        return [x.focalId for x in self.distance]

    def simTerms_focalIds(self) -> List[Union[str, int]]:
        if self.simTerms is None:
            return []
        return [x.focalId for x in self.simTerms]


@dataclass
class BridgerCardTermRankCompare(BridgerCard):
    """A BridgerCard with extra attributes for comparing term ranking strategies"""
    
    dygie_terms_rank_compare: Dict[str, DygieTerms] = dataclasses.field(default_factory=dict)

