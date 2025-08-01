from pathlib import Path
PACKAGE_ROOT = Path(__file__).parent.resolve()
# DATADIR = PACKAGE_ROOT.parent.joinpath('data')
# DATADIR should actually be defined in the .env file

from .config import Config
from .collection_helper import PaperCollectionHelper, AuthorHelper
from .data_helper import DataHelper
from .cards import BridgerCard