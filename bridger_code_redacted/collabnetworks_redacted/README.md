# Backend data processing for Bridger

Repository for research project on Groups and Gaps in Author Collaboration Networks

Author: Jason Portenoy

- [Backend data processing for Bridger](#backend-data-processing-for-bridger)
- [Data processing pipeline](#data-processing-pipeline)
- [Generating data for Bridger](#generating-data-for-bridger)
- [DataHelper documentation](#datahelper-documentation)
  - [Loading data](#loading-data)
  - [Using data](#using-data)
    - [MAG Data](#mag-data)
    - [Mapping between S2 and MAG papers](#mapping-between-s2-and-mag-papers)
    - [Dygie terms extracted from paper titles and abstracts](#dygie-terms-extracted-from-paper-titles-and-abstracts)
    - [Term embeddings](#term-embeddings)
- [Author-level analysis](#author-level-analysis)


# Data processing pipeline

Details of the data processing pipeline, starting with raw data from MAG and S2, can be found at [notes/full_pipeline.md](notes/full_pipeline.md)

# Generating data for Bridger

Once the data is processed, use the `DataHelper` class (in [data_helper.py](data_helper.py)) to generate data for Bridger.

Example:
```python
from data_helper import DataHelper, DEFAULTS_V2
from cards import save_author_card

data_helper = DataHelper.from_defaults(DEFAULTS_V2, min_year=2015, max_year=2021)
# loading the data_helper will take ~40 minutes

outdir = Path('tmp/test_authorcard')
if not outdir.exists():
    outdir.mkdir()

author_id = 00000000
save_author_card(outdir, data_helper, name='test author', author_id=author_id, min_year=2015, max_year=2021)
# saving the author card will take ~30 seconds
# the data will be in `tmp/test_authorcard/testauthor_2015-2021.json`

```

# DataHelper documentation
(written 2020-11-23)

The DataHelper class consolidates the data for author and paper analysis, such as 
+ mapping of S2 paper IDs to MAG paper IDs
+ paper and author metadata from MAG
+ dygie terms from titles and abstracts (tasks, methods, etc.)
+ embeddings for papers (e.g., specter) and terms

A DataHelper instance takes as input a collection of paths to flat data files, 
and optionally a minimum and maximum year to limit the time period.

## Loading data

Using the `from_defaults` class method will load an instance with the most recent data 
including dygie term embeddings. This will take around 30 minutes to load 
and will use a large amount of RAM (around 100-200GB).
```python
# Example usage, from defaults
from data_helper import DataHelper
from util import curr_mem_usage
data_helper = DataHelper.from_defaults(DEFAULTS_V2, min_year=2015, max_year=2021)
print(curr_mem_usage())
```

Alternatively, you can load a slimmer DataHelper with only the data you need:
```python
# Example usage, slim
from data_helper import DataHelper
fnames = defaults['data_fnames']
data_helper = DataHelper(min_year=min_year, max_year=max_year)
data_helper.load_mag_data(fnames['mag_data'])
data_helper.load_s2_mapping(fnames['s2_mapping'])
```
see the `from_defaults` class method for more guidance on how to load data.

## Using data

Once your DataHelper is loaded, the important data can be accessed via the following attributes:
+ `mag_data`: paper, author, affiliation data (and more) from MAG
+ `df_s2_id`: mapping between S2 paper IDs and MAG paper IDs
+ `df_ner`: dygie terms, extracted from titles and abstracts (tasks, methods, etc.)
+ `embeddings` and `embeddings_terms`: embeddings for tasks, methods, materials, metrics

### MAG Data
`mag_data` is a MagData object (see [collabnetworks/mag_data.py](collabnetworks/mag_data.py)). It contains some useful dataframes, such as 
`papers`, `authors`, and `paper_authors`. Entities are identified by `PaperId`, `AuthorId`, `AffiliationId`

### Mapping between S2 and MAG papers
Use the `df_s2_id` attribute:
```python
data_helper.df_s2_id.info()
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 15490732 entries, 0 to 27744598
# Data columns (total 2 columns):
#  #   Column   Dtype
# ---  ------   -----
#  0   PaperId  int64
#  1   s2_id    int64
# dtypes: int64(2)
# memory usage: 354.6 MB
```

### Dygie terms extracted from paper titles and abstracts
Use the `df_ner` attribute:
```python
data_helper.df_ner.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 82688925 entries, 1057 to 233623890
# Data columns (total 9 columns):
#  #   Column           Dtype  
# ---  ------           -----  
#  0   PaperId          int64  
#  1   s2_id            int64  
#  2   label            object 
#  3   term             object 
#  4   softmax_score    float64
#  5   term_cleaned     object 
#  6   term_normalized  object 
#  7   term_display     object 
#  8   term_idx         float64
# dtypes: float64(2), int64(2), object(5)
# memory usage: 6.2+ GB

data_helper.df_ner['label'].value_counts()
# OtherScientificTerm    27486315
# Method                 24596265
# Task                   11437253
# Generic                 9519997
# Material                5818660
# Metric                  3830435
# Name: label, dtype: int64
```

### Term embeddings
The `embeddings` attribute is a numpy array containing embeddings for the terms in `df_ner`. 
Each row is a term. The `embeddings_terms` attribute is a 1d numpy array of the string representations 
of each term, in the same order as in `embeddings`. The `embeddings_terms_to_idx` attribute is 
a dict which maps the string representation of each term to the index in `embeddings` and `embeddings_terms`.


# Author-level analysis

The AuthorHelper object, defined in [collabnetworks/util.py](collabnetworks/util.py), takes as input a MAG AuthorId, a DataHelper instance, 
and optionally a minimum and maximum year.

An author is a collection of papers, weighted by a paper relevance score.

```python
author_id = 0000000000
author = AuthorHelper(
    author_id,
    data=data,
    min_year=2015,
    max_year=2021,
    name='Example Name',
)

print(author.paper_ids)  # list of MAG paper IDs
print(author.paper_weights)  # corresponding weights for papers

card = author.to_card()  # get a BridgerCard object

```