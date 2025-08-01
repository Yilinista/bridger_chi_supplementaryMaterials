# Full pipeline

This will document the full pipeline, starting with downloading the most recent data from MAG and S2.

- [Full pipeline](#full-pipeline)
  - [Set data directory](#set-data-directory)
  - [Convert MAG files to parquet format](#convert-mag-files-to-parquet-format)
  - [Identify the subset of papers to use for the corpus](#identify-the-subset-of-papers-to-use-for-the-corpus)
  - [Get specter embeddings](#get-specter-embeddings)
  - [Get MAG data subsets](#get-mag-data-subsets)
  - [Extract terms (methods, tasks, etc.) from titles and abstracts](#extract-terms-methods-tasks-etc-from-titles-and-abstracts)
    - [Get abstracts](#get-abstracts)
    - [Format data for dygie](#format-data-for-dygie)
    - [Run dygie predictions](#run-dygie-predictions)
    - [Parse dygie predictions](#parse-dygie-predictions)
    - [Normalize terms](#normalize-terms)
    - [Get term embeddings](#get-term-embeddings)
    - [Final processing](#final-processing)
    - [Get average embeddings](#get-average-embeddings)
  - [Identify groups by clustering the co-authorship network](#identify-groups-by-clustering-the-co-authorship-network)
    - [Construct co-authorship network](#construct-co-authorship-network)
    - [TODO: Run local clustering on ego-nets; create persona graph](#todo-run-local-clustering-on-ego-nets-create-persona-graph)
    - [TODO: Run global clustering on persona graph](#todo-run-global-clustering-on-persona-graph)
  - [TF-IDF](#tf-idf)

## Set data directory
```shell
# need to set this first, and again if you use new shells
# use absolute paths, and no trailing slashes
MAG_DATADIR=<path_to_mag_base_directory>
MAG_SUBDIR_NAME=mag-2021-03-01

# new bridger data directory will be created
# e.g., data/computer_science_papers_20201002
BRIDGER_DATADIR=<path_to_new_bridger_data_directory>

# existing bridger data directory
OLD_BRIDGER_DATADIR=<path_to_old_bridger_data_directory>
```

## Convert MAG files to parquet format
+ This takes about 4 hours and creates ~100-200GB new data.
```shell
# in directory: data/mag
# with virtualenv activated (source venv/bin/activate)
NOW=$(date +'%Y%m%dT%H%M'); mkdir $MAG_DATADIR/parquet/$MAG_SUBDIR_NAME && python csv_to_parquet.py $MAG_DATADIR/$MAG_SUBDIR_NAME $MAG_DATADIR/parquet/$MAG_SUBDIR_NAME --spark-mem 500g --debug >& $MAG_DATADIR/parquet/$MAG_SUBDIR_NAME/csv_to_parquet_$NOW.log &
```

## Identify the subset of papers to use for the corpus
+ This gets papers labeled as Computer Science in either MAG or S2 (only papers that have MAG ID).
+ This takes about 15 minutes and uses 1-2GB.
```shell
# in collabnetworks directory
# with virtualenv activated (source venv/bin/activate)
NOW=$(date +'%Y%m%dT%H%M'); mkdir $BRIDGER_DATADIR && python scripts/data_pipeline/pipeline010-get_cs_papers.py $MAG_DATADIR/parquet/$MAG_SUBDIR_NAME $S2_ID_FILENAME $BRIDGER_DATADIR/computer_science_papers.parquet --out-ids $BRIDGER_DATADIR/computer_science_paper_ids.txt --debug >& logs/get_cs_papers_$NOW.log &
```

Collect all the S2 paper IDs:
```shell
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline011-get_s2ids.py $BRIDGER_DATADIR/computer_science_papers.parquet $BRIDGER_DATADIR/computer_science_papers_s2ids.csv.gz --debug >& logs/get_s2ids_$NOW.log &

```

## Get specter embeddings
+ TODO
```shell
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline015-get_specter_embeddings.py $BRIDGER_DATADIR/computer_science_papers_s2ids.csv.gz $BRIDGER_DATADIR/specter_embeddings --existing $OLD_BRIDGER_DATADIR/specter_embeddings --debug >& logs/get_specter_embeddings_$NOW.log &
```

## Get MAG data subsets
+ This takes about 2 hours. Outputs about 15GB.
```shell
# in collabnetworks directory
# with virtualenv activated (source venv/bin/activate)
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline020-get_mag_subsets.py $BRIDGER_DATADIR/computer_science_paper_ids.txt $BRIDGER_DATADIR/${MAG_SUBDIR_NAME}_CSsubset --datadir $MAG_DATADIR/parquet/$MAG_SUBDIR_NAME --debug >& logs/get_mag_subsets_$NOW.log &
```

## Extract terms (methods, tasks, etc.) from titles and abstracts

### Get abstracts
+ This takes 1-2 days and outputs ~6 GB (titles and abstracts).
  + The time can be reduced by excluding papers that have already been processed (using `--existing` option)
```shell
# in collabnetworks directory
# with virtualenv activated (source venv/bin/activate)
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline030-get_abstracts.py $BRIDGER_DATADIR/computer_science_papers.parquet $BRIDGER_DATADIR/computer_science_papers_abstracts.parquet --existing $OLD_BRIDGER_DATADIR/final_processed/dygie_terms_to_s2_id_softmaxThreshold0.90.parquet --debug >& logs/get_abstracts_$NOW.log &
```

### Format data for dygie
+ This splits up the titles and abstracts into multiple files for multiprocessing, then parses them using scispacy.
+ This takes about 2 hours and outputs ~17-18GB.
  + 10677125 records
```shell
NOW=$(date +'%Y%m%dT%H%M'); mkdir $BRIDGER_DATADIR/dygie_predictions && python scripts/data_pipeline/pipeline040-multiprocessing_format_dygie.py $BRIDGER_DATADIR/computer_science_papers_abstracts.parquet $BRIDGER_DATADIR/dygie_predictions/input_data --processes 80 --chunksize 5000 --debug >& logs/multiprocessing_format_dygie_$NOW.log &

```

### Run dygie predictions
+ This uses a pretrained scierc model to extract entities (e.g., Methods, Tasks) and relations (e.g., USED-FOR relations) from the titles and abstracts.
+ Using multiprocessing on 80 CPUs, this takes 4-5 days. If merely updating the data, it would be best to skip the ones that have already been done.
+ Outputs about 31GB (JSONL files)
```shell
# in dygiepp directory
# with dygiepp virtualenv activated (source venv/bin/activate)
# make sure nltk package has downloaded data before running: 'punkt' and 'wordnet'
NOW=$(date +'%Y%m%dT%H%M');DIRPATH=$BRIDGER_DATADIR/dygie_predictions/predictions; mkdir $DIRPATH && mkdir $DIRPATH/logs && python run_predict_multiprocessing.py $BRIDGER_DATADIR/dygie_predictions/input_data $DIRPATH --logdirpath $DIRPATH/logs --num-processes 80 --debug >& run_predict_multiprocessing_$NOW.log &
```

### Parse dygie predictions
+ TODO
```shell
# in collabnetworks directory
# with virtualenv activated (source venv/bin/activate)
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline050-parse_dygie_predictions.py $BRIDGER_DATADIR/dygie_predictions/predictions --processes 64 --debug >& logs/parse_dygie_predictions_multiprocessing_$NOW.log &
```

### Normalize terms
The previous step gave a set of terms extracted from titles and abstracts. Now we normalize the terms, to try to group together different forms of the same term (e.g. plural and singular). We do this by applying lower-casing, removing punctuation, and removing multiple consecutive spaces, then lemmatizing using a spacy model (`en_core-sci-sm` from `scispacy`).

The normalized terms have a problem: the lemma is not always the best term to display or use for downstream tasks (e.g. embedding). An example is that the term "data science" will be normalized to "datum science". So as a final step, we map each normalized term to its most frequently occurring cleaned (lower-cased, no-punctuation) term.

+ Takes 2-3 hours using 80 CPUs. Outputs a ~10GB parquet file.
```shell
# in collabnetworks directory
# with virtualenv activated (source venv/bin/activate)
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline060-normalize_dygie_terms.py $BRIDGER_DATADIR/dygie_predictions/predictions/predicted_terms/df_ner.parquet $BRIDGER_DATADIR/dygie_predictions/predictions/predicted_terms/terms_lemmatized_cleaned.parquet --processes 80 --debug >& logs/normalize_dygie_terms_$NOW.log &

```

### Get term embeddings
+ TODO
```shell
# on s2-server4
# ! IMPORTANT: modify the .slurm file to point to the right directories/files
sbatch scripts/get_sentence_transformer_embeddings.slurm

# Now combine the chunked files into a single embeddings file and a single terms file:
# on any server
# in collabnetworks directory
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline070-combine_embeddings.py $BRIDGER_DATADIR/dygie_predictions/predictions/predicted_terms/embeddings_cs_roberta_finetuneSTS --debug >& logs/combine_embeddings_$NOW.log &

```


### Final processing
+ TODO
+ takes 10-15 minutes
```shell
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline080-final_processing_dygie_terms.py $BRIDGER_DATADIR/dygie_predictions/predictions/predicted_terms/terms_lemmatized_cleaned.parquet $BRIDGER_DATADIR/final_processed --old-data $OLD_BRIDGER_DATADIR/final_processed/dygie_terms_to_s2_id_softmaxThreshold0.90.parquet --debug >& logs/final_processing_dygie_terms_$NOW.log &

NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline085-final_processing_dygie_embeddings.py $BRIDGER_DATADIR/dygie_predictions/predictions/predicted_terms/terms_lemmatized_cleaned.parquet $BRIDGER_DATADIR/dygie_predictions/predictions/predicted_terms/embeddings_cs_roberta_finetuneSTS $BRIDGER_DATADIR/final_processed/embeddings --existing $OLD_BRIDGER_DATADIR/final_processed/embeddings --debug >& logs/final_processing_dygie_embeddings_$NOW.log &
```


### Get average embeddings
+ TODO: get average embeddings for authors
```shell
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline110-get_average_dygie_embeddings.py $BRIDGER_DATADIR/final_processed/embeddings/author_average_embeddings_2015-2022 --min-year 2015 --max-year 2022 --debug >& logs/average_embeddings_$NOW.log &

NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline111-get_average_specter_embeddings.py $BRIDGER_DATADIR/specter_embeddings $BRIDGER_DATADIR/specter_embeddings/average_author_specter_embeddings_2015-2022_pandas.pickle --min-year 2015 --max-year 2022 --debug >& logs/average_specter_embeddings_$NOW.log &
```


## Identify groups by clustering the co-authorship network

### Construct co-authorship network
+ Takes 10-15 minutes
```shell
NOW=$(date +'%Y%m%dT%H%M'); DIRPATH=$BRIDGER_DATADIR/coauthor_2015-2022_minpubs3_collabweighted; mkdir $DIRPATH && python scripts/data_pipeline/pipeline090-get_coauthorship_network.py $BRIDGER_DATADIR/${MAG_SUBDIR_NAME}_CSsubset $DIRPATH/coauthor_edgelist.csv --min-year 2015 --max-year 2022 --min-pubs 3 --save-papers $DIRPATH/paper_ids.txt --pickle $DIRPATH/coauthor_graph.pickle --debug >& $DIRPATH/get_coauthorship_network_$NOW.log &
```


### TODO: Run local clustering on ego-nets; create persona graph
+ This takes 6-7 hours and outputs 300-400MB. Output is JSONL files indicating cluster memberships of the ego-nets, as well as metadata CSV files that can identify nodes that had timeout errors.
```shell
NOW=$(date +'%Y%m%dT%H%M'); DIRPATH=$BRIDGER_DATADIR/coauthor_2015-2022_minpubs3_collabweighted/components_local; mkdir $DIRPATH && mkdir $DIRPATH/logs && python scripts/data_pipeline/pipeline100-egosplit_local_clustering.py $BRIDGER_DATADIR/coauthor_2015-2022_minpubs3_collabweighted/coauthor_edgelist.csv $DIRPATH --min-weight 0.02 --process 24 --debug >& $DIRPATH/logs/egosplit_local_clustering_$NOW.log &

```

+ Redo local clustering for nodes that had timeout errors
```shell
NOW=$(date +'%Y%m%dT%H%M'); python scripts/data_pipeline/pipeline105-egosplit_local_timeout_redo.py $BRIDGER_DATADIR/coauthor_2015-2022_minpubs3_collabweighted/coauthor_edgelist.csv $BRIDGER_DATADIR/coauthor_2015-2022_minpubs3_collabweighted/components_local --process 10 --debug >& logs/egosplit_local_clustering_redo_multiprocessing_$NOW.log &
```

### TODO: Run global clustering on persona graph


## TF-IDF
```shell
NOW=$(date +'%Y%m%dT%H%M'); python scripts/get_author_dygie_tfidf.py $BRIDGER_DATADIR/author_dygie_tfidf_2015-2022 --min-year 2015 --max-year 2022 --debug >& logs/get_author_dygie_tfidf_$NOW.log &

NOW=$(date +'%Y%m%dT%H%M'); python scripts/get_author_dygie_tfidf.py $BRIDGER_DATADIR/author_dygie_tfidf_dropDuplicateTitles_2015-2022 --min-year 2015 --max-year 2022 --drop-duplicate-titles --debug >& logs/get_author_dygie_tfidf_dropDuplicateTitles_$NOW.log &
```