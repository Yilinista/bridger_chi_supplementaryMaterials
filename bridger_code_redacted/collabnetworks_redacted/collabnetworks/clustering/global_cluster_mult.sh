MIN_YEAR=2015
MAX_YEAR=2021
BASE_DIRECTORY=data/computer_science_papers_20201002/coauthor_${MIN_YEAR}-${MAX_YEAR}_minpubs3_collabweighted
COMPONENTS_DIRECTORY=${BASE_DIRECTORY}/components_local
OUTPUT_DIRECTORY=${COMPONENTS_DIRECTORY}/infomap_runs
echo "creating directory ${OUTPUT_DIRECTORY}"
mkdir $OUTPUT_DIRECTORY
echo "creating directory ${OUTPUT_DIRECTORY}/logs"
mkdir ${OUTPUT_DIRECTORY}/logs
for SEED_VAL in {11..50}; do
	SEED_PADDED=${(l:5::0:)SEED_VAL}
        echo "running clustering for seed ${SEED_PADDED} "
	python clustering/egosplit_global_clustering.py ${BASE_DIRECTORY}/coauthor_edgelist.csv $COMPONENTS_DIRECTORY ${OUTPUT_DIRECTORY}/memberships_minWeight02_seed${SEED_PADDED}.pickle --seed $SEED_VAL --min-weight 0.02 --debug >& ${OUTPUT_DIRECTORY}/logs/egosplit_global_clustering_minWeight02_seed${SEED_PADDED}.log &
done

