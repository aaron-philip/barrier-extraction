#!/bin/bash

DATA_DIR=/mnt/home/philipaa/scratch/barrier-paper-datasets
RESULT_DIR=/mnt/home/philipaa/scratch/barrier-paper-results

DATASETS=(
    cv0.00_spacing0.5.h5
    cv0.25_spacing0.5.h5
    cv0.50_spacing0.5.h5
    cv0.75_spacing0.5.h5
    cv0.00_spacing1.0.h5
    cv0.25_spacing1.0.h5
    cv0.50_spacing1.0.h5
    cv0.75_spacing1.0.h5
    res_0.25MeV_noise0.01.h5
    res_0.50MeV_noise0.01.h5
    res_1.00MeV_noise0.01.h5
    res_2.00MeV_noise0.01.h5
    res_0.25MeV_noise0.1.h5
    res_0.50MeV_noise0.1.h5
    res_1.00MeV_noise0.1.h5
    res_2.00MeV_noise0.1.h5
)

for FILE in "${DATASETS[@]}"; do

    DATAFILE="$DATA_DIR/$FILE"
    OUTFILE="$RESULT_DIR/${FILE%.h5}_bnn.h5"

    echo "Submitting array (0–999) for dataset: $FILE"

    sbatch \
        --export=DATAFILE="$DATAFILE",OUTFILE="$OUTFILE" \
        train_bnns.slurm

done
