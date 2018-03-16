#!/bin/bash

SEED=1234
EPOCHS=(10 20)
WEIGHT=("[0.6,0.3,0.1]" "[0.4,0.3,0.3]" "[0.4,0.4,0.2]" "[0.3,0.4,0.3]" "[0.3,0.3,0.4]" "[0.2,0.5,0.3]" )
INPUT_DIR="./cumc12_experiment"
OUTPUT_BASE_DIR="/playpen/mn/mermaid_results"
MAIN_JSON="./test3d_025.json"
VALIDATION_DATASET_DIR="/playpen/data/quicksilver_data/testdata/CUMC12"

COUNTER=0
CUDA_VISIBLE_DEVICES=1

for i in "${WEIGHT[@]}"
do
    W_KEY="model.registration_model.forward_model.smoother.multi_gaussian_weights=${i}"

    OUT_DIR="${OUTPUT_BASE_DIR}/out_${COUNTER}"

    CMD="python multi_stage_smoother_learning.py \
        --input_image_directory ${INPUT_DIR} \
        --nr_of_image_pairs 10 \
        --output_directory ${OUT_DIR} \
        --nr_of_epochs 50,10,5 \
        --config ${MAIN_JSON} \
        --stage_nr 0 \
        --config_kvs ${W_KEY} \
        --seed ${SEED}"
    echo $CMD
    $CMD

    CMDVIZ="python visualize_multi_stage.py \
        --config ${MAIN_JSON} \
        --output_directory ${OUT_DIR} \
        --stage_nr 0"

    echo $CMDVIZ
    $CMDVIZ

    CMDVAL="python compute_validation_results.py \
         --output_directory ${OUT_DIR} \
          --stage_nr 0 \
          --dataset_directory ${VALIDATION_DATASET_DIR} \
          --do_not_visualize \
          --save_overlap_filename overlaps.txt"

    echo $CMDVAL
    $CMDVAL
    
    COUNTER=$[$COUNTER +1]
done
