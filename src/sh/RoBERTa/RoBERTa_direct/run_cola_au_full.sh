#!/bin/bash

for i in {0..99}
do
    export CUDA_VISIBLE_DEVICES=0
    export TASK=cola
    export DATA_DIR=../../../../ceph_data/intermediate/roberta-AAE-v2-only-dot-direct-cola-au-full/${i}
    export MAX_LENGTH=256
    export BERT_MODEL=roberta-large
    export BATCH_SIZE=2
    export NUM_EPOCHS=2
    export SEED=42
    export OUTPUT_DIR=../../../../ceph_data/output/roberta-AAE-v2-only-dot-direct-cola-au-full/${i}

    # Make output directory if it doesn't exist
    mkdir -p $OUTPUT_DIR
    # Add parent directory to python path to access lightning_base.py
    export PYTHONPATH="<YOUR-PATH-TO>/src/repo/transformers/examples":"${PYTHONPATH}"

    python3 ../../../repo/transformers/examples/text-classification/run_pl_LS.py --data_dir $DATA_DIR \
    --gpus 1 \
    --task $TASK \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_train

done