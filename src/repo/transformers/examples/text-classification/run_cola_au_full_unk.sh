#!/bin/bash

for i in {0..99}
do
    export CUDA_VISIBLE_DEVICES=0
    export TASK=cola
    export DATA_DIR=/scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-cola-au-full-unk/${i}
    export MAX_LENGTH=256
    export BERT_MODEL=roberta-large
    export BATCH_SIZE=2
    export NUM_EPOCHS=2
    export SEED=42
    export OUTPUT_DIR=/scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/ceph_data/output/bert-AAE-v2-only-dot-direct-cola-au-full-unk/${i}

    # Make output directory if it doesn't exist
    mkdir -p $OUTPUT_DIR
    # Add parent directory to python path to access lightning_base.py
    export PYTHONPATH="/scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/src/repo/transformers/examples":"${PYTHONPATH}"

    python3 /scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/src/repo/transformers/examples/text-classification/run_pl_LS.py --data_dir $DATA_DIR \
    --gpus 1 \
    --task $TASK \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_train

    #rm -r /scratch/hpc-prf-arguana/tgurcke/lightning_logs
done