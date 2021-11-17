for i in {0..9}
do
    export CUDA_VISIBLE_DEVICES=0
    export TASK=cola
    export DATA_DIR=/workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-generated-claim-cola/${i}
    export MAX_LENGTH=128
    export BERT_MODEL=bert-base-cased
    export BATCH_SIZE=2
    export NUM_EPOCHS=3
    export SEED=2
    export OUTPUT_DIR=/workspace/ceph_data/output/bart-AAE-v2-dot-sub-generated-claim-cola/${i}

    # Make output directory if it doesn't exist
    mkdir -p $OUTPUT_DIR
    # Add parent directory to python path to access lightning_base.py
    export PYTHONPATH="../":"${PYTHONPATH}"

    python3 run_pl_LS.py --data_dir $DATA_DIR \
    --gpus 1 \
    --task $TASK \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_train

    rm -r /workspace/src/repo/transformers/examples/text-classification/lightning_logs
done