export MAX_LENGTH=512
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=4
export NUM_EPOCHS=3
export SEED=1
export LEARNING_RATE=5e-5
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=/workspace/ceph_data/output/bert-IBM-iob
export DATA_DIR=/workspace/ceph_data/intermediate/bert-IBM-iob


# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python3 run_pl_IBM.py --data_dir $DATA_DIR \
--labels /workspace/ceph_data/intermediate/bert-IBM-iob/labels.txt \
--learning_rate $LEARNING_RATE \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--gpus 1 \
--do_train \
--do_predict
