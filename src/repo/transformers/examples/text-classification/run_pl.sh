# Install example requirements
#pip install -r ../requirements.txt

# Download glue data
#python3 ../../utils/download_glue_data.py

# export CUDA_VISIBLE_DEVICES=6
# export TASK=mrpc
# export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/9
# export MAX_LENGTH=128
# export LEARNING_RATE=3e-5
# export BERT_MODEL=bert-base-cased
# export BATCH_SIZE=2
# export NUM_EPOCHS=3
# export SEED=2
# export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/9

# # Make output directory if it doesn't exist
# mkdir -p $OUTPUT_DIR
# # Add parent directory to python path to access lightning_base.py
# export PYTHONPATH="../":"${PYTHONPATH}"

# python3 run_pl_LS.py --data_dir $DATA_DIR \
# --gpus 1 \
# --task $TASK \
# --model_name_or_path $BERT_MODEL \
# --output_dir $OUTPUT_DIR \
# --max_seq_length  $MAX_LENGTH \
# --num_train_epochs $NUM_EPOCHS \
# --train_batch_size $BATCH_SIZE \
# --seed $SEED \
# --do_train 


export CUDA_VISIBLE_DEVICES=6
export TASK=cola
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-cola-au/3
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-cola-au/13

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/1
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/1

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/1
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/11

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/2
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/12

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/3
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/13

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/4
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/14

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/5
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/15

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/6
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/16

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/7
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/17

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/8
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/18

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

export CUDA_VISIBLE_DEVICES=6
export TASK=mrpc
export DATA_DIR=/workspace/ceph_data/intermediate/bert-AAE-v2-only-dot-direct-mrpc/9
export MAX_LENGTH=128
export LEARNING_RATE=3e-5
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=2
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR=/workspace/ceph_data/output/bert-AAE-v2-only-dot-direct-mrpc/19

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