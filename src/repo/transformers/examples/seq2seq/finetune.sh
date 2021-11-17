# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
python finetune.py \
    --learning_rate=3e-5 \
    --fp16 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --val_check_interval 0.5 \
    --data_dir /workspace/ceph_data/intermediate/bart-AAE-v2-icle-dot-sub \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --output_dir=/workspace/ceph_data/output/bart-AAE-v2-icle-dot-sub \
    --num_train_epochs 6 \
    --model_name_or_path facebook/bart-large
    "$@"
