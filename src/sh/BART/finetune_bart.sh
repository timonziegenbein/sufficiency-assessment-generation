#!/bin/bash
export PYTHONPATH="../../repo/transformers/examples":"${PYTHONPATH}"

for i in {0..4}
do
    python <YOUR-PATH-TO>/src/repo/transformers/examples/seq2seq/finetune_args.py \
        --gpus 1 \
        --do_predict \
        --data_dir ../../../ceph_data/intermediate/bart-AAE-v2-only-dot-direct-cola-au-full-mask-gen/${i} \
        --train_batch_size=2 \
        --eval_batch_size=2 \
        --warmup_steps 50 \
        --output_dir= ../../../ceph_data/output/bart-AAE-v2-only-dot-direct-cola-au-full-mask-gen/${i} \
        --num_train_epochs 3 \
        --max_target_length=128 --val_max_target_length=128 --test_max_target_length=128 \
        --model_name_or_path facebook/bart-large
        
done