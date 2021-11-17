#!/bin/bash
export PYTHONPATH="/scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/src/repo/transformers/examples":"${PYTHONPATH}"

for i in {0..4}
do
    python /scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/src/repo/transformers/examples/seq2seq/run_eval.py \
          /scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/ceph_data/output/bart-AAE-v2-only-dot-direct-cola-au-full-mask-gen/${i}/best_tfmr2 \
          /scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/ceph_data/intermediate/bart-AAE-v2-only-dot-direct-cola-au-full-mask-gen/${i}/test.source \
          /scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/ceph_data/intermediate/bart-AAE-v2-only-dot-direct-cola-au-full-mask-gen/${i}/test_supervised.hypo \
        --reference_path /scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/ceph_data/intermediate/bart-AAE-v2-only-dot-direct-cola-au-full-mask-gen//${i}/test.target \
        --score_path /scratch/hpc-prf-arguana/tgurcke/evaluating-the-local-sufficiency-of-arguments-through-generation/masterthesis/ceph_data/intermediate/bart-AAE-v2-only-dot-direct-cola-au-full-mask-gen/${i}/metrics_supervised.json \
        --task summarization \
        --device cuda \
        --bs 4
done