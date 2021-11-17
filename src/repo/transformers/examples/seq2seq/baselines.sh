export PYTHONPATH="../":"${PYTHONPATH}"

for i in {0..9}
do
    python run_eval.py facebook/bart-large-cnn /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub/${i}/test.source /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/cnn_test.hypo \
        --reference_path /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub/${i}/test.target \
        --score_path /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/cnn_metrics.json \
        --task summarization \
        --device cuda \
        --bs 4

    python run_eval.py facebook/bart-large-xsum /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub/${i}/test.source /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/xsum_test.hypo \
        --reference_path /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub/${i}/test.target \
        --score_path /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/xsum_metrics.json \
        --task summarization \
        --device cuda \
        --bs 4

     python run_eval.py facebook/bart-large /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub/${i}/test.source /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/base_test.hypo \
        --reference_path /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub/${i}/test.target \
        --score_path /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/base_metrics.json \
        --task summarization \
        --device cuda \
        --bs 4

    python run_eval.py facebook/bart-large-cnn /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-sufficient/${i}/test.source /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/cnn_test_only_suff.hypo \
        --reference_path /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-sufficient/${i}/test.target \
        --score_path /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/cnn_metrics_only_suff.json \
        --task summarization \
        --device cuda \
        --bs 4

    python run_eval.py facebook/bart-large-xsum /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-sufficient/${i}/test.source /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/xsum_test_only_suff.hypo \
        --reference_path /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-sufficient/${i}/test.target \
        --score_path /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/xsum_metrics_only_suff.json \
        --task summarization \
        --device cuda \
        --bs 4

     python run_eval.py facebook/bart-large /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-sufficient/${i}/test.source /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/base_test_only_suff.hypo \
        --reference_path /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-sufficient/${i}/test.target \
        --score_path /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/base_metrics_only_suff.json \
        --task summarization \
        --device cuda \
        --bs 4

    python run_eval.py facebook/bart-large-cnn /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-insufficient/${i}/test.source /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/cnn_test_only_insuff.hypo \
        --reference_path /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-insufficient/${i}/test.target \
        --score_path /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/cnn_metrics_only_insuff.json \
        --task summarization \
        --device cuda \
        --bs 4

    python run_eval.py facebook/bart-large-xsum /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-insufficient/${i}/test.source /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/xsum_test_only_insuff.hypo \
        --reference_path /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-insufficient/${i}/test.target \
        --score_path /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/xsum_metrics_only_insuff.json \
        --task summarization \
        --device cuda \
        --bs 4

     python run_eval.py facebook/bart-large /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-insufficient/${i}/test.source /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/base_test_only_insuff.hypo \
        --reference_path /workspace/ceph_data/intermediate/bart-AAE-v2-dot-sub-only-insufficient/${i}/test.target \
        --score_path /workspace/ceph_data/intermediate/bart-AAE-vs-dot-sub-baselines/${i}/base_metrics_only_insuff.json \
        --task summarization \
        --device cuda \
        --bs 4
done