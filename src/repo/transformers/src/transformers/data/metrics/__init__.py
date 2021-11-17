# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
        f1_suf = f1_score(y_true=labels, y_pred=preds, average='binary', pos_label=1)
        f1_insuf = f1_score(y_true=labels, y_pred=preds, average='binary', pos_label=0)
        recall_suf = recall_score(y_true=labels, y_pred=preds, average='binary', pos_label=1)
        recall_insuf = recall_score(y_true=labels, y_pred=preds, average='binary', pos_label=0)
        precision_suf = precision_score(y_true=labels, y_pred=preds, average='binary', pos_label=1)
        precision_insuf = precision_score(y_true=labels, y_pred=preds, average='binary', pos_label=0)
        return {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_suf": f1_suf,
            "f1_insuf": f1_insuf,
            "recall_suf": recall_suf,
            "recall_insuf": recall_insuf,
            "precision_suf": precision_suf,
            "precision_insuf": precision_insuf
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(
            labels
        ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
        if task_name == "cola":
            return acc_and_f1(preds, labels)
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"mnli/acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"mnli-mm/acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(
            labels
        ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
