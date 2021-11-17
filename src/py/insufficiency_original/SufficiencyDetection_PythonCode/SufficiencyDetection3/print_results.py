from numpy import std
from numpy import average
import os
import cPickle
from confusion_matrix import ConfusionMatrix


def confusion_matrices_from_report(path):
    f = file(path, 'rb')
    report = cPickle.load(f)
    f.close()
    matrices = []
    for e in report:
        matrices.append(e['confusion_matrix'])
    return matrices


def confusion_matrices_from_directory(path):
    matrices = []
    for i in os.listdir(path):
        if i.endswith(".csv"):
            matrices.append(ConfusionMatrix(file=path+'/'+i))
    return matrices


# Determine scores from python experiment
#report_path = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Results-CNN/report_CNN.bin'
#report_path = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Baseline-majority/baseline-majority.bin'
#conf_matrices = confusion_matrices_from_report(report_path)
#labels = ['noFlaw', 'sufficiency']

# Determine scores from java experiment
conf_path = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Results-SVM-all/matrices'
#conf_path = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Baseline-SVM-bow/matrices'
conf_matrices = confusion_matrices_from_directory(conf_path)
labels = ['NoFallacy', 'Sufficiency']

accuracy = []
macro_f1 = []
precision = []
recall = []
f1_sufficient = []
f1_insufficient = []
recall_insufficient = []
precision_insufficient = []

for cm in conf_matrices:
    accuracy += [cm.a()]
    macro_f1 += [cm.macro_f_impl1()]
    precision += [cm.p()]
    recall += [cm.r()]
    f1_sufficient += [cm.f(labels[0])]
    f1_insufficient += [cm.f(labels[1])]
    recall_insufficient += [cm.r(labels[1])]
    precision_insufficient += [cm.p(labels[1])]


print "Accuracy       \t %.3f +- %.3f" % (average(accuracy), std(accuracy))
print "Macro F1       \t %.3f +- %.3f" % (average(macro_f1), std(macro_f1))
#print "Precision      \t %.3f +- %.3f" % (average(precision), std(precision))
#print "Recall         \t %.3f +- %.3f" % (average(recall), std(recall))
#print "F1 Sufficient  \t %.3f +- %.3f" % (average(f1_sufficient), std(f1_sufficient))
print "F1 Insufficient\t %.3f +- %.3f" % (average(f1_insufficient), std(f1_insufficient))
print "Precision Insufficient  \t %.3f +- %.3f" % (average(precision_insufficient), std(precision_insufficient))
print "Recall Insufficient  \t %.3f +- %.3f" % (average(recall_insufficient), std(recall_insufficient))


