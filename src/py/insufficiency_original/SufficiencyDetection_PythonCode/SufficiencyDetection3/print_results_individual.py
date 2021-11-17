from numpy import std
from numpy import average
import cPickle

report_path = '/workspace/ceph_data/output/insufficiency-original-sub-pc/report_CNN_pc.bin'
labels = ['noFlaw', 'sufficiency']

accuracy = []
macro_f1 = []
precision = []
recall = []
f1_sufficient = []
f1_insufficient = []
epochs = []

f = file(report_path, 'rb')
report = cPickle.load(f)
f.close()
matrices = []
for e in report:
    cm = (e['confusion_matrix'])
    print "iteration=" + str(e['iteration']) + "; fold=" + str(e['fold']) + "; best_epoch = " + str(e['best_epoch'])
    print "  Accuracy = %.3f" % cm.a()
    print "  Macro F1 = %.3f" % cm.macro_f_impl1()
    epochs += [e['best_epoch']]
    accuracy += [cm.a()]
    macro_f1 += [cm.macro_f_impl1()]
    precision += [cm.p()]
    recall += [cm.r()]
    f1_sufficient += [cm.f(labels[0])]
    f1_insufficient += [cm.f(labels[1])]


print "\n\nAccuracy       \t %.3f +- %.3f" % (average(accuracy), std(accuracy))
print "Macro F1       \t %.3f +- %.3f" % (average(macro_f1), std(macro_f1))
print "Precision      \t %.3f +- %.3f" % (average(precision), std(precision))
print "Recall         \t %.3f +- %.3f" % (average(recall), std(recall))
print "F1 Insufficient\t %.3f +- %.3f" % (average(f1_insufficient), std(f1_insufficient))
print "F1 Sufficient  \t %.3f +- %.3f\n" % (average(f1_sufficient), std(f1_sufficient))

print "Epochs (avg)   \t %.3f " % average(epochs)
print "Epochs (std)   \t %.3f " % std(epochs)
