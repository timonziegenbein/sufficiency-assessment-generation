import cPickle
from scipy.stats import wilcoxon
import os
from confusion_matrix import ConfusionMatrix


def get_f1_list_from_report(path):
    f = file(path, 'rb')
    report = cPickle.load(f)
    f.close()
    f1_scores = []
    for x in report:
        f1_scores.append(x['confusion_matrix'].macro_f_impl1())
    return f1_scores


def get_f1_list_from_files(path):
    f1_scores = []
    for i in os.listdir(path):
        if i.endswith(".csv"):
            cm = ConfusionMatrix(file=path+'/'+i)
            f1_scores.append(cm.macro_f_impl1())
    return f1_scores


#result_path1 = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Results-DEV/Results-svm-All-struct/matrices'
#result_path1 = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Baseline-majority/baseline-majority.bin'
result_path1 = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Baseline-SVM-bow/matrices'
f1_scores1 = get_f1_list_from_files(result_path1)
#f1_scores1 = get_f1_list_from_report(result_path1)

result_path2 = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Results-SVM-all/matrices'
f1_scores2 = get_f1_list_from_files(result_path2)
# f1_scores2 = get_f1_list_from_report(result_path2)


# significance 'level' (according to (Peldszus and Stede 2015))
alpha = 0.005

z_statistics, p_value = wilcoxon(f1_scores1, f1_scores2)

print "paired wilcoxon-test", p_value

if p_value > alpha:
    print 'NOT SIGNIFICANT'
else:
    print 'SIGNIFICANT'

# if p_value is bigger  than 0.005 there is NO significant difference
# if p_value is smaller than 0.005 there is A  significant difference

