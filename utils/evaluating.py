# ------------------------------------------------------------ #
#
# file : tools/evaluator.py
# author : CM
# Evaluate matching between an image and its ground truth
#
# ------------------------------------------------------------ #
import os
import sys
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, \
    jaccard_similarity_score, f1_score
	
# Prediction
filename_pr = sys.argv[1]
if(not os.path.isfile(filename_pr)):
    sys.exit(1)
# Input
filename_in = sys.argv[2]
if(not os.path.isfile(filename_in)):
    sys.exit(1)
# Ground truth
filename_gd = sys.argv[3]
if(not os.path.isfile(filename_gd)):
    sys.exit(1)

data_pr = nib.load(filename_pr).get_data()
data_in = nib.load(filename_in).get_data()
data_gd = nib.load(filename_gd).get_data()

data_pr = data_pr.flatten()
data_in = data_in.flatten()
data_gd = data_gd.flatten()

data_in = data_in/data_in.max()

# F1 binary score
f1_score_pr = []
f1_score_in = []
f1_threshold = []
mult = 0.01

for i in range(0, 101):
    threshold = i*mult
    data_pr_threshold = data_pr > threshold
    data_in_threshold = data_in > threshold

    f1_threshold.append(threshold)
    f1_score_pr.append(f1_score(data_gd, data_pr_threshold))
    f1_score_in.append(f1_score(data_gd, data_in_threshold))

max_in = np.max(f1_score_in)
x_in = np.isin(f1_score_in,max_in)
th_in = np.where(x_in)
th_in = mult * th_in[0][0]

max_pr = np.max(f1_score_pr)
x_pr = np.isin(f1_score_pr,max_pr)
th_pr = np.where(x_pr)
th_pr = mult * th_pr[0][0]

f1_curve = pyplot.figure()


pyplot.plot(f1_threshold, f1_score_pr, '-', label='Uception th = %0.2f' % th_pr)
pyplot.plot(f1_threshold, f1_score_in, '-', label='Original th = %0.2f' % th_in)
pyplot.scatter(th_pr,max_pr)
pyplot.scatter(th_in,max_in)

print th_pr
print th_in

pyplot.title('Dice')
pyplot.xlabel("threshold")
pyplot.ylabel("Dice")
pyplot.legend(loc="lower right")
pyplot.savefig("Dice_.png")
