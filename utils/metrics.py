'''
metrics for the network

TP = True positives
FP = False positives
TN = True negatives
FN = False negatives

sensitivity and specificity are good metrics because one can easily see if the training is diverging to a local minima 
where it won't be able to leave. For example, sometimes the network makes everything black, since we have really sparse data,
in this case the sensitivity/recall goes to 0 and the specificity goes to close to 1

'''
from keras import backend as K
import tensorflow as tf

# Sensitivity (true positive rate) or Recall
def sensitivity(truth, prediction):
	TP = K.sum(K.round(K.clip(truth * prediction, 0, 1))) #the clipping is just to make sure that the values are between 0 and 1
	P = K.sum(K.round(K.clip(truth, 0, 1)))
	return TP / (P + K.epsilon())

# Specificity (true negative rate)
def specificity(truth, prediction):
	TN = K.sum(K.round(K.clip((1-truth) * (1-prediction), 0, 1)))
	N = K.sum(K.round(K.clip(1-truth, 0, 1)))
	return TN / (N + K.epsilon())

# Precision (positive prediction value)
def precision(truth, prediction):
	TP = K.sum(K.round(K.clip(truth * prediction, 0, 1)))
	FP = K.sum(K.round(K.clip((1-truth) * prediction, 0, 1)))
	return TP / (TP + FP + K.epsilon())

# DSC = 2*TP / (2*TP + FP + FN)
def dice_coef(y_true, y_pred, smooth=0.1):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
