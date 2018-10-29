import keras.backend as K
import metrics
import tensorflow as tf

def dice_coef_loss(y_true, y_pred):
    return 1-metrics.dice_coef(y_true, y_pred)
def dice_coef_loss_neg(y_true, y_pred):
    return -metrics.dice_coef(y_true, y_pred)

def dice_log_loss(y_true, y_pred):
    return -tf.log(metrics.dice_coef(y_true, y_pred))
    
def dice_sens_loss(y_true, y_pred,alpha= 0.5):
    return -metrics.dice_coef(y_true, y_pred)-alpha*metrics.sensitivity(y_true, y_pred)

def xent(truth, prediction):
	xent_elem_wise = tf.nn.sigmoid_cross_entropy_with_logits(labels=truth,logits=prediction)
	return tf.reduce_mean(xent_elem_wise)


## losses for adversarial training
def out_mae(truth, prediction):
	N = tf.count_nonzero(prediction,dtype='float32')
	return tf.reduce_sum(tf.abs(prediction))/N

def neg_out_mae(truth, prediction):
	return -out_mae(truth, prediction)

def logcosh_disc(y_true, y_pred):
	N = tf.count_nonzero(y_pred,dtype='float32')
	def _logcosh(x):
		return x + K.softplus(-2. * x) - K.log(2.)
	return tf.reduce_sum(_logcosh(y_pred))/N

def neg_logcosh_disc(y_true, y_pred):
	return -logcosh_disc(y_true, y_pred)
