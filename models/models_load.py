'''
loads models for evaluation or fine tunning
'''

import sys
sys.path.append('../data_manipulation')
sys.path.append('../utils')

import tensorflow as tf
import numpy as np
import glob
import keras
from keras.models import *
from keras.utils import plot_model

from losses import *
from metrics import *

#snapshot ensemble
# load the best models of a training with the validation loss as a reference
def load_model_ensemble(folder_name,num_models):
	path = '../../my_weights/'+folder_name+'/64deepVal*'
	models_path = glob.glob(path)
	dice = []
	for paths in models_path:
		dice.append(paths[-5:-3])
	models_path_dict = dict(zip(models_path,dice))
	models_path_sorted = sorted(models_path_dict, key=models_path_dict.__getitem__)

	weights = []
	for i in range(num_models):
		model = load_model(models_path_sorted[-i],custom_objects={'dice_coef_loss_neg': dice_coef_loss_neg,
			'sensitivity':sensitivity,'specificity':specificity})
		weights.append(model.get_weights())
		#print 'weights shape: ',np.ndarray(model.get_weights()[0])
	
	weights_mean = np.mean(weights,axis=0)
	model.set_weights(weights_mean)
	print 'loaded weights mean into model'
	model.summary()
	
	return model
	
if __name__ == '__main__':
	model = load_model_ensemble('UceptionWD',3)
	
	model.summary()
	plot_model(model, show_shapes=True, to_file='UceptionSens.png')
