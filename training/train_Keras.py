import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os 
import sys
import numpy as np
import keras
from keras import metrics
from keras.models import *
from keras.engine import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from keras import backend as K

sys.path.append('../data_manipulation')
sys.path.append('../models')
sys.path.append('../utils')
"""
train the models with patchs or full images
"""
from data_generator import *
from data_Load import *
from losses import *
from metrics import *
import models_Keras
from models_load import *
import pandas as pd

	
#train with patch, using generator for training set and loading every single patch of one image for validation 
def train_generator_patch(lr_init,output_path,steps,num_epoch,batch_size=2,patch_size=64,deph = 6,dropout_rate = 0.2,
		batch_norm = 1,kernel_size = 5,spatialdrop = 1,net='vnet',fine_tunning = 0):
	if fine_tunning:
		model = load_model_ensemble(folder_name = 'FineT_uception_patchS64_batchS2_deph6_sdrop0_drop0.25_kernel2_lr0.002_cont100',num_models = 5)
	else:
		model = models_Keras.Models(batch_size,patch_size,deph,dropout_rate,
			batch_norm,kernel_size,spatialdrop)(net)
	
	metric = [sensitivity,specificity]
	model.compile(optimizer = Adam(lr = lr_init,amsgrad=True), loss = dice_coef_loss_neg , metrics=metric)

	my_generator = Generator(patch_len=patch_size , batch_size=batch_size,viva=True)
	my_loader = Load(patch_len = patch_size ,batch_size = batch_size,viva=True)
	valIm,valSeg = my_loader.load_val()

	print('Fitting model...')
	history = model.fit_generator(generator= my_generator.generatorRandomPatchs('train'),steps_per_epoch = steps, epochs = num_epoch,
	callbacks = callbacks(output_path,batch_size,0),
	validation_data = (valIm,valSeg))
	
#Starts with a cyclic learning rate and then an exponential decay with gamma
def lr_scheduler(epoch):
	lr_max = 0.001
	lr_min = 0.0001
	step_size = 30
	gamma = 0.999
	change1 = 200
	change2 = 51*step_size
	if epoch<=change1:
		lr = lr_max
	elif change1<epoch<change2:
		lr = lr_cycle(epoch,lr_max,lr_min,step_size)
	else:
		lr = lr_exp(epoch,gamma,lr_min,change2)	
	print 'Learning rate: ',lr
	return lr

def lr_exp(epoch,gamma,init,change):
	return init*(gamma**(epoch-change))
	
#setting cyclic learning rate as a triangular function
def lr_cycle(epoch,lr_max= 0.001,lr_min = 0.0001,step_size= 15):
	step = epoch%(2*step_size)
	if step<step_size:
		lr = lr_max - step*(lr_max-lr_min)/step_size
	else:
		lr = lr_min + (step-step_size)*(lr_max-lr_min)/step_size
	return lr

def lr_finder(epoch):
	lr1 = 1e-8*(0.6**(-epoch)) #30 epochs
	print lr1
	return lr1
	
def save_csv(array,name_col,folder_name):
	
	df = pd.DataFrame(dict(sorted(zip(name_col,np.asarray(array)))))
	filepath='../../my_csv/'+folder_name+'.csv'
	df.to_csv(filepath)

def files_updater(folder_name,model,epoch,dice):

	if not os.path.exists('../../my_weights/'+folder_name):
		os.makedirs('../../my_weights/'+folder_name)
	filepath='../../my_weights/'+folder_name+'/64deepVal-{0:02d}-{1:.2f}.h5'.format(epoch,dice)
	model.save(filepath)

## callbacks
def callbacks(folder_name,batch_size,iD):
	if iD == 0:
		lr = lr_scheduler
	else:
		lr = lr_fine_tunning
	if not os.path.exists('../../my_weights/'+folder_name):
		os.makedirs('../../my_weights/'+folder_name)

	return [
	TensorBoard(
	log_dir='../../my_logs/'+folder_name+'/',
	batch_size= batch_size,
	#histogram_freq = 1
	),
	
	#save the model that has the biggest validation loss until the moment 
	ModelCheckpoint(filepath='../../my_weights/'+folder_name+'/64deepVal-{epoch:02d}-{val_loss:.2f}.h5', 
	monitor='val_loss',
	verbose=0,
	save_best_only=True),
	
	ModelCheckpoint(filepath='../../my_weights/'+folder_name+'/64deep-{epoch:02d}-{loss:.2f}.h5', 
	monitor='loss',
	verbose=0,
	save_best_only=True),	

	CSVLogger('../../my_csv/'+folder_name+'.csv'),
	
	LearningRateScheduler(lr)
	]

def hyper_param_search():
	
	batch_norm = 1
	spatialdrop = 0
	dropouts = 0.25
	lr = 0.001
	cont = 50
	for kernel_size in [5,3]:
		for patch_size in [64,48,80]:
			for batch_size in [2,3,4]:
				for deph in [6,10,14]:
					for net in ['unet' , 'vnet_original', 'uception']:
						cont +=1
						file_name = 'hs_{}_patchS{}_batchS{}_deph{}_sdrop{}_drop{}_kernel{}_lr{}_cont{}'.format(net,patch_size,
							batch_size,deph,spatialdrop,dropouts,kernel_size,lr,cont)
						print file_name
						train_generator_patch(lr_init=lr,output_path=file_name,steps=120,num_epoch=100,batch_size=batch_size,
						patch_size=patch_size,deph = deph,dropout_rate = dropouts,
						batch_norm = 1,kernel_size = kernel_size,spatialdrop = spatialdrop,net=net)


if __name__ == '__main__':
	
	'''
	it's important to specify the GPU that one wants, otherwise, the program will use the memory of all GPUs available 
	and do processing in just one of them
	'''
	
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	

	batch_norm = 1
	kernel_size = 5
	patch_size = 64
	batch_size = 2
	deph = 10
	lr = 0.001
	net = 'uception'
	spatialdrop = 0
	dropouts = 0.25
	cont = 1
	file_name = 'VivaM_{}_patchS{}_batchS{}_deph{}_sdrop{}_drop{}_kernel{}_lr{}_cont{}'.format(net,patch_size,
								batch_size,deph,spatialdrop,dropouts,kernel_size,lr,cont)
	print file_name
	train_generator_patch(lr_init=lr,output_path=file_name,steps=100,num_epoch=2000,batch_size=batch_size,
								patch_size=patch_size,deph = deph,dropout_rate = dropouts,
								batch_norm = 1,kernel_size = kernel_size,spatialdrop = spatialdrop,net=net,fine_tunning = 0)

	#hyper_param_search()
	
	#train_generator_patch(lr_init=0.001,output_path='unet_test_hyper',steps=100,num_epoch=100,batch_size=1,
	#							patch_size=64,deph = 6,dropout_rate = 0.25,
	#							batch_norm = 2,kernel_size = 5,spatialdrop = 1,net='unet')
	#train_segAn(patch_size = 64,batch_size= 2,num_epoch = 500,steps=50,output_path = 'SegAn_Net11_BS2_',niter_disc = 1)
