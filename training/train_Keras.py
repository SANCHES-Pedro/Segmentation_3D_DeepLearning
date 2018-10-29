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
	
#train with patch, using generator for training set and loading every single patch of one image for validation 
def train_segAn(patch_size,batch_size,num_epoch,steps,output_path,niter_disc=1):
	
	
	lr_seg_pre = 0.001
	lr_disc = 0.0005
	lr_seg = 0.001
	combined_losses = [logcosh_disc,dice_coef_loss]  ## [out_mae,dice_coef_loss]
	weights_mae  = 1.
	weights_dice = 1.
	combined_weights = [weights_mae , weights_dice] ##weights of the L1 (mean absolute error) and dice losses
	
	model = models_Keras.Models(batch_size = batch_size,patch_size = patch_size,spatialdrop=1)
	my_generator = Generator(patch_len=patch_size , batch_size=batch_size) #generate train images randomly
	my_loader = Load(patch_len = patch_size ,batch_size = batch_size)
	valIm,valGt = my_loader.load_val() # predict with all image at each time
	
	sess = tf.Session() # init tensorflow session for Dice calculations in validation
	
	## place holders for the inputs
	ground_truth_patch = Input(batch_shape=(batch_size,patch_size,patch_size,patch_size,1))
	image_patch = Input(batch_shape=(batch_size,patch_size,patch_size,patch_size,1))
	
	#optimizers
	opt = Adam(lr = lr_seg, beta_1=0.5)
	dopt = Adam(lr = lr_disc,beta_1=0.5)
	
	# get models for discriminator and segmentor, the fixed models are Networks(keras class) objects
	discriminator,discriminator_fixed = model.get_discriminator()
	segmentor,segmentor_fixed = model.get_segmentor()

	# Debug: discriminator and segmentor weights
	n_disc_trainable = len(discriminator.trainable_weights)
	n_seg_trainable = len(segmentor.trainable_weights)
	
	##model that trains the discriminator maximizing the MAE loss
	segmentor_fixed.trainable = False
	predictions_patch_fixed = segmentor_fixed(image_patch)
	output_error_disc = discriminator([image_patch,predictions_patch_fixed,ground_truth_patch])
	combined_discriminator = Model(inputs=[image_patch,ground_truth_patch],outputs = output_error_disc)
	combined_discriminator.compile(optimizer = dopt, loss = neg_logcosh_disc) ##neg_out_mae
	
	
	##model that train the segmentor minimizing the MAE loss and the dice loss from the predictions
	discriminator_fixed.trainable = False
	predictions_patch = segmentor(image_patch)
	output_error_seg = discriminator_fixed([image_patch,predictions_patch,ground_truth_patch])
	combined_segmentor = Model(inputs=[image_patch,ground_truth_patch],outputs = [output_error_seg,predictions_patch])
	combined_segmentor.compile(optimizer = opt, loss = combined_losses, loss_weights = combined_weights, metrics = combined_losses)
	
	
	output_fake = np.zeros(combined_segmentor.output_shape[0])

	
	# Debug: compare if trainable weights correct
	assert(len(combined_discriminator._collected_trainable_weights) == n_disc_trainable)
	assert(len(combined_segmentor._collected_trainable_weights) == n_seg_trainable)

	print 'assert before prefit'
	## Segmentor pretraining
	
	#segmentor.compile(optimizer = Adam(lr = lr_seg_pre,amsgrad=True), loss = dice_coef_loss)
	#print 'Segmentor Pretraining'
	#history = segmentor.fit_generator(generator= my_generator.generatorRandomPatchs('train'),steps_per_epoch= 100, epochs = 50)
	

	assert(len(combined_discriminator._collected_trainable_weights) == n_disc_trainable)
	assert(len(combined_segmentor._collected_trainable_weights) == n_seg_trainable)
	print 'assert after prefit'

	## GAN training
	dice_val_init = 0
	step = 0
	epoch = 0
	dice_train = []
	avg_loss_disc = []
	avg_loss_seg = []
	avg_l1_seg = []
	
	'''csv file
	 epoch,dice train ,dice_val, l1_segmentor, loss_seg, lossl1_disc ,
	 lr_seg, lr_disc, lr_combined, weights_mae , weights_dice, batch_size, niter_disc
	'''
	metrics_names = ['epoch','dice_loss_train' ,'dice_coef_val', 'l1_segmentor', 'loss_seg', 'lossl1_disc']
	params_names = ['lr_seg', 'lr_disc', 'lr_seg_pre', 'weights_mae ', 'weights_dice', 'batch_size', 'niter_disc']
	metrics = [[],[],[],[],[],[]] 
	params = [[lr_seg] , [lr_disc] , [lr_seg_pre] , [weights_mae] , [weights_dice] , [batch_size] , [niter_disc]]
	
	save_csv(params,params_names,output_path+'_params')

	for batch_imgs, batch_gt in my_generator.generatorRandomPatchs('train'):
		step += 1
		
		## train the discrimator niter times before training the segmentor
		loss_discriminator = combined_discriminator.train_on_batch(x = [batch_imgs,batch_gt], y = output_fake)
		#if step%niter_disc == 0 or step == 1:
		loss_segmentor = combined_segmentor.train_on_batch(x = [batch_imgs, batch_gt],y = [output_fake,batch_gt])
		
		avg_loss_disc.append(loss_discriminator)
		avg_loss_seg.append(loss_segmentor[0])
		avg_l1_seg.append(loss_segmentor[1])
		dice_train.append(loss_segmentor[2])
		
		sys.stdout.write(' step %d / %d ; Discriminator: loss %.5f  ; combined: loss %.5f mae %.5f dice %.5f ; train avg_dice %.5f \r'% 
		(step,steps, loss_discriminator,loss_segmentor[0],loss_segmentor[1],loss_segmentor[2],np.asarray(dice_train).mean()))
		sys.stdout.flush()

		if step == steps:
			step = 0
			
			print ''
			val_prediction = segmentor.predict(valIm,batch_size,1)
			dice_val = sess.run(dice_coef(val_prediction,valGt,0))
			print 'dice_val',dice_val
		
			if dice_val>dice_val_init:
				files_updater(output_path,segmentor,epoch,dice_val)
				dice_val_init = dice_val
			
			metrics[0].append(epoch)
			metrics[1].append(np.asarray(dice_train).mean())
			metrics[2].append(dice_val)
			metrics[3].append(np.asarray(avg_l1_seg).mean())
			metrics[4].append(np.asarray(avg_loss_seg).mean())
			metrics[5].append(np.asarray(avg_loss_disc).mean())

			save_csv(metrics,metrics_names,output_path)
			
			'''
			print ' lr : '
			K.set_value(combined_model.optimizer.lr,lr_finder(epoch))
			K.set_value(discriminator.optimizer.lr,lr_finder(epoch))
			'''
			
			print 'epoch ',epoch,'/',num_epoch
			epoch +=1
			
			dice_train = []
			avg_loss_disc = []
			avg_loss_seg = []
			avg_l1_seg = []
			
		if epoch == num_epoch:
			break


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
