'''
Code for load the images, load the model, make the predictions on patchs and rebuild the image
'''

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import keras
from keras.models import *
import nibabel as nib
from medpy.io import load,save
import sys
sys.path.append('../data_manipulation')
sys.path.append('../utils')
sys.path.append('../models')

import glob

from losses import *
from metrics import *
from losses import *
from data_Load import *
from data_generator import *
from models_load import *
import patch_Extraction


def predict_patch(path,batch_size,patch_size,num_models):
	
	print 'loading model'
	model = load_model_ensemble(path,num_models)
	
	print 'model loaded'
	my_loader= Load(batch_size=batch_size,patch_len=patch_size)
	
	testIm_patch,testSeg_patch,testIm,image_shape = my_loader.load_test()
	
	print("Predicting")
	#metricas = model.evaluate(x=testIm_patch, y=testSeg_patch, batch_size=batch_size, verbose=1)
	output_Array = model.predict(testIm_patch,batch_size=batch_size,verbose=1)
	
	sess = tf.Session()
	print "Dice: ",sess.run(dice_coef(output_Array,testSeg_patch,0))	
	#print output_Array.shape, '  ',testSeg_patch.shape
	return output_Array,testSeg_patch,testIm_patch,image_shape


def rebuild_image_unif(output_Array,patch_size,image_shape):
	print("array to image")
	import math
	image_shape = np.asarray(image_shape).astype('float16')
	a = math.ceil(image_shape[0]/patch_size)
	b = math.ceil(image_shape[1]/patch_size)
	c = math.ceil(image_shape[2]/patch_size)
	n_paches = np.array([a,b,c]).astype('uint16')
	image_shape = image_shape.astype('uint16')

	output_Image = np.zeros((image_shape[0],image_shape[1],image_shape[2],1), dtype=np.float32)
	print "n_patchs ",n_paches

	cont = 0
		
	dict_shape = {'i':0,'j':1,'k':2}
	print n_paches[0]
	for i in range(n_paches[0]):
		for j in range(n_paches[1]):
			for k in range(n_paches[2]):
				
				index = []
				index_patch = []
				dict_num = {'i':i,'j':j,'k':k}
				for a in ['i','j','k']:
					#print a
					index.append(dict_num[a]*patch_size) #index init
					
					if dict_num[a] == n_paches[dict_shape[a]]-1:
						index_patch.append(image_shape[dict_shape[a]]-(n_paches[dict_shape[a]]-1)*patch_size)
						index.append(None)
					else:
						index.append((dict_num[a]+1)*patch_size)
						index_patch.append(None)

				output_Image[index[0]:index[1],index[2]:index[3],index[4]:index[5]] = output_Array[cont,:index_patch[0],:index_patch[1],:index_patch[2]]
				cont += 1
	return output_Image
	
	

def predict_ISBI(model_name,test_name,batch_size,patch_size = 64,num_models=1):
	
	model = load_model_ensemble(model_name,num_models)
	my_loader= Load(batch_size=batch_size,patch_len=patch_size)
	testIm,testSeg = my_loader.loadImages(iD='test',downsampling = 1,isotrope=True)
	a = testIm[1].shape
	patch = patch_Extraction.Patch(patch_size)

	for i in range(0,4):
		
		testIm_patch,testSeg_patch = patch.load_Patch(testIm[i],testSeg[i])
		testIm_patch = testIm_patch.astype('float32')

		output_Array = model.predict(testIm_patch,batch_size=batch_size,verbose=1)
		
		I_image = rebuild_image_unif(testIm_patch,patch_size=patch_size,image_shape=a)
		I_prediction = rebuild_image_unif(output_Array,patch_size=patch_size,image_shape=a)
		I_seg = rebuild_image_unif(testSeg_patch,patch_size=patch_size,image_shape=a)	
		
		save_nii(I_image,'../../Results/'+test_name+'_'+str(i)+'_original.nii')
		save_nii(I_prediction,'../../Results/'+test_name+'_'+str(i)+'_pred.nii')
		save_nii(I_seg,'../../Results/'+test_name+'_'+str(i)+'_gt.nii')
		

def predict_build_patch(model_name,test_name,batch_size,patch_size = 64,num_models=1):
	
	prediction,seg,a = predict_patch(model_name,batch_size = batch_size, patch_size=patch_size,num_models=num_models)
	# reconstruction
	I_image = rebuild_image_unif(image,patch_size=patch_size,image_shape=a)
	I_prediction = rebuild_image_unif(prediction,patch_size=patch_size,image_shape=a)
	I_seg = rebuild_image_unif(seg,patch_size=patch_size,image_shape=a)
	save_nii(I_prediction,'../../Results/'+test_name+'_pred.nii')
	save_nii(I_seg,'../../Results/'+test_name+'_gt.nii')

def save_nii(output_Image,path):
	save(output_Image,path)



if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = ''

	#predict_build_patch('FineT_uception_patchS64_batchS2_deph6_sdrop0_drop0.25_kernel2_lr0.002_cont100','testFinalBullit',batch_size = 2,patch_size = 64,num_models = 2)
	predict_ISBI('FineT_uception_patchS64_batchS2_deph6_sdrop0_drop0.25_kernel2_lr0.002_cont100','ISBI_results_d_',batch_size = 1,patch_size = 64,num_models = 5)
