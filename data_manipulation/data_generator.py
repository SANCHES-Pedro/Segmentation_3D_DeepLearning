'''
uses pythons generator to fed the neural network training
'''
import sys
import numpy as np 
import os
import glob
from medpy.io import load,save
import tensorflow as tf
import keras
from random import randint
import nibabel as nib

sys.path.append('../io')

from data_Augmentation import *
import patch_Extraction
import preprocessing
import read
import data_Load

## original shape (448,448,128)
class Generator(object):

	def __init__(self,patch_len = 64 ,batch_size = 4,viva = True):	
		self.patch_len = patch_len
		self.batch_size = batch_size
		self.patch= patch_Extraction.Patch(self.patch_len)
		self.loader = data_Load.Load(patch_len = patch_len ,batch_size = batch_size,viva=viva)
				
	def generatorImages(self,iD):
		images,segmentations = self.loader.loadImages(iD, cont = 1,downsampling=1,isotrope= True,pad = True )
		while True:
			i = randint(0,len(images)-1)
			im = images[np.newaxis,i]
			#if iD == 'train':
			#	im = data_augmentation(im)
			seg = segmentations[np.newaxis,i]
			yield im,seg
				

	def generatorRandomPatchs(self,iD):
		
		images,segmentations = self.loader.loadImages(iD,downsampling=1,isotrope= True)
		patch = patch_Extraction.Patch(self.patch_len)
		
		while True:
			batch_features = []
			batch_labels = []
			i = 0
			while i<self.batch_size:
				
				features,labels = patch.RandomPatch(images,segmentations)
				#if labels.max() != 0:
				batch_features.append(features)
				batch_labels.append(labels)
				i+=1
					
			batch_features = np.asarray(batch_features,dtype='float32')
			batch_labels = np.asarray(batch_labels,dtype='float32')
			yield batch_features, batch_labels
