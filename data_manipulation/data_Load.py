'''
load data from a folder
'''

import numpy as np 
import os
import glob
from medpy.io import load,save,header
import random
import sys
sys.path.append('../io')

import patch_Extraction
from preprocessing import prepro
import read


class Load:

	def __init__(self,patch_len = 64 ,batch_size = 1, train_Im_path = "train_Images", train_Seg_path = "train_Seg", test_Im_path = "test_Images",
	 test_Seg_path = "test_Seg", validation_Im_path = "validation_Images",validation_Seg_path = "validation_Seg",viva = False):	

		self.patch_len = patch_len
		self.train_Im_path = train_Im_path
		self.train_Seg_path = train_Seg_path
		self.test_Im_path = test_Im_path
		self.test_Seg_path = test_Seg_path
		self.validation_Im_path = validation_Im_path
		self.validation_Seg_path = validation_Seg_path
		self.batch_size = batch_size
		self.patch= patch_Extraction.Patch(self.patch_len)
		if viva:
			self.train_Im_path = 'v_'+train_Im_path
			self.train_Seg_path = 'v_'+train_Seg_path
			self.test_Im_path = 'v_'+test_Im_path
			self.test_Seg_path = 'v_'+test_Seg_path
			self.validation_Im_path = 'v_'+validation_Im_path
			self.validation_Seg_path = 'v_'+validation_Seg_path
			
	def feedImage(self,images,segmentations):
		i = random.randint(0,images.shape[0]-1)
		return images[np.newaxis,i],segmentations[np.newaxis,i]

	def loadImages(self,iD,cont = None,downsampling=1,isotrope= False,pad = False):
		'''
		iD is the the part of the data: train, validation or test
		cont is the number of images to load
		'''
		
		if iD == 'train':
			Im_path = self.train_Im_path
			Seg_path = self.train_Seg_path
		elif iD == 'val':
			Im_path = self.validation_Im_path
			Seg_path = self.validation_Seg_path
		elif iD == 'test':
			Im_path = self.test_Im_path
			Seg_path = self.test_Seg_path
			
		imgs = glob.glob('../../Data/'+Im_path+'/*')
		segs = glob.glob('../../Data/'+Seg_path+'/*')
		imgs = np.sort(imgs)
		segs = np.sort(segs)
		
		x =len(imgs)
		
		images = []
		segmentations= []
		if cont == None:
			cont = x
		for i in range(cont):
			print 'ID: ',iD,'---- imgs ',imgs[i],' ---- seg',segs[i]
			images.append(read.Reader(isotrope)(imgs[i],downsampling,preprocessing = True,pad = pad))
			segmentations.append(read.Reader(isotrope)(segs[i],downsampling,pad = pad))
			
		images = np.asarray(images,dtype='float16')
		segmentations = np.asarray(segmentations,dtype='float16')

		print 'images loaded with shape: ',images.shape
		
		return images,segmentations
		
		
	def load_val(self,downsampling=1):
		
		valIm,valSeg = self.loadImages(iD='val',cont = 1,downsampling=downsampling,isotrope= True)
		patch = patch_Extraction.Patch(self.patch_len)
		valIm,valSeg = patch.load_Patch(valIm[0],valSeg[0])
		
		s = np.arange(valIm.shape[0])
		np.random.shuffle(s)
		valIm = valIm[s]
		valSeg = valSeg[s]
		
		return valIm,valSeg

	def load_test(self):
		testIm,testSeg = self.loadImages(iD='test',downsampling = 1,isotrope=True)
		patch = patch_Extraction.Patch(self.patch_len)
		testIm_patch,testSeg_patch = patch.load_Patch(testIm[0],testSeg[0])
		return testIm_patch,testSeg_patch,testIm[0],testIm[0].shape

if __name__ == "__main__":

	mydata = Load(viva=True)	
	data,seg = mydata.loadImages('val',downsampling = 1,isotrope= True)
	print data.shape
	'''
	data1,seg1=mydata.feedImage(data,seg)
	print data.shape,'  ',seg.shape
	seg1 = seg1.astype('float32')	
	data1 = data1.astype('float32')	
	dataA = mydata.data_augmentation(data1)
	save(seg1[0],'../../Results/testData_AugSeg.nii')
	save(data1[0],'../../Results/testData_Aug_norm.nii')
	
	'''
