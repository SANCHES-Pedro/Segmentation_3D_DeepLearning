import sys
import numpy as np 
import os
import glob
from medpy.io import load,save,header
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import random
import nibabel as nib
import time
from dipy.align.reslice import reslice
import scipy
from skimage import util,filters,exposure

## original shape (448,448,128)
class dataPre(object):

	def __init__(self,patch_len = 64 ,batch_size = 1, train_Im_path = "train_Images", train_Seg_path = "train_Seg", test_Im_path = "test_Images",
	 test_Seg_path = "test_Seg", validation_Im_path = "validation_Images",validation_Seg_path = "validation_Seg"):	
		 
		self.patch_len = patch_len
		self.train_Im_path = train_Im_path
		self.train_Seg_path = train_Seg_path
		self.test_Im_path = test_Im_path
		self.test_Seg_path = test_Seg_path
		self.validation_Im_path = validation_Im_path
		self.validation_Seg_path = validation_Seg_path
		self.batch_size = batch_size

	def niiToNp(self,filename,isotrope = True):

		img = nib.load(filename)
		data = img.get_data()
		affine = img.affine
		zooms = img.header.get_zooms()[:3]
		new_zooms = (1,1,1)
		if isotrope:
			data, affine = reslice(data, affine, zooms, new_zooms)
			data= data[::2,::2,::2]
			pad = []
			for i in range(3):
				if (data.shape[i]%8)%2 == 0:
					pad.append(((8-data.shape[i]%8)/2,(8-data.shape[i]%8)/2))
				else:
					pad.append(((8-data.shape[i]%8)/2,(8-data.shape[i]%8)/2+1))

			padding_array =  (pad[0],pad[1],pad[2])
			
			data = np.pad(data,padding_array,mode='constant')
			
		else:
			data= data[::2,::2,:]

		data= data[:,:,:,np.newaxis]
		data = data.astype('float16')
		#print data.shape
		return data

	def preprocessing(self,data):
		
		clipping = (data.max()-data.min())/4
		data = data.clip(min=None, max=clipping)
		data = data/data.max() #normalization
		data = np.power(data,2)

		return data
		
	def feedImage(self,images,segmentations):
		i = random.randint(0,images.shape[0]-1)
		return images[np.newaxis,i],segmentations[np.newaxis,i]

	def extractPatch(self,d, patch_size,i, x, y, z):
		patch = d[i,x:x+patch_size,y:y+patch_size,z:z+patch_size]
		return patch

	def RandomPatch(self,d,f, patch_size ,superposed = False,padding=0):
		i = random.randint(0, d.shape[0]-1)
		x = random.randint(0, d.shape[1]-patch_size)
		y = random.randint(0, d.shape[2]-patch_size)
		z = random.randint(0, d.shape[3]-patch_size)
		#print ' i '  ,i,' x ',x,' y ',y,' z ',z
		data = self.extractPatch(d, patch_size,i, x, y, z)
		if superposed:
			seg = self.extractPatch(f, patch_size-2*padding,i, x+padding, y+padding, z+padding)
		else:
			seg = self.extractPatch(f, patch_size,i, x, y, z)
		return data,seg

	def loadImages(self,iD):
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
		
		x=2
		if iD=='train':
			x =len(imgs)
			
		images = []
		segmentations= []
		for i in range(x):
			print 'ID: ',iD,'---- imgs ',imgs[i],' ---- seg',segs[i]
			images.append(self.preprocessing(self.niiToNp(imgs[i])))
			segmentations.append(self.niiToNp(segs[i]))
		
		images = np.asarray(images,dtype='float16')
		segmentations = np.asarray(segmentations,dtype='float16')

		print 'images loaded with shape: ',images.shape
		
		return images,segmentations
		
		
	def data_augmentation(self,image):
		#print image.shape

		filt_sigma = random.uniform(0, 0.5)
		gamma_y = random.uniform(0.9, 1.1)
		gamma_G = random.uniform(0.9, 1.1)
		sigmoid_G = random.uniform(0, 3)
		shape = image.shape
		image = image.reshape((shape[1],shape[2],shape[3]))
		
		image = exposure.adjust_gamma(image, gamma=gamma_y, gain=gamma_G)
		image = exposure.adjust_sigmoid(image,gain = sigmoid_G )
		image = filters.gaussian(image, sigma = filt_sigma, preserve_range=True)
		image_var = np.var(image)
		noise_var = random.uniform(0, image_var/100)
		image = util.random_noise(image,var = noise_var)
		
		image = image[np.newaxis,:,:,:,np.newaxis]
		return image
		
	def returnRandomPatchs(self,images,segmentations):
		batch_features = np.zeros((self.batch_size, self.patch_len, self.patch_len, self.patch_len, 1), dtype='float16')
		batch_labels = np.zeros((self.batch_size, self.patch_len, self.patch_len, self.patch_len, 1), dtype='float16')
		
		i = 0
		while i<self.batch_size:
			features,labels = self.RandomPatch(images,segmentations,64)
			if labels.max() != 0:
				batch_features[i],batch_labels[i] = features,labels
				i+=1
		return batch_features, batch_labels

	def load_data_patch_unif64(self,Im_path,Seg_path,cont):
				
		imgs = glob.glob('../../Data/'+Im_path+'/*')
		segs = glob.glob('../../Data/'+Seg_path+'/*')
		data = self.niiToNp(imgs[cont])
		dataSeg = self.niiToNp(segs[cont])

		print 'img: ',imgs[cont], '---- seg: ',segs[cont]
		## cubic patchs		
		patch_len = self.patch_len
		image_shape = np.asarray(data.shape).astype('float16')
		n_paches = np.array([round(image_shape[0]/self.patch_len),round(image_shape[1]/self.patch_len),round(image_shape[2]/self.patch_len)]).astype('uint8')
		#rint 'n_paches: ',n_paches,' image_shape: ',image_shape, ' patch_len : ',patch_len
		imgdata = np.zeros(( np.prod(n_paches) ,patch_len,patch_len,patch_len,1), dtype=np.float32)
		imgSeg = np.zeros(( np.prod(n_paches) ,patch_len,patch_len,patch_len,1), dtype=np.float32)
		
		#print 'n_paches: ',n_paches,' image_shape: ',image_shape,' imgData shape: ',imgdata.shape
		#print "n_patchs ",n_paches," data shape ",image_shape
		
		
		conter = 0
		for i in range(n_paches[0]):
			for j in range(n_paches[1]):
				for k in range(n_paches[2]):
					if i == n_paches[0]-1 and j ==n_paches[1]-1:
						imgdata[cont,:32,:32,:]  = data[i*patch_len:i*patch_len+(patch_len/2),j*patch_len:j*patch_len+(patch_len/2),
						k*patch_len:(k+1)*patch_len] 
						imgSeg[cont,:32,:32,:]  = dataSeg[i*patch_len:i*patch_len+(patch_len/2),j*patch_len:j*patch_len+(patch_len/2),
						k*patch_len:(k+1)*patch_len] 
					elif i == n_paches[0]-1 and j !=n_paches[1]-1:
						imgdata[cont,:32,:,:] = data[i*patch_len:i*patch_len+(patch_len/2),j*patch_len:(j+1)*patch_len,
						k*patch_len:(k+1)*patch_len]
						imgSeg[cont,:32,:,:] = dataSeg[i*patch_len:i*patch_len+(patch_len/2),j*patch_len:(j+1)*patch_len,
						k*patch_len:(k+1)*patch_len]
					elif i != n_paches[0]-1 and j == n_paches[1]-1:
						imgdata[conter,:,:32,:] = data[i*patch_len:(i+1)*patch_len,j*patch_len:j*patch_len+(patch_len/2),
						k*patch_len:(k+1)*patch_len]
						imgSeg[conter,:,:32,:] = dataSeg[i*patch_len:(i+1)*patch_len,j*patch_len:j*patch_len+(patch_len/2),
						k*patch_len:(k+1)*patch_len]
					else:
						imgdata[conter] = data[i*patch_len:(i+1)*patch_len,j*patch_len:(j+1)*patch_len,k*patch_len:(k+1)*patch_len]
						imgSeg[conter] = dataSeg[i*patch_len:(i+1)*patch_len,j*patch_len:(j+1)*patch_len,k*patch_len:(k+1)*patch_len]
					#print conter
					conter += 1
		return imgdata,imgSeg


	def load_val(self):
		
		valIm,valSeg = self.load_data_patch_unif64(self.validation_Im_path,self.validation_Seg_path,1)
		s = np.arange(valIm.shape[0])
		np.random.shuffle(s)
		valIm = valIm[s]
		valSeg = valSeg[s]
		return valIm[:2],valSeg[:2]

	def load_test(self,iD = 1):
		testIm,testSeg = self.load_data_patch_unif64(self.test_Im_path,self.test_Seg_path,iD)
		return testIm[:4],testSeg[:4]

if __name__ == "__main__":
	
	mydata = dataPre()	
	data,seg = mydata.loadImages('val')
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
