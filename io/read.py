import nibabel as nib
import numpy as np
from dipy.align.reslice import reslice
from preprocessing import prepro
import matplotlib.pyplot as plt

class Reader:
	def __init__(self,isotrope=True):
		self.isotrope = isotrope
		
	def __call__(self,filename,downsampling = 1,preprocessing = False,pad = False):
		img = nib.load(filename)
		data = img.get_data()
		affine = img.affine
		zooms = img.header.get_zooms()[:3]
		#new_zooms = (0.5,0.5,0.5)
		new_zooms = (1,1,1)
		
		#make the image isotrope
		if self.isotrope:
			data, affine = reslice(data, affine, zooms, new_zooms) #Trilinear interpolation
		if preprocessing:
			data = prepro(data)

		data= data[::downsampling,::downsampling,::downsampling]
		
		# padding can be done if one wants to input a full image in the network
		# in this case, it's convenient to have a shape that is multiple of a power of 2, in this case 16
		
		if pad:
			pad = []
			deph = 16
			for i in range(3):
				if (data.shape[i]%deph)%2 == 0:
					pad.append(((deph-data.shape[i]%deph)/2,(deph-data.shape[i]%deph)/2))
				else:
					pad.append(((deph-data.shape[i]%deph)/2,(deph-data.shape[i]%deph)/2+1))

			padding_array =  (pad[0],pad[1],pad[2])
			
			data = np.pad(data,padding_array,mode='constant')

		data= data[:,:,:,np.newaxis]
		data = data.astype('float16')
		#print 'data max',data.max() ,' data mean : ',data.mean()
		
		return data
