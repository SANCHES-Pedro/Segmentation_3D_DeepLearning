'''
code to extract patchs from full images, either randomly, for using in the generators, 
or loading all possible non-superposed patchs in a image, for predictions.
'''

import random
import sys
sys.path.append('../io')
import read
import numpy as np
import math
from skimage.transform import rescale

class Patch:
	
	def __init__(self,patch_size):
		self.patch_size=patch_size

	def extractPatch(self,d,i, x, y, z):
		patch = d[i,x:x+self.patch_size,y:y+self.patch_size,z:z+self.patch_size]
		return patch

	def RandomPatch(self,d,f,superposed = False,padding=0):
		i = random.randint(0, d.shape[0]-1)
		x = random.randint(0, d.shape[1]-self.patch_size)
		y = random.randint(0, d.shape[2]-self.patch_size)
		z = random.randint(0, d.shape[3]-self.patch_size)
		#print ' i '  ,i,' x ',x,' y ',y,' z ',z
		
		data = self.extractPatch(d,i, x, y, z)
		
		if superposed:
			seg = self.extractPatch(f, self.patch_size-2*padding,i, x+padding, y+padding, z+padding)
		else:
			seg = self.extractPatch(f,i, x, y, z)
		return data,seg
		
	def Random_Patch_Multiresolution(self,data,segmentation):
		i = random.randint(0, data.shape[0]-1)
		x = random.randint(0, data.shape[1]-self.patch_size)
		y = random.randint(0, data.shape[2]-self.patch_size)
		z = random.randint(0, data.shape[3]-self.patch_size)
		#print ' i '  ,i,' x ',x,' y ',y,' z ',z
		
		center_data = data[i,x:x+self.patch_size,y:y+self.patch_size,z:z+self.patch_size]
		center_seg = segmentation[i,x:x+self.patch_size,y:y+self.patch_size,z:z+self.patch_size]
		imgdata_context_down = np.zeros((self.patch_size,self.patch_size,self.patch_size,1))
		context_size = 2.
		pad = int(math.ceil(self.patch_size / context_size))
		pad_data = np.pad(data, ((0,0),(pad, pad),(pad, pad),(pad, pad),(0,0)), mode = 'constant')
		context_data = pad_data[i,x:x+2*self.patch_size,y:y+2*self.patch_size,z:z+2*self.patch_size]
		
		context_data=context_data.astype('float32')
		imgdata_context_down[:,:,:,0] = rescale(context_data[:,:,:,0], 1.0 / context_size, mode = 'constant',multichannel = False,anti_aliasing=True)

		return center_data, imgdata_context_down ,center_seg


	def load_Patch(self,data,dataSeg):
		## cubic patchs		
		patch_len = self.patch_size
		image_shape = np.asarray(data.shape).astype('float16')
		#print 'image_shape',image_shape

		n_paches = np.array([math.ceil(image_shape[0]/patch_len),math.ceil(image_shape[1]/patch_len),math.ceil(image_shape[2]/patch_len)]).astype('uint8')
		#print 'n_paches: ',n_paches,' image_shape: ',image_shape, ' patch_len : ',patch_len
		image_shape = image_shape.astype('uint16')

		imgdata = np.zeros(( np.prod(n_paches) ,patch_len,patch_len,patch_len,1), dtype=np.float32)
		imgSeg = np.zeros(( np.prod(n_paches) ,patch_len,patch_len,patch_len,1), dtype=np.float32)
		
		#print 'n_paches: ',n_paches,' image_shape: ',image_shape,' imgData shape: ',imgdata.shape
		#print "n_patchs ",n_paches," data shape ",imgdata.shape, 'image_shape',image_shape
		
		cont = 0
		
		dict_shape = {'i':0,'j':1,'k':2}

		for i in range(n_paches[0]):
			for j in range(n_paches[1]):
				for k in range(n_paches[2]):
					index = []
					index_patch = []
					dict_num = {'i':i,'j':j,'k':k}
					for a in ['i','j','k']:
						#print a
						index.append(dict_num[a]*patch_len) #index init
						
						if dict_num[a] == n_paches[dict_shape[a]]-1:
							x = image_shape[dict_shape[a]]-(n_paches[dict_shape[a]]-1)*patch_len
							index_patch.append(x)
							if x == -240:
								print a
								print image_shape[dict_shape[a]]
								print(n_paches[dict_shape[a]]-1)*patch_len
								print x
							index.append(None)
						else:
							index.append((dict_num[a]+1)*patch_len)
							index_patch.append(None)
					#print index
					#print index_patch
					imgdata[cont,:index_patch[0],:index_patch[1],:index_patch[2]] = data[index[0]:index[1],index[2]:index[3],index[4]:index[5]]
					imgSeg[cont,:index_patch[0],:index_patch[1],:index_patch[2]] = dataSeg[index[0]:index[1],index[2]:index[3],index[4]:index[5]]
					cont += 1
		print 'patched image shape: ',imgdata.shape
		return imgdata,imgSeg

	def load_Patch_Multiresolution(self,data,dataSeg):
		## cubic patchs		
		patch_len = self.patch_size
		image_shape = np.asarray(data.shape).astype('float16')
		print 'image_shape',image_shape

		n_paches = np.array([math.ceil(image_shape[0]/patch_len),math.ceil(image_shape[1]/patch_len),math.ceil(image_shape[2]/patch_len)]).astype('uint8')
		#print 'n_paches: ',n_paches,' image_shape: ',image_shape, ' patch_len : ',patch_len
		image_shape = image_shape.astype('uint16')
		pad = int(math.ceil(self.patch_size / 2.))
		pad_data = np.pad(data, ((pad, pad),(pad, pad),(pad, pad),(0,0)), mode = 'constant')
		context_size = 2
		
		imgdata = np.zeros(( np.prod(n_paches) ,patch_len,patch_len,patch_len,1), dtype=np.float32)
		imgdata_context_down = np.zeros(( np.prod(n_paches) ,patch_len,patch_len,patch_len,1), dtype=np.float32)
		imgdata_context = np.zeros(( context_size*patch_len,context_size*patch_len,context_size*patch_len,1), dtype=np.float32)
		imgSeg = np.zeros(( np.prod(n_paches) ,patch_len,patch_len,patch_len,1), dtype=np.float32)
		
		#print 'n_paches: ',n_paches,' image_shape: ',image_shape,' imgData shape: ',imgdata.shape
		#print "n_patchs ",n_paches," data shape ",imgdata.shape, 'image_shape',image_shape

		cont = 0
			
		dict_shape = {'i':0,'j':1,'k':2}

		for i in range(n_paches[0]):
			for j in range(n_paches[1]):
				for k in range(n_paches[2]):
					index = []
					index_patch = []
					index_context = []
					index_patch_context = []
					dict_num = {'i':i,'j':j,'k':k}
					for a in ['i','j','k']:
						#print a
						index.append(dict_num[a]*patch_len) #index init
						index_context.append(dict_num[a]*patch_len) #index init

						if dict_num[a] == n_paches[dict_shape[a]]-1:
							x = image_shape[dict_shape[a]]-(n_paches[dict_shape[a]]-1)*patch_len
							index_patch.append(x)
							index_patch_context.append(x+2*pad)
							
							index_context.append(None)
							index.append(None)
						else:
							index_context.append((dict_num[a]+1)*patch_len+2*pad)
							index.append((dict_num[a]+1)*patch_len)
							
							index_patch.append(None)
							index_patch_context.append(None)

					imgdata[cont,:index_patch[0],:index_patch[1],:index_patch[2]] = data[index[0]:index[1],index[2]:index[3],index[4]:index[5]]
					
					imgdata_context[:index_patch_context[0],:index_patch_context[1],:index_patch_context[2]] = pad_data[index_context[0]:index_context[1],
							index_context[2]:index_context[3],index_context[4]:index_context[5]]

					imgSeg[cont,:index_patch[0],:index_patch[1],:index_patch[2]] = dataSeg[index[0]:index[1],index[2]:index[3],index[4]:index[5]]
					imgdata_context_down[cont,:,:,:,0] = rescale(imgdata_context[:,:,:,0], 1.0 / float(context_size), mode = 'constant',multichannel = False,anti_aliasing=True)

					cont += 1
		print 'patched image shape: ',imgdata_context_down.shape

		return imgdata,imgdata_context_down,imgSeg

