"""
class to return the models, many different architectures are possible
"""

import keras
from keras.models import *
from keras.layers import Input, Concatenate,concatenate, Conv3D,MaxPooling3D, UpSampling3D, Dropout, Cropping3D,BatchNormalization , ZeroPadding3D, add,ELU,Activation,PReLU,SpatialDropout3D,LeakyReLU,Multiply,Subtract,Conv3DTranspose,Add
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from keras.utils import plot_model,multi_gpu_model
from tensorflow.python.client import device_lib
from keras import regularizers
import numpy as np
from keras.engine.network import *
from keras import backend as K

class Models:

	def __init__(self,batch_size=1,patch_size=64,deph = 6,dropout_rate = 0.2,batch_norm = 1,kernel_size = 3,spatialdrop = 1):
		
		self.padding = 'same' # padding in the convolutions, use same with you want to keep the features map size after each convolution
		self.batch_size = batch_size
		self.patch_size = patch_size
		self.deph = deph
		self.batch_norm = batch_norm
		self.kernel_size = kernel_size
		self.spatialdrop = spatialdrop
		self.dropout_rate = dropout_rate

	#just pass the name of the architecture for return the right model
	def __call__(self,net):
		
		inputs = Input(batch_shape=(self.batch_size,self.patch_size,self.patch_size,self.patch_size,1))		## 64

		if net == 'unet':
			output = self.get_unet(inputs)
		elif net == 'vnet':
			output = self.get_vnet(inputs)
		elif net == 'vnet_original':
			output = self.get_vnet_original(inputs)
		elif net == 'uception':
			output = self.get_Uception(inputs)
			
		model = Model(inputs,output)
		
		model.summary()
		#plot_model(model, show_shapes=True, to_file= net+'.png')
		print model.count_params()
		return model
	
	#Inception module that halves the shape of the features map in each dimension
	def inception_reduction(self,deph,X):
		Y1 = Conv3D(deph, 1,strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(X)
		Y1 = self.drop(Y1)
		Y1 = Conv3D(deph, (1,1,5), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y1)
		Y1 = Conv3D(deph, (1,5,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y1)
		Y1 = Conv3D(deph, (5,1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y1)

		Y2 = Conv3D(deph, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(X)
		Y2 = self.drop(Y2)
		Y2 = Conv3D(deph, 5,strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(X)

		Y3 = MaxPooling3D(pool_size=2)(X)
		Y3 = Conv3D(deph, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y3)

		Y = Concatenate()([Y1,Y2,Y3])
		return Y
	
	#Inception module that keeps the feature map shape
	def inception_deep(self,deph,X):
		Y1 = Conv3D(deph, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(X)
		Y1 = self.drop(Y1)

		Y1 = Conv3D(deph, (1,1,7), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y1)
		Y1 = Conv3D(deph, (1,7,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y1)
		Y1 = Conv3D(deph, (7,1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y1)

		Y2 = Conv3D(deph, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(X)
		Y2 = self.drop(Y2)

		Y2 = Conv3D(deph, (1,1,7), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y2)
		Y3 = Conv3D(deph, (1,7,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y2)
		Y4 = Conv3D(deph, (7,1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y2)
		
		Y5 = Conv3D(deph, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(X)
		Y5 = self.drop(Y5)

		Y5 = Conv3D(deph, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(Y5)
		
		Y = Concatenate()([Y1,Y2,Y3,Y4,Y5])
		
		return Y
	
	#Encoder-Decoder network with Inception modules and shortcuts inspired in V/U net's topology
	def get_Uception(self,inputs):
		
		Y = Conv3D(64, 5, padding = self.padding, kernel_initializer = 'he_normal')(inputs)
		Y = Activation('relu')(Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = Conv3D(64, 5, padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)

		jump1 = Activation('relu')(Y)
		jump2 = self.inception_reduction(self.deph*2,jump1)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		jump2 = self.drop(jump2)
		jump3 = self.inception_reduction(self.deph*4,jump2)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		jump3 = self.drop(jump3)
		
		Y = self.inception_reduction(self.deph*8,jump3)
		Y = self.drop(Y)
		
		Y = self.inception_deep(self.deph*16,Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = SpatialDropout3D(self.dropout_rate)(Y)
		
		Y = UpSampling3D(size=(2, 2, 2))(Y)
		Y = Concatenate()([jump3,Y])
		Y = self.inception_deep(self.deph*8,Y)
		Y = self.drop(Y)

		Y = UpSampling3D(size=(2, 2, 2))(Y)
		Y = Concatenate()([jump2,Y])
		Y = self.inception_deep(self.deph*4,Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = self.drop(Y)

		Y = UpSampling3D(size=(2, 2, 2))(Y)
		Y = Concatenate()([jump1,Y])
		Y = self.inception_deep(self.deph*2,Y)
		Y = self.drop(Y)

		Y = Conv3D(64, 5, padding = self.padding, kernel_initializer = 'he_normal')(Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = Activation('relu')(Y)
		Y = self.drop(Y)
		Y = Conv3D(1, 1, activation = 'sigmoid')(Y)

		return Y
		

	def get_vnet(self,inputs):
		
		Y = Conv3D(32, self.kernel_size, padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(inputs)
		Y = Activation('relu')(Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = self.drop(Y)
		Y = Conv3D(32, self.kernel_size, padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		jump1 = Activation('relu')(Y)

		Y = Conv3D(self.deph*2, 2, padding = self.padding, kernel_initializer = 'he_normal', strides=(2, 2, 2))(jump1)
		Down = Activation('relu')(Y)
		Y = Conv3D(self.deph*2, self.kernel_size,  padding = self.padding, kernel_initializer = 'he_normal')(Down)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(self.deph*2, self.kernel_size,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		jump2 = Activation('relu')(Y)
		
		Y = Conv3D(self.deph*4, 2,  padding = self.padding, kernel_initializer = 'he_normal', strides=(2, 2, 2))(Concatenate()([Down,jump2]))
		Down = Activation('relu')(Y)
		Y = Conv3D(self.deph*4, self.kernel_size, padding = self.padding, kernel_initializer = 'he_normal')(Down)
		Y = self.drop(Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(self.deph*4, self.kernel_size,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		jump3 = Activation('relu')(Y)
		
		
		Y = Conv3D(self.deph*8, 2,  padding = self.padding, kernel_initializer = 'he_normal', strides=(2, 2, 2))(Concatenate()([Down,jump3]))
		Down = Activation('relu')(Y)
		
		Y = Conv3D(self.deph*16, self.kernel_size,padding = self.padding, kernel_initializer = 'he_normal')(Down)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(self.deph*16, self.kernel_size, padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		Y = self.drop(Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = Activation('relu')(Y)
		Y = Concatenate()([Down,Y])
		
		Y = Conv3D(self.deph*8, 2,  padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = 2)(Y))
		Up = Activation('relu')(Y)
		
		Y = Concatenate()([jump3,Up])
		
		Y = Conv3D(self.deph*4, self.kernel_size,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(self.deph*4, self.kernel_size,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		
		Y = Concatenate()([Y,Up])

		Y = Conv3D(self.deph*4, 2,  padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = 2)(Y))
		Up = Activation('relu')(Y)
		
		Y = Concatenate()([jump2,Up])
		
		Y = Conv3D(self.deph*4, self.kernel_size,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(self.deph*2, self.kernel_size,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)

		Y = Concatenate()([Y,Up])

		Y = Conv3D(self.deph*2, 2,  padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = 2)(Y))
		Y = Activation('relu')(Y)
		Y = Concatenate()([jump1,Y])
		Y = self.drop(Y)
		Y = Conv3D(64, self.kernel_size, padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(64, self.kernel_size, padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		Y = self.drop(Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(1, 1, activation = 'sigmoid')(Y)
		return Y

	def get_vnet_original(self,inputs):
		
		Y = Conv3D(self.deph, 5, padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(inputs)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = PReLU()(Y)
		jump1 = self.drop(Y)

		Y = Conv3D(self.deph*2, 2, padding = self.padding, kernel_initializer = 'he_normal', strides=(2, 2, 2))(jump1)
		Down = Activation('relu')(Y)
		Y = Conv3D(self.deph*2, 5,  padding = self.padding, kernel_initializer = 'he_normal')(Down)
		Y = Activation('relu')(Y)
		Y = self.drop(Y)
		Y = Conv3D(self.deph*2, 5,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = Activation('relu')(Y)
		jump2 = self.drop(Y)
		
		Y = Conv3D(self.deph*4, 2,  padding = self.padding, kernel_initializer = 'he_normal', strides=(2, 2, 2))(Add()([Down,jump2]))
		Down = Activation('relu')(Y)
		Y = Conv3D(self.deph*4, 5, padding = self.padding, kernel_initializer = 'he_normal')(Down)
		Y = Activation('relu')(Y)
		Y = self.drop(Y)
		Y = Conv3D(self.deph*4, 5,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		jump3 = Activation('relu')(Y)
		
		
		Y = Conv3D(self.deph*8, 2,  padding = self.padding, kernel_initializer = 'he_normal', strides=(2, 2, 2))(Add()([Down,jump3]))
		Down = Activation('relu')(Y)
		
		Y = Conv3D(self.deph*8, 5,padding = self.padding, kernel_initializer = 'he_normal')(Down)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = Activation('relu')(Y)
		Y = self.drop(Y)
		Y = Conv3D(self.deph*8, 5, padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = Activation('relu')(Y)
		Y = self.drop(Y)
		Y = Add()([Down,Y])
		
		Y = Conv3DTranspose(filters=self.deph*4,kernel_size =4,strides=2,padding='same',kernel_initializer='he_normal')(Y)
		Up = Activation('relu')(Y)
		
		Y = Add()([jump3,Up])
		
		Y = Conv3D(self.deph*4, 5,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(self.deph*4, 5,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		
		Y = Add()([Y,Up])
		
		Y = Conv3DTranspose(filters=self.deph*2,kernel_size =4,strides=2,padding='same',kernel_initializer='he_normal')(Y)
		Up = Activation('relu')(Y)
		
		Y = Add()([jump2,Up])
		
		Y = Conv3D(self.deph*2, 5,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(self.deph*2, 5,  padding = self.padding, kernel_initializer = 'he_normal')(Y)
		Y = self.drop(Y)
		Y = Activation('relu')(Y)
		
		Y = Add()([Y,Up])
		
		Y = Conv3DTranspose(filters=self.deph,kernel_size =4,strides=2,padding='same',kernel_initializer='he_normal')(Y)
		Y = PReLU()(Y)
		Y = Add()([jump1,Y])
		Y = self.drop(Y)
		Y = Conv3D(self.deph, 5, padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		if self.batch_norm:
			Y = BatchNormalization()(Y)
		Y = PReLU()(Y)
		Y = self.drop(Y)	
		Y = Conv3D(1, 1, activation = 'sigmoid')(Y)
		return Y
		
	def get_unet(self,inputs):

		#
		conv_1 = Conv3D(self.deph*2, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
		conv_1 = self.drop(conv_1)
		conv_1 = Conv3D(self.deph*2, self.kernel_size, padding='same', kernel_initializer='he_normal',use_bias=False)(conv_1)
		if self.batch_norm:
			conv_1 = BatchNormalization()(conv_1)
		conv_1 = Activation('relu')(conv_1)
		pool_1 = MaxPooling3D(2)(conv_1)

		#
		conv_2 = Conv3D(self.deph*4, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
		conv_2 = self.drop(conv_2)
		conv_2 = Conv3D(self.deph*4, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)
		pool_2 = MaxPooling3D(2)(conv_2)

		#
		conv_3 = Conv3D(self.deph*8, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal',use_bias=False)(pool_2)
		conv_3 = self.drop(conv_3)
		if self.batch_norm:
			conv_3 = BatchNormalization()(conv_3)
		conv_3 = Activation('relu')(conv_3)
		conv_3 = Conv3D(self.deph*8, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)
		pool_3 = MaxPooling3D(2)(conv_3)

		#
		conv_4 = Conv3D(self.deph*12, self.kernel_size, padding='same', kernel_initializer='he_normal',use_bias=False)(pool_3)
		conv_4 = self.drop(conv_4)
		if self.batch_norm:
			conv_4 = BatchNormalization()(conv_4)
		conv_4 = Activation('relu')(conv_4)

		conv_4 = Conv3D(self.deph*12, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_4)

		#
		up_1 = UpSampling3D(size=2)(conv_4)
		up_1 = concatenate([conv_3, up_1], axis=4)
		conv_6 = Conv3D(self.deph*8, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(up_1)
		conv_6 = self.drop(conv_6)
		conv_6 = Conv3D(self.deph*8, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_6)

		#
		up_2 = UpSampling3D(size=2)(conv_6)
		up_2 = concatenate([conv_2, up_2], axis=4)
		conv_7 = Conv3D(self.deph*4, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(up_2)
		conv_7 = self.drop(conv_7)
		conv_7 = Conv3D(self.deph*4, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_7)

		#
		up_3 = UpSampling3D(size=2)(conv_7)
		up_3 = concatenate([conv_1, up_3], axis=4)
		conv_8 = Conv3D(self.deph*2, self.kernel_size, padding='same', kernel_initializer='he_normal')(up_3)
		conv_8 = self.drop(conv_8)
		if self.batch_norm:
			conv_8 = BatchNormalization()(conv_8)
		conv_8 = Activation('relu')(conv_8)

		conv_8 = Conv3D(self.deph*2, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_8)
		#
		conv_10 = Conv3D(32, self.kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_8)
		
		conv_11 = Conv3D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_10)

		return conv_11
		
	def get_segmentor(self):
		inputs = Input(batch_shape=(self.batch_size,self.patch_size,self.patch_size,self.patch_size,1))		## 64
		self.deph = 16
		
		encoder_1 = self.SegAn_encoder(inputs,self.deph,downsample_kernel = 7)		##32
		encoder_1 = self.drop(encoder_1)
		encoder_2 = self.SegAn_encoder(encoder_1,self.deph*2,downsample_kernel = 5)	##16
		encoder_2 = self.drop(encoder_2)
		encoder_3 = self.SegAn_encoder(encoder_2,self.deph*4,downsample_kernel = 4)	##8
		encoder_3 = self.drop(encoder_3)
		encoder_4 = self.SegAn_encoder(encoder_3,self.deph*8,downsample_kernel = 3)	##4
		encoder_4 = self.drop(encoder_4)
		
		decoder_4 = self.SegAn_decoder(encoder_4,self.deph*8)	##8
		decoder_4 = Concatenate()([decoder_4,encoder_3])
		decoder_3 = self.SegAn_decoder(decoder_4,self.deph*4)	##16
		decoder_3 = Concatenate()([decoder_3,encoder_2])
		decoder_2 = self.SegAn_decoder(decoder_3,self.deph*2)	##32
		decoder_2 = Concatenate()([decoder_2,encoder_1])
		decoder_1 = self.SegAn_decoder(decoder_2,self.deph)	##64
		
		Y = Conv3D(16, 3 , padding = self.padding, kernel_initializer = 'he_normal')(decoder_1)
		Y = Activation('relu')(Y)

		output = Conv3D(1, 1, activation = 'sigmoid')(decoder_1)
		
		model = Model(inputs,output)
		model_fixed = Network(inputs,output)
		model.summary()
		plot_model(model, show_shapes=True, to_file='v-net_original.png')
		return model,model_fixed
	
	def get_discriminator(self):
		self.deph = 50

		ground_truth = Input(batch_shape=(self.batch_size,self.patch_size,self.patch_size,self.patch_size,1))
		prediction = Input(batch_shape=(self.batch_size,self.patch_size,self.patch_size,self.patch_size,1))
		image = Input(batch_shape=(self.batch_size,self.patch_size,self.patch_size,self.patch_size,1))

		mult = Multiply()
		gt = mult([image,ground_truth])
		pred = mult([image,prediction])
		
		conv1 = self.SegAn_discriminator_conv(channels = 1,deph=self.deph,kernel_size = 15)
		gt1 = conv1(gt)
		pred1 = conv1(pred)
		
		conv2 = self.SegAn_discriminator_conv(channels = self.deph,deph=self.deph*2,kernel_size = 11)
		gt2 = conv2(gt1)
		pred2 = conv2(pred1)
		
		conv3 = self.SegAn_discriminator_conv(channels = self.deph*2,deph=self.deph*4,kernel_size = 9)
		gt3 = conv3(gt2)
		pred3 = conv3(pred2)
		
		gt1 = ZeroPadding3D(mult.output_shape[1]/4)(gt1)
		gt2 = ZeroPadding3D(mult.output_shape[1]*3/8)(gt2)
		gt3 = ZeroPadding3D(mult.output_shape[1]*7/16)(gt3)
		gt_conc = Concatenate()([gt,gt1,gt2,gt3])
		
		pred1 = ZeroPadding3D(mult.output_shape[1]/4)(pred1)
		pred2 = ZeroPadding3D(mult.output_shape[1]*3/8)(pred2)
		pred3 = ZeroPadding3D(mult.output_shape[1]*7/16)(pred3)
		pred_conc = Concatenate()([pred,pred1,pred2,pred3])
		
		
		
		out = Subtract()([gt_conc,pred_conc])
		
		model = Model(inputs=[image,prediction,ground_truth],outputs = out)
		model_fixed = Network(inputs=[image,prediction,ground_truth],outputs = out)
		
		#model.summary()
		#plot_model(model, show_shapes=True, to_file='discriminator_big.png')
		

		return model,model_fixed
		
	def SegAn_discriminator_conv(self,channels,deph = 8,kernel_size=3,LR_alpha = 0.01):
		
		inputs = Input(batch_shape=(None,None,None,None,channels))
		Y = Conv3D(deph, 3 , padding = self.padding, kernel_initializer = 'he_normal',strides=(2, 2, 2),use_bias=False)(inputs)
		Y = BatchNormalization()(Y)
		Y = LeakyReLU(LR_alpha)(Y)
		Y = Conv3D(deph, 5 , padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(inputs)
		Y = BatchNormalization()(Y)
		Y = LeakyReLU(LR_alpha)(Y)
		Y = Conv3D(deph, (1,1,kernel_size),padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		Y = Conv3D(deph, (1,kernel_size,1),padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		Y = Conv3D(deph, (kernel_size,1,1),padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		Y = BatchNormalization()(Y)
		Y = LeakyReLU(LR_alpha)(Y)
		model = Model(inputs,Y)
		return model
		
	def SegAn_encoder(self,X,deph,downsample_kernel,LR_alpha=0.01):
		
		Y = Conv3D(deph, downsample_kernel , padding = self.padding, kernel_initializer = 'he_normal',strides=(2, 2, 2),use_bias=False)(X)
		Y = BatchNormalization()(Y)
		Y1 = LeakyReLU(LR_alpha)(Y)
		
		Y = Conv3D(deph, 1, padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y1)
		Y = BatchNormalization()(Y)
		Y = LeakyReLU(LR_alpha)(Y)
		Y = Conv3D(deph, 5, padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		Y = BatchNormalization()(Y)
		Y = LeakyReLU(LR_alpha)(Y)
		Y = Conv3D(deph, 1, padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		Y = LeakyReLU(LR_alpha)(Y)
		Y = Concatenate()([Y,Y1])

		return Y
		
	def SegAn_decoder(self,X,deph):
		
		Y1 = UpSampling3D(size = 2)(X)
		
		Y = Conv3D(deph, 1,  padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y1)
		#Y = BatchNormalization()(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(deph, 3,  padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		#Y = BatchNormalization()(Y)
		Y = Activation('relu')(Y)
		Y = Conv3D(deph, 1,  padding = self.padding, kernel_initializer = 'he_normal',use_bias=False)(Y)
		Y = BatchNormalization()(Y)
		Y = Activation('relu')(Y)
		
		#Y = Concatenate()([Y,Y1])
		
		return Y

	def drop(self,Y):
		if self.spatialdrop:
			return SpatialDropout3D(self.dropout_rate)(Y)
		else:
			return Dropout(self.dropout_rate)(Y)
if __name__ == '__main__':
	'''
	for deph in [6,10,14]:
		for net in ['unet','vnet_original','uception']:
			print net, ' deph ',deph
			model = Models(batch_size=2,patch_size=64,deph = deph,dropout_rate = 0.2,batch_norm = 1,kernel_size = 5,spatialdrop = 1)(net)
			'''
	model = Models(batch_size=2,patch_size=64,deph = 10,dropout_rate = 0.2,batch_norm = 1,kernel_size = 5,spatialdrop = 1)('uception')
	'''
	m = Models()
	inputs = Input(batch_shape=(1,8,8,8,1))
	y = m.inception_deep(128,inputs)
	model = Model(inputs=inputs, outputs=y)
	model.summary()

	plot_model(model, show_shapes=True, to_file='inception_deep.png')
	'''
