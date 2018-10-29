import os 
import tensorflow as tf
import sys
from medpy.io import load,save

import numpy as np
import keras
from keras import metrics
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from preprocessing_TF import dataPre
from model_TF import *
from keras import backend as K

from ..data_manipulation.data_generator import *
from ..data_manipulation.data_Load import *
from ..utils.losses import *
from ..utils.metrics import *
from ..models.models_Keras import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Trainer:

	def __init__(self,num_epoch=10,steps_per_epoch=1,deph=8,multi_gpu=False,name_file='test'):
		
		self.name_file = name_file
		self.saveDir="../../my_weights/"+name_file
		if not os.path.exists(self.saveDir):
			os.makedirs(self.saveDir)
			
		self.steps_per_epoch = steps_per_epoch
		self.num_epoch = num_epoch
		self.data = tf.placeholder(name="data", dtype=tf.float32, shape=[1,None,None,None,1])
		self.seg = tf.placeholder(name="seg", dtype=tf.float32, shape=[1,None,None,None,1]) #ground truth
		
		self.lr = tf.get_variable("learningRate", initializer=0.001, trainable=False)
		tf.summary.scalar("learningRate",self.lr)

		self.output = myModel(deph = deph)(self.data,iD = 'vnet')
		
		with tf.name_scope("loss_dice"):
			self.loss = dice_coef_loss(self.seg,self.output)
			tf.summary.scalar("loss_dice",self.loss)

		with tf.name_scope("train"):
			self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		self.verbose=True
		self.merge = tf.summary.merge_all()

	def saveWeights(self,savedir):
		#tf.saved_model.simple_save(self.sess,savedir,inputs={"data": self.data},outputs={"output": self.output})
		save_path = self.saver.save(self.sess, savedir)
		
	def close(self):
		self.sess.close()
		tf.reset_default_graph()

	def fit(self, X_train, Y_train):
		[loss_val,opt] = self.sess.run([self.loss, self.opt],feed_dict={self.data:X_train,self.seg:Y_train})
		return	loss_val

	def validate(self, X_valid, Y_valid):
		[loss_val] = self.sess.run([self.loss], feed_dict={self.data:X_valid,self.seg:Y_valid})
		return loss_val
	
	def load_Predict(self):
		#data
		mydata = dataPre()
		data_test,seg_test = mydata.loadImages("test")

		#model
		self.saver.restore(self.sess, self.saveDir)
		#tf.saved_model.loader.load(self.sess,[tag_constants.SERVING],self.saveDir)
		#predict
		output,loss_val = self.sess.run([self.output,self.loss], feed_dict={self.data: data_test,self.seg:seg_test})
		save(output[0],'../../Results/test21.nii')
		print loss_val
		return output,loss_val

	def lr_scheduler(self,epoch):
		lr_max = 0.004
		lr_min = 0.00002
		step_size = 10
		gamma = 0.99
		change = 5*step_size
		if epoch<change:
			lr = self.lr_cycle(epoch,lr_max,lr_min,step_size)
		else:
			lr = self.lr_exp(epoch,gamma,lr_min,change)	
		print 'Learning rate: ',lr
		return lr
		
	def lr_exp(self,epoch,gamma,init,change):
		return init*(gamma**(epoch-change))
		
	def lr_cycle(self,epoch,lr_max,lr_min,step_size):
		step = epoch%(2*step_size)
		if step<step_size:
			lr = lr_max - step*(lr_max-lr_min)/step_size
		else:
			lr = lr_min + (step-step_size)*(lr_max-lr_min)/step_size
		return lr
	def lr_finder(self,epoch):
		lr1 = 1e-9*(0.5**(-epoch)) #30 epochs
		print lr1
		return lr1
		
	def train(self):
		print 'init_train'
		#init variables 
		val_loss_max = 0
		loss_max = 0
		
		self.sess.run(tf.global_variables_initializer())
		
		#tensorboard Writer
		writer = tf.summary.FileWriter('../../my_logs/' + self.name_file)
		writer.add_graph(self.sess.graph)
		print 'init_data_load'
		
		#Preprocessing images load
		mydata = dataPre()
		data_val,seg_val = mydata.loadImages("val")
		data_train,seg_train = mydata.loadImages("train")
		print 'data_loaded'

		try:
			itr = 0
			for epoch in range(self.num_epoch):
				self.sess.run(self.lr.assign(self.lr_scheduler(epoch))) #scheduler ->> self.lr_scheduler(epoch)
				for step in range(self.steps_per_epoch):
					
					#feeds the patchs used for training
					data,seg = mydata.feedImage(data_train,seg_train)
					
					#data augmentation white noise and gaussian filter
					#data = mydata.data_augmentation(data)
					
					#train
					loss = self.fit(data,seg)
					print '-',
					#for ploting
					itr += 1

				#validation

				data1,seg1 = mydata.feedImage(data_val,seg_val)
				val_loss = self.validate(data1,seg1)
				
				[_, s] = self.sess.run([self.loss, self.merge], feed_dict={self.data:data1,self.seg:seg1})
				writer.add_summary(s, epoch)

				## save best model
				if loss<loss_max:
					self.saveWeights(self.saveDir)
					print 'Model saved'
					loss_max=loss
				if val_loss<val_loss_max:
					self.saveWeights(self.saveDir+"_val")
					print 'Model Val saved'
					val_loss_max=val_loss
				print ' '
				print 'epoch => [%d / %d] , train_loss : %f , val_loss : %f' % (epoch,self.num_epoch,loss,val_loss)
		except  KeyboardInterrupt:
			print("close")
		self.close()

if __name__ == '__main__':
	
	unet_train = Trainer(num_epoch =200 ,steps_per_epoch=20,deph=8,name_file='tfFullImageVnet')
	unet_train.train()
	#output,loss = unet_train.load_Predict()
	#print output.shape
