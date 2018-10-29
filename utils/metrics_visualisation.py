import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.figure(1)

file_names = ['Uception_Bullit_multi_SD02D04_BS1','Unet_Bullit_D02_BS1']
for file_name in file_names:
	file_csv = pd.read_csv('../../my_csv/'+file_name+'.csv')
	
	plt.subplot(311)
	plt.plot(file_csv['epoch'],file_csv['loss'], label=file_name)
	plt.xlabel('step')
	plt.ylabel('losses')
	plt.title('Dice coef loss')
	plt.legend()

	plt.subplot(312)
	plt.plot(file_csv['epoch'],file_csv['sensitivity'], label=file_name)
	plt.xlabel('step')
	plt.ylabel('sensitivity')
	plt.title('sensitivity')
	plt.legend()

	plt.subplot(313)
	plt.plot(file_csv['epoch'],file_csv['val_loss'], label=file_name)
	plt.xlabel('step')
	plt.ylabel('val_loss')
	plt.title('val_loss')
	plt.legend()

plt.show()
