import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.figure(1)

#raw_hist	raw_edges	clip_hist	clip_edges	norm_hist	norm_edges	pow_hist	pow_edges
names = ['hs_uception_patchS64_batchS2_deph6_sdrop0_drop0.25_kernel5_lr0.001_cont17',
		'hs_vnet_patchS64_batchS2_deph6_sdrop1_drop0.15_kernel5_lr0.001_cont7',
		'hs_unet_patchS64_batchS2_deph6_sdrop1_drop0.25_kernel5_lr0.001_cont2']
epoch = range(100)
colors = ['r','b','g','y','c','m']
for i in range(len(names)):
	file_csv = pd.read_csv(names[i]+'.csv')
	plt.plot(epoch,file_csv['loss'],colors[i])

plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('Best runs by model - training')
plt.legend('Uception','Unet','Vnet')

'''
plt.subplot(311)
plt.plot(np.round(file_csv['raw_edges']),file_csv['raw_hist'])
plt.xlabel('Intensities')
plt.ylabel('number of voxels')
plt.title('Raw Data')
plt.legend()

plt.subplot(312)
plt.plot(np.round(file_csv['clip_edges']),file_csv['clip_hist'])
plt.xlabel('step')
plt.ylabel('sensitivity')
plt.title('sensitivity')
plt.legend()

plt.subplot(313)
plt.plot(np.round(file_csv['norm_edges']),file_csv['norm_hist'])
plt.xlabel('step')
plt.ylabel('val_loss')
plt.title('val_loss')
plt.legend()
'''
plt.show()
