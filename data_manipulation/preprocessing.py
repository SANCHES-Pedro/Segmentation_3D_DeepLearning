import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepro(data,verbose = False):
	
	binss = 50
	data = np.asarray(data)
	data = data.astype('float32')
	
	if verbose:
		plt.figure(1)
		plt.subplot(221)
		plt.hist(data.flatten(),bins= binss)
		#plt.xlabel('Intensities')
		plt.ylabel('number of voxels')
		plt.title('Raw Data')
		plt.grid(True)

	clipping = (data.max()-data.min())/4
	data = data.clip(min=0, max=clipping)
	
	if verbose:
		plt.subplot(222)
		plt.hist(data.flatten(),bins= binss)
		#plt.xlabel('Intensities')
		plt.ylabel('number of voxels')
		plt.title('Clipped')
		#plt.text(60, .025, 'clipping value = ')
		plt.grid(True)
	
	data = data/np.max(data)
	#data = (data-np.mean(data))/np.var(data) #normalization


	if verbose:
		plt.subplot(223)
		plt.hist(data.flatten(),bins= binss)
		plt.xlabel('Intensities')
		plt.ylabel('number of voxels')
		plt.title('Normalized')
		plt.grid(True)

	data = np.power(data,2)
	
	if verbose:
		plt.subplot(224)
		plt.hist(data.flatten(),bins= binss)
		plt.xlabel('Intensities')
		plt.ylabel('number of voxels')
		#plt.text(0,7, 1, 'p = 2')
		plt.title('Projected')
		plt.grid(True)
		plt.subplots_adjust(top=0.95, bottom=0.12, left=0.2, right=0.95, hspace=0.25,
                    wspace=0.55)
		#plt.show()
		plt.savefig('preprocessing_histograms.pdf',orientation = 'landscape')

	'''
	a = np.concatenate([raw_hist.reshape([len(raw_hist),1]), raw_edges[1:].reshape([len(raw_hist),1]),clip_hist.reshape([len(raw_hist),1]),
	 clip_edges[1:].reshape([len(raw_hist),1]),norm_hist.reshape([len(raw_hist),1]), norm_edges[1:].reshape([len(raw_hist),1]),
	 pow_hist.reshape([len(raw_hist),1]), pow_edges[1:].reshape([len(raw_hist),1])],axis=1)

	df = pd.DataFrame(a,
	columns=['raw_hist', 'raw_edges','clip_hist', 'clip_edges','norm_hist', 'norm_edges','pow_hist', 'pow_edges' ])
	
	df.to_csv('preprocessing_hist.csv')
	'''
	return data
