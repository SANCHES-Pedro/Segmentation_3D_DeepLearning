import random
import numpy as np
from skimage import util,filters,exposure

def data_augmentation(image):
	
	gamma_y = random.uniform(0.9, 1.1)
	gamma_G = random.uniform(0.9, 1.1)
	sigmoid_G = random.uniform(0, 3)
	filt_sigma = random.uniform(0, 0.5)

	shape = image.shape
	image = image.reshape((shape[1],shape[2],shape[3]))

	# constrast random changing
	image = exposure.adjust_gamma(image, gamma=gamma_y, gain=gamma_G)
	image = exposure.adjust_sigmoid(image,gain = sigmoid_G )

	#blur
	image = filters.gaussian(image, sigma = filt_sigma, preserve_range=True)

	#noise
	image_var = np.var(image)
	noise_var = random.uniform(0, image_var/100)
	image = util.random_noise(image,var = noise_var)

	image = image[np.newaxis,:,:,:,np.newaxis]
	return image
