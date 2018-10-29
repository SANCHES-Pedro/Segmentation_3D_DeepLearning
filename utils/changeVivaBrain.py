import glob
import nibabel as nib
import numpy as np
from dipy.align.reslice import reslice


imgs = glob.glob('/media/pedro/Data/Estagio/ICube/Code/Data/vivabrain/isotrope/*')


for i in range(len(imgs)):
	image_name = imgs[i][-9:]
	path_img = imgs[i]
	img = nib.load(path_img)
	seg = nib.load('/media/pedro/Data/Estagio/ICube/Code/Data/vivabrain/rorpo/seg_'+image_name)

	data = seg.get_data()
	affine = img.affine
	zooms = img.header.get_zooms()[:3]
	#new_zooms = (1,1,1)

	data, affine = reslice(data, affine, zooms, zooms)

	data_nii = nib.Nifti1Image(data, affine)

	nib.save(data_nii,'/media/pedro/Data/Estagio/ICube/Code/Data/vivabrain/rorpo_rot/seg_'+image_name)

