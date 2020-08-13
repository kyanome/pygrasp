import os
import shutil

def create_img_dir(labels):
	path = "../train/"
	# remove dir if dir exists
	for label in labels:
		if os.path.isdir(path+label):
			shutil.rmtree(path+label)

	for label in labels:
		if not os.path.isdir(path+label):
			os.makedirs(path+label)

def create_anno_file():
	save_anno_path = "../train/annotation.csv"
	
	if os.path.isfile(save_anno_path)==True:
		os.remove(save_anno_path)

	# save annotation file
	if os.path.isfile(save_anno_path)==False:
		with open(save_anno_path,"w"):pass