import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd


def get_bbox(output):
	bbox = {}
	conf = output[1]
	center_x = output[2][0]
	center_y = output[2][1]
	bbox_width = output[2][2]
	bbox_height = output[2][3]

	if conf > 0.6:
		bbox["topleft_x"] = center_x - (bbox_width/2)
		bbox["topleft_y"] = center_y - (bbox_height/2)
		bbox["bottomright_x"] = center_x + (bbox_width/2)
		bbox["bottomright_y"] = center_y + (bbox_height/2)
		bbox["label"] = output[0].decode()
		return bbox


def get_bbox_img(img, bounding_box_info):
	# opencv to pillow 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_pil = Image.fromarray(img)

	if bounding_box_info==None:
		return img, "None"
	else:
		# crop img
		topleft_x, topleft_y, bottomright_x, bottomright_y, label = bounding_box_info.values()
		img_crop = img_pil.crop((topleft_x, topleft_y, bottomright_x, bottomright_y))
		img_crop = img_crop.resize((64, 64), Image.LANCZOS)

		# pillow to opencv
		img_crop = np.asarray(img_crop)
		img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)

		return img_crop, label


def save_img(img, label, index):
	save_anno_path = "../train/annotation.csv"
	img_path = "data/train/"+label+"/"+ str(index) +".png"
	save_img_path = "../train/"+label+"/"+ str(index) +".png"

	df = pd.DataFrame([[img_path,  index]])
	df.to_csv(save_anno_path, mode='a', header=False, index=False)

	# save an image
	cv2.imwrite(save_img_path, img)

