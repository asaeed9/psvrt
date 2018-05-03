from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import cv2

list_of_imgs = []
img_dir_same = "./gen_script/data_sample/1"
img_dir_diff = "./gen_script/data_sample/0"
def load_data(img_dir):
	for img in os.listdir(img_dir):
		print(img)
    		img = os.path.join(img_dir, img)
    		print(img)	
		if not img.endswith(".png"):
        		continue
    		a = cv2.imread(img)
    		
		if a is None:
        		print ("Unable to read image{}".format(img))
        		continue
    		
		list_of_imgs.append(a.flatten())
	train_data = np.array(list_of_imgs)
	return train_data
	
if __name__ == "__main__":
	train_data = load_data(img_dir_same)
	print(train_data.shape()))

