import psvrt
import numpy as np
import matplotlib.pyplot as plt
raw_input_size = [60,60,1] # Size of whole image
batch_size = 1000000

# Change `problem_type` to SR for spatial relation labels
data_parameters = {'problem_type': 'SD',
		   'item_size': [4,4],
		   'box_extent': [40,40],
		   'num_items': 2,
		   'num_item_pixel_values': 1,
		   'SD_portion': 0,
		   'SR_portion': 1,
		   'display': False}

data_generator = psvrt.psvrt(raw_input_size=raw_input_size, batch_size=batch_size) # Load generator
data_generator.initialize_vars(**data_parameters) # Initialize it with image parameters

data = data_generator.single_batch() # Fetch a single batch

stimuli = data[0] # Images
labels = data[1] # Labels

# print(np.shape(stimuli))
# print(np.shape(labels))
labels_temp = np.squeeze(labels[:, 0, 0, 0])
#
# print(np.shape(stimuli[0, :, :, 0]))
# print(np.shape(np.squeeze(stimuli[0, :, :, 0])))
# plt.plot(np.squeeze(stimuli[0,:,:,0]))
# plt.imsave('out.png', np.squeeze(stimuli[0,:,:,0]))

# print(labels_temp)
count = 1
for img, lbl in zip(stimuli, labels_temp):
    img = np.squeeze(img[:,:,0])

    if int(lbl) == 1:
        plt.imsave('./data_sample_7/1/' + str(count) + '13_1.png', img)
    else:
        plt.imsave('./data_sample_7/0/' + str(count) + '13_0.png', img)

    count +=1

# plt.imshow(np.squeeze(stimuli[0,:,:,0]))
# plt.show()
# cv2.imwrite("cpImage.bmp", data[0][0])
