import numpy as np
import matplotlib.pyplot as plt

img = np.random.rand(300,200,3)
img[240:250,120:200]=0

mask = np.zeros((300,200))
mask[220:260,120:300]=0.7
mask[250:270,140:170]=0.3

f, axarr = plt.subplots(1,2, figsize = (10, 5))
axarr[0].imshow(img)
axarr[1].imshow(mask)
plt.show()

IM_SIZE = 60     # Patch size

x_min, y_min = 0,0
x_max = img.shape[0] - IM_SIZE
y_max = img.shape[1] - IM_SIZE

# print(img.shape)
# print(x_max, y_max)

xd, yd, x, y = 0,0,0,0

if (mask.max() > 0):
    xd, yd = np.where(mask>0)

    # print(np.where(mask>0))
    x_min = xd.min()
    y_min = yd.min()
    x_max = min(xd.max()- IM_SIZE-1, img.shape[0] - IM_SIZE-1)
    y_max = min(yd.max()- IM_SIZE-1, img.shape[1] - IM_SIZE-1)

    print(x_min, y_min, x_max, y_max)

    if (y_min >= y_max):

        y = y_max
        if (y + IM_SIZE >= img.shape[1] ):
            print('Error')

    else:
        y = np.random.randint(y_min,y_max)

    if (x_min>=x_max):

        x = x_max
        if (x+IM_SIZE >= img.shape[0] ):
            print('Error')

    else:
        x = np.random.randint(x_min,x_max )

print(x,y)
n_img = img[x:x+IM_SIZE, y:y+IM_SIZE,:]
n_mask = mask[x:x+IM_SIZE, y:y+IM_SIZE]

f, axarr = plt.subplots(2,2, figsize = (10, 5))
axarr[0][0].imshow(img)
axarr[0][1].imshow(mask)

axarr[1][0].imshow(n_img)
axarr[1][1].imshow(n_mask)
plt.show()