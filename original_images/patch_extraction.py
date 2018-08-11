import numpy as np


patch_size = 4
data = np.array([9,8,6,9,8,2,6,3,4,5], dtype=np.float32)
mask = np.array([0,0,0,0,1,1,1,1,0,0], dtype=np.float32)
padded_mask = np.pad(mask, (0, patch_size - 1), 'constant')

position = np.arange(0, 10)

iterate_len = len(data) - patch_size +1
print(iterate_len)
# ri = mi * m(i+1) * m(i+2) * m(i+3) (for i in 0 to 6)
# 0123432100

# print(position)

# print(type(m))
# for i in range(0, iterate_len):
# print(np.sum(m[0:4]))


# print(list(map(lambda i: np.sum(m[i: i+patch_size]), np.arange(len(data) - patch_size))))
# np_conv = np.convolve(mask,np.ones(patch_size,dtype=int),'valid')
# print(np_conv)
# pad_len = len(data) - len(np_conv)
np_conv = np.convolve(padded_mask,np.ones(patch_size,dtype=int),'valid')
print(np_conv)