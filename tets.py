
import numpy as np
import cv2
a = np.array([[1,2,3],[4,5,6]])
y = np.transpose(a, [1, 0])
print(y)

a = np.zeros([30, 30])
a[15:, :] = 255

img_T = np.transpose(a, [1,0])
left_flip_img = np.flip(img_T, axis=[0])
right_flip_img = np.flip(img_T, axis=[1])
imgs = np.concatenate((a, left_flip_img, right_flip_img), axis=0)
cv2.imshow('im', imgs)
cv2.waitKey()

