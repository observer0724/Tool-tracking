import cv2 as c
import numpy as np
from matplotlib import pyplot as plt

img = c.imread('leftcamera_15cm.avi_000000.674.jpg')
hsv = c.cvtColor(img, c.COLOR_RGB2HSV)
lower = np.array([150,0,0])
upper = np.array([170,30,30])

filted_img = c.inRange(hsv,lower,upper)
# filted_img = c.resize(filted_img,(50,50))
# filted_img = c.inRange(filted_img,np.array([100]),np.array([255]))
# m,n  = np.where(filted_img ==np.max(filted_img))
# print (len(m))
fig = plt.figure()
fig.add_subplot(121)
plt.imshow(hsv)
fig.add_subplot(122)
plt.imshow(filted_img)
plt.show()
