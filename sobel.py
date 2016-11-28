import cv2
import numpy as np


filename = 'leftcamera_15cm.avi_000004.074.jpg'
img = cv2.imread(filename)
img_g = cv2.GaussianBlur( img,(3,3), 0, 0, cv2.BORDER_DEFAULT )
gray = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)


sobelx = cv2.Sobel(gray,cv2.CV_16S,1,0,ksize=3)
sobely = cv2.Sobel(gray,cv2.CV_16S,0,1,ksize=3)




abs_x = cv2.convertScaleAbs(sobelx)
abs_y = cv2.convertScaleAbs(sobely)

grad = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

cv2.imwrite('grad.jpg',grad)
