import cv2 as c
import numpy as np
import glob


def do_filter(img):
    img_n = img[5:,:,:]
    hsv = c.cvtColor(img_n, c.COLOR_BGR2HSV)
    lower = np.array([80,30,60])
    upper = np.array([100,50,80])

    filted_img = c.inRange(hsv,lower,upper)
    # print (np.shape(canny_2))
    return filted_img

folder = '/home/yida/Desktop/leftcamera_15cm/'
names = glob.glob(folder + '*.jpg')

for i in range(len(names)):
    filename = names[i]
    savename = '/home/yida/Desktop/inrange/'+ str(i)+ '_inrange.jpg'
    img = c.imread(filename)
    filted_img = do_filter(img)
    img_c = np.zeros(np.shape(img))
    img_c[5:,:,2] = filted_img
    new_img = img_c + img

    c.imwrite(savename,new_img)
