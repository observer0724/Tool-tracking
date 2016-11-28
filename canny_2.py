import cv2 as c
import numpy as np
import glob


def do_canny(img):
    img_g = c.GaussianBlur( img,(5,5), 0, 0, c.BORDER_DEFAULT )
    gray = c.cvtColor(img_g, c.COLOR_BGR2GRAY)
    gray2 = gray[5:,:]
    canny = c.Canny(gray2,250,300)
    lower = np.array([100])
    upper = np.array([255])
    canny_2 = c.inRange(canny,lower ,upper)
    # print (np.shape(canny_2))
    return canny_2,gray

folder = '/home/yida/Desktop/leftcamera_15cm/'
names = glob.glob(folder + '*.jpg')

for i in range(len(names)):
    filename = names[i]
    savename = '/home/yida/Desktop/canny/'+ str(i)+ '_canny.jpg'
    img = c.imread(filename)
    canny_2,gray = do_canny(img)
    img_c = np.zeros(np.shape(gray))
    img_c[5:,:] = canny_2
    new_img = img_c + gray

    c.imwrite(savename,new_img)
