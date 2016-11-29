import cv2 as c
import numpy as np
import glob
import random as r
from matplotlib import pyplot as plt

folder = '/home/observer0724/Desktop/leftcamera_15cm/pictures/'
names = glob.glob(folder + '*.jpg')
size = 102
def do_canny(img):
    img_g = c.GaussianBlur( img,(5,5), 0, 0, c.BORDER_DEFAULT )
    gray = c.cvtColor(img_g, c.COLOR_BGR2GRAY)
    gray2 = gray[5:,:]
    canny = c.Canny(gray2,300,450)
    M,N = np.where(canny == np.max(canny))
    r.shuffle(M)
    m = M[0:10]
    r.shuffle(N)
    n = N[0:10]
    index_m = sum(m)/len(m)
    index_n = sum(n)/len(n)
    max_m = np.max(M)
    max_n = np.max(N)
    min_m = np.min(M)
    min_n = np.min(N)
    return index_m,index_n,max_m,max_n,min_m,min_n,canny
    # return canny



def do_filter(img):
    img_n = img[5:,:,:]
    hsv = c.cvtColor(img_n, c.COLOR_BGR2HSV)
    lower = np.array([80,30,60])
    upper = np.array([100,50,80])

    filted_img = c.inRange(hsv,lower,upper)
    M,N  = np.where(filted_img ==np.max(filted_img))
    r.shuffle(M)
    m = M[0:10]
    r.shuffle(N)
    n = N[0:10]
    index_m = sum(m)/len(m)
    index_n = sum(n)/len(n)
    return index_m,index_n,filted_img
    # return filted_img



def choose_area(canny_m,canny_n,max_m,max_n,min_m,min_n,filter_m,filter_n):
    if canny_m >= filter_m and canny_n >= filter_n:
        cut_m_1 = max_m+5
        cut_m_2 = max_m+5+size
        cut_n_1 = max_n
        cut_n_2 = max_n+size
    elif canny_m >= filter_m and canny_n < filter_n:
        cut_m_1 = max_m+5
        cut_m_2 = max_m+5+size
        cut_n_1 = min_n-size
        cut_n_2 = min_n
    elif canny_m < filter_m and canny_n >= filter_n:
        cut_m_1 = min_m+5-size
        cut_m_2 = min_m+5
        cut_n_1 = max_n
        cut_n_2 = max_n+size
    elif canny_m< filter_m and canny_n < filter_n:
        cut_m_1 = min_m+5-size
        cut_m_2 = min_m+5
        cut_n_1 = min_n-size
        cut_n_2 = min_n


    if cut_m_2 >= 480:
        cut_m_2 = 479
    if cut_m_1 < 0:
        cut_m_1 = 0
    if cut_n_2 >= 640:
        cut_n_2 = 639
    if cut_n_1 < 0:
        cut_n_1 = 0

    return cut_m_1,cut_m_2,cut_n_1,cut_n_2


for i in range(len(names)):

    filename = names[i]
    savename = '/home/observer0724/Desktop/leftcamera_15cm/cut/' + str(i) + '_cut.jpg'
    img = c.imread(filename)
    # canny = do_canny(img)
    # filted_img = do_filter(img)
    canny_m,canny_n,max_m,max_n,min_m,min_n,canny = do_canny(img)
    filter_m,filter_n,filted_img = do_filter(img)

    cut_m_1,cut_m_2,cut_n_1,cut_n_2 = choose_area(canny_m, canny_n,max_m,max_n,min_m,min_n,filter_m, filter_n)
    # print (cut_m_1,cut_m_2,cut_n_1,cut_n_2)
    # cut_img = img[cut_m_1:cut_m_2,cut_n_1:cut_n_2,:]
    img[cut_m_1:cut_m_2,cut_n_1,1] = 255
    img[cut_m_1:cut_m_2,cut_n_2,1] = 255
    img[cut_m_1,cut_n_1:cut_n_2,1] = 255
    img[cut_m_2,cut_n_1:cut_n_2,1] = 255
    canny_2 = np.zeros(np.shape(img))
    canny_2[5:,:,1] = canny
    filted_2 = np.zeros(np.shape(img))
    filted_2[5:,:,2] = filted_img

    new_img = canny_2+filted_2+img






    c.imwrite(savename,new_img)
