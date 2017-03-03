import cv2 as c
import numpy as np
import glob
import random as r
from matplotlib import pyplot as plt

folder = '/home/yida/Desktop/8/not_touch/'
names = glob.glob(folder + '*.png')
size = 102
def do_canny(img):
    img_g = c.GaussianBlur( img,(5,5), 0, 0, c.BORDER_DEFAULT )
    # img_g = img
    gray = c.cvtColor(img_g, c.COLOR_BGR2GRAY)
    gray2 = gray[5:,:]
    canny = c.Canny(gray2,200,300)
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
def do_sobel(img):
    img = c.imread(filename)
    img_g = c.GaussianBlur( img,(1,1), 0, 0, c.BORDER_DEFAULT )
    gray = c.cvtColor(img_g, c.COLOR_BGR2GRAY)


    sobelx = c.Sobel(gray,c.CV_16S,1,0,ksize=3)
    sobely = c.Sobel(gray,c.CV_16S,0,1,ksize=3)




    abs_x = c.convertScaleAbs(sobelx)
    abs_y = c.convertScaleAbs(sobely)

    grad = c.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    grad_1 = grad[5:,:]
    grad_2 = c.inRange(grad_1,np.array([180]),np.array([255]))
    M,N = np.where(grad_2 == np.max(grad_2))
    r.shuffle(M)
    m = M[0:50]
    r.shuffle(N)
    n = N[0:50]
    index_m = sum(m)/len(m)
    index_n = sum(n)/len(n)
    max_m = np.max(M)
    max_n = np.max(N)
    min_m = np.min(M)
    min_n = np.min(N)
    return index_m,index_n,max_m,max_n,min_m,min_n,grad_2


def do_filter(img):
    img_n = img[5:,:,:]
    hsv = c.cvtColor(img_n, c.COLOR_BGR2HSV)
    lower = np.array([0,0,0])
    upper = np.array([100,40,75])

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
        cut_m_1 = min_m+5-size+20
        cut_m_2 = min_m+5+20
        cut_n_1 = max_n-20
        cut_n_2 = max_n+size-20
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
    savename = '/home/yida/Desktop/8/cut_2/' + str(i) + '_cut.png'
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
