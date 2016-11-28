import numpy as np
import cv2 as c
from matplotlib import pyplot as plt
kernel_size = 102

img = c.imread('leftcamera_15cm.avi_000004.074.jpg')
sobel = c.imread('grad.jpg')
sobel_2 = sobel[5:,:,:]
gray = c.cvtColor( sobel_2, c.COLOR_BGR2GRAY)
# gray_g = c.GaussianBlur( gray,(35,35), 0, 0, c.BORDER_DEFAULT )


m,n = np.shape(gray)

kernel = np.empty([kernel_size, kernel_size],dtype = int)
map_list = np.empty([m- kernel_size+1, n- kernel_size +1],dtype = int)


for i in range (m - kernel_size +1):
    for j in range (n- kernel_size +1):
        kernel = gray[i:i+ kernel_size-1, j:j+kernel_size -1]
        map_list[i,j] = sum(sum(kernel))

map_2 = map_list*255/np.max(map_list)
# np.savetxt('new_2.csv', map_2, delimiter = ',')

in_m = np.where(map_list == np.max(map_list))[0][0]
in_n = np.where(map_list == np.max(map_list))[1][0]
print(in_m,in_n)
plt.imshow(map_2, cmap = 'spectral')
plt.show()
# new_img = img[in_m:in_m+kernel_size-1,in_n:in_n+kernel_size-1,:]
#
# c.imwrite('cut_out.jpg',new_img)
