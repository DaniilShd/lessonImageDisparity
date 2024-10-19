import cv2
from matplotlib import pyplot

imgL = cv2.imread('Stereo_Image_4.png', 0)
imgR = cv2.imread('Stereo_Image_3.png', 0)

stereo=cv2.StereoBM_create(numDisparities=96, blockSize=17)
disparity = stereo.compute(imgL, imgR)

pyplot.figure(figsize = (20,30))
pyplot.imshow(disparity, 'gray')
pyplot.xticks([])
pyplot.yticks([])
pyplot.show()
