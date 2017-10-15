import numpy as np
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from scipy import ndimage as ndi
from skimage import feature
from PIL import Image


img = color.rgb2gray(
    io.imread('Colosseum_in_Rome,_Italy_-_April_2007.jpg', as_grey=True))
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
# Note the 0 sigma for the last axis, we don't wan't to blurr the color
# planes together!
img = ndimage.gaussian_filter(img, sigma=(5, 6), order=0)
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
print img.shape
img = img.astype('float32')
print "check1"
dx = ndimage.sobel(img, 0)  # horizontal derivative
print "check2"
dy = ndimage.sobel(img, 1)  # vertical derivative
mag = np.hypot(dx, dy)  # magnitude
print mag;
mag *= 255.0 / np.max(mag)  # normalize (Q&D)
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
scipy.misc.imsave('sobel.jpg', mag)

im = io.imread('sobel.jpg')
# Compute the Canny filter for two values of sigma
edges = np.uint8(feature.canny(im, sigma=1) * 255)


print edges.shape

newArr = [];

for y in range(edges.shape[1]):
    for x in range(edges.shape[0]):
        if(edges[x][y] == 255):
            tempVar = str(x)+","+str(y)
            newArr.append(tempVar)


plt.imshow(edges)

plt.show()
scipy.misc.imsave('canny.jpg', edges)

kernel_finalSharpen = np.array([[1., 1., 1.],
                                [1., 10., 1.],
                                [1., 1., 1.]])


shape = img.shape
supershape = (shape[0] + 2, shape[1] + 2)
supermatrix = np.zeros(supershape, dtype=np.float)
supermatrix[1:-1, 1:-1] = img


def neighbors(r, c, supermatrix):
    imageSlicing = supermatrix[r-1:r+2, c-1:c+2]
    print imageSlicing.shape
    return imageSlicing



def convolution(imageMatrix, kernel, arrayPos, supermatrix):
    returnVal=imageMatrix
    for y in range(imageMatrix.shape[1]):
        for x in range(imageMatrix.shape[0]):
            returnStr=str(x) + "," + str(y)
            if returnStr in arrayPos:
                imageSlice=neighbors(x, y, supermatrix)
                if imageSlice.shape == kernel.shape:
                    print "ImageSlice.shape: ",imageSlice.shape
                    print "ReturnStr: ",returnStr
                    imageSlicerAsArray = np.squeeze(np.asarray(imageSlice))
                    kernelAsArray = np.squeeze(np.asarray(kernel))
                    matrixProd = imageSlicerAsArray*kernelAsArray
                    print "Matrix Product.sum(): ", matrixProd.sum()
                    print "Kernel.sum(): ", kernel.sum()
                    summation = matrixProd.sum()/kernel.sum()
                    returnVal[x][y] = summation

    return returnVal

returnVal1=convolution(img, kernel_finalSharpen, newArr, supermatrix)

plt.imshow(returnVal1, cmap=plt.get_cmap('gray'))
plt.show()
