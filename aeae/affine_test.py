import matplotlib.pyplot as plt
import cv2
import numpy as np
from affine import Affine


def param2affine(sigma):
    '''
    Conversion of affine parameter into matrix form through the use of module 'affine'
    sigma[0] = degree of anti-clockwise rotation (unit = degrees)
    sigma[1] = translation rightwards
    sigma[2] = translation down
    sigma[3] = scaling factor
    sigma[4] = degree of shear transformation
    
    Input:
        sigma = affine transformation parameter
    Output:
        (2,3) np.array representing transformation
    '''
    aff_mat =  Affine.translation(sigma[1],sigma[2]) * Affine.scale(sigma[3]) * Affine.rotation(sigma[0])* Affine.shear(sigma[4],sigma[4]) 
    return np.array(aff_mat)[:6].reshape(2,3)


img = cv2.imread('test.png',0)
rows,cols = img.shape
sigma = [0,0,0,0.8,0]

print(img.shape)
M = param2affine(sigma)
dst = cv2.warpAffine(img,M,(cols,rows))

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(img)
fig.add_subplot(1,2,2)
plt.imshow(dst)
plt.show()
plt.close()