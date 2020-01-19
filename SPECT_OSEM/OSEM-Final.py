# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import math

# read 3D SPECT raw data
imgData = np.fromfile('whole_bone.raw', dtype="float32")
# According to Document, the data was captured in 60 frames,
# projection of a single frame is 128*128 matrix.
proj = imgData.reshape(60,128,128)
#for convenience
square = 128*128

# theta is rotation angle from the starting -x, around z axis
def get_spect_tran_m(theta):
    bias = 1e-6
    # {cij} system matrix 
    c = np.zeros((square,128),dtype="float32")
    # np.linspace is a method for arithmetic progression
    # i is the coordinate at r axis on detector
    for i in np.linspace(-63.5,63.5,128):
        # ang is rotation angle from +x, couter-clockwise
        ang = math.pi-theta/180*math.pi
        # vertical distance line from FOV center to detector
        slope = math.tan(ang)
        # intercept on y axis
        intercept = i/math.cos(ang+bias)
        # delta model of {cij}
        # ray driven ergodic, No. c_j detector cell
        c_j = int(i+63.5)
        if slope<=0:  
            for x in range(-63,65):
                for y in range(-63,65):
                    x2 = x-1
                    y2 = y-1
                    if (slope*x+intercept-y)*(slope*x2+intercept-y2) <=0:
                        c_i = (x+63)*128+y+63
                        c[c_i,c_j] = 1
        else:
            for x in range(-63,65):
                for y in range(-63,65):
                    x1 = x-1
                    y1 = y
                    x2 = x
                    y2 = y-2
                    if (slope*x1+intercept-y1)*(slope*x2+intercept-y2) <=0:
                        c_i = (x+63)*128+y+63
                        c[c_i,c_j] = 1
    return c

# actual rotation angle, 6 degree a frame
theta_sample = np.arange(0,60,dtype="float32")*6
bias = 1e-6
# when theta = n*90 deg
theta_sample[np.linspace(0,45,4,dtype = "int")] = theta_sample[np.linspace(0,45,4,dtype="int")] + bias
# system matrix for 60 frames, same for every layer along z axis
frame_c = {}
i = 0

# tqdm to create a progress bar 0-60, based on loop
for n in tqdm(range(0,60)):
    num = str(i)
    frame_c[num] = get_spect_tran_m(theta_sample[i])
    i = i + 1

# to pile the 60 frames of system matrix in one matrix
total_c = np.zeros((square,128*60),dtype="float32")
for i in range(0,59):
    range1 = 128*i
    range2 = 128*(i+1) 
    # store system matrix of 60 frames
    total_c[:,range1:range2] = frame_c[str(i)]

# reconstrunction image
osem_spect = np.zeros((128,128,128),dtype="float32")

# iter means iteration times
# divide the dataset to sub subsets
# sub should be the factor of 60
def OSEM_SPECT(i,sub,iter):
    a=np.linspace(0,128*60,sub+1,dtype="int")
    # No. i slice on z axis
    proj_slice = proj[:,i,:]
    total_p = proj_slice.reshape(1,128*60).squeeze(0)
    f_old = np.ones((1,square),dtype="float32").squeeze(0)
    f_new = np.ones((1,square),dtype="float32").squeeze(0)
    for index in range(0,iter):
        for j in range(1,sub):
            f_new = (f_old/np.sum(total_c[:,a[j]:a[j+1]]+bias,axis=1))*total_c[:,a[j]:a[j+1]].dot(total_p[a[j]:a[j+1]]/(f_old.dot(total_c[:,a[j]:a[j+1]]+bias)))
            f_old = f_new
    return f_old.reshape(128,128)
# i in progress bar represents the i th layer on z axis
for i in tqdm(range(0,128)):
    osem_spect[i] = OSEM_SPECT(i,4,4)

# to view in Amide or Matlab
osem_spect.tofile("osem_mine.raw")
# to view a transversal profile
#plt.imshow(osem_spect[64,:,:])
#plt.show()

# According to ICRP Publication 210, representative profiles should be:
# transversal no. 147, 158, 168, 207
# coronal no. 87
# sagittal no. 109