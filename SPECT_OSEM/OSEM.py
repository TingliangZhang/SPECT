# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import math

# read 3D SPECT raw data
imgData = np.fromfile('whole_bone.raw', dtype="float32")
# Chart 2 in Document
# distance between detector to rotation center
df=pd.read_excel('angle_r.xlsx')
# df.values返回多位数组
angle_r = df.values
# According to Document, the data was captured in 60 frames,
# projection of a single frame is 128*128 matrix.
pro = imgData.reshape(60,128,128)
end = 128*128
p1 = pro[:,70,:]

# theta is rotation angle from the starting -x, around z axis
def get_spect_tran_m(theta):
    bias = 1e-6
    # {cij} system matrix 
    c = np.zeros((end,128),dtype="float32")
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
theta_range = np.arange(0,60,dtype="float32")*6
bias = 1e-6
# when theta = n*90 deg
theta_range[np.linspace(0,45,4,dtype = "int")] = theta_range[np.linspace(0,45,4,dtype="int")] + bias
# system matrix for 60 frames
dic_c = {}
i = 0

#tqdm to create a progress bar 0-60, based on loop
for n in tqdm(range(0,60)):
    num = str(i)
    dic_c[num] = get_spect_tran_m(theta_range[i])
    i = i + 1


total_c = np.zeros((end,128*60),dtype="float32")
total_p = np.zeros((1,128*60),dtype="float32").squeeze(0)
for i in range(0,59):
    range1 = 128*i
    range2 = 128*(i+1) 
    # store system matrix of 60 frames
    total_c[:,range1:range2] = dic_c[str(i)]
    total_p[range1:range2] = p1[i]

# reconstrunction image
osem_spect = np.zeros((128,128,128),dtype="float32")

# iter means iteration numbers
# divide the dataset to iter subsets
# iter should be the factor of 60
def OSEM_SPECT(i,iter):
    a=np.linspace(0,128*60,iter+1,dtype="int")
    pro_splice = pro[:,i,:]
    total_p = pro_splice.reshape(1,128*60).squeeze(0)
    f0 = np.ones((1,end),dtype="float32").squeeze(0)
    temp = np.ones((1,end),dtype="float32").squeeze(0)
    for i in range(0,iter):
        for j in range(1,iter):
            temp = total_c[:,a[j]:a[j+1]].dot(total_p[a[j]:a[j+1]]/(f0.dot(total_c[:,a[j]:a[j+1]]+bias)))*(f0/np.sum(total_c[:,a[j]:a[j+1]]+bias,axis=1))
            f0 = temp
    
    return f0.reshape(128,128)
# i in progress bar represents the i th layer on z axis
for i in tqdm(range(0,128)):
    osem_spect[i] = OSEM_SPECT(i,4)

# to view in Amide
osem_spect.tofile("osem_mine.raw")
# to view a transversal profile
plt.imshow(osem_spect[64,:,:])
plt.show()