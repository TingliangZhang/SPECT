import numpy as np
# import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import pandas as pd
# import tqdm
from scipy.fftpack import fft,ifft
# import PIL
# import matlab
# import matlab.engine
import math

#engine = matlab.engine.start_matlab() # Start MATLAB process


imgData = np.fromfile('whole_bone.raw', dtype="float32")
#imgData = imgData.reshape(width, height, channels)
df=pd.read_excel('angle_r.xlsx')
# df.values返回多位数组
angle_r = df.values
a_r = np.zeros((2,60))
a_r[0] = angle_r[[0,2,4,6],:].reshape(1,60)
#a_r[1] = angle_r[[1,3,5,7],:].reshape(1,60)

print(f"{a_r[0]}")