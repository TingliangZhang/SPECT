# 导入必要的库函数
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import math

# 读取whole_bone.raw
imgData = np.fromfile('whole_bone.raw', dtype="float32")
# 读取角度值
df=pd.read_excel('angle_r.xlsx')
# df.values返回多位数组
angle_r = df.values
# 投影数据
pro = imgData.reshape(60,128,128)
end = 128*128
p1 = pro[:,70,:]

# 旋转theta角时的 SPECT 变换 
def get_spect_tran_m(theta):
    #rotate detector theta from initial ang
    c = np.zeros((end,128),dtype="float32")
    # i = -63.5 -62.5 ... 63.5
    for i in np.linspace(-63.5,63.5,128):
        ang = math.pi-theta*math.pi/180
        a = math.tan(ang)
        b = i/math.cos(theta*math.pi/180+1e-6)
        c_y = int(i+63.5)
        if a<=0:
            for x in range(-63,65):
                for y in range(-63,65):
                    x2 = x-1
                    y2 = y-1

                    if (a*x+b-y)*(a*x2+b-y2) <=0:
                        pos = (x+63)*128+y+63
                        c[pos,c_y] = 1
        else:
            for x in range(-63,65):
                for y in range(-63,65):
                    x1 = x-1
                    y1 = y
                    x2 = x
                    y2 = y-2
                    if (a*x1+b-y1)*(a*x2+b-y2) <=0:
                        pos = (x+63)*128+y+63
                        c[pos,c_y] = 1
    return c

theta_range = np.arange(0,60,dtype="float32")*6
bias = 1e-6
theta_range[np.linspace(0,45,4,dtype = "int")] = theta_range[np.linspace(0,45,4,dtype="int")] + bias
dic_c = {}
i = 0

for n in tqdm(range(0,60)):
    num = str(i)
    dic_c[num] = get_spect_tran_m(theta_range[i])
    i = i + 1

bias = 1e-6
a1 = 0
a2 = 128*15
a3 = 128*30
a4 = 128*45
a5 = 128*60
iter = 4
total_c = np.zeros((128*128,128*60),dtype="float32")
total_p = np.zeros((1,128*60),dtype="float32").squeeze(0)
for i in range(0,59):
    range1 = 128*i
    range2 = 128*(i+1) 
    total_c[:,range1:range2] = dic_c[str(i)]
    total_p[range1:range2] = p1[i]
osem_spect = np.zeros((128,128,128),dtype="float32")

def OSEM_SPECT(i,idata):
    pro_splice = pro[:,i,:]
    total_p = pro_splice.reshape(1,128*60).squeeze(0)
    f0 = np.ones((1,end),dtype="float32").squeeze(0)
    temp = np.ones((1,end),dtype="float32").squeeze(0)
    for i in range(0,iter):
        temp= total_c[:,a1:a2].dot(total_p[a1:a2]/(f0.dot(total_c[:,a1:a2]+bias)))*(f0/np.sum(total_c[:,a1:a2]+bias,axis=1))
        f0 = temp
        temp= total_c[:,a2:a3].dot(total_p[a2:a3]/(f0.dot(total_c[:,a2:a3]+bias)))*(f0/np.sum(total_c[:,a2:a3]+bias,axis=1))
        f0 = temp
        temp= total_c[:,a3:a4].dot(total_p[a3:a4]/(f0.dot(total_c[:,a3:a4]+bias)))*(f0/np.sum(total_c[:,a3:a4]+bias,axis=1))
        f0 = temp
        temp= total_c[:,a4:a5].dot(total_p[a4:a5]/(f0.dot(total_c[:,a4:a5]+bias)))*(f0/np.sum(total_c[:,a4:a5]+bias,axis=1))
        f0 = temp
    return f0.reshape(128,128)
for i in tqdm(range(0,128)):
    osem_spect[i] = OSEM_SPECT(i,4)

# 写入到.raw文件中，用Amide查看即可
osem_spect.tofile("osem_mine.raw")
# 预览某一横截面图像
plt.imshow(osem_spect[64,:,:])
plt.show()