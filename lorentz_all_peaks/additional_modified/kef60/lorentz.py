import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def check(index, arr):
    kp = 20
    for j in range(1, kp):
        if not ((arr[index] > arr[index - j]) and (arr[index] > arr[index + j])):
            return False
    return True


def lorentzsum(x, gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, scale0, scale1, scale2, scale3, scale4, scale5, shift0, shift1, shift2, shift3, shift4, shift5):
    return (scale0 * ((1/(math.pi * gamma0))*((gamma0 * gamma0) / ((x - shift0) * (x - shift0) + gamma0 * gamma0)))) + (scale1 * ((1/(math.pi * gamma1))*((gamma1 * gamma1) / ((x - shift1) * (x - shift1) + gamma1 * gamma1)))) + (scale2 * ((1/(math.pi * gamma2))*((gamma2 * gamma2) / ((x - shift2) * (x - shift2) + gamma2 * gamma2)))) + (scale3 * ((1/(math.pi * gamma3))*((gamma3 * gamma3) / ((x - shift3) * (x - shift3) + gamma3 * gamma3)))) + (scale4 * ((1/(math.pi * gamma4))*((gamma4 * gamma4) / ((x - shift4) * (x - shift4) + gamma4 * gamma4)))) + (scale5 * ((1/(math.pi * gamma5))*((gamma5 * gamma5) / ((x - shift5) * (x - shift5) + gamma5 * gamma5))))


xx = []
yy = []
x0 = []

name = 'kef60'
ext = '.mdrs'

with open((name + ext), 'r') as inf:
    while True:
        line = inf.readline().strip()
        if not line:
            break
        yy.append(float(line))
    
mainsize = len(yy)  
    
for i in range(mainsize):
    xx.append(i * 2 + 1000)
    
for i in range(20, mainsize - 20):
    if check(i, yy):
        x0.append(i * 2 + 1000)
        
x = np.array(xx)
y = np.array(yy)

print(x0)
shift1 = 1188
shift2 = 1252
shift3 = 1328
shift4 = 1420
shift5 = 1536

shift0 = 1536

params = curve_fit(lambda x, gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, scale0, scale1, scale2, scale3, scale4, scale5 : lorentzsum(x, gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, scale0, scale1, scale2, scale3, scale4, scale5, shift0, shift1, shift2, shift3, shift4, shift5), x, y, maxfev=1000000)
print(params[0])

gamma0 = params[0][0]
gamma1 = params[0][1]
gamma2 = params[0][2]
gamma3 = params[0][3]
gamma4 = params[0][4]
gamma5 = params[0][5]
scale0 = params[0][6]
scale1 = params[0][7]
scale2 = params[0][8]
scale3 = params[0][9]
scale4 = params[0][10]
scale5 = params[0][11]

lorentz0 = [] 
lorentz1 = [] 
lorentz2 = [] 
lorentz3 = [] 
lorentz4 = [] 
lorentz5 = [] 
lorentzall = [] 
for i in range(mainsize):
    currx = i * 2 + 1000
    lorentz0.append(scale0 * ((1/(math.pi * gamma0))*((gamma0 * gamma0) / ((currx - shift0) * (currx - shift0) + gamma0 * gamma0))))
    lorentz1.append(scale1 * ((1/(math.pi * gamma1))*((gamma1 * gamma1) / ((currx - shift1) * (currx - shift1) + gamma1 * gamma1))))
    lorentz2.append(scale2 * ((1/(math.pi * gamma2))*((gamma2 * gamma2) / ((currx - shift2) * (currx - shift2) + gamma2 * gamma2))))
    lorentz3.append(scale3 * ((1/(math.pi * gamma3))*((gamma3 * gamma3) / ((currx - shift3) * (currx - shift3) + gamma3 * gamma3))))
    lorentz4.append(scale4 * ((1/(math.pi * gamma4))*((gamma4 * gamma4) / ((currx - shift4) * (currx - shift4) + gamma4 * gamma4))))
    lorentz5.append(scale5 * ((1/(math.pi * gamma5))*((gamma5 * gamma5) / ((currx - shift5) * (currx - shift5) + gamma5 * gamma5))))
    lorentzall.append(lorentz0[i] + lorentz1[i] + lorentz2[i] + lorentz3[i] + lorentz4[i] + lorentz5[i])

plt.xlabel('длина волны, нм')
plt.ylabel('интенсивность, отн. ед.')
plt.plot(x, y)

plt.plot(x, lorentz0)
plt.plot(x, lorentz1)
plt.plot(x, lorentz2)
plt.plot(x, lorentz3)
plt.plot(x, lorentz4)
plt.plot(x, lorentz5)
plt.plot(x, lorentzall)

ax = plt.subplot(111)
ax.plot(x, y, label='Data')
ax.plot(x, lorentz5, label='D1')
ax.plot(x, lorentz4, label='D2')
ax.plot(x, lorentz3, label='D3')
ax.plot(x, lorentz2, label='D4')
ax.plot(x, lorentz1, label='КЛ')
ax.plot(x, lorentz0, label='Доп')
ax.plot(x, lorentzall, label='Сумм')
ax.legend()

plt.grid()
plt.savefig(name + '_additional.png', dpi=600)
plt.show()
