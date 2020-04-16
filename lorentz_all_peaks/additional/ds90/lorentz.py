import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def check(ielt, arr):
    kp = [1, 5, 10, 20, 35]
    for j in range(len(kp)):
        if not ((arr[ielt] > arr[ielt-kp[j]]) and (arr[ielt] > arr[ielt+kp[j]])):
            return False
    return True


def halfwidth(_lorentz):
    lorentzSize = len(_lorentz)
    yMax = 0
    xMax = 0
    for i in range(lorentzSize):
        if _lorentz[i] > yMax:
            yMax = _lorentz[i]
            xMax = i
    halfInt = yMax / 2
    xCurr = xMax
    while True:
        xCurr -= 1
        if _lorentz[xCurr] < halfInt:
            break
    leftX = xCurr * 2 + 1000
    xCurr = xMax
    while True:
        xCurr += 1
        if _lorentz[xCurr] < halfInt:
            break
    rightX = xCurr * 2 + 1000
    return rightX - leftX


def lorentzsum(x, gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, scale0, scale1, scale2, scale3, scale4, scale5, shift0, shift1, shift2, shift3, shift4, shift5):
    return (scale0 * ((1/(math.pi * gamma0))*((gamma0 * gamma0) / ((x - shift0) * (x - shift0) + gamma0 * gamma0)))) + (scale1 * ((1/(math.pi * gamma1))*((gamma1 * gamma1) / ((x - shift1) * (x - shift1) + gamma1 * gamma1)))) + (scale2 * ((1/(math.pi * gamma2))*((gamma2 * gamma2) / ((x - shift2) * (x - shift2) + gamma2 * gamma2)))) + (scale3 * ((1/(math.pi * gamma3))*((gamma3 * gamma3) / ((x - shift3) * (x - shift3) + gamma3 * gamma3)))) + (scale4 * ((1/(math.pi * gamma4))*((gamma4 * gamma4) / ((x - shift4) * (x - shift4) + gamma4 * gamma4)))) + (scale5 * ((1/(math.pi * gamma5))*((gamma5 * gamma5) / ((x - shift5) * (x - shift5) + gamma5 * gamma5))))


xx = []
yy = []
x0 = []

name = 'ds90'
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
shift1 = x0[0]
shift2 = x0[1]
shift3 = x0[2]
shift4 = x0[3]
shift5 = x0[4]

params = curve_fit(lambda x, gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, scale0, scale1, scale2, scale3, scale4, scale5, shift0 : lorentzsum(x, gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, scale0, scale1, scale2, scale3, scale4, scale5, shift0, shift1, shift2, shift3, shift4, shift5), x, y, maxfev=1000000)
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
shift0 = params[0][12]

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

with open('result.txt', 'w') as ouf:
    ouf.write('D1 peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(shift5) + '\n')
    ouf.write('Energy, eV: ' + str(1243.125 / shift5) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz5)) + '\n')
    ouf.write('\n')
    ouf.write('D2 peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(shift4) + '\n')
    ouf.write('Energy, eV: ' + str(1243.125 / shift4) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz4)) + '\n')
    ouf.write('\n')
    ouf.write('D3 peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(shift3) + '\n')
    ouf.write('Energy, eV: ' + str(1243.125 / shift3) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz3)) + '\n')
    ouf.write('\n')
    ouf.write('D4 peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(shift2) + '\n')
    ouf.write('Energy, eV: ' + str(1243.125 / shift2) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz2)) + '\n')
    ouf.write('\n')
    ouf.write('Band-edge peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(shift1) + '\n')
    ouf.write('Energy, eV: ' + str(1243.125 / shift1) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz1)) + '\n')
    ouf.write('\n')
