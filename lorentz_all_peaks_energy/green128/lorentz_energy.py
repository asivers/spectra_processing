import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def check(index, arr):
    kp = 11
    for j in range(1, kp):
        if not ((arr[index] > arr[index - j]) and (arr[index] > arr[index + j])):
            return False
    return True


def energy(_waveLength):
    for i in range(len(_waveLength)):
        _waveLength[i] = 1243.125 / _waveLength[i]
    return _waveLength


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


def lorentzsum(x, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, gamma8,
               scale1, scale2, scale3, scale4, scale5, scale6, scale7, scale8,
               shift1, shift2, shift3, shift4, shift5, shift6, shift7, shift8):
    return (scale1 *
            ((1 / (math.pi * gamma1)) * ((gamma1 * gamma1) / ((x - shift1) * (x - shift1) + gamma1 * gamma1)))) + \
           (scale2 *
            ((1 / (math.pi * gamma2)) * ((gamma2 * gamma2) / ((x - shift2) * (x - shift2) + gamma2 * gamma2)))) + \
           (scale3 *
            ((1 / (math.pi * gamma3)) * ((gamma3 * gamma3) / ((x - shift3) * (x - shift3) + gamma3 * gamma3)))) + \
           (scale4 *
            ((1 / (math.pi * gamma4)) * ((gamma4 * gamma4) / ((x - shift4) * (x - shift4) + gamma4 * gamma4)))) + \
           (scale5 *
            ((1 / (math.pi * gamma5)) * ((gamma5 * gamma5) / ((x - shift5) * (x - shift5) + gamma5 * gamma5)))) + \
           (scale6 *
            ((1 / (math.pi * gamma6)) * ((gamma6 * gamma6) / ((x - shift6) * (x - shift6) + gamma6 * gamma6)))) + \
           (scale7 *
            ((1 / (math.pi * gamma7)) * ((gamma7 * gamma7) / ((x - shift7) * (x - shift7) + gamma7 * gamma7)))) + \
           (scale8 *
            ((1 / (math.pi * gamma8)) * ((gamma8 * gamma8) / ((x - shift8) * (x - shift8) + gamma8 * gamma8))))

name = 'green128'
ext = '.mdrs'

waveLengthArrInv = []
intensityArrInv = []
maxWaveLengthArrInv = []

with open((name + ext), 'r') as inf:
    while True:
        line = inf.readline().strip()
        if not line:
            break
        intensityArrInv.append(float(line))

mainsize = len(intensityArrInv)

for i in range(mainsize):
    waveLengthArrInv.append(i * 2 + 1000)

for i in range(26, mainsize - 12):
    if check(i, intensityArrInv):
        maxWaveLengthArrInv.append(i * 2 + 1000)

print(maxWaveLengthArrInv)

energyNp = np.array(list(reversed(energy(waveLengthArrInv))))
intensityNp = np.array(list(reversed(intensityArrInv)))
maxEnergyArr = list(reversed(energy(maxWaveLengthArrInv)))

print(maxEnergyArr)
shift1 = maxEnergyArr[1-1]
shift2 = maxEnergyArr[2-1]
shift3 = maxEnergyArr[3-1]
shift4 = maxEnergyArr[4-1]
shift5 = maxEnergyArr[5-1]
shift6 = maxEnergyArr[6-1]
shift7 = maxEnergyArr[7-1]
shift8 = maxEnergyArr[8-1]

min_gamma = 0.0001
max_gamma = 0.01
min_scale = 0.01
max_scale = 5

params = curve_fit(
    lambda x, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, gamma8,
           scale1, scale2, scale3, scale4, scale5, scale6, scale7, scale8:
    lorentzsum(x, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, gamma8,
               scale1, scale2, scale3, scale4, scale5, scale6, scale7, scale8,
               shift1, shift2, shift3, shift4, shift5, shift6, shift7, shift8),
    energyNp, intensityNp,
    bounds=((min_gamma, min_gamma, min_gamma, min_gamma, min_gamma, min_gamma, min_gamma, min_gamma,
             min_scale, min_scale, min_scale, min_scale, min_scale, min_scale, min_scale, min_scale),
            (max_gamma, max_gamma, max_gamma, max_gamma, max_gamma, max_gamma, max_gamma, max_gamma,
             max_scale, max_scale, max_scale, max_scale, max_scale, max_scale, max_scale, max_scale)),
    maxfev=1000000)

print(params[0])

gamma1 = params[0][0]
gamma2 = params[0][1]
gamma3 = params[0][2]
gamma4 = params[0][3]
gamma5 = params[0][4]
gamma6 = params[0][5]
gamma7 = params[0][6]
gamma8 = params[0][7]
scale1 = params[0][8]
scale2 = params[0][9]
scale3 = params[0][10]
scale4 = params[0][11]
scale5 = params[0][12]
scale6 = params[0][13]
scale7 = params[0][14]
scale8 = params[0][15]

lorentz1 = []
lorentz2 = []
lorentz3 = []
lorentz4 = []
lorentz5 = []
lorentz6 = []
lorentz7 = []
lorentz8 = []
lorentzall = []

count = 0
for i in range(mainsize - 1, -1, -1):
    currx = 1243.125 / (i * 2 + 1000)
    lorentz1.append(scale1 * ((1 / (math.pi * gamma1)) * (
                (gamma1 * gamma1) / ((currx - shift1) * (currx - shift1) + gamma1 * gamma1))))
    lorentz2.append(scale2 * ((1 / (math.pi * gamma2)) * (
                (gamma2 * gamma2) / ((currx - shift2) * (currx - shift2) + gamma2 * gamma2))))
    lorentz3.append(scale3 * ((1 / (math.pi * gamma3)) * (
                (gamma3 * gamma3) / ((currx - shift3) * (currx - shift3) + gamma3 * gamma3))))
    lorentz4.append(scale4 * ((1 / (math.pi * gamma4)) * (
                (gamma4 * gamma4) / ((currx - shift4) * (currx - shift4) + gamma4 * gamma4))))
    lorentz5.append(scale5 * ((1 / (math.pi * gamma5)) * (
                (gamma5 * gamma5) / ((currx - shift5) * (currx - shift5) + gamma5 * gamma5))))
    lorentz6.append(scale6 * ((1 / (math.pi * gamma6)) * (
                (gamma6 * gamma6) / ((currx - shift6) * (currx - shift6) + gamma6 * gamma6))))
    lorentz7.append(scale7 * ((1 / (math.pi * gamma7)) * (
                (gamma7 * gamma7) / ((currx - shift7) * (currx - shift7) + gamma7 * gamma7))))
    lorentz8.append(scale8 * ((1 / (math.pi * gamma8)) * (
                (gamma8 * gamma8) / ((currx - shift8) * (currx - shift8) + gamma8 * gamma8))))
    lorentzall.append(lorentz1[count] + lorentz2[count] + lorentz3[count] + lorentz4[count] +
                      lorentz5[count] + lorentz6[count] + lorentz7[count] + lorentz8[count])
    count += 1

plt.xlabel('энергия, эВ')
plt.ylabel('интенсивность, отн. ед.')
plt.plot(energyNp, intensityNp)

plt.plot(energyNp, lorentz1)
plt.plot(energyNp, lorentz2)
plt.plot(energyNp, lorentz3)
plt.plot(energyNp, lorentz4)
plt.plot(energyNp, lorentz5)
plt.plot(energyNp, lorentz6)
plt.plot(energyNp, lorentz7)
plt.plot(energyNp, lorentz8)
plt.plot(energyNp, lorentzall)

ax = plt.subplot(111)
ax.plot(energyNp, intensityNp, label='Измерения')
ax.plot(energyNp, lorentz1, label='D1')
ax.plot(energyNp, lorentz2, label='Неизв. пик')
ax.plot(energyNp, lorentz3, label='D2')
ax.plot(energyNp, lorentz4, label='D3')
ax.plot(energyNp, lorentz5, label='D4')
ax.plot(energyNp, lorentz6, label='Неизв. пик')
ax.plot(energyNp, lorentz7, label='Краевая')
ax.plot(energyNp, lorentz8, label='Неизв. пик')
ax.plot(energyNp, lorentzall, label='Сумма')
ax.legend()

plt.grid()
plt.savefig(name + '_simple.png', dpi=600)
plt.show()


with open('result.txt', 'w') as ouf:
    ouf.write('D1 peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(1243.125 / shift1) + '\n')
    ouf.write('Energy, eV: ' + str(shift1) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz1)) + '\n')
    ouf.write('\n')
    ouf.write('Unrecognized peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(1243.125 / shift2) + '\n')
    ouf.write('Energy, eV: ' + str(shift2) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz2)) + '\n')
    ouf.write('\n')
    ouf.write('D2 peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(1243.125 / shift3) + '\n')
    ouf.write('Energy, eV: ' + str(shift3) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz3)) + '\n')
    ouf.write('\n')
    ouf.write('D3 peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(1243.125 / shift4) + '\n')
    ouf.write('Energy, eV: ' + str(shift4) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz4)) + '\n')
    ouf.write('\n')
    ouf.write('D4 peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(1243.125 / shift5) + '\n')
    ouf.write('Energy, eV: ' + str(shift5) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz5)) + '\n')
    ouf.write('\n')
    ouf.write('Unrecognized peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(1243.125 / shift6) + '\n')
    ouf.write('Energy, eV: ' + str(shift6) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz6)) + '\n')
    ouf.write('\n')
    ouf.write('BE peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(1243.125 / shift7) + '\n')
    ouf.write('Energy, eV: ' + str(shift7) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz7)) + '\n')
    ouf.write('\n')
    ouf.write('Unrecognized peak' + '\n')
    ouf.write('Wavelength, nm: ' + str(1243.125 / shift8) + '\n')
    ouf.write('Energy, eV: ' + str(shift8) + '\n')
    ouf.write('Halfwidth, nm: ' + str(halfwidth(lorentz8)) + '\n')
    ouf.write('\n')
