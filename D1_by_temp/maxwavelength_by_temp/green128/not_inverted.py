import os
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


def normalize(arr):
    maximum = max(arr)
    arr_out = []
    for j in range(len(arr)):
        arr_out.append(arr[j] / maximum)
    return arr_out


def energy(_waveLength):
    _energy = []
    for i in range(len(_waveLength)):
        _energy.append(1243.125 / _waveLength[i])
    return _energy


def approx(_temp, _dif):
    return 1.17 - 0.000473 * _temp * _temp / (636 + _temp) - _dif


data = dict()

for filename in os.listdir('files'):
    fileData = []
    localMaxWavelengthArr = []
    localMaxIntArr = []
    with open(os.path.join('files', filename), 'r') as inf:
        while True:
            line = inf.readline().strip()
            if not line:
                break
            fileData.append(float(line))
    mainSize = len(fileData)
    for i in range(mainSize - 100, mainSize - 20):
        if check(i, fileData):
            localMaxWavelengthArr.append(i * 2 + 1000)
            localMaxIntArr.append(fileData[i])
    localMaxInt = 0
    localMaxWaveLength = 0
    for i in range(len(localMaxIntArr)):
        if localMaxIntArr[i] > localMaxInt:
            localMaxInt = localMaxIntArr[i]
            localMaxWaveLength = localMaxWavelengthArr[i]
    data[filename.split('K')[0]] = localMaxWaveLength

tempGivenArr = []
waveLengthGivenArr = []

for pair in sorted(data.items(), key=lambda x: (len(x[0]), x[0])):
    tempGivenArr.append(float(pair[0]))
    waveLengthGivenArr.append(float(pair[1]))

tempGiven = np.array(tempGivenArr)
energyGivenArr = energy(waveLengthGivenArr)
energyGiven = np.array(energyGivenArr)

params = curve_fit(approx, tempGiven, energyGiven)
dif = params[0][0]

tempApproxArr = []
energyApproxArr = []
for T in range(0, 215):
    tempApproxArr.append(T)
    energyApproxArr.append(1.17 - 0.000473 * T * T / (636 + T) - dif)
tempApprox = np.array(tempApproxArr)
energyApprox = np.array(energyApproxArr)

energyRoom = approx(293, dif)
waveLengthRoom = 1243.125 / energyRoom
energy10K = approx(10, dif)
waveLength10K = 1243.125 / energy10K
changeEnergy = energyRoom - energy10K
changeWaveLength = waveLengthRoom - waveLength10K
with open('result.txt', 'w') as ouf:
    ouf.write('dif = ' + str(dif) + '\n')
    ouf.write('energyRoom = ' + str(energyRoom) + '\n')
    ouf.write('waveLengthRoom = ' + str(waveLengthRoom) + '\n')
    ouf.write('changeEnergy = ' + str(changeEnergy) + '\n')
    ouf.write('changeWaveLength = ' + str(changeWaveLength) + '\n')

plt.plot(tempGiven, energyGiven, 'ks')
plt.plot(tempApprox, energyApprox, 'r')

plt.xlabel('T, K')
plt.ylabel('E, эВ')

plt.grid()
plt.savefig('not_inverted.png', dpi=600)
plt.show()
