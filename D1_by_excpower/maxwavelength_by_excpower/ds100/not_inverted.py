import os
import math
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
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
    for i in range(len(_waveLength)):
        _waveLength[i] = 1243.125 / _waveLength[i]
    return _waveLength


# def approx(_temp, _Ea, _B, _C):
#     return _B / (1. + _C * _temp**1.5 / 2.71828**(-_Ea * 11594. / _temp))


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
    excPowerStr = (filename.split('mW')[0]).split('mv')[1]
    data[excPowerStr] = localMaxWaveLength

excPowerGivenArr = []
waveLengthGivenArr = []

for pair in sorted(data.items(), key=lambda x: (len(x[0]), x[0])):
    excPowerGivenArr.append(float(pair[0]))
    waveLengthGivenArr.append(float(pair[1]))

trueExcPowerGivenArr = [5.625, 11.25, 22.5, 45.0, 67.5]
excPowerGiven = np.array(trueExcPowerGivenArr)
wavelengthGiven = np.array(waveLengthGivenArr)
energyGiven = np.array(energy(waveLengthGivenArr))
with open('result', 'w') as ouf:
    ouf.write('wavelength (nm): ' + str(wavelengthGiven) + '\n')
    ouf.write('energy (eV): ' + str(energyGiven) + '\n')

plt.plot(excPowerGiven, energyGiven, 'ks')
plt.plot(excPowerGiven, energyGiven, 'r')

plt.xlabel('мощность возбуждения, мВт')
plt.ylabel('E, эВ')

plt.grid()
plt.savefig('not_inverted.png', dpi=600)
plt.show()
