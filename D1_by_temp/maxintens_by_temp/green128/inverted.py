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


def single_approx(_temp, _Ea, _I0, _B):
    return _I0 / (1. + _B * (_temp ** 1.5) * (2.71828 ** (-_Ea * 11594. / _temp)))


def single_count(_tempGivenArr, _maxGivenArr):
    tempGiven = np.array(_tempGivenArr)
    maxGiven = np.array(_maxGivenArr)
    params = curve_fit(single_approx, tempGiven, maxGiven,
                       p0=[0, 0, 0],
                       bounds=((0, 0, 0),
                               (np.inf, np.inf, np.inf)),
                       maxfev=99999)
    Ea = params[0][0]
    I0 = params[0][1]
    B = params[0][2]
    print(Ea, I0, B)
    with open('single_result.txt', 'w') as ouf:
        ouf.write('Ea = ' + str(Ea) + '\n')
        ouf.write('I0 = ' + str(I0) + '\n')
        ouf.write('B = ' + str(B) + '\n')
    leftT = int(tempGiven[0])
    rightT = int(tempGiven[-1]) + 450
    _tempApproxArr = []
    _maxApproxArr = []
    for iTemp in range(leftT, rightT):
        T = float(iTemp)
        _tempApproxArr.append(T)
        curr = I0 / (1. + B * (T ** 1.5) * (2.71828 ** (-Ea * 11594. / T)))
        _maxApproxArr.append(curr)
    returnArr = []
    returnArr.append(_tempApproxArr)
    returnArr.append(_maxApproxArr)
    return returnArr


def double_approx(_temp, _Ea1, _Ea2, _I0, _B1, _B2):
    return _I0 / (1. + _B1 * (_temp ** 1.5) * (2.71828 ** (-_Ea1 * 11594. / _temp)) +
                  _B2 * (_temp ** 1.5) * (2.71828 ** (-_Ea2 * 11594. / _temp)))


def double_count(_tempGivenArr, _maxGivenArr):
    tempGiven = np.array(_tempGivenArr)
    maxGiven = np.array(_maxGivenArr)
    params = curve_fit(double_approx, tempGiven, maxGiven,
                       p0=[0, 0, 0, 0, 0],
                       bounds=((0, 0, 0, 0, 0),
                               (np.inf, np.inf, np.inf, np.inf, np.inf)),
                       maxfev=99999)
    Ea1 = params[0][0]
    Ea2 = params[0][1]
    I0 = params[0][2]
    B1 = params[0][3]
    B2 = params[0][4]
    print(Ea1, Ea2, I0, B1, B2)
    with open('double_result.txt', 'w') as ouf:
        ouf.write('Ea1 = ' + str(Ea1) + '\n')
        ouf.write('Ea2 = ' + str(Ea2) + '\n')
        ouf.write('I0 = ' + str(I0) + '\n')
        ouf.write('B1 = ' + str(B1) + '\n')
        ouf.write('B2 = ' + str(B2) + '\n')
    leftT = int(tempGiven[0])
    rightT = int(tempGiven[-1]) + 200
    _tempApproxArr = []
    _maxApproxArr = []
    for iTemp in range(leftT, rightT):
        T = float(iTemp)
        _tempApproxArr.append(T)
        curr = I0 / (1. + B1 * (T ** 1.5) * (2.71828 ** (-Ea1 * 11594. / T)) + B2 * (T ** 1.5) * (2.71828 ** (
                    -Ea2 * 11594. / T)))
        _maxApproxArr.append(curr)
    returnArr = []
    returnArr.append(_tempApproxArr)
    returnArr.append(_maxApproxArr)
    return returnArr


data = dict()

for filename in os.listdir('files'):
    fileData = []
    localMaxArr = []
    with open(os.path.join('files', filename), 'r') as inf:
        while True:
            line = inf.readline().strip()
            if not line:
                break
            fileData.append(float(line))
    mainSize = len(fileData)
    for i in range(mainSize - 100, mainSize - 20):
        if check(i, fileData):
            localMaxArr.append(fileData[i])
    data[filename.split('K')[0]] = max(localMaxArr)

tempGivenArr = []
maxGivenArr = []

for pair in sorted(data.items(), key=lambda x: (len(x[0]), x[0])):
    tempGivenArr.append(float(pair[0]))
    maxGivenArr.append(float(pair[1]))

tempGiven = np.array(tempGivenArr)
maxGiven = np.array(maxGivenArr)

resultSingle = single_count(tempGivenArr, maxGivenArr)
tempApproxSingle = resultSingle[0]
maxApproxSingle = resultSingle[1]

resultDouble = double_count(tempGivenArr, maxGivenArr)
tempApproxDouble = resultDouble[0]
maxApproxDouble = resultDouble[1]

for i in range(len(tempGiven)):
    tempGiven[i] = 1000 / tempGiven[i]
    maxGiven[i] = math.log10(maxGiven[i])
for i in range(len(tempApproxSingle)):
    tempApproxSingle[i] = 1000 / tempApproxSingle[i]
    maxApproxSingle[i] = math.log10(maxApproxSingle[i])
for i in range(len(tempApproxDouble)):
    tempApproxDouble[i] = 1000 / tempApproxDouble[i]
    maxApproxDouble[i] = math.log10(maxApproxDouble[i])

plt.plot(tempGiven, maxGiven, 'ks')
plt.plot(tempApproxSingle, maxApproxSingle, color='g', label='Аппроксимация по формуле (5)')
plt.plot(tempApproxDouble, maxApproxDouble, color='r', label='Аппроксимация по формуле (6)')
plt.legend()

plt.xlabel('1000/T, (1/K)')
plt.ylabel('log(I), отн. ед.')

plt.grid()
plt.savefig('inverted.png', dpi=600)
plt.show()
