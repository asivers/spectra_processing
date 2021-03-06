import os
import math
import numpy as np
from scipy.interpolate import splrep, splev
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


def approx(_temp, _Ea, _B, _C):
    return _B / (1. + _C * _temp**1.5 / 2.71828**(-_Ea * 11594. / _temp))


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
    data[(filename.split('K')[1]).split('times')[0]] = max(localMaxArr)

numGivenArr = []
maxGivenArr = []

for pair in sorted(data.items(), key=lambda x: (len(x[0]), x[0])):
    numGivenArr.append(float(pair[0]))
    maxGivenArr.append(float(pair[1]))

numGiven = np.array(numGivenArr)
maxGiven = np.array(maxGivenArr)

# p0 = [0.002, 15, 0.01]
# params = []
# params = curve_fit(approx, numGiven, maxGiven, p0, bounds=((0, -np.inf, -np.inf), (1, np.inf, np.inf)), maxfev=99999)
# Ea = params[0][0]
# B = params[0][1]
# C = params[0][2]
# print(Ea, B, C)
# with open('eabc.txt', 'w') as ouf:
#     ouf.write('Ea = ' + str(Ea) + '\n')
#     ouf.write('B = ' + str(B) + '\n')
#     ouf.write('C = ' + str(C) + '\n')
#
# leftT = int(numGiven[0] + 3)
# rightT = int(numGiven[-1])
#
# tempApproxArr = []
# maxApproxArr = []
# for iTemp in range(leftT, rightT):
#     T = float(iTemp)
#     tempApproxArr.append(T)
#     curr = B / (1. + C * T**1.5 / 2.71828**(-Ea * 11594. / T))
#     maxApproxArr.append(curr)

numGiven = np.array(numGivenArr)
maxGiven = np.array(maxGivenArr)
# tempApprox = np.array(tempApproxArr)
# maxApprox = np.array(maxApproxArr)

plt.plot(numGiven, maxGiven, 'ks')
plt.plot(numGiven, maxGiven, 'g')
# plt.plot(tempApprox, maxApprox, 'r')
# bspl = splrep(numGiven, maxGiven, s=1000000)
# maxGivenBspl = splev(numGiven, bspl)
# plt.plot(numGiven, maxGivenBspl, 'b')

plt.xlabel('Количество сканирующих проходов')
plt.ylabel('I, отн. ед.')

plt.grid()
plt.savefig('not_inverted.png', dpi=600)
plt.show()
