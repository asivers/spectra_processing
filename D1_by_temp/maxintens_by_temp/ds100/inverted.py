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
    data[filename.split('K')[0]] = max(localMaxArr)

tempGivenArr = []
maxGivenArr = []

for pair in sorted(data.items(), key=lambda x: (len(x[0]), x[0])):
    tempGivenArr.append(float(pair[0]))
    maxGivenArr.append(float(pair[1]))

tempGiven = np.array(tempGivenArr)
maxGiven = np.array(maxGivenArr)

p0 = [0.002, 15, 0.01]
params = []
params = curve_fit(approx, tempGiven, maxGiven, p0, bounds=((0, -np.inf, -np.inf), (1, np.inf, np.inf)), maxfev=99999)
Ea = params[0][0]
B = params[0][1]
C = params[0][2]
print(Ea, B, C)
with open('eabc.txt', 'w') as ouf:
    ouf.write('Ea = ' + str(Ea) + '\n')
    ouf.write('B = ' + str(B) + '\n')
    ouf.write('C = ' + str(C) + '\n')

leftT = int(tempGiven[0])
rightT = int(tempGiven[-1]) + 1200

tempApproxArr = []
maxApproxArr = []
for iTemp in range(leftT, rightT):
    T = float(iTemp)
    tempApproxArr.append(T)
    curr = B / (1. + C * T**1.5 / 2.71828**(-Ea * 11594. / T))
    maxApproxArr.append(curr)

tempGiven = np.array(tempGivenArr)
maxGiven = np.array(maxGivenArr)
tempApprox = np.array(tempApproxArr)
maxApprox = np.array(maxApproxArr)

for i in range(len(tempGiven)):
    tempGiven[i] = 1000 / tempGiven[i]
    maxGiven[i] = math.log10(maxGiven[i])
for i in range(len(tempApprox)):
    tempApprox[i] = 1000 / tempApprox[i]
    maxApprox[i] = math.log10(maxApprox[i])

plt.plot(tempGiven, maxGiven, 'ks')
plt.plot(tempApprox, maxApprox, 'r')

plt.xlabel('1000/T, (1/K)')
plt.ylabel('log(I), отн. ед.')

plt.grid()
plt.savefig('inverted.png', dpi=600)
plt.show()
