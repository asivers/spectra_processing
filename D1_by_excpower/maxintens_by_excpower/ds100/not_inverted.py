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


# def approx(_temp, _Ea, _B, _C):
#     return _B / (1. + _C * _temp**1.5 / 2.71828**(-_Ea * 11594. / _temp))


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
    excPowerStr = (filename.split('mW')[0]).split('mv')[1]
    data[excPowerStr] = max(localMaxArr)

excPowerGivenArr = []
maxGivenArr = []

for pair in sorted(data.items(), key=lambda x: (len(x[0]), x[0])):
    excPowerGivenArr.append(float(pair[0]))
    maxGivenArr.append(float(pair[1]))

trueExcPowerGivenArr = [5.625, 11.25, 22.5, 45.0, 67.5]
excPowerGiven = np.array(trueExcPowerGivenArr)
maxGiven = np.array(maxGivenArr)

# params = curve_fit(approx, excPowerGiven, maxGiven)
# print(params[0])
# with open('result.txt', 'w') as ouf:
#     ouf.write(params[0][0])

plt.plot(excPowerGiven, maxGiven, 'ks')
# plt.plot(excPowerGiven, maxGiven, 'r')
bspl = splrep(excPowerGiven, maxGiven, s=1000000)
maxGivenBspl = splev(excPowerGiven, bspl)
plt.plot(excPowerGiven, maxGivenBspl, 'b')

plt.xlabel('мощность возбуждения, мВт')
plt.ylabel('I, отн. ед.')

plt.grid()
plt.savefig('not_inverted.png', dpi=600)
plt.show()
