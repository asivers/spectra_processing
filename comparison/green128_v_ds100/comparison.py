import os
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


data = dict()

green128 = []
with open('10K128times30mv45mW2slit33nm.mdrs', 'r') as inf:
    while True:
        line = inf.readline().strip()
        if not line:
            break
        green128.append(float(line))

ds100 = []
with open('10K100W100mv45mW2slit33nm.mdrs', 'r') as inf:
    while True:
        line = inf.readline().strip()
        if not line:
            break
        ds100.append(float(line))

waveLength = []
for i in range(len(ds100)):
    waveLength.append(i * 2 + 1000)

ax = plt.subplot(111)
ax.plot(waveLength, green128, label='f = 50 кГц')
ax.plot(waveLength, ds100, label='f = 900 кГц')
ax.legend()

plt.plot(waveLength, green128, 'g')
plt.plot(waveLength, ds100, 'gray')

plt.xlabel('длина волны, нм')
plt.ylabel('I, отн. ед.')

plt.grid()
plt.savefig('not_inverted.png', dpi=600)
plt.show()
