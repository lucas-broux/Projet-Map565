#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The purpose of this script is to analyse data with a time series approach.
Authors : LACOMBE Armand, BROUX Lucas.
Python version : 3.*
"""

# Imports.
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats

# Open file.
f = open(join("data", "clean_data_bordeaux_min.csv"), "r")
d = np.loadtxt(f, delimiter=',')

# Arrange data.
y = []
for (date, temperature) in d:
    year = int(str(int(date))[0:4])
    month = int(str(int(date))[4:6])
    day = int(str(int(date))[6:8])
    if ((month == 12 and day >= 15) or (month == 1 and day <= 15)):
        y.append(temperature)
x = range(len(y))

# Compute qqplot.
stats.probplot(y, dist="norm", plot=pylab)
pylab.show()

# Plot series.
plt.figure(0)
plt.plot(x, y, label = "Température estivale à Bordeaux depuis 1946")
plt.title("Température estivale à Bordeaux depuis 1946")
plt.xlabel("Temps")
plt.ylabel("Température")
plt.legend(loc = 'best')
# plt.show()

# Compute Hill estimator.
def hill_estimator(X, k):
    """
    Compute hill estimator with parameter k.

    :param X: The considered values.
    :param k: The parameter of the computation.
    """
    # Sort values.
    X.sort()
    # Compute sum.
    n = len(X)
    s = sum([np.log(X[i]) for i in range(n - k, n)]) / k - np.log(X[n - k])
    return s

# Compute Pickands estimator.
def pickands_estimator(X, k):
    """
    Compute pickands estimator with parameter k.

    :param X: The considered values.
    :param k: The parameter of the computation.
    """
    # Sort values.
    X.sort()
    # Compute estimator.
    n = len(X)
    if k < n / 4:
        s = (1 / np.log(2)) * (np.log( (X[n - k] - X[n - 2 * k]) / (X[n - 2 * k] - X[n - 4 * k])))
    else:
        s = 0
    return s


xi_p = [pickands_estimator(y, k) for k in range(1, int(len(y) / 4))]
x = range(1, int(len(y) / 4))
plt.figure(2)
plt.plot(x, xi_p, label = "Estimateur de Pickands")
plt.title(r"Estimateur de $\xi_k$ en fonction de k")
plt.xlabel("k")
plt.ylabel("Estimateurs")
plt.legend(loc = 'best')
plt.show()
