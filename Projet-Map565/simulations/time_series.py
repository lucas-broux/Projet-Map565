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

# Open file.
f = open(join("data", "clean_data_bordeaux.csv"), "r")
d = np.loadtxt(f, delimiter=',')

# Plot time series.
y = d[:, 1]
x = range(len(d[:, 1]))
plt.figure(1)
plt.plot(x, y, label = "Température à Bordeaux depuis 1946")
plt.title("Température à Bordeaux depuis 1946")
plt.xlabel("Temps")
plt.ylabel("Température")
plt.legend(loc = 'best')
plt.show()
