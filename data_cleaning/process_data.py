#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The purpose of this script is to export data in an exploitable .csv format.
Authors : LACOMBE Armand, BROUX Lucas.
Python version : 3.*
"""

# Imports.
from os.path import join
import numpy as np

# Open file.
f = open(join("Bordeaux", "data3.txt"), "r")
d = np.loadtxt(f, delimiter=',')

# Process data.
d = np.delete(d, 0, 1) # Remove id column.
d = np.delete(d, range(1 + np.argwhere(d[:,2]==9)[-1][0]), 0) # Keep data only after last invalid value (invalidity index == 9).
d = np.delete(d, 2, 1) # Remove last column (invalidity index).

# Save as clean_data.txt
np.savetxt('clean_data_bordeaux2.csv', d, delimiter=',')
