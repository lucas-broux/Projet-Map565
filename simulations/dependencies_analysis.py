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
from scipy.optimize import minimize

# Open file.
f = open(join("data", "clean_data_bordeaux_max.csv"), "r")
d = np.loadtxt(f, delimiter=',')

# Arrange data.
y = []
for (date, temperature) in d:
    year = int(str(int(date))[0:4])
    month = int(str(int(date))[4:6])
    day = int(str(int(date))[6:8])
    if ((month == 7 and day >= 15) or (month == 8 and day <= 15)):
        y.append(temperature)
x = range(len(y))

# Compute qqplot.
# stats.probplot(y, dist="norm", plot=pylab)
# pylab.show()

# Plot series.
# plt.figure(0)
# plt.plot(x, y, label = "Température estivale à Bordeaux depuis 1946")
# plt.title("Température estivale à Bordeaux depuis 1946")
# plt.xlabel("Temps")
# plt.ylabel("Température")
# plt.legend(loc = 'best')
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


# xi_p = [pickands_estimator(y, k) for k in range(1, int(len(y) / 4))]
# x = range(1, int(len(y) / 4))
# plt.figure(2)
# plt.plot(x, xi_p, label = "Estimateur de Pickands")
# plt.title(r"Estimateur de $\xi_k$ en fonction de k")
# plt.xlabel("k")
# plt.ylabel("Estimateurs")
# plt.legend(loc = 'best')


# Compute N(x).
def Nu(X, x):
    """
    Compute the number of elements in X greater than x.

    :param X: Array of observations.
    :param x: Variable.
    """
    return len([1 for v in X if v > x])

# Compute en(x).
def en(X, x):
    """
    Compute the value of en(x) for the array X and the variable x.

    :param X: Array of observations.
    :param x: Variable.
    """
    if Nu(X, x) != 0:
        return sum([v - x for v in X if v > x]) / Nu(X, x)
    else:
        return 0

# V = sorted(y)
# N = [en(y, x) for x in V]
# plt.figure(2)
# plt.plot(V, N, label = r"$e_n (x)$")
# plt.title(r"$e_n(x)$ en fonction de la température $x$")
# plt.xlabel("Température")
# plt.ylabel(r"$e_n$")
# plt.legend(loc = 'best')

# We take u = 225 for the remaining computations.
u = 225

# Compute the likelihood.
def likelihood(X, xi, beta):
    """
    Return the considered likelihood.

    :param X: The array of observations.
    :param xi: Parameter.
    :param beta: Parameter.
    """
    n = len(X)
    t1 = - n * np.log(beta)
    try:
        t2 = - (1 + 1 / xi) * sum([np.log(1 + (v - u) * xi / beta) for v in X])
    except Exception as e:
        t2 = - 100000
    if t2 != t2:
        t2 = - 100000
    return t2


# Extimate xi and beta by likelihood optimization.
L = lambda x :  - likelihood(y, x[0], x[1])
bnds = ((None, None), (0.1, None))
x0 = (-0.2, 150)
res = minimize(L, x0, bounds=bnds)
[xi_1, beta_1] = res.x


# Estimate xi and beta by linear regression.
V = [v for v in sorted(y) if (v > u and v < 350)]
slope, intercept, r_value, p_value, std_err = stats.linregress(V, [en(y, x) for x in V])
xi_2 = slope / (1 + slope)
beta_2 = (1 - xi_2) * intercept
print("r-squared:", r_value**2)

# Print results.
print(xi_1, xi_2)
print(beta_1, beta_2)


# Compute the repartition function.
def F(X, y, xi, beta):
    """
    Compute F(u + y) with estimated parameters xi and beta.

    :param X: Observations.
    :param y: Positive value.
    :param xi: Parameter.
    :param beta: Parameters.
    :return: F(u + y)
    """
    n = len(X)
    # print((1 + xi * y / beta))
    return 1 - (Nu(X, u) / n) * ((1 + xi * y / beta) ** (- 1 / xi))

# Compute empirical repartition function.
def F_empirical(X, y):
    """
    Compute empirical repartition function.

    :param X: Observations.
    :param y: Variable.
    """
    n = len(X)
    return (len([1 for x in X if x <= y]) / n)


# Plot both repartition functions.
V = range(1, 200)
F_1 = [F(y, v, xi_1, beta_1) for v in V]
F_2 = [F(y, v, xi_2, beta_2) for v in V]
F_3 = [F_empirical(y, u + v) for v in V]
plt.figure(2)
plt.plot([v + u for v in V], F_1, label = r"$\hat{F}(u + y)$ pour l'estimateur par m.v.")
plt.plot([v + u for v in V], F_2, label = r"$\hat{F}(u + y)$ pour l'estimateur par régression linéaire.")
plt.plot([v + u for v in V], F_3, label = r"$\hat{F}(u + y)$ empirique.")
plt.title(r"$\hat{F}(u + y)$ en fonction de la température $u + y$")
plt.xlabel("Température u + y")
plt.ylabel(r"$\hat{F}$")
plt.legend(loc = 'best')
plt.show()

# Print wanted risk values.
risk_1 = F(y, 350 - u, xi_1, beta_1)
risk_2 = F(y, 350 - u, xi_2, beta_2)
risk_3 = F_empirical(y, 350)
print("Risque obtenu par maximum de vraisemblance : " + str(risk_1))
print("Risque obtenu par régression linéaire : " + str(risk_2))
print("Risque obtenu par répartition empirique : " + str(risk_3))
print(F(y, 420 - u, xi_1, beta_1))
print(F_empirical(y, 420))
