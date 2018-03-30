# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 12:52:39 2018

@author: LACOMBE Armand, BROUX Lucas.
"""

# Imports.
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import csv

# Open file.
f = open(join("data", "clean_data_bordeaux.csv"), "r")
d = np.loadtxt(f, delimiter=',')

# Does the Buys-Ballot decomposotion.
y = d[:, 1]
m1 = np.ones(len(y))
m2 = np.arange(1, len(y)+1)
s = (len(y),365)
S = np.zeros(s)
for t in range(len(y)):
    S[t , int(np.rint(t - np.floor((t+1)/365.24)*365.24)  % 365)] = 1
for j in range(364):
    S[:,j] = S[:,j]-S[:,364]
    
M = np.array((m1,m2)).T
S = np.delete(S,364,1)

mat1 = np.bmat([[np.dot(M.T,M), np.dot(M.T,S)],[np.dot(S.T,M), np.dot(S.T,S)]])
mat2 = np.concatenate((np.dot(M.T,y),np.dot(S.T,y)), axis=0)
matFinal = np.dot(np.linalg.inv(mat1),mat2)
matFinal = np.array(matFinal)[0]

y_trend = matFinal[0] * m1 + matFinal[1] * m2
y_season = np.dot(S,matFinal[2:])
y_res = y - y_trend - y_season
y_pred = y - y_res

# Exports a csv file of the predictions, because of reasons.

L_trend = np.array([y_trend.tolist()])
L_season = np.array([y_season.tolist()])
L_res = np.array([y_res.tolist()])
L_pred = np.array([y_pred.tolist()])

data = np.concatenate((L_trend, L_season, L_res, L_pred), axis=0).T.tolist()

with open("Buysballot.csv", "w", newline='') as f_write:
    writer = csv.writer(f_write)
    writer.writerows(data)
f.close()

# Plots error = f(predicted) so as to guess wether an additive model is relevant or not.

plt.figure(0)
plt.figure(figsize=(20,5))
plt.scatter(y_pred,abs(y_res),s=1)
plt.title('erreur en fonction de la prediction')
plt.xlabel('prediction')
plt.ylabel('erreur')
plt.show()
print("On valide donc un modèle additif.")

# Plots stuff.

x = range(len(y))

plt.figure(1)
plt.figure(figsize=(20,5))
plt.plot(x, y_trend, label = "trend de la température à Bordeaux depuis 1946")
plt.title("température à Bordeaux depuis 1946")
plt.xlabel("Temps")
plt.ylabel("Température")
plt.legend(loc = 'best')
plt.show()
plt.figure(2)
plt.figure(figsize=(20,5))
plt.plot(x, y_season, label = "saisonnalité de la température à Bordeaux depuis 1946")
plt.title("température à Bordeaux depuis 1946")
plt.xlabel("Temps")
plt.ylabel("Température")
plt.legend(loc = 'best')
plt.show()
plt.figure(3)
plt.figure(figsize=(20,5))
plt.plot(x, y_res, label = "résidus de la température à Bordeaux depuis 1946")
plt.title("température à Bordeaux depuis 1946")
plt.xlabel("Temps")
plt.ylabel("Température")
plt.legend(loc = 'best')
plt.show()

plt.figure(figsize=(20,20))
plt.figure(1)
plt.subplot(311)
plt.plot(x, y_trend)
plt.title("trend de la température à Bordeaux depuis 1946")
plt.subplot(312)
plt.plot(x, y_season)
plt.title("saisonnalité de la température à Bordeaux depuis 1946")
plt.subplot(313)
plt.plot(x, y_res, label = "résidus de la température à Bordeaux depuis 1946")
plt.title("résidu de la température à Bordeaux depuis 1946")
plt.show()
