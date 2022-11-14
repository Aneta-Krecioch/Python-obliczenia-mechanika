# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:33:55 2021

@author: Admin
"""

#podłoże Winkerowskie -> do zmienienia

import numpy as np
import matplotlib.pyplot as plt

# Dane ukladu
m = 1.0 #[kg]
g = 9.81
yg = 50. #[m]

# IC - initial conditions
x0 = np.array[0 , 100] #[m]
v0 = np.array[100 , 25] #[m]

# parametry rozwiazania
delta_t = 0.01 #[s] -> krok czasowy
steps = 1000 # -> liczba krokow

# ROZWIAZANIE
t_macierz = np.zeros(steps)
x_macierz = np.zeros(steps)


vcur = v0
xcur = x0

for i in range(steps):
    # wyznaczenie sily dzialajacej na cialo
    #######################################
    #lcur = xcur - b
    #delta_l = lcur - lswob
    #Fs = - k * delta_l
    
    Fwyp = np.zeros(2)
    #wersor prędkosci
    #opor powietrza
    vcur_w = np.linalg.norm(vcur)
    Fop_kier = - vcur / np.linalg.norm(vcur)
    Fop_w = 0.3 * vcur_w * vcur_w
    Fop = Fop_kier * Fop_w
    
    Fg = - m * g
    Fwyp[0] +=  Fop
    Fwyp[1] += Fg
    
    if xcur[1] <= yg:
        vcur[1] = - vcur[1]
    #######################################
    
    vnext = Fwyp / m * delta_t + vcur
    xnext = vnext * delta_t + xcur

    # zapis wynikow
    x_macierz[i] = xnext
    
    t = delta_t * (i + 1)
    t_macierz[i] = t
    
    
    # wyswietlanie wynikow
    # print("vnext:", vnext, "xnext:", xnext, " || t:", t)

    # nadpisanie aktualnego stanu ukladu
    vcur = vnext
    xcur = xnext
        
plt.plot(t_macierz, x_macierz)
plt.xlabel("t [s]")
plt.ylabel("x [m]")
plt.show()

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(t_macierz, x_macierz, y_macierz)
plt.show()