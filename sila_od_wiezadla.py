# -*- coding: utf-8 -*-
"""
Created on Wed Dec 01 9:33:55 2021

@author: Admin
"""

#podłoże Winkerowskie -> do zmienienia

import numpy as np
import matplotlib.pyplot as plt

def wyznacz_sile_od_wiez(a, b, k, l_swob):
    wek_kier = b - a

    l_akt = np.linalg.norm(wek_kier)
    # //do wyznaczenia dlugosci wektora wykorzystywana jest
    # //funkcja norm z bilbioteki numpy, ktora
    # //znajduje sie w module linalg
    
    # 2) wyznaczenie przyrostu dlugosci wiezadla
    delta_l = l_akt - l_swob # przyrost dlugosci
    
    # 3) warunek if/else
    if delta_l > 0.0:
        F_wart = k * delta_l**2 # wartosc sily

        F_wer = wek_kier / l_akt # wersor sily
        # wykorzystujemy policzone wczesniej:
        # wek_kier oraz l_akt; to przyspieszy obliczenia
        
        F = F_wer * F_wart # wektor sily [N]
    
    else: # -- jezeli przyrost dlugosci jest mniejszy od zera --
        F = np.array([0.0, 0.0]) # wektor zerowy
    
    return F

# Dane ukladu
m = 1.0 #[kg]
g = 9.81
yg = 50. #[m]

b = np.array([1. , 1.]) #[m]
lswob = 1. #[m]
k = 1. #[N/m^2]

# IC - initial conditions
x0 = np.array[2. , 1.] #[m]
v0 = np.array[0. , 0.] #[m]

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
    
    if vcur_w > 0.:
        Fop_kier = - vcur / np.linalg.norm(vcur)
        Fop_w = 0.3 * vcur_w * vcur_w
        Fop = Fop_kier * Fop_w
    else:
        Fop = np.array([0., 0.])
        
    #grawitacja
    Fg = - m * g
    
    #sprezyna
    Fspr = wyznacz_sile_wiez(xcur, b, k, lswob)
    
    #sila wypadkowa
    Fwyp[0] +=  Fop
    Fwyp[1] += Fg
    Fwyp += Fspr
    
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