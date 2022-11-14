# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 07:54:50 2021
 
@author: m-1
"""

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
        # F = np.array([0.0, 0.0]) # wektor zerowy
 
        F_wart = k * delta_l**2 # wartosc sily
        F_wer = wek_kier / l_akt # wersor sily
        F = - F_wer * F_wart # wektor sily [N]
 
    return F
 
 
# Dane ukladu
m = 1.0  #[kg]
g = 9.81 #[m/s^2]
yg = 0.0 #[m]
 
lswob = 1.0 #[m]
k = 10.0 #[N/m^2]
b = np.array([1.0, 1.0]) #[m]
 
# IC - initial conditions
x0 = np.array([2.0, 1.0]) #[m]
v0 = np.array([0.0, 0.0]) #[m/s]
 
# parametry rozwiazania
delta_t = 0.01 #[s] -> krok czasowy
steps = 5000 # -> liczba krokow
 
# ROZWIAZANIE
t_macierz = np.zeros(steps)
x_macierz = np.zeros(steps)
y_macierz = np.zeros(steps)
 
vcur = v0
xcur = x0
 
for i in range(steps):
    # wyznaczenie sily dzialajacej na cialo
    #######################################
    Fwyp = np.zeros(2)
 
    #sila oporu powietrza
    vcur_w = np.linalg.norm(vcur)
 
    if vcur_w > 0.0:
        Fop_kier = - vcur / vcur_w
        Fop_w = 0.05 * vcur_w * vcur_w
        Fop = Fop_kier * Fop_w
    else:
        Fop = np.array([0.0, 0.0])
 
    #grawitacja
    Fg = - m * g
 
    #sila od sprezyny
    Fspr = wyznacz_sile_od_wiez(xcur, b, k, lswob)
 
    #sila wypadkowa
    Fwyp += Fop
    Fwyp[1] += Fg
    Fwyp += Fspr
 
 
    # Fwyp[0] = Fop[0]
    # Fwyp[1] = Fop[1] + Fg
 
    # if xcur[1] <= yg:
    #     vcur[1] = - vcur[1]
 
    #######################################
 
    vnext = Fwyp / m * delta_t + vcur
    xnext = vnext * delta_t + xcur
 
    # zapis wynikow
    x_macierz[i] = xnext[0]
    y_macierz[i] = xnext[1]
 
    t = delta_t * (i + 1)
    t_macierz[i] = t
 
    # wyswietlanie wynikow
    # print("vnext:", vnext, "xnext:", xnext, " || t:", t)
 
    # nadpisanie aktualnego stanu ukladu
    vcur = vnext
    xcur = xnext
 
# plt.plot(x_macierz, y_macierz)
# plt.show()
 
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(t_macierz, x_macierz, y_macierz)
# plt.show()
 
 
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque
 
history_len = 500  # how many trajectory points to display
 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(autoscale_on=False, xlim=(-1., 3.), ylim=(-3., 1.5))
ax.set_aspect('equal')
ax.grid()
 
line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)
 
dt = delta_t
def animate(i):
    thisx = [b[0], x_macierz[i]]
    thisy = [b[1], y_macierz[i]]
 
    if i == 0:
        history_x.clear()
        history_y.clear()
 
    history_x.appendleft(x_macierz[i])
    history_y.appendleft(y_macierz[i])
 
    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text
 
 
ani = animation.FuncAnimation(
    fig, animate, len(y_macierz), interval=dt*1000, blit=True)
plt.show()