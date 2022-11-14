#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dane ukladu
m = 1.0 #[kg]
F = 1.0 #[N]

# IC - initial conditions
x0 = 0.0 #[m]
v0 = 0.0 #[m/s]

# parametry rozwiazania
delta_t = 0.1 #[s] -> krok czasowy
steps = 10 # -> liczba krokow

# ROZWIAZANIE
vcur = v0
xcur = x0

vnext = F / m * delta_t + vcur
xnext = vcur * delta_t + xcur

print("vnext:", vnext, "xnext:", xnext)


vcur = vnext
xcur = xnext

vnext = F / m * delta_t + vcur
xnext = vcur * delta_t + xcur

print("vnext2:", vnext, "xnext2:", xnext)


# In[2]:


# Dane ukladu
m = 1.0 #[kg]
F = 1.0 #[N]

# IC - initial conditions
x0 = 0.0 #[m]
v0 = 0.0 #[m/s]

# parametry rozwiazania
delta_t = 0.1 #[s] -> krok czasowy
steps = 10 # -> liczba krokow

# ROZWIAZANIE
vcur = v0
xcur = x0

for i in range(steps):
    vnext = F / m * delta_t + vcur
    xnext = vcur * delta_t + xcur

    print("vnext:", vnext, "xnext:", xnext)

    vcur = vnext
    xcur = xnext


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# Dane ukladu
m = 1.0 #[kg]
F = 1.0 #[N]

# IC - initial conditions
x0 = 0.0 #[m]
v0 = 0.0 #[m/s]

# parametry rozwiazania
delta_t = 0.1 #[s] -> krok czasowy
steps = 10 # -> liczba krokow

# ROZWIAZANIE
t_macierz = np.zeros(steps)
x_macierz = np.zeros(steps)

vcur = v0
xcur = x0

for i in range(steps):
    vnext = F / m * delta_t + vcur
    xnext = vcur * delta_t + xcur

    # zapis wynikow
    x_macierz[i] = xnext
    
    t = delta_t * (i + 1)
    t_macierz[i] = t
    
    # wyswietlanie wynikow
    print("vnext:", vnext, "xnext:", xnext, " || t:", t)

    # nadpisanie aktualnego stanu ukladu
    vcur = vnext
    xcur = xnext
    
plt.plot(t_macierz, x_macierz)
plt.xlabel("t [s]")
plt.ylabel("x [m]")
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Dane ukladu
m = 1.0 #[kg]
F = 1.0 #[N]

# IC - initial conditions
x0 = 0.0 #[m]
v0 = 0.0 #[m/s]

# parametry rozwiazania
delta_t = 0.01 #[s] -> krok czasowy
steps = 10 # -> liczba krokow

# ROZWIAZANIE
t_macierz = np.zeros(steps)
x_macierz = np.zeros(steps)

x_analityczne_macierz = np.zeros(steps)


vcur = v0
xcur = x0

for i in range(steps):
    vnext = F / m * delta_t + vcur
    xnext = vcur * delta_t + xcur

    # zapis wynikow
    x_macierz[i] = xnext
    
    t = delta_t * (i + 1)
    t_macierz[i] = t
    
    
    ## rozwiazanie analityczne
    x_analityczne_macierz[i] = (F / m) * t * t / 2.
    
    
    # wyswietlanie wynikow
    print("vnext:", vnext, "xnext:", xnext, " || t:", t)

    # nadpisanie aktualnego stanu ukladu
    vcur = vnext
    xcur = xnext
    
plt.plot(t_macierz, x_macierz, 'g')
plt.plot(t_macierz, x_analityczne_macierz, 'r')
plt.xlabel("t [s]")
plt.ylabel("x [m]")
plt.show()


# In[5]:


## Sprezyna

import numpy as np
import matplotlib.pyplot as plt

# Dane ukladu
m = 1.0 #[kg]
F = 1.0 #[N]
k = 10. #[N/m]
lswob = 1. #[m]
b = -1. #[m] //nieruchomy przyczep sprezyny

# IC - initial conditions
x0 = 0.0 #[m]
v0 = 0.0 #[m/s]

# parametry rozwiazania
delta_t = 0.05 #[s] -> krok czasowy
steps = 700 # -> liczba krokow

# ROZWIAZANIE
t_macierz = np.zeros(steps)
x_macierz = np.zeros(steps)

vcur = v0
xcur = x0

for i in range(steps):
    # wyznaczenie sily dzialajacej na cialo
    #######################################
    lcur = xcur - b
    delta_l = lcur - lswob
    Fs = - k * delta_l
    
    Fwyp = F + Fs
    #######################################
    
    vnext = Fwyp / m * delta_t + vcur
    xnext = vcur * delta_t + xcur

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


# In[6]:


## Sprezyna

import numpy as np
import matplotlib.pyplot as plt

# Dane ukladu
m = 1.0 #[kg]
F = 1.0 #[N]
k = 10. #[N/m]
lswob = 1. #[m]
b = -1. #[m] //nieruchomy przyczep sprezyny

# IC - initial conditions
x0 = 0.0 #[m]
v0 = 0.0 #[m/s]

# parametry rozwiazania
delta_t = 0.05 #[s] -> krok czasowy
steps = 700 # -> liczba krokow

# ROZWIAZANIE
t_macierz = np.zeros(steps)
x_macierz = np.zeros(steps)

vcur = v0
xcur = x0

for i in range(steps):
    # wyznaczenie sily dzialajacej na cialo
    #######################################
    lcur = xcur - b
    delta_l = lcur - lswob
    Fs = - k * delta_l
    
    Fwyp = F + Fs
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


# In[16]:


## Sprezyna

import numpy as np
import matplotlib.pyplot as plt

# Dane ukladu
m = 1.0 #[kg]
F = 100. #[N]
k = 10. #[N/m]
lswob = 1. #[m]
b = -1. #[m] //nieruchomy przyczep sprezyny

# IC - initial conditions
x0 = 0.0 #[m]
v0 = 0.0 #[m/s]

# parametry rozwiazania
delta_t = 0.05 #[s] -> krok czasowy
steps = 1000 # -> liczba krokow

# ROZWIAZANIE
t_macierz = np.zeros(steps)
x_macierz = np.zeros(steps)

vcur = v0
xcur = x0

for i in range(steps):
    # wyznaczenie sily dzialajacej na cialo
    #######################################
    lcur = xcur - b
    delta_l = lcur - lswob
    Fs = - k * delta_l
    
    Fwyp = F + Fs
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


# In[17]:


## Punkt 2D - dynamika

import numpy as np
import matplotlib.pyplot as plt

# Dane ukladu
m = 1.0 #[kg]
F = 100.0 #[N]
k = 10. #[N/m]
lswob = 1. #[m]
b = -1. #[m] //nieruchomy przyczep sprezyny

Fy = -0.01 #[N]

# IC - initial conditions
x0 = 0.0 #[m]
v0 = 0.0 #[m/s]

x0y = 0.0 #[m]
v0y = 0.0 #[m/s]

# parametry rozwiazania
delta_t = 0.05 #[s] -> krok czasowy
steps = 1000 # -> liczba krokow

# ROZWIAZANIE
t_macierz = np.zeros(steps)
x_macierz = np.zeros(steps)
y_macierz = np.zeros(steps)

vcur = v0
xcur = x0

vcury = v0y
xcury = x0y

for i in range(steps):
    # wyznaczenie sily dzialajacej na cialo - "x"
    #######################################
    lcur = xcur - b
    delta_l = lcur - lswob
    Fs = - k * delta_l
    
    Fwypx = F + Fs
    #######################################
    
    # wyznaczenie sily dzialajacej na cialo - "y"
    #######################################
    Fwypy = Fy
    #######################################
    
    
    # "x"
    vnext = Fwyp / m * delta_t + vcur
    xnext = vnext * delta_t + xcur
    
    # "y"
    vnexty = Fwypy / m * delta_t + vcury
    xnexty = vnexty * delta_t + xcury

    
    # zapis wynikow
    x_macierz[i] = xnext
    y_macierz[i] = xnexty
    
    t = delta_t * (i + 1)
    t_macierz[i] = t
    
    # wyswietlanie wynikow
    # print("vnext:", vnext, "xnext:", xnext, " || t:", t)

    # nadpisanie aktualnego stanu ukladu
    vcur = vnext
    xcur = xnext
    
    vcury = vnexty
    xcury = xnexty
    
plt.plot(x_macierz, y_macierz)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.show()


# In[13]:


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(t_macierz, x_macierz, y_macierz)
plt.show()


# In[27]:


## Sprezyna

import numpy as np
import matplotlib.pyplot as plt

# Dane ukladu
m = 1.0 #[kg]
F = 100. #[N]
k = 10. #[N/m]
lswob = 1. #[m]
b = -1. #[m] //nieruchomy przyczep sprezyny

Fy = -0.03 #[N]

# IC - initial conditions
x0 = np.array([0.0, 0.0]) #[m]
v0 = np.array([0.0, 0.0]) #[m/s]

# parametry rozwiazania
delta_t = 0.05 #[s] -> krok czasowy
steps = 1000 # -> liczba krokow

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
    
    lcur = xcur[0] - b
    delta_l = lcur - lswob
    Fs = - k * delta_l
    
    Fwyp[0] = F + Fs
    Fwyp[1] = Fy
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
    
plt.plot(x_macierz, y_macierz)
plt.show()


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(t_macierz, x_macierz, y_macierz)
plt.show()


# In[23]:


print(t_macierz.max())


# In[37]:


# parametry wiezadla
b = np.array([1., 2.])  # [m]
k = 10.    # [N/m^2]

lswob = 1. # [m]

# punkt 2d
xcur = np.array([5., 2.5])

# wyznaczenie sily od wiezadla
## a == xcur
wekt_wiez = b - xcur
lakt = np.linalg.norm(wekt_wiez)

delta_l = lakt - lswob

Fwiez_wart = k * delta_l * delta_l

Fwiez_kier = wekt_wiez / lakt



# wiezadlo / ciegno
if delta_l > 0.:
    Fwiez = Fwiez_kier * Fwiez_wart
else:
    Fwiez = np.array([0., 0.])

    
# sprezyna 2d
if delta_l > 0.:
    Fwiez = Fwiez_kier * Fwiez_wart
else:
    Fwiez = - Fwiez_kier * Fwiez_wart
    
    

print(wekt_wiez)
print(lakt)
print(delta_l)
print(Fwiez_wart)
print(Fwiez_kier, np.linalg.norm(Fwiez_kier))
print(Fwiez)


# In[45]:


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

#Dane:
# [m] - dwuwymiarowy wektor polozenia przyczepu ruchomego A
a = np.array([2.0, 3.0])
# [m] - dwuwymiarowy wektor polozenia przyczepu nieruchomego B
b = np.array([-1.0, 0.2])
# [N/m^2] - wspolczynnik sztywnosci wiezadla w modelu kwadratowym
k = 11.0
# [m] - dlugosc swobodna wiezadla
l_swob = 2.1

print("--- TEST 1 ---")
print("Sila dla zadanych parametrow wiezadla wynosi: ")
print(wyznacz_sile_od_wiez(a, b, k, l_swob), "N")


# In[ ]:




