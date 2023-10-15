import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.optimize import minimize
from scipy.special import jv

a = 2e-3
l = 1064e-9
c = 0 + 1j
E0 = 15
k = 2 * np.pi / l
R = 3 * a
z = 4
u1 = 5e-6
v1 = 5e-6

f1 = random.uniform(-np.pi, np.pi)
f2 = random.uniform(-np.pi, np.pi)
f3 = random.uniform(-np.pi, np.pi)
f4 = random.uniform(-np.pi, np.pi)


def intensity_distribution(individual):
    df1, df2, df3, df4 = individual
    uv = u1 ** 2 + v1 ** 2
    E_00 = np.pi * a ** 2 * np.exp(c * k * (uv) / (2 * z)) / (c * l * z) * E0 * 2 * 2 * jv(1,
                                                                                           k * a / z * np.sqrt(uv)) / (
                   k * a / z * np.sqrt(uv))
    E_res = E_00 * (np.exp(-c * k * R / 2 / z * (u1 + v1)) * np.exp(-c * (f1 + df1)) + np.exp(
        -c * k * R / 2 / z * (u1 - v1)) * np.exp(-c * (f2 + df2)) + np.exp(-c * k * R / 2 / z * (v1 - u1)) * np.exp(
        -c * (f3 + df3)) + np.exp(-c * k * R / 2 / z * (u1 + v1)) * np.exp(-c * (f4 + df4)))
    I = E_res * np.conj(E_res) * 26.55e-4
    return -I.real


res = minimize(intensity_distribution, [0, 0, 0, 0], method="L-BFGS-B",
               bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])

optimal_params = res.x
list = [f1, f2, f3, f4]
list_1 = np.array(list)
list_2 = np.array(list + optimal_params)

print('Начальные фазы лучей:', *list_1)
print('Сдвиги фаз на молуляторах:', *optimal_params)
print('Конечные фазы лучей:', *list_1 + optimal_params)

# создание сетки значений x, y, z
u = np.linspace(-100e-5, 100e-5, 1000)
v = np.linspace(-100e-5, 100e-5, 1000)

phase1 = list_2[0]
phase2 = list_2[1]
phase3 = list_2[2]
phase4 = list_2[3]

initphase1 = list_1[0]
initphase2 = list_1[1]
initphase3 = list_1[2]
initphase4 = list_1[3]


def intensity_distribution1(f1, f2, f3, f4, R, u, v, z):
    uv = u ** 2 + v ** 2
    E_00 = np.pi * a ** 2 * np.exp(c * k * (uv) / (2 * z)) / (c * l * z) * E0 * 2 * 2 * jv(1,
                                                                                           k * a / z * np.sqrt(uv)) / (
                   k * a / z * np.sqrt(uv))
    E_res = E_00 * (
            np.exp(-c * k * R / 2 / z * (u + v)) * np.exp(-c * f1) + np.exp(-c * k * R / 2 / z * (u - v)) * np.exp(
        -c * f2) + np.exp(-c * k * R / 2 / z * (v - u)) * np.exp(-c * f3) + np.exp(
        -c * k * R / 2 / z * (u + v)) * np.exp(-c * f4))
    I = E_res * np.conj(E_res) * 26.55e-4
    return I.real


# Создание сетки значений X и Y
U, V = np.meshgrid(u, v)

# Вычисление интенсивности для каждой точки сетки
intensity_opt = intensity_distribution1(phase1, phase2, phase3, phase4, R, U, V, z)
intensity_init = intensity_distribution1(initphase1, initphase2, initphase3, initphase4, R, U, V, z)

central_intensity_opt = intensity_distribution1(phase1, phase2, phase3, phase4, R, u1, v1, z)
central_intensity_init = intensity_distribution1(initphase1, initphase2, initphase3, initphase4, R, u1, v1, z)
print('Отношение интенсивностей в центре картины до и после фазировки:', central_intensity_init / central_intensity_opt)

# Создание фигуры и осей
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Отображение первого графика
im1 = ax1.imshow(intensity_opt, extent=[np.min(u), np.max(u), np.min(v), np.max(v)], origin='lower', cmap='viridis')
ax1.set_xticks([-100e-5, -50e-5, 0, 50e-5, 100e-5])  # Установка соответствующих значений на оси x
ax1.set_yticks([-100e-5, -50e-5, 0, 50e-5, 100e-5])
ax1.set_xlabel('u, m')
ax1.set_ylabel('v, m')
ax1.set_title('Optimized Intensity in the uv Plane at z={} m'.format(z))
fig.colorbar(im1, ax=ax1)

# Отображение второго графика
im2 = ax2.imshow(intensity_init, extent=[np.min(u), np.max(u), np.min(v), np.max(v)], origin='lower', cmap='viridis')
ax2.set_xticks([-100e-5, -50e-5, 0, 50e-5, 100e-5])  # Установка соответствующих значений на оси x
ax2.set_yticks([-100e-5, -50e-5, 0, 50e-5, 100e-5])
ax2.set_xlabel('u, m')
ax2.set_ylabel('v, m')
ax2.set_title('Initial Intensity in the uv Plane at z={} m'.format(z))
fig.colorbar(im2, ax=ax2)
plt.show()
