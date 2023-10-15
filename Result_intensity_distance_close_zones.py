import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
import random as rand


def generate_random_phases():  # генерация случайных фаз
    random_phases = [rand.uniform(-np.pi, np.pi) for _ in range(4)]
    return random_phases


# создание сетки значений x, y, z
u = np.linspace(-100e-5, 100e-5, 1000)
v = np.linspace(-100e-5, 100e-5, 1000)
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
z = 4  # Фиксированное значение z для визуализации влияния на интенсивность

# задаем случайные значения фаз для каждого пучка
phase1, phase2, phase3, phase4 = generate_random_phases()
# phase1, phase2, phase3, phase4 = np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4
# phase1, phase2, phase3, phase4 = 0, 0, 0, 0
print(phase1, phase2, phase3, phase4)

# константы
a = 2e-3
l = 1064e-9
c = 0 + 1j
E0 = 15
k = 2 * np.pi / l
R = 3 * a
const = 26.55e-4


def intensity_distribution(f1, f2, f3, f4, R, u, v, z):
    uv = u ** 2 + v ** 2
    E_00 = np.pi * a ** 2 * np.exp(c * k * (uv) / (2 * z)) / (c * l * z) * E0 * 2 * 2 * jv(1,
                                                                                           k * a / z * np.sqrt(uv)) / (
                   k * a / z * np.sqrt(uv))
    E_res = E_00 * (
            np.exp(-c * k * R / 2 / z * (u + v)) * np.exp(c * f1) + np.exp(-c * k * R / 2 / z * (u - v)) * np.exp(
        c * f2) + np.exp(-c * k * R / 2 / z * (v - u)) * np.exp(c * f3) + np.exp(
        c * k * R / 2 / z * (u + v)) * np.exp(c * f4))
    I = E_res * np.conj(E_res)
    return I.real * const


def intensity_in_the_close_zone(x, y):
    R, a = 5, 3  # для масштабирования
    sigma = 0.8
    gaussian = np.exp(-((x - R / 2) ** 2 + (y - R / 2) ** 2) / (2 * sigma ** 2))
    gaussian += np.exp(-((x - R / 2) ** 2 + (y + R / 2) ** 2) / (2 * sigma ** 2))
    gaussian += np.exp(-((x + R / 2) ** 2 + (y - R / 2) ** 2) / (2 * sigma ** 2))
    gaussian += np.exp(-((x + R / 2) ** 2 + (y + R / 2) ** 2) / (2 * sigma ** 2))
    result = E0 ** 2 * np.maximum(gaussian, 0)  # Предотвращает отрицательные значения
    return result


# Создание сетки значений X и Y

U, V = np.meshgrid(u, v)
X, Y = np.meshgrid(x, y)

# Создание фигуры и осей
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Вычисление интенсивности для каждой точки сетки
intensity = intensity_distribution(phase1, phase2, phase3, phase4, R, U, V, z)

# Вычисление интенсивности в ближней зоне
inten_close_zone = intensity_in_the_close_zone(X, Y)

# Отображение первого графика
im1 = ax1.imshow(intensity, extent=[np.min(u), np.max(u), np.min(v), np.max(v)], origin='lower', cmap='viridis')
ax1.set_xticks([-100e-5, -50e-5, 0, 50e-5, 100e-5])  # Установка соответствующих значений на оси x
ax1.set_yticks([-100e-5, -50e-5, 0, 50e-5, 100e-5])
ax1.set_xlabel('u, м')
ax1.set_ylabel('v, м')
ax1.set_title('Intensity Distribution in the uv Plane at z={}m'.format(z))
fig.colorbar(im1, ax=ax1)

# Отображение второго графика
im2 = ax2.imshow(inten_close_zone, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], origin='lower', cmap='viridis')
ax2.set_xlabel('x, cm')
ax2.set_ylabel('y, cm')
ax2.set_title('Intensity in the Close Zone')
fig.colorbar(im2, ax=ax2)
plt.show()
