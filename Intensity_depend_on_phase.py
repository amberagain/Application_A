import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
import random as rand

# константы
a = 2e-3
l = 1064e-9
c = 0 + 1j
E0 = 15
k = 2 * np.pi / l
R = 3 * a
const = 26.55e-4
z = 4


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


def generate_random_phases():  # генерация случайных фаз
    random_phases = [rand.uniform(-np.pi, np.pi) for _ in range(4)]
    return random_phases


# Создание сетки значений X и Y
c = 0 + 1j
u = np.linspace(-0.003, 0.003, 500)  # Изменим диапазон для X
v = np.linspace(-0.003, 0.003, 500)  # Изменим диапазон для Y

# Вычисление интенсивности для каждой точки сетки
phase_1, phase_2, phase_3, phase_4 = generate_random_phases()
phase_01, phase_02, phase_03, phase_04 = 0, 0, 0, 0
I1 = intensity_distribution(phase_1, phase_2, phase_3, phase_4, R, u, 0, z)  # Интенсивность для случайных фаз
I2 = intensity_distribution(phase_01, phase_02, phase_03, phase_04, R, u, 0, z)  # Интенсивность для фиксированных фаз

# Визуализация графика
plt.plot(v, I1, label=f'I (Phase: {phase_1:.2f}, {phase_2:.2f}, {phase_3:.2f}, {phase_4:.2f})', linestyle='--')
plt.plot(v, I2, label=f'I (Phase: {phase_01:.2f}, {phase_02:.2f}, {phase_03:.2f}, {phase_04:.2f})')
plt.xlabel('V')
plt.ylabel('Intensity (I)')
plt.title('Intensity vs V')
plt.grid()
plt.legend()
plt.show()
