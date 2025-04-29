

"Example response to a step function (Example 3.2)"
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
# Параметры сигнала
T0 = 10.0
g0 = np.e / T0  # чтобы пиковое значение v = 1 при t = T0

# Временная ось от 0 до 100 с с шагом 0.5 с
t = np.arange(0, 100.5, 0.5)

# Импульсная характеристика (Example 3.2)
v = g0 * t * np.exp(-t / T0)

# Построение графика
plt.figure(figsize=(6, 4))
plt.plot(t, v, color='k')
plt.xlabel('Time (s)')
plt.ylabel('v')
plt.xlim(0, 100)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


# Parameters for discretization
m = n = 150
T0 = 10
Δt = 0.5
t = np.linspace(-1, 10, n)

# Define G matrix using the formula (3.104)
G = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        if i <= j:
            G[i, j] = (t[i] - t[j]) * np.exp(-(t[i] - t[j]) / T0) * Δt

# Compute singular values
U, s, Vt = np.linalg.svd(G, full_matrices=False)
# s - сингулярные значения

# Удаляем последнее значение
s = s[:-1]  # обрезаем последний элемент

# Строим график
plt.semilogy(s, 'o-')
plt.xlabel('j')
plt.ylabel('$s_j$')
plt.title('Singular values')
plt.grid(True)
plt.show()

from scipy.ndimage import gaussian_filter1d

import numpy as np
import matplotlib.pyplot as plt

# Параметры
A = 25  # Амплитуда в pT
tau = 0.1  # Временная постоянная в секундах
t = np.logspace(-3, 0, 500)  # Логарифмически распределенные значения времени от 0.001 до 1 с

# Без шума
B = A * np.exp(-t / tau)
dB_dt = -(A / tau) * np.exp(-t / tau)

# С шумом
np.random.seed(0)  # для воспроизводимости
noise_B = np.random.normal(0, 3, size=t.shape)  # Шум 3 pT
noise_dB_dt = np.random.normal(0, 0.3, size=t.shape)  # Шум 0.3 nT/s

B_noisy = B + noise_B
dB_dt_noisy = dB_dt + noise_dB_dt

# Создадим два отдельных графика: один для B(t), второй для dB/dt(t)

fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# График для B(t)
axs[0].loglog(t, B, label='B(t) без шума (pT)', color='blue')
axs[0].loglog(t, B_noisy, label='B(t) с шумом (pT)', linestyle='--', color='cyan')
axs[0].set_ylabel('B (pT)')
axs[0].set_title('Экспоненциальный спад B(t)')
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[0].legend()

# График для dB/dt(t)
axs[1].loglog(t, dB_dt, label='dB/dt(t) без шума (nT/s)', color='red')
axs[1].loglog(t, dB_dt_noisy, label='dB/dt(t) с шумом (nT/s)', linestyle='--', color='orange')
axs[1].set_xlabel('Время (с)')
axs[1].set_ylabel('dB/dt (nT/s)')
axs[1].set_title('Экспоненциальный спад dB/dt(t)')
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1].legend()

plt.tight_layout()
plt.show()
