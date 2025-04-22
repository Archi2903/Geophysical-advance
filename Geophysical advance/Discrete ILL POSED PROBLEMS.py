

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




# Параметры сигнала
T0 = 10.0  # Постоянная времени
t = np.linspace(0, 100, 500)  # Временная ось с более мелким шагом для гладкости

# Отклик на ступенчатую функцию (Example 3.2)
v = 1 - np.exp(-t / T0)  # Скорость стремится к 1 м/с
# Истинная модель: импульсы на 10 и 30 секундах
m_true = np.exp(-0.5*((t-10)/1.5)**2) + 0.5*np.exp(-0.5*((t-30)/2.5)**2)

# Сглаживание истинной модели
d_true = gaussian_filter1d(m_true, sigma=15)

# Добавление шума с дисперсией (0.05V)^2 (std = 0.05V)
V = np.max(d_true)
noise_std = (0.05 * V) # Стандартное отклонение
noise = np.random.normal(0, noise_std, size=d_true.shape)
d_noisy = d_true + noise

# Построение графиков
plt.figure(figsize=(10, 6))

# Рис. 3.11: Истинная модель
plt.subplot(2, 1, 1)
plt.plot(t, m_true)
plt.title("Figure 3.11: The true model")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.xlim(0, 100)
plt.ylim(0, 1.2)

# Рис. 3.12: Зашумленные данные
plt.subplot(2, 1, 2)
plt.plot(t, d_noisy, color='darkred')
plt.title("Figure 3.12: Noisy data ($N(0, (0.05V)^2)$)")  # Исправлена подпись
plt.xlabel("Time (s)")
plt.ylabel("V")
plt.xlim(0, 100)
plt.ylim(0, 1.2)

plt.tight_layout()
plt.show()