import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Создание матрицы G для примера с томографией (3x3 сетка, 8 измерений)
G = np.array([
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [2, 0, 0, 0, 2, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2]
])

# Вычисление SVD
U, s, Vt = np.linalg.svd(G, full_matrices=False)
V = Vt.T

# Анализ сингулярных значений
plt.figure(figsize=(10, 4))
plt.semilogy(s, 'ko-')
plt.title('Сингулярные значения матрицы G')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.grid(True)
plt.show()

# Определение ранга (порог для малых сингулярных значений)
rank = np.sum(s > 1e-10)
print(f"Ранг матрицы G: {rank}")

# Построение обобщенной обратной матрицы G†
S_inv = np.diag(1.0 / s[:rank])
G_pinv = V[:, :rank] @ S_inv @ U[:, :rank].T

# Ковариационная матрица решения (σ=1)
Cov_m = G_pinv @ G_pinv.T

# Матрица разрешения модели
Rm = G_pinv @ G

# Визуализация диагонали матрицы разрешения
diag_Rm = np.diag(Rm).reshape(3, 3)

plt.figure(figsize=(8, 6))
plt.imshow(diag_Rm, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Разрешение')
plt.title('Диагональные элементы матрицы разрешения модели')
plt.xticks(np.arange(3))
plt.yticks(np.arange(3))
plt.show()

# Тестовая модель: спайк в центральном блоке (s22)
m_test = np.zeros(9)
m_test[4] = 1  # s22
d_test = G @ m_test

# Восстановление модели
m_rec = G_pinv @ d_test

# Визуализация тестовой и восстановленной модели
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(m_test.reshape(3, 3), cmap='viridis', vmin=0, vmax=1)
axes[0].set_title('Истинная модель (s22=1)')
axes[1].imshow(m_rec.reshape(3, 3), cmap='viridis', vmin=0, vmax=1)
axes[1].set_title('Восстановленная модель')
plt.show()

# Анализ устойчивости: добавление шума к данным
np.random.seed(42)
noise_level = 0.1
d_noisy = d_test + noise_level * np.random.randn(len(d_test))
m_rec_noisy = G_pinv @ d_noisy

# Визуализация решения с шумом
plt.figure(figsize=(8, 6))
plt.imshow(m_rec_noisy.reshape(3, 3), cmap='viridis')
plt.colorbar(label='Скорость')
plt.title('Решение с добавлением шума (σ=0.1)')
plt.show()

# Усеченное SVD (TSVD) для стабилизации
trunc_rank = 5  # Эмпирический выбор
S_inv_trunc = np.diag(1.0 / s[:trunc_rank])
G_pinv_trunc = V[:, :trunc_rank] @ S_inv_trunc @ U[:, :trunc_rank].T
m_rec_trunc = G_pinv_trunc @ d_noisy

# Визуализация TSVD решения
plt.figure(figsize=(8, 6))
plt.imshow(m_rec_trunc.reshape(3, 3), cmap='viridis')
plt.colorbar(label='Скорость')
plt.title('TSVD решение (p=5)')
plt.show()