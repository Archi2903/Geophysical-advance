import numpy as np
import matplotlib.pyplot as plt

# 1. Определяем параметры распределения
mu = 0      # Среднее значение (μ)
sigma = 0.5  # Стандартное отклонение (σ)
N = 10000    # Количество случайных точек

# 2. Генерируем Z ~ N(0,1)
Z = np.random.randn(N)

# 3. Вычисляем X = μ + σ * Z
X = mu + sigma * Z

# 4. Строим гистограмму распределения X
plt.figure(figsize=(8,5))
plt.hist(X, bins=50, density=True, alpha=0.6, color='b', edgecolor='black')
plt.xlabel('Значения X')
plt.ylabel('Плотность')
plt.title(f'Гистограмма нормального распределения X ~ N({mu}, {sigma**2})')
plt.grid(True)
plt.show()
