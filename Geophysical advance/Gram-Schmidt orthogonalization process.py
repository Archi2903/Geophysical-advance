import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Преобразуем векторы в тип float для корректных вычислений
V = np.array([[1, 2, 2], [1, 3, 3], [1, 1, 2]], dtype=float)

# Функция для применения процесса Грама-Шмидта
def gram_schmidt(V):
    n = len(V)
    W = np.zeros_like(V)  # Массив для ортогональных векторов
    W[0] = V[0]  # Первый вектор остается без изменений
    for i in range(1, n):
        W[i] = V[i]
        for j in range(i):
            # Вычитаем проекцию вектора на уже ортогональные векторы
            W[i] -= np.dot(V[i], W[j]) / np.dot(W[j], W[j]) * W[j]
    return W

# Получаем ортогональные векторы
W = gram_schmidt(V)

# Визуализация в 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Отображаем исходные векторы
ax.quiver(0, 0, 0, V[0, 0], V[0, 1], V[0, 2], color="r", label="v1")
ax.quiver(0, 0, 0, V[1, 0], V[1, 1], V[1, 2], color="g", label="v2")
ax.quiver(0, 0, 0, V[2, 0], V[2, 1], V[2, 2], color="b", label="v3")

# Отображаем ортогональные векторы
ax.quiver(0, 0, 0, W[0, 0], W[0, 1], W[0, 2], color="purple", label="w1")
ax.quiver(0, 0, 0, W[1, 0], W[1, 1], W[1, 2], color="black", label="w2")
ax.quiver(0, 0, 0, W[2, 0], W[2, 1], W[2, 2], color="yellow", label="w3")

# Настройка графика
ax.set_xlim([0, 4])
ax.set_ylim([0, 6])
ax.set_zlim([0, 2])

# Подписи и легенда
ax.legend()

# Показываем сетку
ax.grid(True)

plt.show()
