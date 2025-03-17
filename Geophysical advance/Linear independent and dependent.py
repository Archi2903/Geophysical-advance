import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Линейно независимые векторы (2D)
vectors_independent_2d = np.array([[1, 0], [0, 1]])

# Линейно зависимые векторы (2D) - один вектор является кратным другого
vectors_dependent_2d = np.array([[1, 0], [2, 0]])

# Линейно независимые векторы (3D)
vectors_independent_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Линейно зависимые векторы (3D) - один из векторов является линейной комбинацией двух других
vectors_dependent_3d = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])

# Функция для рисования 2D векторов
def plot_2d_vectors(ax, vectors, title, colors=['r', 'b']):
    origin = np.array([[0, 0], [0, 0]])
    ax.quiver(*origin, vectors[:, 0], vectors[:, 1], angles='xy', scale_units='xy', scale=1, color=colors)
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_title(title)

# Функция для рисования 3D векторов
def plot_3d_vectors(ax, vectors, title, colors=['r', 'g', 'b']):
    origin = np.zeros((3, 3))
    ax.quiver(*origin, vectors[:, 0], vectors[:, 1], vectors[:, 2], color=colors)
    ax.set_xlim([0, 1.5])
    ax.set_ylim([0, 1.5])
    ax.set_zlim([0, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

# Создание графиков
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 2D графики
plot_2d_vectors(axes[0], vectors_independent_2d, "Линейно независимые векторы (2D)solution C=0- одно решение имеет")
plot_2d_vectors(axes[1], vectors_dependent_2d, "Линейно зависимые векторы (2D) имеет много решений ", colors=['r', 'r'])

plt.show()

# 3D графики
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
plot_3d_vectors(ax1, vectors_independent_3d, "Линейно независимые векторы (3D) solution C=0- одно решение имеет ")

ax2 = fig.add_subplot(122, projection='3d')
plot_3d_vectors(ax2, vectors_dependent_3d, "Линейно зависимые векторы (3D)", colors=['r', 'g', 'r'])

plt.show()
