import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_linear_system_3d(a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)
    
    Z1 = (d1 - a1 * X - b1 * Y) / c1
    Z2 = (d2 - a2 * X - b2 * Y) / c2
    Z3 = (d3 - a3 * X - b3 * Y) / c3
    
    ax.plot_surface(X, Y, Z1, alpha=0.5, color='r', label=' 1 equation')
    ax.plot_surface(X, Y, Z2, alpha=0.5, color='g', label=' 2 equation')
    ax.plot_surface(X, Y, Z3, alpha=0.5, color='b', label=' 3 equation')
    
    A = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
    B = np.array([d1, d2, d3])
    
    if np.linalg.det(A) != 0:  # Проверка на существование решения
        solution = np.linalg.solve(A, B)
        ax.scatter(solution[0], solution[1], solution[2], color='black', s=100, label=f'Solution: {solution}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(' 3D Linear System')
    plt.legend()
    plt.show()

# Пример использования
plot_linear_system_3d(1, 2, 3, 14, 1, 2, 2, 11, 1, 3, 4, 19)
