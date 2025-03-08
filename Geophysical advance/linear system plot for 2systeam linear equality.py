import numpy as np
import matplotlib.pyplot as plt


"""a1x+b1y=c1
   a2x+b2y=c2"""
def plot_linear_system(a1, b1, c1, a2, b2, c2):
    x = np.linspace(-2, 2, 500)
    y1 = (c1 - a1 * x) / b1
    y2 = (c2 - a2 * x) / b2
    
    plt.plot(x, y1, label=f'{a1}x + {b1}y = {c1}')
    plt.plot(x, y2, label=f'{a2}x + {b2}y = {c2}')
    
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([c1, c2])
    
    if np.linalg.det(A) != 0:  # Проверка на существование решения
        solution = np.linalg.solve(A, B)
        plt.scatter(solution[0], solution[1], color='red', zorder=3, label=f'Solution: {solution}')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graphical Representation of Linear System')
    plt.show()

# Пример использования
plot_linear_system(2, -1, 0, -1, 2, 3)
