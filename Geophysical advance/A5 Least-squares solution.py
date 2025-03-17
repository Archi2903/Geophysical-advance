import numpy as np
import matplotlib.pyplot as plt

# Данные
x = np.array([1, 2, 3, 4])  # Значения x
y = np.array([1.2, 1.9, 3.1, 4.0])  # Значения y

# Строим матрицу A (добавляем столбец единичных значений для b)
A = np.vstack([np.ones_like(x), x]).T

# Решаем систему наименьших квадратов: (A^T A) x = A^T b
A_T_A = np.dot(A.T, A)  # A^T A
A_T_b = np.dot(A.T, y)  # A^T b

# Находим параметры b и m
params = np.linalg.solve(A_T_A, A_T_b)

# Параметры прямой
b, m = params

print(f"Найденные параметры: b = {b}, m = {m}")


# Исходные данные
plt.scatter(x, y, color='blue', label='Данные')

# Построение прямой регрессии
y_pred = b + m * x
plt.plot(x, y_pred, color='red', label='Прямая регрессии')

# Оформление графика
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()