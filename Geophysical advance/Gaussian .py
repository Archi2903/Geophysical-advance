import numpy as np
import matplotlib.pyplot as plt



# Ввод размера матрицы
size = eval(input("matrix size: "))

# Инициализация матрицы нулями размером size         
O = np.zeros((size[0], size[1]))

# Заполнение матрицы значениями
for i in range(size[0]):
    O[i] = np.asarray(eval(input()))  # Пример: 2,-1,2,10  или 1,-2,1,8  и т.д.


n = len(O)

# Преобразование матрицы в верхнюю треугольную форму методом Гаусса с проверкой деления на ноль
for i in range(n - 1):
    if O[i, i] == 0:  # Если элемент на главной диагонали равен нулю
        # Переставляем строки
        for j in range(i + 1, n):
            if O[j, i] != 0:  # Ищем строку, где элемент не равен нулю
                O[[i, j]] = O[[j, i]]  # Меняем строки местами
                break
    # Преобразуем строку
    for k in range(i + 1, n):
        if O[i, i] != 0:  # Проверка на деление на ноль
            O[k] = O[k] - O[k, i] / O[i, i] * O[i]

# Решение системы уравнений
x = np.zeros((n, 1))

# Преобразование последней строки в вектор b и создаём матрицу A
A = O[:, :-1]  # Берём все строки и все столбцы, кроме последнего
b = O[:, -1]  # Последний столбец — это вектор b

# Прямой ход для получения решения
if A[n - 1, n - 1] != 0:  # Проверка на деление на ноль
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]
for i in range(n - 2, -1, -1):
    summ = 0
    for k in range(i + 1, n):
        summ += A[i, k] * x[k]
    if A[i, i] != 0:  # Проверка на деление на ноль
        x[i] = (b[i] - summ) / A[i, i]

print("Решение системы: x1, x2, x3\n", x) 