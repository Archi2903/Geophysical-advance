import numpy as np

# Определяем матрицу
A = np.array([[3, 2, 5], [1, 4, 7], [2, 6, 8]])

# 1. Через формулу для 3x3 (через определитель и алгебраические дополнения)
DetA = np.linalg.det(A) # Определитель матрицы
A_inv_formula = np.linalg.inv(A) if DetA != 0 else "Matrix is singular" # Если определитель не равен 0, то находим обратную матрицу
print("Inverse using formula:")
print(A_inv_formula)

# 2. Метод Гаусса-Жордана (приведение к единичной матрице)
A_ext = np.hstack((A, np.eye(3)))  # Расширенная матрица
for i in range(3):
    A_ext[i] = A_ext[i] / A_ext[i, i]  # Делаем ведущий элемент 1
    for j in range(3):
        if i != j:
            A_ext[j] -= A_ext[i] * A_ext[j, i]  # Обнуляем остальные элементы столбца
A_inv_gauss = A_ext[:, 3:]
print("Inverse using Gauss-Jordan:")
print(A_inv_gauss)

# 3. С использованием numpy
A_inv_numpy = np.linalg.inv(A) if DetA != 0 else "Matrix is singular"
print("Inverse using numpy:")
print(A_inv_numpy)
