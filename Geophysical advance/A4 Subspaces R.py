import numpy as np
from scipy.linalg import null_space

# Определяем матрицу A
A = np.array([
    [3, 1, 9, 4],
    [2, 1, 7, 3],
    [5, 2, 16, 7]
])

# Вычисляем нулевое пространство (null space)
N = null_space(A)

# Выводим результат
print("Null space of A:")
print(N)

## Ручной метод
import numpy as np

def gaussian_elimination(A):
    """ Приведение матрицы к ступенчатому виду методом Гаусса """
    A = A.astype(float)
    rows, cols = A.shape
    for i in range(min(rows, cols)):
        # Поиск максимального элемента в колонке
        max_row = max(range(i, rows), key=lambda r: abs(A[r, i]))
        A[[i, max_row]] = A[[max_row, i]]
        if A[i, i] == 0:
            continue
        # Приведение к 1 и обнуление элементов ниже
        A[i] /= A[i, i]
        for j in range(i + 1, rows):
            A[j] -= A[i] * A[j, i]
    return A

def back_substitution(A):
    """ Нахождение решений для нулевого пространства """
    rows, cols = A.shape
    free_vars = []
    for i in range(rows):
        if all(A[i, j] == 0 for j in range(cols)):
            free_vars.append(i)
    null_space = np.zeros((cols, len(free_vars)))
    for idx, var in enumerate(free_vars):
        null_space[var, idx] = 1
    return null_space

# Определяем матрицу A
A = np.array([
    [3, 1, 9, 4],
    [2, 1, 7, 3],
    [5, 2, 16, 7]
])

# Приводим к ступенчатому виду
A_reduced = gaussian_elimination(A)

# Вычисляем нулевое пространство вручную
N = back_substitution(A_reduced)

# Выводим результат
print("Null space of A:")
print(N)
