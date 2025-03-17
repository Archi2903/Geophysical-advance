import numpy as np

def gauss_jordan_inverse(matrix):
    """
    Функция находит обратную матрицу методом Гаусса-Жордана.
    Вход: квадратная матрица (список списков или np.array).
    Выход: обратная матрица, если она существует.
    """
    n = len(matrix)  # Размерность матрицы
    A = np.array(matrix, dtype=float)  # Преобразуем в float (чтобы избежать ошибок с int)
    I = np.eye(n, dtype=float)  # Создаём единичную матрицу

    # 1. Прямой ход - приводим A к верхнетреугольному виду
    for col in range(n):
        # Выбираем главный элемент (по диагонали)
        pivot = A[col, col]
        if pivot == 0:
            raise ValueError("Матрица вырожденная, обратной не существует")
        
        # Нормализуем строку (делим на ведущий элемент)
        A[col] /= pivot
        I[col] /= pivot  # То же самое делаем с единичной матрицей

        # Зануляем все элементы ниже и выше ведущего
        for row in range(n):
            if row != col:
                factor = A[row, col]
                A[row] -= factor * A[col]
                I[row] -= factor * I[col]

    return I  # Возвращаем обратную матрицу

# Пример использования
A = [[21, 1, 2], 
     [3, 2, 4],
     [5, 6, 7]]

inverse_A = gauss_jordan_inverse(A)

# Выводим результат
print("Обратная матрица(Inverse matrix):")
print(np.array(inverse_A))
