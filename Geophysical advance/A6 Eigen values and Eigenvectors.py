import numpy as np

# Матрица A
A = np.array([[4, 1],
              [2, 3]])

# Нахождение собственных значений и собственных векторов
eigenvalues, eigenvectors = np.linalg.eig(A)

# Вывод собственных значений
print("Собственные значения:", eigenvalues)

# Вывод собственных векторов
print("Собственные векторы:")
print(eigenvectors)
