import sympy as sp

# Определяем матрицу A
A = sp.Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Приводим матрицу к редуцированной ступенчатой форме
rref_matrix, pivot_columns = A.rref()

# Выводим результаты
print("Редуцированная ступенчатая форма (RREF):")
sp.pprint(rref_matrix)

# Проверяем количество ведущих (основных) столбцов
if len(pivot_columns) == A.shape[1]:
    print("Столбцы линейно независимы.")
else:
    print("Столбцы линейно зависимы.")
