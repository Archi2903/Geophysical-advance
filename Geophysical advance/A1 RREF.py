#LINEAR ALGEBRA

"A2 Reducing a matrix to row-echelon form RREF"
import sympy as sp # SymPy is a Python library for symbolic mathematics
import numpy as np 

A = np.array([
    [1, 3, 3],
    [1, 8 ,2],
    [1, 3 ,4]
], dtype=float)

A = sp.Matrix(A) # Convert the array to a sympy matrix
B = np.array([14, 11, 19], dtype=float)

rref_matrix, _ = A.rref() # rref - метод, который приводит матрицу к ступенчатому виду
# rref_matrix - матрица в ступенчатом виде (RREF), pivot_columns - индексы ведущих столбцов (пивоты)

# Выводим результат
print("Редуцированная строковая ступенчатая форма (RREF):")
print(rref_matrix)
