#LINEAR ALGEBRA

"A2 Reducing a matrix to row-echelon form RREF"
import sympy as sp # SymPy is a Python library for symbolic mathematics
import numpy as np 
#weeee
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8 ,9]
], dtype=float)

A = sp.Matrix(A) # Convert the array to a sympy matrix
B = np.array([0, 0, 0], dtype=float)

rref_matrix, _ = A.rref() # rref - метод, который приводит матрицу к ступенчатому виду
# rref_matrix - матрица в ступенчатом виде (RREF), pivot_columns - индексы ведущих столбцов (пивоты)

# Выводим результат
print("Редуцированная строковая ступенчатая форма (RREF):")
print(rref_matrix)
