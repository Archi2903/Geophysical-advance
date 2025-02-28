import numpy as np

# Матрица A
A = np.array([[1, 2], [3, 4], [5, 6]])

# QR-факторизация
Q, R = np.linalg.qr(A)

print("Q:")
print(Q)
print("R:")
print(R)
