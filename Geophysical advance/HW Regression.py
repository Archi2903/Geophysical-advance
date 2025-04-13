import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D

# Заданная ковариационная матрица
Cov = np.array([
    [88.53, -33.60, -5.33],
    [-33.60, 15.44, 2.67],
    [-5.33, 2.67, 0.48]
])

# Оценённые параметры
mL2 = np.array([16.42, 96.97, 9.41])

# Критическое значение χ² для 3 степеней свободы и 95% доверия
chi2_crit = chi2.ppf(0.95, 3)  # ≈7.815

# Вычисление обратной матрицы ковариации
Cinv = np.linalg.inv(Cov)

# Собственные значения и векторы Cinv
eigenvals, eigenvecs = np.linalg.eigh(Cinv)
# Сортировка в порядке убывания собственных значений
eigenvals = eigenvals[::-1]
eigenvecs = eigenvecs[:, ::-1]

# Длины полуосей эллипсоида
semi_axes = np.sqrt(chi2_crit / eigenvals)

# Параметризация единичной сферы
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
sphere = np.stack((x.flatten(), y.flatten(), z.flatten()))

# Преобразование сферы в эллипсоид
ellipsoid = eigenvecs @ np.diag(semi_axes) @ sphere
X = ellipsoid[0, :].reshape(x.shape) + mL2[0]
Y = ellipsoid[1, :].reshape(y.shape) + mL2[1]
Z = ellipsoid[2, :].reshape(z.shape) + mL2[2]

# Построение графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.3, linewidth=0, antialiased=False)
# Отметка точки оценки
ax.scatter([mL2[0]], [mL2[1]], [mL2[2]], color='orange', s=50)
ax.set_xlabel('m1 (м)')
ax.set_ylabel('m2 (м/с)')
ax.set_zlabel('m3 (м/с²)')
ax.set_title('95% доверительный эллипсоид и оценка параметров')
plt.tight_layout()
plt.show()
