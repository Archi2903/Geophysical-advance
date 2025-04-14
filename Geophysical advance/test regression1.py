import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

# Данные
# Оценка параметров:
mL2 = np.array([16.42, 96.97, 9.41])

# Исправленная матрица ковариации (с положительными корреляциями)
Cov = np.array([
    [88.53, 33.60, 5.33],
    [33.60, 15.44, 2.67],
    [5.33, 2.67, 0.48]
])

# Вычисление обратной матрицы ковариации
Cinv = np.linalg.inv(Cov)

# Вычисление собственных значений и векторов обратной матрицы
eigenvals, eigenvecs = np.linalg.eigh(Cinv)
# Сортировка по убыванию
eigenvals = eigenvals[::-1]
eigenvecs = eigenvecs[:, ::-1]

# Критическое значение chi2 для 95% доверительного уровня и 3 степеней свободы
chi2_crit = chi2.ppf(0.95, 3)  # примерно 7.81

# Длины полуосей эллипсоида (уравнение 2.49)
semi_axes = np.sqrt(chi2_crit / eigenvals)

# -------------------- Генерация точек эллипсоида --------------------
# Параметризация единичной сферы
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
sphere_points = np.vstack([x.flatten(), y.flatten(), z.flatten()])

# Преобразование единичной сферы в эллипсоид:
scaled_points = sphere_points * semi_axes[:, np.newaxis]
rotated_points = eigenvecs @ scaled_points
ellipsoid_points = rotated_points + mL2[:, np.newaxis]

# -------------------- Вычисление bounding box --------------------
# Определяем максимальное отклонение от центра для каждого параметра
delta = np.max(np.abs(ellipsoid_points - mL2[:, np.newaxis]), axis=1)
# Интервалы для каждого параметра
intervals = np.vstack([mL2 - delta, mL2 + delta]).T

# Вывод интервалов для сравнения с уравнением 2.50
print("Вычисленные доверительные интервалы (mL2 ± delta):")
print("m1: {:.2f} ± {:.2f}  => [{:.2f}, {:.2f}]".format(mL2[0], delta[0],
      mL2[0]-delta[0], mL2[0]+delta[0]))
print("m2: {:.2f} ± {:.2f}  => [{:.2f}, {:.2f}]".format(mL2[1], delta[1],
      mL2[1]-delta[1], mL2[1]+delta[1]))
print("m3: {:.2f} ± {:.2f}  => [{:.2f}, {:.2f}]".format(mL2[2], delta[2],
      mL2[2]-delta[2], mL2[2]+delta[2]))

# -------------------- 3D-визуализация --------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Отрисовка эллипсоида (точки)
ax.scatter(ellipsoid_points[0], ellipsoid_points[1], ellipsoid_points[2], 
           c='blue', alpha=0.1, s=1, label='Confidence Ellipsoid')

# Построение bounding box
# Получаем 8 углов прямоугольника через произведение всех комбинаций min/max
corners = np.array(list(product(
    [intervals[0, 0], intervals[0, 1]],
    [intervals[1, 0], intervals[1, 1]],
    [intervals[2, 0], intervals[2, 1]]
)))
# Определяем рёбра bounding box по индексам вершин
edges = [
    [0,1], [0,2], [0,4],
    [1,3], [1,5],
    [2,3], [2,6],
    [3,7],
    [4,5], [4,6],
    [5,7],
    [6,7]
]
for edge in edges:
    ax.plot(*zip(*corners[edge]), color='red', linestyle='--', linewidth=1)

# Подписи осей и заголовок
ax.set_xlabel('m₁ (м)', fontsize=12)
ax.set_ylabel('m₂ (м/с)', fontsize=12)
ax.set_zlabel('m₃ (м/с²)', fontsize=12)
ax.set_title('3D Confidence Ellipsoid with Bounding Box', fontsize=14)
ax.view_init(elev=25, azim=45)
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()