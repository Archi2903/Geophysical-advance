import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D

"Стандартные ошибки параметров"
# Covariace matrix
#Cov(θ) = σ^2 * ((G^T)*G)^-1 evequation
Cov = np.array([
    [88.53, -33.60, -5.33],
    [-33.60, 15.44, 2.67],
    [-5.33, 2.67, 0.48]
])

# mtrue = [10m, 100m/s, 9.8m/s^2]
# σ=8м - noise
# mL2 = (((G^T)*G)^-1)(G^T)*d
# Оценённые параметры

#95% ловерительные интервалы
# m+-1.96*
mL2 = np.array([16.42, 96.97, 9.41])
# m1=16.4m-+ 18.4m ->[-2.0, 34.8]
# m2=97.0m/s-+ 7.7m/s ->[89.3, 104.7]
# m3=9.4m/s^2-+ 1.4m/s^2 ->[8.0, 10.8]

"Interpretation:mtrue values fall within the Confidence Intervals, Estimate is consistent despite the noise!"
# Result mtrue and mL2 estimate a little bit diffrent because σ=8м - noise

# Критическое значение χ² для 3 степеней свободы и 95% доверия
chi2_crit = chi2.ppf(0.95, 3)  # ≈7.815
"""
Проверка Критическое значение χ²
# Исходные данные
t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_observed = np.array([109.4, 187.5, 267.5, 331.9, 386.1, 428.4, 452.2, 498.1, 512.3, 513.0])
sigma = 8

# Параметры модели
m1, m2, m3 = 16.42, 96.97, 9.41

# Прогнозы модели
y_predicted = m1 + m2*t - 0.5*m3*t**2

# 1. Вычисление статистики χ²
chi_squared = np.sum(((y_observed - y_predicted)/sigma)**2)
print(f"χ² статистика: {chi_squared:.2f}")

# 2. Расчет p-значения (исправленная строка)
degrees_of_freedom = len(t) - 3
p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)  # Используем chi2 из scipy.stats
print(f"p-значение: {p_value:.2f}")

# 3. Визуализация
import matplotlib.pyplot as plt
x = np.linspace(0, 20, 500)
plt.plot(x, chi2.pdf(x, degrees_of_freedom), label=f'χ² (df={degrees_of_freedom})')
plt.axvline(chi_squared, color='red', linestyle='--', label=f'χ² = {chi_squared:.2f}')
plt.fill_between(x[x>=chi_squared], chi2.pdf(x[x>=chi_squared], degrees_of_freedom), 
                 color='red', alpha=0.2, label=f'p = {p_value:.2f}')
plt.legend()
plt.show()
"""


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
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Create 3 graphs
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# For every projections
for i, (pair, ax) in enumerate(zip([(0,1), (0,2), (1,2)], axs)):
    # Index parametrs for every projections
    idx1, idx2 = pair
    
    # Выделяем подматрицу ковариации для выбранных параметров
    Cov_sub = Cov[[[idx1, idx1], [idx2, idx2]], [[idx1, idx2], [idx1, idx2]]]
    
    # Create ellips for projections
    L = np.linalg.cholesky(Cov_sub)  # Разложение Холецкого
    ellipse = (L @ (circle * np.sqrt(chi2_crit)) + mL2[[idx1, idx2]].reshape(2,1))
    
    # ellips
    ax.plot(ellipse[0], ellipse[1], color='blue', alpha=0.5)
    ax.scatter(mL2[idx1], mL2[idx2], color='red', s=50)
    
    # LAbes
    labels = ['m1 (m)', 'm2 (m/s)', 'm3 (m/s²)']
    ax.set_xlabel(labels[idx1])
    ax.set_ylabel(labels[idx2])
    
    # Названия графиков
    titles = [' (m1, m2)', ' (m1, m3)', ' (m2, m3)']
    ax.set_title(titles[i])
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
""" ------------------------- 3D vizual -------------------------"""
from itertools import product

# paramets for sphere
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
sphere_points = np.vstack([x.flatten(), y.flatten(), z.flatten()])

# Transform sphere to ellipsoid
scaled_points = sphere_points * semi_axes[:, np.newaxis] # scale
# Rotate points using eigenvectors
rotated_points = eigenvecs @ scaled_points
ellipsoid_points = rotated_points + mL2[:, np.newaxis]

""" bounding box """
deltas = np.ptp(ellipsoid_points, axis=1) / 2
intervals = np.vstack([mL2 - deltas, mL2 + deltas]).T

# Output of intervals for comparison with equation 2.50
print("Compute mL2 (mL2 ± delta):")
print("m1: {:.2f} ± {:.2f}  => [{:.2f}, {:.2f}]".format(mL2[0], deltas[0],
      mL2[0]-deltas[0], mL2[0]+deltas[0]))
print("m2: {:.2f} ± {:.2f}  => [{:.2f}, {:.2f}]".format(mL2[1], deltas[1],
      mL2[1]-deltas[1], mL2[1]+deltas[1]))
print("m3: {:.2f} ± {:.2f}  => [{:.2f}, {:.2f}]".format(mL2[2], deltas[2],
      mL2[2]-deltas[2], mL2[2]+deltas[2]))

# Plot 3d
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# ellipsoid
ax.scatter(ellipsoid_points[0], ellipsoid_points[1], ellipsoid_points[2], 
           c='blue', alpha=0.1, s=1, label='Confidence Ellipsoid')

# Углы bounding box
corners = np.array(list(product(
    [intervals[0, 0], intervals[0, 1]],
    [intervals[1, 0], intervals[1, 1]],
    [intervals[2, 0], intervals[2, 1]]
)))

# edge bounding box
for i, edge in enumerate([
    [0,1], [0,2], [0,4],
    [1,3], [1,5],
    [2,3], [2,6],
    [3,7],
    [4,5], [4,6],
    [5,7],
    [6,7]
]):
    if i == 0:
        ax.plot(*zip(*corners[edge]), color='red', linestyle='--', linewidth=1, label='Bounding Box')
    else:
        ax.plot(*zip(*corners[edge]), color='red', linestyle='--', linewidth=1)


# label
bbox_center = np.mean(corners, axis=0)
annotation_text = ("Compute mL2 (mL2 ± delta):\n" +
                   "m1: 16.42 ± {:.2f}  => [{:.2f}, {:.2f}]\n".format(deltas[0], mL2[0]-deltas[0], mL2[0]+deltas[0]) +
                   "m2: 96.97 ± {:.2f}  => [{:.2f}, {:.2f}]\n".format(deltas[1], mL2[1]-deltas[1], mL2[1]+deltas[1]) +
                   "m3: 9.41 ± {:.2f}  => [{:.2f}, {:.2f}]".format(deltas[2], mL2[2]-deltas[2], mL2[2]+deltas[2]))
ax.text(bbox_center[0], bbox_center[1], bbox_center[2], annotation_text, 
        fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.6))

# label and titels
ax.set_xlabel('m1 (м)', fontsize=12)
ax.set_ylabel('m2 (м/s)', fontsize=12)
ax.set_zlabel('m3 (м/s²)', fontsize=12)
ax.set_title('3D Confidence Ellipsoid with Bounding Box', fontsize=14)

ax.view_init(elev=25, azim=45)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()