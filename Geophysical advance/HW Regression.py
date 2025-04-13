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
# ml2 ист знач попадают в интервалы, а занчит оценка согласуется несмотря на шум

# наблюдается смещение 
# Ковариационная матрица отражает неопределённость оценок из-за шума в данных.
# Результат близок к истинным значениям, но с отклонениями из-за ошибок измерений.
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

# Создаем 3 отдельных графикa
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Для каждой проекции
for i, (pair, ax) in enumerate(zip([(0,1), (0,2), (1,2)], axs)):
    # Выбираем индексы параметров для проекции
    idx1, idx2 = pair
    
    # Выделяем подматрицу ковариации для выбранных параметров
    Cov_sub = Cov[[[idx1, idx1], [idx2, idx2]], [[idx1, idx2], [idx1, idx2]]]
    
    # Создаем эллипс для проекции
    L = np.linalg.cholesky(Cov_sub)  # Разложение Холецкого
    ellipse = (L @ (circle * np.sqrt(chi2_crit)) + mL2[[idx1, idx2]].reshape(2,1))
    
    # Рисуем эллипс
    ax.plot(ellipse[0], ellipse[1], color='blue', alpha=0.5)
    ax.scatter(mL2[idx1], mL2[idx2], color='red', s=50)
    
    # Подписи осей
    labels = ['m1 (м)', 'm2 (м/с)', 'm3 (м/с²)']
    ax.set_xlabel(labels[idx1])
    ax.set_ylabel(labels[idx2])
    
    # Названия графиков
    titles = [' (m1, m2)', ' (m1, m3)', ' (m2, m3)']
    ax.set_title(titles[i])
    
    # Сетка и оформление
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()