""" 
Rank Deficiency and Ill–Conditioning
задача томографии с недостаточным рангом. 
Нам нужно применить сингулярное разложение (SVD) для анализа и решения проблемы

Шаг 1: Понимание структуры задачи
Матрица G имеет размер 8x9, но её ранг равен 7, что меньше n=9 → задача рангово дефицитна.
Это означает, что существует ненулевое нулевое пространство (модельное пространство), 
и решение не будет уникальным!!!

Шаг 2: Сингулярное разложение (SVD) матрицы G
- SVD матрицы G: G = U · S · V^T.
- Сингулярные значения: s1 ≥ s2 ≥ ... ≥ s8. Последнее сингулярное значение s8 ≈ 0 (из-за ошибок округления).
- Эффективный ранг p=7.

Шаг 3: Обобщённая обратная матрица G†
- Формула: G† = V_p · S_p^{-1} · U_p^T.
- Используем только первые p=7 сингулярных значений.

Шаг 4: Решение m+ = G†·d
- Пример данных d (время прохождения лучей): d = [t1, t2, ..., t8].
- Решение m+ будет вектором из 9 элементов, но с неопределённостью в нулевом пространстве.

Шаг 5: Анализ разрешающей способности
- Матрица разрешения модели: Rm = V_p · V_p^T.
- Диагональные элементы Rm показывают, насколько хорошо разрешены параметры.
- В примере диагональный элемент для s33 (m9) равен 1 → идеальное разрешение, остальные <1 → размытие.

Шаг 6: Ковариация решения
- Cov(m+) = σ² · V_p · S_p^{-2} · V_p^T.
- Параметры с малыми сингулярными значениями будут иметь большую неопределённость.

Пример визуализации нулевого пространства:
- Нулевое пространство N(G) определяется последними 2 столбцами V (V[:, 7] и V[:, 8]).
- При добавлении любой линейной комбинации этих векторов к решению, данные не изменятся.
**Тест на разрешение (spike test):**

- Если m_test = [0, 0, ..., 1 (на 5-й позиции), ..., 0], то восстановленное решение m+ = Rm[:,4] (5-й столбец Rm).
- Визуализация показывает размытие информации в соседние ячейки.

Выводы:
1. Решение m+ не уникально из-за нулевого пространства.
2. Разрешающая способность ограничена (кроме s33).
3. Стабильность решения зависит от усечения малых сингулярных значений.
4. Нулевое пространство приводит к неопределённости в модели, которая не влияет на данные.
Иллюстрация с Python:

- Построение матрицы G на основе лучевых путей.

- Визуализация сингулярных значений, матрицы разрешения и ковариации.

- Пример восстановления тестовой модели (spike test).
"""
"____________________________________________________________________________________________________"

"Шаг 1: Построение матрицы прямого оператора G"
""" 
m = 9(s11,s12,s13,s21,s22,s23,s31,s32,s33) params model
d = 8(t1,t2,t3,t4,t5,t6,t7,t8)             times data
G = np.zeros((8, 9))
"""
import numpy as np

G = np.array([
    [1,0,0,1,0,0,1,0,0],
    [0,1,0,0,1,0,0,1,0],
    [0,0,1,0,0,1,0,0,1],
    [1,1,1,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,1,1,1],
    [np.sqrt(2),0,0,0,np.sqrt(2),0,0,0,np.sqrt(2)],
    [0,0,0,0,0,0,0,0,np.sqrt(2)],
   # [0,0,0,0,0,0,0,0,0]
])
# print("Matrix G:")
# print(G)
""" [1 0 0 1 0 0 1 0 0]
    [0 1 0 0 1 0 0 1 0]
    [0 0 1 0 0 1 0 0 1]
    [1 1 1 0 0 0 0 0 0]
    [0 0 0 1 1 1 0 0 0]
    [0 0 0 0 0 0 1 1 1]
    [1.41 0 0 0 1.41 0 0 0 1.41]
    [0 0 0 0 0 0 0 0 1.41]]
    [0 0 0 0 0 0 0 0 0]
    """
"____________________________________________________________________________________________________"

"Шаг 2: Сингулярное разложение (SVD) матрицы G"
"eight singular values of G are, numerically evaluated"
U, S, Vt = np.linalg.svd(G, full_matrices=False)
p = np.sum(S > 1e-10)  # Ранг p = 7 (s8 ≈ 0) m=n=p

print("Singular values of G:")
for i, s_val in enumerate(S, 1):
 print(f"s{i}: {s_val:.3f}")

""" 
Singular values of G:
s1: 3.180
s2: 2.000
s3: 1.732
s4: 1.732
s5: 1.732
s6: 1.607
s7: 0.554
s8: 0.000
s9: 0.000
"""
"____________________________________________________________________________________________________"
"Шаг 3: Null Space (нулевое пространство)"
# Столбцы V (transport Vt)
V = Vt.T  # Vt - Transport matrix V

V0 = V[:, -2:]  # извлекаем 8-й и 9-й столбцы (индексы 7 и 8)

print("\nModel Null Space V₀ (V₈, V₉):")
print("   v8\t\t   v9")
print(V0)

V0_book = np.array([
    [-0.0620, -0.4035],
    [-0.4035,  0.0620],
    [ 0.4655,  0.3415],
    [ 0.4035, -0.0620],
    [ 0.0620,  0.4035],
    [-0.4655, -0.3415],
    [-0.3415,  0.4655],
    [ 0.3415, -0.4655],
    [ 0.0000,  0.0000]
])
""" 
from real data           book version 
   v8       v9           v8       v9
-0.2177  0.1788        [-0.0620, -0.4035]
 0.5374  0.3670        [-0.4035,  0.0620]
-0.2374 -0.5458        [ 0.4655,  0.3415]
 0.5374 -0.3670        [ 0.4035, -0.0620]
-0.2177 -0.1788        [ 0.0620,  0.4035]
-0.2374  0.5458        [-0.4655, -0.3415]
-0.2374  0.1882        [-0.3415,  0.4655]
-0.2374 -0.1882        [ 0.3415, -0.4655]
 0.3197 -0.0000        [ 0.0000,  0.0000]
 
 """
"____________________________________________________________________________________________________"
# Step 5: Reshape vectors into 3x3 blocks
v8_real = V0[:, 0].reshape(3, 3)
v9_real = V0[:, 1].reshape(3, 3)

v8_book = V0_book[:, 0].reshape(3, 3)
v9_book = V0_book[:, 1].reshape(3, 3)

# Step 6: Display reshaped 3x3 matrices
# For real data
print("\nReshaped real V8:")
print(np.round(v8_real, 4))

print("\nReshaped real V9:")
print(np.round(v9_real, 4))
# for book data
print("\nReshaped book V8 (eq. 3.95):")
print(v8_book)

print("\nReshaped book V9 (eq. 3.96):")
print(v9_book)
""" Reshaped from data V8:        Reshaped from data V9
[-0.2177   0.5374  -0.2374]   [ 0.1788   0.367  -0.5458]    
[ 0.5374  -0.2177  -0.2374]   [-0.367   -0.1788   0.5458]     
[-0.2374  -0.2374   0.3197]   [ 0.1882  -0.1882  -0.0000]      
    Reshaped from book V8:        Reshaped from book V9:
[-0.062  -0.4035   0.4655]    [-0.4035   0.062    0.3415]
[ 0.4035   0.062   -0.4655]   [-0.062    0.4035  -0.3415]
[-0.3415   0.3415   0.    ]   [ 0.4655  -0.4655   0.    ]
"""
"____________________________________________________________________________________________________"
"Plotting Reshaped Vectors V8 and V9"
import matplotlib.pyplot as plt
# Plotting the heatmaps
# Построение графиков
fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

# V8
im1 = axs[0].imshow(v8_book, cmap='gray', vmin=-0.5, vmax=0.5)
axs[0].set_title("V₈ (book)")
axs[0].set_xlabel("j")
axs[0].set_ylabel("i")
axs[0].set_xticks([0, 1, 2])
axs[0].set_xticklabels(['1', '2', '3'])
axs[0].set_yticks([0, 1, 2])
axs[0].set_yticklabels(['1', '2', '3'])

# V9
im2 = axs[1].imshow(v9_book, cmap='gray', vmin=-0.5, vmax=0.5)
axs[1].set_title("V₉ (book)")
axs[1].set_xlabel("j")
axs[1].set_ylabel("i")
axs[1].set_xticks([0, 1, 2])
axs[1].set_xticklabels(['1', '2', '3'])
axs[1].set_yticks([0, 1, 2])
axs[1].set_yticklabels(['1', '2', '3'])

# Добавляем отдельный colorbar справа
cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), orientation='vertical', shrink=0.8, pad=0.02)
cbar.set_label("V0", rotation=270, labelpad=15)
cbar.set_ticks([-0.5, 0, 0.5])

plt.show()
"Result of plotting  Image of the null space model V8 and V9"

U8 = -U[:, 7] # 8-й столбец U (индекс 7)

# Вывод U₈
print("U₈ (null data U):")
for i, val in enumerate(U8, 1):
    print(f"u{i}: {val:.3f}")
    """ 
U₈ (null data U):
u1: -0.408
u2: -0.408
u3: -0.408
u4: 0.408
u5: 0.408
u6: 0.408
u7: -0.000
u8: 0.000
"""
"____________________________________________________________________________________________________"
"Plotting Rm"
# Строим Rm = V[:, :p] @ V[:, :p].T
Vp = V[:, :p]
Rm = Vp @ Vp.T

# Vizualize the model resolution matrix Rm
plt.figure(figsize=(6, 5))
im = plt.imshow(Rm, cmap='gray', vmin=0, vmax=1)
plt.title("Model Resolution Matrix $\\mathbf{R}_m$")
plt.xlabel("j")
plt.ylabel("i")
plt.xticks(ticks=np.arange(9), labels=[str(i+1) for i in range(9)])
plt.yticks(ticks=np.arange(9), labels=[str(i+1) for i in range(9)])
plt.colorbar(im, label="$R_{m_{ij}}$")
plt.grid(False)
plt.tight_layout()
plt.show()

"diag Rm (формула 3.98)"
diag_Rm = np.diag(Rm)                # 9 элементов на диагонали
reshaped_diag = diag_Rm.reshape(3, 3).T  # транспонируем как в формуле 3.98

"# Reshaped diag(Rm) (формула 3.98):"
# [[1.000 0.000 0.000]
#  [0.000 0.000 0.000]
"Plotting diag(Rm)"
plt.figure(figsize=(5, 4))
im = plt.imshow(reshaped_diag, cmap='gray', vmin=0, vmax=1)
plt.title("Diagonal of $\\mathbf{R}_m$ in geometric layout")
plt.xlabel("j")
plt.ylabel("i")
plt.xticks([0, 1, 2], ['1', '2', '3'])
plt.yticks([0, 1, 2], ['1', '2', '3'])
plt.colorbar(im, label="diag($\\mathbf{R}_m$)")
plt.tight_layout()
plt.show()
"____________________________________________________________________________________________________"

print("Reshaped diag(Rm) (формула 3.98):")
print(np.round(reshaped_diag, 3))
""" 
Reshaped diag(Rm) (формула 3.98):
 [0.833 0.833 0.667]
 [0.833 0.833 0.667]
 [0.667 0.667 1.000]
"""
"____________________________________________________________________________________________________"
# ─── ШАГ 4: Извлекаем 5‑й столбец Rm — это обобщённо‑обратная реакция на spike ─
col5 = Rm[:, 4]       # индекс 4 ← пятый элемент
col5_column = col5.reshape(-1, 1)
print("dtest = Gm")
print(np.round(col5_column, 3))
""" 
 [ 0.167]
 [-0.   ]
 [-0.167]
 [ 0.   ]
 [ 0.833]
 [ 0.167]
 [-0.167]
 [ 0.167]
 [ 0.   ]
 """

# ─── ШАГ 5: Reshape в 3×3 и транспонируем, чтобы получить формулу (3.100) ────
m_plus = col5.reshape(3, 3).T
print("\nReshaped 5-й столбец Rm (формула 3.100):")
print(np.round(m_plus, 3))
# ─── РЕЗУЛЬТАТ ───────────────────────────────────────────────────────────────
print("Восстановленная модель (формула 3.100):")
print(np.round(m_plus, 3))
""" 
Reshaped 5-й столбец Rm (формула 3.100):
 [ 0.167  0.    -0.167]
 [-0.     0.833  0.167]
 [-0.167  0.167  0.   ]
 """
 
 
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 4))
im = plt.imshow(m_plus, cmap='gray', vmin=0, vmax=1)
plt.title("Spike Test Model (Figure 3.8)")
plt.xlabel("j")
plt.ylabel("i")
plt.xticks([0, 1, 2], ['1', '2', '3'])
plt.yticks([0, 1, 2], ['1', '2', '3'])
plt.colorbar(im, label="$\\mathbf{m}^+$")
plt.tight_layout()
plt.show()