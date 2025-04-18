
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


import numpy as np
from sklearn.linear_model import LinearRegression

# 1. 데이터 정의
t = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([109.4, 187.5, 267.5, 331.9, 386.1, 428.4, 452.2, 498.1, 512.3, 513.0])

# 2. 디자인 행렬 구성: [1, t, -0.5 * t^2]
G = np.hstack([
    np.ones_like(t),  # c (절편)
    t,             # b 계수
    -0.5 * t**2   # a 계수와 곱해질 항
])

# 3. 선형 회귀 모델 (절편 직접 넣었으므로 fit_intercept=False)
model = LinearRegression(fit_intercept=False)
model.fit(G, y)
m_L2 = model.coef_

# 4. 알려진 오차 표준편차
sigma = 8
sigma_squared = sigma ** 2

# 5. 공분산 행렬 계산
GtG_inv = np.linalg.inv(G.T @ G)
cov_m_L2 = sigma_squared * GtG_inv

# 6. 결과 출력
print("추정된 회귀 계수 [a, b, c]:", m_L2)
print("\n오차 분산 σ² =", sigma_squared)
print("\n계수 공분산 행렬 (Cov[m_L2]):\n", cov_m_L2)
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from scipy.stats import norm

# z 값 (정규분포 기준 95% 신뢰수준)
z = norm.ppf(0.975)  # 약 1.96

# 신뢰구간 계산
conf_intervals = []
for i in range(len(m_L2)):
    se = np.sqrt(cov_m_L2[i, i])  # 표준 오차
    lower = m_L2[i] - z * se
    upper = m_L2[i] + z * se
    conf_intervals.append((lower, upper))

# 결과 출력
labels = ['m_1', 'm_2', 'm_3']
for i, (ci, est) in enumerate(zip(conf_intervals, m_L2)):
    print(f"{labels[i]} = {est:.4f},  95% 신뢰구간: ({ci[0]:.4f}, {ci[1]:.4f})")
from scipy.stats import chi2

# 예측 및 잔차
y_pred = model.predict(G)
residuals = y - y_pred

# 카이제곱 통계량 계산: RSS / sigma^2
sigma = 8
RSS = np.sum(residuals**2)
chi_squared = RSS / (sigma**2)

# 자유도
df = len(y) - G.shape[1]  # 10 - 3 = 7

# p-value 계산 (우측 누적 확률)
p_value = 1 - chi2.cdf(chi_squared, df=df)

# 출력
print(f"Chi-squared statistic: {chi_squared:.4f}")
print(f"Degrees of freedom: {df}")
print(f"p-value: {p_value:.4e}")

import pandas as pd

# 표준편차 벡터 (각 계수의 standard error)
std_errors = np.sqrt(np.diag(cov_m_L2))

# 상관계수 행렬 계산
corr_matrix = cov_m_L2 / (std_errors[:, None] * std_errors[None, :])
corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)

# 결과 출력
labels = ['m_1', 'm_2', 'm_3']
print("계수 상관계수 행렬:")
print(corr_df.round(4))  # 소수점 4자리까지 보기 좋게 출력
# 공분산 행렬의 역행렬
inv_cov = np.linalg.inv(cov_m_L2)

# 역행렬의 고유값 분해 (diagonalization)
eigvals, eigvecs = np.linalg.eig(inv_cov)

# 고유값 오름차순 정렬
idx = np.argsort(eigvals)
eigvals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:, idx]

# 신뢰수준 (95%)에 해당하는 카이제곱 분위값
p = 3  # 계수 개수
chi2_val = chi2.ppf(0.95, df=p)

# 95% 신뢰 타원체의 반축 길이 (semiaxis)
semiaxes = np.sqrt(chi2_val / eigvals_sorted)

# 출력
np.set_printoptions(precision=4, suppress=True)
print("고유값 (오름차순):", eigvals_sorted)
print("95% 신뢰수준 Chi² 값:", chi2_val)
print("신뢰 타원체의 반축 길이 (semiaxes):", semiaxes)
# 정렬된 고유값/고유벡터로 역행렬 복원
reconstructed_sorted = eigvecs_sorted @ np.diag(eigvals_sorted) @ np.linalg.inv(eigvecs_sorted)

# 출력 (예쁘게 보기 위해 옵션 설정)
np.set_printoptions(precision=4, suppress=True)
print("역행렬 (Cov⁻¹):\n", inv_cov)
print("정렬된 고유값 (Λ):\n", eigvals_sorted)
print("\n정렬된 고유벡터 행렬 (Q):\n", eigvecs_sorted)
print("\nQ Λ Q⁻¹ 복원 결과 (정렬된 고유값 기준):\n", reconstructed_sorted)
# 구면 생성
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
sphere = np.stack([x, y, z], axis=-1).reshape(-1, 3)

# 회전, 스케일, 이동
trans_axes = eigvecs_sorted @ np.diag(semiaxes)
ellipsoid = (trans_axes @ sphere.T).T + m_L2

# 투영
projections = [
    (ellipsoid[:, 0], ellipsoid[:, 1]),  # m1-m2
    (ellipsoid[:, 0], ellipsoid[:, 2]),  # m1-m3
    (ellipsoid[:, 1], ellipsoid[:, 2])   # m2-m3
]
labels = [('$m_1$', '$m_2$'), ('$m_1$', '$m_3$'), ('$m_2$', '$m_3$')]
lims = [[-50, 50, 85, 110], [-50, 50, 7, 12], [80, 120, 7, 12]]
colors = ['r', 'b', 'g']

# GridSpec으로 2:1:1:1 비율 subplot 배치
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 1, 1], figure=fig)

# 3D Plot
ax3d = fig.add_subplot(gs[0], projection='3d')
ax3d.scatter(ellipsoid[:, 1], ellipsoid[:, 0], np.full_like(ellipsoid[:, 2], 7),
             s=0.5, color='r', alpha=0.2, label='(m1,m2)')
ax3d.scatter(np.full_like(ellipsoid[:, 1], 80), ellipsoid[:, 0], ellipsoid[:, 2],
             s=0.5, color='b', alpha=0.2, label='(m1,m3)')
ax3d.scatter(ellipsoid[:, 1], np.full_like(ellipsoid[:, 0], -50), ellipsoid[:, 2],
             s=0.5, color='g', alpha=0.2, label='(m2,m3)')
ax3d.scatter(ellipsoid[:, 1], ellipsoid[:, 0], ellipsoid[:, 2],
             s=0.5, color='gray', alpha=0.05, label="Ellipsoid")

ax3d.set_ylabel('$m_1$ (m)')
ax3d.set_xlabel('$m_2$ (m/s)')
ax3d.set_zlabel('$m_3$ (m/s²)')
ax3d.set_ylim(-50, 50)
ax3d.set_xlim(110, 80)
ax3d.set_zlim(7, 12)
ax3d.view_init(elev=30, azim=115)
ax3d.set_box_aspect((1, 1, 1.7))
ax3d.legend(markerscale=10)

# 2D 평면 슬라이스
for i, (proj, (lx, ly), lim, color) in enumerate(zip(projections, labels, lims, colors), start=1):
    ax = fig.add_subplot(gs[i])
    ax.scatter(proj[0], proj[1], s=0.5, color=color, alpha=0.5)
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])

plt.tight_layout()
plt.show()