import numpy as np
import matplotlib.pyplot as plt

#------------------------P1.Random sampling---------------------------------
# Define μ,σ,N
u = 175      # Expected value E[X] - "μ" 
sigma = 3   # Expected variance Var - "σ"
N = 10000  # Sample

# Generate random samples  Z ~ N(0,1) 
Z = np.random.randn(N)

# Transform the samples X ~ N(μ, σ^2) using X = μ + σ * Z for given values of μ and σ
X = u + sigma * Z

# Vizualization Plot a histogram
plt.figure(figsize=(8, 5))
plt.hist(X, bins=50, density=True, alpha=0.6, color='gray', edgecolor='black')
plt.axvline(u, color='green', linestyle='solid', linewidth=2, label=f'Theoretical values μ: {u:.2f}')
plt.axvline(np.mean(X), color='red', linestyle='dashed', linewidth=2, label=f'Sample values μ: {np.mean(X):.2f}')
plt.axvline(u - sigma**2, color='blue', linestyle='solid', linewidth=2, label=f'Theoretical σ^2: {(sigma**2):.2f}')
plt.axvline(np.mean(X) - sigma**2, color='red', linestyle='dashed', linewidth=2, label=f'Sample σ^2: {(np.var(X)):.2f}')
plt.axvline(u + sigma**2, color='blue', linestyle='solid', linewidth=2)
plt.axvline(np.mean(X) + sigma**2, color='red', linestyle='dashed', linewidth=2)
plt.xlabel("X")
plt.ylabel("Density of Normal Distribution")
plt.title("Histogram of Transformed Normal Distribution  X = μ + σ * Z")
plt.legend()
plt.show()

# Calculate the sample mean and variance X∼N(μ,σ^2)
"1. Sample mean E[X]=E[μ+σZ]=μ+σE[Z]=μ -> E[X] = μ"
EX = np.mean(X)
"2. Sample variance Var[X]=Var[μ+σZ]=σ^2Var[Z]=σ^2 -> Var[X] = σ^2"
Var = np.var(X)
print(f"Theoretical values μ: {u}, Sample values μ: {EX:.2f}")
print(f"Theoretical σ: {sigma**2}, Sample σ: {Var:.2f}")

#------------------------P2.Bayes' theorem simulation------------------------
# Define
#P(A) - probability of true positive
#P(B) - probability of disease 
PAB1 = 0.99  # P(A|B) - True positive
PAB2 = 1 - PAB1  # False positive

PB1 = 0.0001  # P(B) - prior probability of disease
PB2 = 1 - PB1  # prior probability of no disease

# posterior probability Bayes' Theorem
#P(A)= SUM(P(A|B)*P(B)) - Law of total probability
PA = PAB1 * PB1 + PAB2 * PB2  # P(A)
# Bayes' Theorem
# P(B|A) = (P(A|B)*P(B)) / P(A)
PBA = (PAB1 * PB1) / PA  # P(B|A)

# Plot
labels = ["P(A | B)", "P(B)"]
values1 = [PAB2, PAB1]  # Probability Disease if test true positive
values2 = [PB1, PB2]  # Вероятности болезни (Probability Disease) 

x_positions = [0, 1]
width = 1.1  
colors1 = ['blue', 'red']
colors2 = ['darkgreen', 'yellow']
text_labels1 = ["No Disease | False positive P(A|B)1%", "Disease | True positive P(A|B)99%"]
text_labels2 = ["Disease P(B)0.01%", "No disease P(B)99.99%"]

# histograms and graphics
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Tests Столбец теста
bottom_val = 0
for i, val in enumerate(values1):
    axes[0].bar(x_positions[0], val, bottom=bottom_val, color=colors1[i], alpha=1, width=width)
    axes[0].text(x_positions[0], bottom_val + val / 2, text_labels1[i], ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    bottom_val += val

# Diseases Столбец болезни 
bottom_val = 0
for i, val in enumerate(values2):
    axes[0].bar(x_positions[1], val, bottom=bottom_val, color=colors2[i], alpha=1 if i == 0 else 0.7, width=width, edgecolor='black' if i == 0 else None, linewidth=1.5)
    axes[0].text(x_positions[1], bottom_val + val / 2, text_labels2[i], ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    bottom_val += val

axes[0].axhline()
axes[0].legend([f'Posterior probability if actually has the disease: {PBA:.5f}'], loc='upper right')
axes[0].set_xlabel("Test result")
axes[0].set_ylabel("P")
axes[0].set_title("Probability of disease given test result")
axes[0].set_xticks(x_positions)
axes[0].set_xticklabels(labels)
axes[0].set_ylim(0, 1)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Explore how the posterior probability changes when the following parameters  Визуализация зависимостей P(B|A) от P(A|B) и P(B)
PAB = np.linspace(0.001, 0.99, 100)  # P(A|B)
PB = np.linspace(0.0001, 0.99, 100)  # P(B)

PBconst = 0.01  #  P(B) const
PBA_from_PAB = [(PAB * PBconst) / (PAB * PBconst + (1 - PAB) * (1 - PBconst)) for PAB in PAB]

PABconst = 0.99  #  P(A|B) const
PBA_from_PB = [(PABconst * PB) / (PABconst * PB + (1 - PABconst) * (1 - PB)) for PB in PB]

# Graph probabilities
# Plot P(B|A) от P(A|B)
axes[1].plot(PAB, PBA_from_PAB, color='blue', label='P(B|A) of P(A|B)')
axes[1].set_title('P(B|A) changes of P(A|B)')
axes[1].grid(True)

axes[1].plot(PB, PBA_from_PB, color='red', label='P(B|A) of P(B)')
axes[1].set_xlabel('P(A|B) - Accuracy of the test')
axes[1].set_ylabel('P(B|A) - Posterior probability')
axes[1].set_title('P(B) - Prior probability of disease')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
