import numpy as np
import matplotlib.pyplot as plt

# 1.   Define
#P(A) - вероятность true positive
#P(B) - вероятность disease 
PAB1 = 0.99 # P(A|B) Result True positive  (have the disease) 99% accuracy of the test
PAB2 = 1-PAB1  # Result False positive   (does not have the disease) 1%

PB1 = 0.5  # P(B)prior probability disease 0.01%
PB2 = 1 - PB1  # prior probability NO DISEASE 99.99%

# 2. posterior probability Bayes' Theorem
#P(A)= SUM(P(A|B)*P(B)) - Law of total probability
PA = PAB1*PB1+PAB2*PB2 # posterior
# Bayes' Theorem
# P(B|A) = (P(A|B)*P(B)) / P(A) 
PBA= (PAB1*PB1) / (PA) # posterior probability

# Plot
import matplotlib.pyplot as plt

labels = ["P (A | B)", "P ( B )"]
values1 = [PAB2, PAB1]  # Probability Disease if test true positive
values2 = [PB1, PB2]    # Probability Disease 

x_positions = [0, 1]  
width = 1.1  
colors1 = ['blue', 'red']
colors2 = ['darkgreen', 'yellow']
text_labels1 = ["No Disease | False positive P(A|B)1% ", "Disease | True positive P(A|B)99%"]
text_labels2 = ["Disease P(B)0.01%", "No disease P(B)99.99%"]

plt.figure(figsize=(8, 8))

# Tests 
bottom_val = 0
for i, val in enumerate(values1):
    plt.bar(x_positions[0], val, bottom=bottom_val, color=colors1[i], alpha=1, width=width)
    plt.text(x_positions[0], bottom_val + val / 2, text_labels1[i], ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    bottom_val += val

# Diseases
bottom_val = 0
for i, val in enumerate(values2):
    plt.bar(x_positions[1], val, bottom=bottom_val, color=colors2[i], alpha=1 if i == 0 else 0.7, width=width, edgecolor='black' if i == 0 else None, linewidth=1.5)
    plt.text(x_positions[1], bottom_val + val / 2, text_labels2[i], ha='center', va='center', fontsize=8, color='black' if i == 0 else 'black', fontweight='bold')
    bottom_val += val
    
#posterior probability if has the disease

plt.axhline ()
plt.legend( [f'Posterior probability if actually has the disease: {PBA:.5f}'], loc='upper right')

plt.xlabel("Test result")
plt.ylabel("P")
plt.title("Probability of disease given test result")

plt.xticks(x_positions, labels)
plt.ylabel("P")
plt.title("Probability of disease given test result")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
