# import the required libraries 
import random
import matplotlib.pyplot as plt

# +
# store the random numbers in a list 
nums = []

mean_word_length=6.5  # = μ = α * β
var = 5.0  # σ²=α/β²= α³/μ² => β = σ / sqrt(α)
alpha = (mean_word_length/var)**2 # α = (μ/σ)²
beta = mean_word_length/alpha
print(alpha, beta)

for i in range(10000):
    temp = 1*random.gammavariate(alpha, beta)
    nums.append(temp)

# plotting a graph
plt.hist(nums, bins = 200)
plt.show()
# -


