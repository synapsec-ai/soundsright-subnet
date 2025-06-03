import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def weights_mutation_alg1(weights: list, max_value: int) -> list:
    """Softmax scaling"""
    scaled_weights = [(w * 2.5) for w in weights]
    softmax = np.exp(scaled_weights) / np.sum(np.exp(scaled_weights))
    return ((softmax / np.max(softmax)) * max_value)

def weights_mutation_alg2(weights: list, max_value: int) -> list:
    """Power scaling"""
    if not weights:
        return []
    
    power = 0.63

    min_w = min(weights)
    max_w = max(weights)

    if max_w == min_w:
        return [max_value] * len(weights)  # all values are equal

    # Normalize to [0, 1]
    norm_weights = [(w - min_w) / (max_w - min_w) for w in weights]

    # Apply power transformation
    transformed = [w ** power for w in norm_weights]

    # Rescale to [0, max_value]
    scaled = [w * max_value for w in transformed]

    return scaled

def weights_mutation_alg3(weights: list, max_value: int) -> list:
    """Return weights as is"""
    return [(w * max_value) for w in weights]

max_value = 65535
weights = np.arange(0, 1.01, 0.2).tolist()

mut1 = weights_mutation_alg1(weights, max_value)
mut2 = weights_mutation_alg2(weights, max_value)
mut3 = weights_mutation_alg3(weights, max_value)

print(f"softmax weights: {mut1}")

# Plotting
plt.plot(weights, mut1, label='Softmax Scaling (alg1)', marker='o')
plt.plot(weights, mut2, label='Power scaling (alg2)', marker='x')
plt.plot(weights, mut3, label='Linear Scale (alg3)', marker='s')
plt.title("Comparison of Weight Mutation Algorithms")
plt.xlabel("Original Weights")
plt.ylabel("Mutated Weights")
plt.legend()
plt.grid(True)
plt.show()