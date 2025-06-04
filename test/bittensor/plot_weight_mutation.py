import numpy as np 
import matplotlib.pyplot as plt

def weights_mutation_alg1(weights: list, max_value: int) -> list:
    """Softmax scaling"""
    scaled_weights = [(w * 5.5) for w in weights]
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
        return [max_value] * len(weights)
    norm_weights = [(w - min_w) / (max_w - min_w) for w in weights]
    transformed = [w ** power for w in norm_weights]
    return [w * max_value for w in transformed]

def weights_mutation_alg3(weights: list, max_value: int) -> list:
    """Return weights as is"""
    scaled_weights = [(w * 0.1) for w in weights]
    softmax = np.exp(scaled_weights) / np.sum(np.exp(scaled_weights))
    return ((softmax / np.max(softmax)) * max_value)

max_value = 65535
weights = np.arange(0, 1.01, 0.2).tolist()

mut1 = weights_mutation_alg1(weights, max_value)
mut1 = [m/max(mut1) for m in mut1]
mut2 = weights_mutation_alg2(weights, max_value)
mut2 = [m/max(mut2) for m in mut2]
mut3 = weights_mutation_alg3(weights, max_value)
mut3 = [m/max(mut3) for m in mut3]

print(f"softmax weights: {mut1}")

# Plotting
plt.plot(weights, mut1, label='Softmax Scaling (=2.5) (alg1)', marker='o')
plt.plot(weights, mut2, label='Power Scaling (=0.63) (alg2)', marker='x')
plt.plot(weights, mut3, label=' (alg3)', marker='s')
plt.title("Comparison of Weight Mutation Algorithms")
plt.xlabel("Original Weights")
plt.ylabel("Mutated Weights")
plt.legend()
plt.grid(True)
plt.show()