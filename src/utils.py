
import numpy as np

# Esse é nosso Tweak function usado no Hill Climbing e outros
def bounded_uniform_convolution(v, p=1.0, r=10.0, min_val=-100, max_val=100):
    """
    Tweak function (Algoritmo 8 do livro do Sean Luke)
    """
    tweaked = v.copy()
    for i in range(len(tweaked)):
        if np.random.random() <= p:
            while True:
                noise = np.random.uniform(-r, r)
                new_value = tweaked[i] + noise
                if min_val <= new_value <= max_val:
                    tweaked[i] = new_value
                    break
    return tweaked

def create_random_individual(dim, min_val, max_val):
    return np.random.uniform(min_val, max_val, dim)
