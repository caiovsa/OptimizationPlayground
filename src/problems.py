
import numpy as np

class Problem:
    def __init__(self, n_var, bounds):
        self.n_var = n_var
        self.bounds = bounds

    def evaluate(self, x):
        raise NotImplementedError

class Sphere(Problem):
    def __init__(self, n_var=10):
        super().__init__(n_var, [-100, 100])

    def evaluate(self, x):
        # Shifted sphere para ficar mais complexo
        return np.sum((x - 0.5) ** 2)

class Ackley(Problem):
    def __init__(self, n_var=10):
        super().__init__(n_var, [-100, 100])
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

    def evaluate(self, x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            sum_sq = np.sum(x**2, axis=1)
            sum_cos = np.sum(np.cos(self.c * x), axis=1)
            
            term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / self.n_var))
            term2 = -np.exp(sum_cos / self.n_var)
            
            # Return scalar if input was 1D
            result = term1 + term2 + self.a + np.exp(1)
            return result[0] if result.size == 1 else result
