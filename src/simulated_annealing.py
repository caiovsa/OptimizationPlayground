import sys
import os
import numpy as np

# Permite rodar este arquivo diretamente ou importá-lo como módulo
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import bounded_uniform_convolution
from src.problems import Sphere, Ackley

def simulated_annealing(problem, initial, max_evals=50000, initial_temp=1000, cooling_rate=0.99, min_temp=0.001, step_size=10):

    current = initial.copy()
    current_eval = problem.evaluate(current)
    T = initial_temp
    evaluations = 1
    history = [current_eval]

    while T > min_temp and evaluations < max_evals:
        neighbor = bounded_uniform_convolution(current, p=0.9, r=step_size, min_val=problem.bounds[0], max_val=problem.bounds[1])
        neighbor_eval = problem.evaluate(neighbor)
        evaluations += 1

        delta = neighbor_eval - current_eval

        if delta < 0:
            current = neighbor
            current_eval = neighbor_eval
            history.append(current_eval)
        else:
            acceptance_prob = np.exp(-delta / T)
            if np.random.rand() < acceptance_prob:
                current = neighbor
                current_eval = neighbor_eval
                history.append(current_eval)
        
        T *= cooling_rate

    return current, current_eval, history

if __name__ == "__main__":
    print("="*60)
    print("TESTE ISOLADO: Simulated Annealing")
    print("="*60)

    # Teste 1: Sphere
    print("\n--- Otimizando função Sphere (10 variáveis) ---")
    problem_sphere = Sphere(n_var=10)
    X = np.random.uniform(-100, 100, problem_sphere.n_var)
    best_sol, best_val, hist = simulated_annealing(problem_sphere, initial=X, step_size=0.1, cooling_rate=0.999)
    print(f"Melhor valor encontrado: {best_val:.6f}")
    print(f"Solução (primeiros 5 valores): {best_sol[:5]}...")
    print(f"Número de melhorias aceitas: {len(hist)}")
    
    # Teste 2: Ackley
    print("\n--- Otimizando função Ackley (10 variáveis) ---")
    problem_ackley = Ackley(n_var=10)
    X = np.random.uniform(-100, 100, problem_ackley.n_var)
    best_sol, best_val, hist = simulated_annealing(problem_ackley, initial=X, step_size=0.5, cooling_rate=0.995, initial_temp=1000)
    print(f"Melhor valor encontrado: {best_val:.6f}")
    print(f"Solução (primeiros 5 valores): {best_sol[:5]}...")
    print(f"Número de melhorias aceitas: {len(hist)}")