import sys
import os
import numpy as np

# Permite rodar
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import bounded_uniform_convolution
from src.problems import Sphere, Ackley

# Hill Climbing, usando o Tweak function definido em utils.py
def hill_climbing(problem, initial, max_evals=30000, step_size=10, p=0.9):
    
    current = initial.copy()
    current_eval = problem.evaluate(current)
    evaluations = 1
    history = [current_eval]

    while evaluations < max_evals:
        # Generate one neighbor
        neighbor = bounded_uniform_convolution(current, p=p, r=step_size, min_val=problem.bounds[0], max_val=problem.bounds[1])
        neighbor_eval = problem.evaluate(neighbor)
        evaluations += 1

        if neighbor_eval < current_eval:
            current = neighbor
            current_eval = neighbor_eval
            history.append(current_eval)
        
    return current, current_eval, history

if __name__ == "__main__":
    print("="*60)
    print("TESTE ISOLADO: Hill Climbing")
    print("="*60)

    # Teste 1: Sphere
    print("\n--- Otimizando função Sphere (10 variáveis) ---")
    problem_sphere = Sphere(n_var=10)
    X = np.random.uniform(-100, 100, problem_sphere.n_var)
    best_sol, best_val, hist = hill_climbing(problem_sphere, initial=X, step_size=0.5, p=0.1)
    
    print(f"Melhor valor encontrado: {best_val:.6f}")
    print(f"Solução (primeiros 5 valores): {best_sol[:5]}...")
    print(f"Número de melhorias aceitas: {len(hist)}")

    # Teste 2: Ackley
    print("\n--- Otimizando função Ackley (10 variáveis) ---")
    problem_ackley = Ackley(n_var=10)
    X = np.random.uniform(-100, 100, problem_ackley.n_var)
    best_sol, best_val, hist = hill_climbing(problem_ackley, initial=X, step_size=0.5, p=0.1)
    
    print(f"Melhor valor encontrado: {best_val:.6f}")
    print(f"Solução (primeiros 5 valores): {best_sol[:5]}...")
    print(f"Número de melhorias aceitas: {len(hist)}")