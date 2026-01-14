import sys
import os
import numpy as np

# Permite rodar este arquivo diretamente ou importá-lo como módulo
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import create_random_individual
from src.problems import Sphere, Ackley

def particle_swarm_optimization(problem, n_particles=30, max_evals=50000, w=0.7, c1=1.5, c2=1.5):
    dim = problem.n_var
    min_val, max_val = problem.bounds
    eval_count = 0
    
    positions = np.array([create_random_individual(dim, min_val, max_val) for _ in range(n_particles)])
    velocities = np.random.uniform(-abs(max_val - min_val) * 0.1, 
                                    abs(max_val - min_val) * 0.1, 
                                    (n_particles, dim))
    
    # Avaliação inicial
    fitness = np.array([problem.evaluate(pos) for pos in positions])
    eval_count += n_particles
    
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()
    
    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]
    
    history = [gbest_fitness]

    while eval_count < max_evals:
        for i in range(n_particles):
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            
            cognitive = c1 * r1 * (pbest_positions[i] - positions[i])
            social = c2 * r2 * (gbest_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social
            
            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], min_val, max_val)
            
            current_fitness = problem.evaluate(positions[i])
            eval_count += 1
            
            if current_fitness < pbest_fitness[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_fitness[i] = current_fitness
                
                if current_fitness < gbest_fitness:
                    gbest_position = positions[i].copy()
                    gbest_fitness = current_fitness
            
            if eval_count >= max_evals:
                break
        
        history.append(gbest_fitness)
        
    return gbest_position, gbest_fitness, history

if __name__ == "__main__":
    print("="*60)
    print("TESTE ISOLADO: Particle Swarm Optimization (PSO)")
    print("="*60)

    # Teste 1: Sphere
    print("\n--- Otimizando função Sphere (10 variáveis) ---")
    problem = Sphere(n_var=10)
    best_sol, best_val, hist = particle_swarm_optimization(problem, n_particles=30)
    print(f"Melhor valor encontrado: {best_val:.6f}")
    print(f"Solução (primeiros 5 valores): {best_sol[:5]}...")
    print(f"Tamanho do histórico: {len(hist)}")

    # Teste 2: Ackley (usando seus parâmetros otimizados)
    print("\n--- Otimizando função Ackley (10 variáveis) ---")
    problem = Ackley(n_var=10)
    best_sol, best_val, hist = particle_swarm_optimization(problem, n_particles=90, w=0.6571, c1=1.6319, c2=0.6239)
    print(f"Melhor valor encontrado: {best_val:.6f}")
    print(f"Solução (primeiros 5 valores): {best_sol[:5]}...")
    print(f"Tamanho do histórico: {len(hist)}")