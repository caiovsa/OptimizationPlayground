import sys
import os
import numpy as np
import random

# Permite rodar este arquivo diretamente ou importá-lo como módulo
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import bounded_uniform_convolution, create_random_individual
from src.problems import Sphere, Ackley

def tournament_selection(population, fitness_values, t=2):
    best_index = random.randint(0, len(population) - 1)
    for _ in range(1, t):
        next_index = random.randint(0, len(population) - 1)
        # Menor valor é melhor (minimização)
        if fitness_values[next_index] < fitness_values[best_index]:
            best_index = next_index
    return population[best_index]

def two_point_crossover(parent1, parent2):
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(size), 2))
    child1 = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]])
    child2 = np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]])
    return child1, child2

def genetic_algorithm(problem, popsize=100, n_elites=10, max_evals=50000, tournament_size=2):
    dim = problem.n_var
    min_val, max_val = problem.bounds
    eval_count = 0
    
    population = [create_random_individual(dim, min_val, max_val) for _ in range(popsize)]
    
    best_overall_individual = None
    best_overall_fitness = np.inf 
    history = []

    while eval_count < max_evals:
        fitness_values = []
        for ind in population:
            val = problem.evaluate(ind)
            fitness_values.append(val)
            eval_count += 1
            if eval_count >= max_evals: break
        
        fitness_values = np.array(fitness_values)
        
        # Atualiza o melhor global
        if len(fitness_values) > 0:
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_overall_fitness:
                best_overall_fitness = fitness_values[min_idx]
                best_overall_individual = population[min_idx]
        
        history.append(best_overall_fitness)

        if eval_count >= max_evals: break

        # Elitismo
        sorted_indices = np.argsort(fitness_values)
        next_population = [population[i] for i in sorted_indices[:n_elites]]

        # Criação de nova geração
        num_children_to_create = popsize - n_elites
        for _ in range(num_children_to_create // 2):
            parent_a = tournament_selection(population, fitness_values, t=tournament_size)
            parent_b = tournament_selection(population, fitness_values, t=tournament_size)
            child_a, child_b = two_point_crossover(parent_a, parent_b)
            
            # Mutação (Tweak)
            child_a = bounded_uniform_convolution(child_a, min_val=min_val, max_val=max_val)
            child_b = bounded_uniform_convolution(child_b, min_val=min_val, max_val=max_val)
            
            next_population.append(child_a)
            next_population.append(child_b)
        
        # Preencher se a população for ímpar
        while len(next_population) < popsize:
                parent_a = tournament_selection(population, fitness_values, t=tournament_size)
                next_population.append(bounded_uniform_convolution(parent_a, min_val=min_val, max_val=max_val))

        population = next_population[:popsize]

    return best_overall_individual, best_overall_fitness, history

if __name__ == "__main__":
    print("="*60)
    print("TESTE ISOLADO: Genetic Algorithm")
    print("="*60)

    # Teste 1: Sphere
    print("\n--- Otimizando função Sphere (10 variáveis) ---")
    problem = Sphere(n_var=10)
    # População um pouco maior para convergir bem
    best_sol, best_val, _ = genetic_algorithm(problem, popsize=100, n_elites=10)
    print(f"Melhor valor encontrado: {best_val:.6f}")

    # Teste 2: Ackley
    print("\n--- Otimizando função Ackley (10 variáveis) ---")
    problem = Ackley(n_var=10)
    best_sol, best_val, _ = genetic_algorithm(problem, popsize=200, n_elites=20)
    print(f"Melhor valor encontrado: {best_val:.6f}")