# Algoritmos de Otimização

Este repositório contém implementações em Python de vários algoritmos de otimização metaheurística. Estas implementações foram desenvolvidas como parte da disciplina "Busca e Otimização" no mestrado.

## Algoritmos Implementados

Os seguintes algoritmos estão localizados no diretório `src/`:

- **Hill Climbing** (`hill_climbing.py`): Um algoritmo de busca local iterativa básica.
- **Simulated Annealing** (`simulated_annealing.py`): Uma técnica probabilística para aproximar o ótimo global de uma função dada.
- **Genetic Algorithm** (`genetic_algorithm.py`): Um algoritmo evolutivo inspirado no processo de seleção natural.
- **Particle Swarm Optimization** (`pso.py`): Um método computacional que otimiza um problema tentando melhorar iterativamente uma solução candidata em relação a uma medida de qualidade.

## Estrutura do Projeto

- `src/`: Contém as implementações principais em Python dos algoritmos e problemas de teste (Sphere, Ackley).
  - `problems.py`: Define os problemas de otimização.
  - `utils.py`: Funções auxiliares para perturbação e manipulação de soluções.
- `notebooks/`: Jupyter notebooks demonstrando o uso e desempenho dos algoritmos.
  - `1_Hill_Climbing.ipynb` a `4_PSO.ipynb`: Análise individual de cada método.
  - `5_Geral_Algoritmos_Otimização.ipynb`: Visão geral.
  - `6_Comparativo_Algoritmos.ipynb`: Análise comparativa dos algoritmos.

## Dependências

O projeto requer as seguintes bibliotecas:

- numpy
- matplotlib
- pymoo
- jupyter

## Instalação

Instale as dependências necessárias usando o pip:

```bash
pip install -r requirements.txt
```

## Uso

Você pode executar os algoritmos diretamente dos arquivos fonte para ver uma demonstração básica:

```bash
python src/hill_climbing.py
```

Alternativamente, utilize os Jupyter notebooks no diretório `notebooks/` para análise visual e experimentação.
