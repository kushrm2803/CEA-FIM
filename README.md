# CEA-FIM: Community-Aware Evolution Algorithm for Fair Influence Maximization

This repository contains an implementation of the Community-Aware Evolution Algorithm for Fair Influence Maximization (CEA-FIM).

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- Python >= 3.7
- numpy >= 1.19.0 
- networkx >= 2.5
- numba >= 0.50.0
- python-louvain >= 0.15
- scipy >= 1.5.0
- gurobipy >= 9.1.0 
- cvxpy >= 1.1.0

## Files Structure

The repository maintains the following structure:

- `CEA-FIM.py`: Main implementation of the evolutionary algorithm
- `algorithms.py`: Core optimization algorithms
- `icm.py`: Independent Cascade Model implementation
- `utils.py`: Utility functions and helpers
- `pickle_graphnattr.py`: Graph and attribute data handling
- `networks/`: Directory containing network data files

## Usage Example

```python
import networkx as nx
from CEA_FIM import initialize_population, run_evolution

# Load network
G = nx.read_edgelist("networks/graph.txt")

# Set parameters
pop_size = 10  # Population size
budget = 5     # Number of seed nodes to select
max_gen = 150  # Number of generations

# Run algorithm
solution = run_evolution(G, pop_size, budget, max_gen)

print(f"Selected seed set: {solution}")
```

## Code Organization

The code is organized as follows:

1. **Main Algorithm (`CEA-FIM.py`):**
   - Implementation of the evolutionary algorithm
   - Population initialization and evolution
   - Fitness evaluation
   - Genetic operators (mutation, crossover)

2. **Optimization Algorithms (`algorithms.py`):**
   - Frank-Wolfe algorithm variants
   - Indicator functions
   - Gradient-based optimization

3. **Influence Model (`icm.py`):**
   - Independent Cascade Model implementation
   - Live-edge graph sampling
   - Influence probability calculations

4. **Utilities (`utils.py`):**
   - Graph visualization
   - Greedy algorithms
   - Projection functions
   - Helper utilities