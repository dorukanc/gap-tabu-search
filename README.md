# Tabu Search Algorithm for Optimization

This repository contains a Python implementation of a Tabu Search Algorithm developed for the **Modeling and Methods in Optimization IE 307** class project. The algorithm is designed to solve assignment problems where jobs must be assigned to agents in a way that minimizes cost while satisfying resource constraints.

## Project Description

The main objective of this project is to assign a set of jobs to a set of agents such that:

- **Every job is assigned to exactly one agent.**
- **The total resource consumption by each agent does not exceed their capacity.**
- **The total assignment cost is minimized.**

The Tabu Search metaheuristic is employed to efficiently explore the solution space and escape local optima by using a tabu list to keep track of recent moves.

## Features

- **Initialization**: Generates an initial feasible solution ensuring each job is assigned to one agent.
- **Feasibility Checks**: Ensures that assignments do not violate agent capacity constraints.
- **Neighbor Generation**: Creates neighboring solutions by swapping job assignments between agents.
- **Tabu List Management**: Implements tabu tenure and aspiration criteria to guide the search.
- **Cost Calculation**: Computes the total cost of assignments, including penalties for capacity violations.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries:
  - `numpy`
  - `random`
  - `matplotlib`

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/dorukanc/gap-tabu-search.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd gap-tabu-search
   ```

3. **Install dependencies**:

   ```bash
   pip install numpy
   pip install matplotlib
   ```

### Running the Algorithm

Execute the main script to run the Tabu Search algorithm:

```bash
code gap_tabu_search.ipynb
```

## Algorithm Overview

1. **Initialization**:

   - An initial solution is generated where each job is assigned to an agent.
   - If a job cannot be assigned without exceeding capacity, it is assigned to the agent with the least capacity overload.

2. **Evaluation**:

   - The cost of the solution is calculated, including penalties for any capacity violations.

3. **Neighborhood Generation**:

   - Neighboring solutions are generated by reassigning jobs to different agents.
   - Only feasible neighbors that respect the assignment constraint (each job assigned exactly once) are considered.

4. **Tabu List Management**:

   - Recent moves are added to the tabu list to prevent cycling.
   - Aspiration criteria allow overriding the tabu status if a better solution is found.

5. **Iteration**:

   - The algorithm iterates through the search space, updating the best solution found.
   - Termination occurs after a maximum number of iterations or when no improvement is observed over a set limit.

## Code Structure

- `gap_tabu_search.ipynb`: Contains the main implementation of the Tabu Search algorithm.

## Functions Description

- **initialize_solution**:

  Initializes a feasible solution by assigning each job to an agent, allowing initial capacity violations which are penalized in the cost function.

- **is_feasible**:

  Checks if a solution is feasible by ensuring resource capacities are not exceeded for any agent.

- **calculate_cost**:

  Calculates the total cost of a solution, including penalties for any capacity violations.

- **get_neighbors**:

  Generates neighboring solutions by changing the agent assignment of jobs.

- **tabu_search**:

  Main function that performs the Tabu Search optimization.

## Parameters

- `num_agents`: Number of agents.
- `num_jobs`: Number of jobs.
- `costs`: A matrix representing the cost of assigning each job to each agent.
- `resource_consumptions`: A matrix representing resource consumption when agents perform jobs.
- `resource_capacities`: A list representing each agent's capacity.
- `max_iterations`: Maximum number of iterations to perform.
- `tabu_tenure`: Number of iterations a move remains in the tabu list.
- `no_improvement_limit`: Number of iterations to continue without improvement before stopping.

## Usage Example

```python
import numpy as np

# Define parameters
num_agents = 5
num_jobs = 25
max_iterations = 100
tabu_tenure = 5
no_improvement_limit = 10

# Generate random costs and resources for demonstration purposes
np.random.seed(0)
costs = np.random.randint(1, 10, size=(num_agents, num_jobs))
resource_consumptions = np.random.randint(1, 5, size=(num_agents, num_jobs))
resource_capacities = np.random.randint(20, 30, size=num_agents)

# Run the Tabu Search algorithm
best_solution, best_cost, cost_history = tabu_search(
    num_agents,
    num_jobs,
    costs,
    resource_consumptions,
    resource_capacities,
    max_iterations,
    tabu_tenure,
    no_improvement_limit
)

print("Best Solution:", best_solution)
print("Best Cost:", best_cost)
```

## Results Interpretation

- **Best Solution**: An array representing the agent assigned to each job.
- **Best Cost**: The lowest total cost achieved during the search, including penalties.
- **Cost History**: A list of costs from each iteration, useful for analyzing convergence.

## Contributions

Contributions to improve the algorithm or adapt it to more complex problems are welcome. Please fork the repository and submit a pull request.

## References

- Tabu Search methodology and implementation techniques discussed in class.
- Python programming concepts for optimization algorithms.

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Professor Hande Küçükaydın** for guidance throughout the IE 307 course.


