import numpy as np
import matplotlib.pyplot as plt  # Importing matplotlib for plots
import time

# Step 1: Parse Problem Data
def parse_problem(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]  # Remove extra spaces and line breaks

    # Parse number of agents and jobs
    agents_jobs = lines[0].split(", ")
    num_agents = int(agents_jobs[0].split()[0])
    num_jobs = int(agents_jobs[1].split()[0])

    # Parse costs
    cost_start_idx = lines.index("cost of allocating job j to agent i") + 1
    cost_end_idx = cost_start_idx + num_agents
    costs = [list(map(int, line.split())) for line in lines[cost_start_idx:cost_end_idx]]

    # Parse resource consumption
    resource_start_idx = lines.index("resource consumed in allocating job j to agent i") + 1
    resource_end_idx = resource_start_idx + num_agents
    resource_consumptions = [list(map(int, line.split())) for line in lines[resource_start_idx:resource_end_idx]]

    # Parse resource capacities
    capacity_idx = lines.index("resource capacity of agent i") + 1
    resource_capacities = list(map(int, lines[capacity_idx].split()))

    return num_agents, num_jobs, np.array(costs), np.array(resource_consumptions), np.array(resource_capacities)
# Step 2: Initialize Solution
def initialize_solution(num_agents, num_jobs, resource_consumptions, resource_capacities):
    solution = []
    resources_used = [0] * num_agents
    for j in range(num_jobs):
        min_overload = float('inf')
        selected_agent = None
        for i in range(num_agents):
            projected_usage = resources_used[i] + resource_consumptions[i][j]
            overload = projected_usage - resource_capacities[i]
            if overload < min_overload:
                min_overload = overload
                selected_agent = i
        if selected_agent is not None:
            solution.append(selected_agent)
            resources_used[selected_agent] += resource_consumptions[selected_agent][j]
        else:
            # Assign to the agent with the most remaining capacity
            max_capacity = max(resource_capacities)
            selected_agent = resource_capacities.index(max_capacity)
            solution.append(selected_agent)
            resources_used[selected_agent] += resource_consumptions[selected_agent][j]
    return solution, resources_used
# Step 3: Calculate Total Cost (Penalty if capacity violation)
def calculate_cost(solution, costs, resource_consumptions, resource_capacities):
    total_cost = 0
    resources_used = [0] * len(resource_capacities)
    for j, agent in enumerate(solution):
        total_cost += costs[agent][j]
        resources_used[agent] += resource_consumptions[agent][j]
    # Add penalty for capacity violations
    penalty = 0
    for i in range(len(resource_capacities)):
        if resources_used[i] > resource_capacities[i]:
            penalty += (resources_used[i] - resource_capacities[i]) * 1000  # Penalty coefficient
    total_cost += penalty
    return total_cost

# Step 4: Check Feasibility of a Solution
# Function to check feasibility
def is_feasible(solution, resource_consumptions, resource_capacities):
    num_agents = len(resource_capacities)
    resources_used = [0] * num_agents
    for j, agent in enumerate(solution):
        resources_used[agent] += resource_consumptions[agent][j]
    for i in range(num_agents):
        if resources_used[i] > resource_capacities[i]:
            return False
    return True

# Step 5: Generate Neighboring Solutions
def get_neighbors(solution, num_agents, resource_consumptions, resource_capacities):
    neighbors = []
    num_jobs = len(solution)
    for j in range(num_jobs):
        current_agent = solution[j]
        for i in range(num_agents):
            if i != current_agent:
                new_solution = solution.copy()
                new_solution[j] = i
                if is_feasible(new_solution, resource_consumptions, resource_capacities):
                    neighbors.append(new_solution)
    return neighbors

# Step 6: Tabu Search Algorithm (Tabu override implemented)
def tabu_search(num_agents, num_jobs, costs, resource_consumptions, resource_capacities, max_iterations, tabu_tenure, no_improvement_limit):

    # Record start time
    start_time = time.time()

    solution, resources_used = initialize_solution(num_agents, num_jobs, resource_consumptions, resource_capacities)
    best_solution = solution
    best_cost = calculate_cost(solution, costs, resource_consumptions, resource_capacities)
    tabu_list = []
    cost_history = []
    no_improvement_counter = 0

    for iteration in range(max_iterations):
        neighbors = get_neighbors(solution, num_agents, resource_consumptions, resource_capacities)
        best_neighbor = None
        best_neighbor_cost = float('inf')
        # aspiration tabu override
        for neighbor in neighbors:
            neighbor_cost = calculate_cost(neighbor, costs, resource_consumptions, resource_capacities)
            if neighbor in tabu_list and neighbor_cost >= best_cost:
                continue
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost

        if best_neighbor is None:
            break

        solution = best_neighbor
        cost_history.append(best_neighbor_cost)

        # Update tabu list
        tabu_list.append(solution)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # Check for improvement
        if best_neighbor_cost < best_cost:
            best_cost = best_neighbor_cost
            best_solution = best_neighbor
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= no_improvement_limit:
                print(f"Terminating search after {iteration+1} iterations with no improvement.")
                break

    
    # Record end time
    end_time = time.time()

    execution_time = end_time - start_time

    return best_solution, best_cost, cost_history, execution_time

def print_initial_solution_and_cost(num_agents, num_jobs, costs, resource_consumptions, resource_capacities):
    # Initialize solution
    solution, resources_used = initialize_solution(num_agents, num_jobs, resource_consumptions, resource_capacities)
    # Calculate initial cost
    initial_cost = calculate_cost(solution, costs, resource_consumptions, resource_capacities)
    
    return solution, initial_cost

# Step 7: Main Execution
file_path_1 = r"problem-instances/problem_instance1.txt"
file_path_2 = r"problem-instances/problem_instance2.txt"
file_path_3 = r"problem-instances/problem_instance3.txt"

file_paths = [file_path_1, file_path_2, file_path_3]
plot_data = []  # To store data for plotting

for file_path in file_paths:
    num_agents, num_jobs, costs, resource_consumptions, resource_capacities = parse_problem(file_path)

    # Parameters
    max_iterations = 1000  # Maximum iterations as a safety net
    tabu_tenure = 10
    no_improvement_limit = 50  # Terminate after 50 iterations without improvement

    # Run the tabu search algorithm
    best_solution, best_cost, cost_history, execution_time = tabu_search(
        num_agents,
        num_jobs,
        costs,
        resource_consumptions,
        resource_capacities,
        max_iterations,
        tabu_tenure,
        no_improvement_limit
    )

    # Print initial solution and cost
    init_sol, init_cost = print_initial_solution_and_cost(num_agents, num_jobs, costs, resource_consumptions, resource_capacities)

    # Print the results in a table format
    print(f"{'File Path':<20}{'Initial Solution':<40}{'Initial Cost':<15}{'Best Solution':<40}{'Best Cost':<15}{'Execution Time':<15}")
    print("-" * 120)
    print(f"{file_path:<20}{str(init_sol):<40}{init_cost:<15}{str(best_solution):<40}{best_cost:<15}{execution_time:<15}")

    plot_data.append((file_path, cost_history))  # Collect data for plotting



# Step 8: Plot Results
plt.figure(figsize=(12, 6))

for i, (file_path, cost_history) in enumerate(plot_data):
    plt.plot(range(1, len(cost_history) + 1), cost_history, label=f"Instance {i+1}")

plt.title("Cost vs. Iterations for Tabu Search")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.grid(True)
plt.show()