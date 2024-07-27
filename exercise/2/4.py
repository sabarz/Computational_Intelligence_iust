import numpy as np

equations = [
    # lambda x: 2*x - 4,
    # lambda x: x**2 - 8*x + 4,
    lambda x: 168*x**3 - 7.22*x**2 + 15.5*x - 13.2
]

population_size = 100
num_generations = 1000
mutation_rate = 0.1

def fitness(solution):
    error = 0
    for equation in equations:
        error += abs(equation(solution))
    return 1 / (1 + error)

def initialize_population():
    return np.random.uniform(low=-100, high=100, size=(population_size,))

def crossover(parent1, parent2):
    alpha = np.random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

def mutate(solution):
    if np.random.random() < mutation_rate:
        mutation_amount = np.random.uniform(low=-1, high=1)
        solution += mutation_amount
    return solution

population = initialize_population()
for generation in range(num_generations):
    fitness_scores = np.array([fitness(sol) for sol in population])

    probabilities = fitness_scores / np.sum(fitness_scores)
    parent_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)

    next_generation = []
    for i in range(0, population_size, 2):
        parent1 = population[parent_indices[i]]
        parent2 = population[parent_indices[i + 1]]
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        next_generation.extend([child1, child2])

    population = np.array(next_generation)


best_solution = population[np.argmax([fitness(sol) for sol in population])]
print("Roots found by genetic algorithm:")
for equation in equations:
    print("Equation:", equation)
    print("Root:", best_solution)
    print("Value of the equation at the root:", equation(best_solution))
