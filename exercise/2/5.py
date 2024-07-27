import numpy as np

class ACO:
    def __init__(self, n_ants, n_iterations, alpha, beta, rho, Q):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

    def initialize(self, distances):
        self.all_distances = distances
        self.n_cities = len(distances)
        self.pheromone = np.ones((self.n_cities, self.n_cities)) / self.n_cities

    def distance(self, city1, city2):
        return self.all_distances[city1][city2]

    def probability(self, city_from, city_to, visited):
        pheromone = np.power(self.pheromone[city_from][city_to], self.alpha)
        distance = np.power(1.0 / self.distance(city_from, city_to), self.beta)
        return pheromone * distance

    def update_pheromones(self, all_routes, all_lengths):
        self.pheromone *= self.rho
        for i, route in enumerate(all_routes):
            for j in range(len(route) - 1):
                self.pheromone[route[j]][route[j + 1]] += self.Q / all_lengths[i]

    def run(self, distances):
        self.initialize(distances)

        best_route = None
        best_length = np.inf

        for iteration in range(self.n_iterations):
            all_routes = []
            all_lengths = []

            for ant in range(self.n_ants):
                route = []
                visited = set()
                current_city = np.random.randint(0, self.n_cities)
                route.append(current_city)
                visited.add(current_city)

                while len(visited) < self.n_cities:
                    probabilities = []
                    for city in range(self.n_cities):
                        if city not in visited:
                            probabilities.append(self.probability(current_city, city, visited))
                        else:
                            probabilities.append(0)
                    
                    probabilities = np.array(probabilities)
                    probabilities = probabilities / probabilities.sum()
                    next_city = np.random.choice(range(self.n_cities), p=probabilities)
                    route.append(next_city)
                    visited.add(next_city)
                    current_city = next_city

                route.append(route[0])  
                all_routes.append(route)

                length = sum(self.distance(route[i], route[i + 1]) for i in range(len(route) - 1))
                all_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_route = route

            self.update_pheromones(all_routes, all_lengths)
        
        return best_route, best_length

if __name__ == "__main__":
    distances = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    n_ants = 10
    n_iterations = 100
    alpha = 1.0
    beta = 5.0
    rho = 0.5
    Q = 100

    aco = ACO(n_ants, n_iterations, alpha, beta, rho, Q)
    best_route, best_length = aco.run(distances)

    print("Best route:", best_route)
    print("Best length:", best_length)
