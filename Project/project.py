import numpy as np
import random

# initialize the grid
def initialize_grid(n, num_balls, num_holes):
    grid = np.zeros((n, n))
    positions = random.sample(range(n * n), num_balls + num_holes)
    balls = positions[:num_balls]
    holes = positions[num_balls:]

    for b in balls:
        grid[b // n][b % n] = 1  # Ball
    for h in holes:
        grid[h // n][h % n] = 2  # Hole

    return grid, balls, holes

# print the grid
def print_grid(grid):
    for row in grid:
        print(' '.join(['A' if cell == -1 else 'B' if cell == 1 else 'H' if cell == 2 else '.' for cell in row]))
    print()

# move the agent
def move_agent(agent_pos, new_pos, grid, base_grid, has_ball):
    grid[agent_pos[0], agent_pos[1]] = base_grid[agent_pos[0], agent_pos[1]]
    
    grid[new_pos[0], new_pos[1]] = -1
    return new_pos

# find next move towards a ball or hole
def find_target(agent_pos, grid, pheromones, target_type, visited):
    neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    max_pheromone = -1
    next_moves = []
    unseen_moves = []

    for move in neighbors:
        nx, ny = agent_pos[0] + move[0], agent_pos[1] + move[1]
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx, ny] == target_type:
                return (nx, ny), True
            elif pheromones[nx, ny] > max_pheromone:
                max_pheromone = pheromones[nx, ny]
                next_moves = [(nx, ny)]
            elif pheromones[nx, ny] == max_pheromone:
                next_moves.append((nx, ny))
            if (nx, ny) not in visited:
                unseen_moves.append((nx, ny))
    
    if max_pheromone == 0 and unseen_moves:
        return random.choice(unseen_moves), False
    elif next_moves:
        return random.choice(next_moves), False
    elif unseen_moves:
        return random.choice(unseen_moves), False
    else:
        all_moves = [(agent_pos[0] + move[0], agent_pos[1] + move[1]) for move in neighbors if 0 <= agent_pos[0] + move[0] < grid.shape[0] and 0 <= agent_pos[1] + move[1] < grid.shape[1]]
        return random.choice(all_moves), False

def main():
    n = 6
    num_balls = 6
    num_holes = 6
    max_moves = 35
    
    grid, balls, holes = initialize_grid(n, num_balls, num_holes)
    base_grid = np.copy(grid)
    position_agent = (0, 0)
    grid[position_agent] = -1
    move_count = 0
    has_ball = grid[position_agent[0], position_agent[1]] == 1

    pheromone_holes = np.zeros((n, n))
    pheromone_balls = np.zeros((n, n))
    neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    visited = set()
    visited.add(position_agent)

    print_grid(grid)

    while move_count < max_moves:
        # update pheromone 
        for move in neighbors:
            nx, ny = position_agent[0] + move[0], position_agent[1] + move[1]
            if 0 <= nx < n and 0 <= ny < n:
                if grid[nx, ny] == 1:
                    pheromone_balls[position_agent] += 1
                elif grid[nx, ny] == 2:
                    pheromone_holes[position_agent] += 1
                if grid[nx, ny] in [1, 2]: 
                    for next_move in neighbors:
                        nnx, nny = nx + next_move[0], ny + next_move[1]
                        if (0 <= nnx < n and 0 <= nny < n) and (nnx, nny) != position_agent:
                            if grid[nx, ny] == 1:
                                pheromone_balls[nnx, nny] += 1
                            elif grid[nx, ny] == 2:
                                pheromone_holes[nnx, nny] += 1

        if has_ball:
            next_move, found = find_target(position_agent, grid, pheromone_holes, 2, visited)
        else:
            next_move, found = find_target(position_agent, grid, pheromone_balls, 1, visited)

        if next_move:
            visited.add(next_move)
            if found:
                if has_ball:
                    # place the ball in the hole
                    has_ball = False
                    base_grid[next_move] = 0  # remove the hole
                    grid[next_move] = 0 
                    # reset pheromones around the hole
                    for move in neighbors:
                        nx, ny = next_move[0] + move[0], next_move[1] + move[1]
                        if 0 <= nx < n and 0 <= ny < n:
                            pheromone_holes[nx, ny] = 0
                else:
                    # pick up the ball
                    has_ball = True
                    base_grid[next_move] = 0  # remove the ball
                    grid[next_move] = 0 
                    # reset pheromones around the ball
                    for move in neighbors:
                        nx, ny = next_move[0] + move[0], next_move[1] + move[1]
                        if 0 <= nx < n and 0 <= ny < n:
                            pheromone_balls[nx, ny] = 0
            position_agent = move_agent(position_agent, next_move, grid, base_grid, has_ball)
            move_count += 1

        print_grid(grid)
        print(f'Moves left: {max_moves - move_count}')

main()
