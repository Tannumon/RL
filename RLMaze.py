import pygame
import sys
import random
import matplotlib.pyplot as plt
import numpy as np


# Initialize Pygame
pygame.init()

# Set up maze dimensions and cell size
maze_width, maze_height = 1000, 900
cell_size = 100  # Adjust this value for larger or smaller cells

# Set up colors
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0, 128)  # Adding alpha for transparency



# Set up maze layout (0 represents walls, 1 represents paths)
maze_layout = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
]

num_rows = len(maze_layout)
num_cols = len(maze_layout[0])


def isTerminatingCell(curRow, curCol):
    if rewards[curRow, curCol] == -1:
        return False
    else: 
        return True

# Epsilon Greedy Algorithm
def nextAction(curRow, curCol, epsilon):
    if (np.random.random() < epsilon):
        return (np.argmax(qValues[curRow, curCol])) # choose the most optimal action
    else:
        return np.random.randint(4) # choose a random action the other times to explore the environment
    
def nextLocation(curRow, curCol, actionidx):
    newRow = curRow
    newCol = curCol
    if ((moves[actionidx] == 'up') and (curRow > 0)):
        newRow -= 1
    elif ((moves[actionidx] == 'right') and (curCol < mazeCol - 1)):
        newCol += 1
    elif ((moves[actionidx] == 'down') and (curRow < mazeRow - 1)):
        newRow += 1
    elif ((moves[actionidx] == 'left') and (curCol > 0)):
        newCol -= 1
    return newRow, newCol


def shortestPath():
    curRow, curCol = 0, 0
    shortest_path = []
    shortest_path.append([curRow, curCol])
    #continue moving along the path until we reach the goal (i.e., the item packaging location)
    while not isTerminatingCell(curRow, curCol):
        #get the best action to take
        action_index = nextAction(curRow, curCol, 1.)
        #move to the next location on the path, and add the new location to the list
        curRow, curCol = nextLocation(curRow, curCol, action_index)
        shortest_path.append([curRow, curCol])
    return shortest_path

mazeRow = 9
mazeCol = 10

qValues = np.zeros((mazeRow, mazeCol, 4))

# corresponds to action code: 0 = up, 1 = right, 2 = down, 3 = left
moves = ["up", "right", "down", "left"] 

rewards = np.full((mazeRow, mazeCol), -100)
rewards[8, 9] = 100 # Final Goal Cheese

# Making the path where Jerry can go
aisle = {}
aisle[0] = [i for i in range(0, 10)]
aisle[1] = [0, 3, 4, 5, 6, 7, 8, 9]
aisle[2] = [i for i in range(0,10)]
aisle[3] = [0, 1, 7, 9]
aisle[4] = [0, 1, 2, 3, 5, 6, 7, 9]
aisle[4].append(0)
aisle[5] = [0, 4, 5, 6, 7, 8, 9]
aisle[6] = [0, 1, 2, 4]
aisle[7] = [0, 1, 2, 4, 5, 6, 7, 8, 9]
aisle[8] = [0, 1, 2, 4, 5, 6, 7, 8]

#rewards for aisle
for row in range(0, 9):
  for col in aisle[row]:
    rewards[row, col] = -1.

# Set up Pygame window
screen = pygame.display.set_mode((maze_width, maze_height))
pygame.display.set_caption("Maze with Q-values")

# Set up clock to control frames per second
clock = pygame.time.Clock()

# Set up Q-values for each cell (initialized to 0 for each action)
q_values = [[{a: 0 for a in range(4)} for _ in range(num_cols)] for _ in range(num_rows)]

rewards = np.full((mazeRow, mazeCol), -100)
rewards[8, 9] = 100 # Final Goal 
#rewards for aisle
for row in range(0, 9):
  for col in aisle[row]:
    rewards[row, col] = -1.

# Set up agent position and direction
agent_row, agent_col = 0, 0
agent_direction = 0  # 0: up, 1: right, 2: down, 3: left

# Variables for plotting
episode_rewards = []
moves_per_episode = []

# Main game loop
# for x in range(500):
while not isTerminatingCell(row, col):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Agent takes a random action
    action = random.choice(range(4))

    # Update Q-value for the current state-action pair
    reward = -0.1  # Example reward (you can modify this based on your scenario)
    q_values[agent_row][agent_col][action] += reward

    # Move the agent based on the chosen action
    if action == 0 and agent_row > 0 and maze_layout[agent_row - 1][agent_col] == 1:
        agent_row -= 1
    elif action == 1 and agent_col < num_cols - 1 and maze_layout[agent_row][agent_col + 1] == 1:
        agent_col += 1
    elif action == 2 and agent_row < num_rows - 1 and maze_layout[agent_row + 1][agent_col] == 1:
        agent_row += 1
    elif action == 3 and agent_col > 0 and maze_layout[agent_row][agent_col - 1] == 1:
        agent_col -= 1

    # Check if the agent reached the end or hit a wall
    if (agent_row, agent_col) == (num_rows - 1, num_cols - 1) or maze_layout[agent_row][agent_col] == 0:
        # Reset the agent to the starting position
        agent_row, agent_col = 0, 0
        episode_rewards.append(sum(sum(q_values[row][col].values()) for row in range(num_rows) for col in range(num_cols)))
        moves_per_episode.append(len(episode_rewards))

    # Draw maze
    screen.fill(white)
    for row in range(num_rows):
        for col in range(num_cols):
            if maze_layout[row][col] == 0:
                pygame.draw.rect(screen, black, (col * cell_size, row * cell_size, cell_size, cell_size))
            else:
                # Draw Q-values in each cell
                for i, action in enumerate(range(4)):
                    q_str = f"{q_values[row][col][action]:.2f}"
                    color = (0, 255 - i * 30, 0)
                    font = pygame.font.SysFont(None, 18)
                    text = font.render(q_str, True, color)
                    screen.blit(text, (col * cell_size + i * 15, row * cell_size + i * 15))

    # Draw agent
    pygame.draw.rect(screen, green, (agent_col * cell_size, agent_row * cell_size, cell_size, cell_size))

    # Plot rewards and moves per episode
    plt.figure(1)
    plt.subplot(211)
    plt.plot(moves_per_episode)
    plt.title('Number of Moves vs. Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Moves')

    plt.subplot(212)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards vs. Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')

    plt.tight_layout()
    plt.draw()

    pygame.display.flip()
    clock.tick(10)  # Increase the speed for visualization
    plt.pause(0.1)  # Allow time for the plots to update

    plt.clf()  # Clear the plots for the next iteration
