import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import random

random.seed(0)

class StaticGridWorld:
    def __init__(self):
        self.size = 100
        self.grid = np.zeros((self.size, self.size))
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.current_state = self.start

        # Define actions: right, down, left, up
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Create irregular obstacle pattern
        self.create_irregular_obstacles()

    def create_irregular_obstacles(self):
        # Generate random obstacles
        for i in range(self.size):
            for j in range(self.size):
                # Ensure we don't place obstacles at start or goal positions
                if (i, j) != self.start and (i, j) != self.goal:
                    if random.random() < 0.3:
                        self.grid[i][j] = 1  # 1 represents obstacle

        # Clear a small area around start and goal to ensure they're accessible
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                start_x, start_y = self.start[0] + dx, self.start[1] + dy
                goal_x, goal_y = self.goal[0] + dx, self.goal[1] + dy

                if 0 <= start_x < self.size and 0 <= start_y < self.size:
                    self.grid[start_x][start_y] = 0
                if 0 <= goal_x < self.size and 0 <= goal_y < self.size:
                    self.grid[goal_x][goal_y] = 0

    def reset(self):
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        next_state = (
            self.current_state[0] + self.actions[action][0],
            self.current_state[1] + self.actions[action][1]
        )

        # Check bounds and obstacles
        if (0 <= next_state[0] < self.size and
                0 <= next_state[1] < self.size and
                self.grid[next_state] != 1):
            self.current_state = next_state

        # Return reward based on new state
        if self.current_state == self.goal:
            return self.current_state, 100, True
        elif self.grid[self.current_state] == 1:
            return self.current_state, -100, True
        else:
            return self.current_state, -1, False


class MDPPolicyIteration:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.values = np.zeros((env.size, env.size))
        self.policy = np.zeros((env.size, env.size), dtype=int)
        self.transition_probs = self.build_transition_matrix()

    def build_transition_matrix(self):
        # Build deterministic transition probabilities
        transitions = {}
        for i in range(self.env.size):
            for j in range(self.env.size):
                if self.env.grid[i, j] == 1:  # Skip obstacles
                    continue

                state = (i, j)
                transitions[state] = {}

                for action in range(4):
                    next_i = i + self.env.actions[action][0]
                    next_j = j + self.env.actions[action][1]

                    # Check bounds and obstacles
                    if (0 <= next_i < self.env.size and
                            0 <= next_j < self.env.size and
                            self.env.grid[next_i, next_j] != 1):
                        transitions[state][action] = {(next_i, next_j): 1.0}
                    else:
                        transitions[state][action] = {state: 1.0}

        return transitions

    def policy_evaluation(self, threshold=1e-4):
        while True:
            delta = 0
            for i in range(self.env.size):
                for j in range(self.env.size):
                    if (i, j) == self.env.goal or self.env.grid[i, j] == 1:
                        continue

                    old_value = self.values[i, j]
                    action = self.policy[i, j]

                    # Calculate new value based on current policy
                    new_value = 0
                    for next_state, prob in self.transition_probs[(i, j)][action].items():
                        if next_state == self.env.goal:
                            reward = 100
                        elif self.env.grid[next_state] == 1:
                            reward = -100
                        else:
                            reward = -1
                        new_value += prob * (reward + self.gamma * self.values[next_state])

                    self.values[i, j] = new_value
                    delta = max(delta, abs(old_value - new_value))

            if delta < threshold:
                break

    def policy_improvement(self):
        policy_stable = True

        for i in range(self.env.size):
            for j in range(self.env.size):
                if (i, j) == self.env.goal or self.env.grid[i, j] == 1:
                    continue

                old_action = self.policy[i, j]
                action_values = np.zeros(4)

                # Calculate value for each action
                for action in range(4):
                    for next_state, prob in self.transition_probs[(i, j)][action].items():
                        if next_state == self.env.goal:
                            reward = 100
                        elif self.env.grid[next_state] == 1:
                            reward = -100
                        else:
                            reward = -1
                        action_values[action] += prob * (reward + self.gamma * self.values[next_state])

                self.policy[i, j] = np.argmax(action_values)

                if old_action != self.policy[i, j]:
                    policy_stable = False

        return policy_stable

    def train(self, max_iterations=100):
        for _ in range(max_iterations):
            self.policy_evaluation()
            if self.policy_improvement():
                break


class QLearning:
    def __init__(self, env, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state])

    def train(self, episodes=10000):
        for _ in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)

                # Q-Learning update
                old_value = self.q_table[state][action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
                self.q_table[state][action] = new_value

                state = next_state


def evaluate_policy(env, policy, num_episodes=100):
    total_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 1000

        while not done and steps < max_steps:
            if isinstance(policy, MDPPolicyIteration):
                action = policy.policy[state]
            else:
                action = np.argmax(policy.q_table[state])

            state, reward, done = env.step(action)
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def visualize_grid(env, policy=None, title="GridWorld Environment"):
    plt.figure(figsize=(12, 12))

    # Create a colormap for the grid
    grid_colors = env.grid.copy()

    # Mark start and goal positions
    grid_colors[env.start] = 2  # Different value for start
    grid_colors[env.goal] = 3  # Different value for goal

    # Create custom colormap
    colors = ['white', 'black', 'green', 'red']
    cmap = plt.cm.colors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(grid_colors, cmap=cmap, norm=norm)

    if policy is not None:
        # Plot policy arrows
        arrow_length = 0.3
        for i in range(env.size):
            for j in range(env.size):
                if env.grid[i, j] == 0:  # Only show arrows in free spaces
                    if isinstance(policy, MDPPolicyIteration):
                        action = policy.policy[i, j]
                    else:
                        action = np.argmax(policy.q_table[(i, j)])

                    dx, dy = env.actions[action]
                    if (i, j) != env.goal:  # Don't show arrow at goal
                        plt.arrow(j, i, dy * arrow_length, dx * arrow_length,
                                  head_width=0.3, color='blue', alpha=0.6)

    # Add legend elements
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Free Space',
                   markerfacecolor='white', markersize=15, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', label='Obstacle',
                   markerfacecolor='black', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='Start',
                   markerfacecolor='green', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='Goal',
                   markerfacecolor='red', markersize=15),
        plt.Line2D([0], [0], color='blue', label='Policy Direction',
                   marker='>', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Run benchmark
env = StaticGridWorld()

# Train and evaluate MDP Policy Iteration
start_time = time.time()
mdp = MDPPolicyIteration(env)
mdp.train()
mdp_time = time.time() - start_time
mdp_mean, mdp_std = evaluate_policy(env, mdp)

# Train and evaluate Q-Learning
start_time = time.time()
ql = QLearning(env)
ql.train()
ql_time = time.time() - start_time
ql_mean, ql_std = evaluate_policy(env, ql)

# Print results
print("\nBenchmark Results:")
print("\nMDP Policy Iteration:")
print(f"Training time: {mdp_time:.2f} seconds")
print(f"Mean reward: {mdp_mean:.2f} ± {mdp_std:.2f}")

print("\nQ-Learning:")
print(f"Training time: {ql_time:.2f} seconds")
print(f"Mean reward: {ql_mean:.2f} ± {ql_std:.2f}")

# Visualize the environment and policies
visualize_grid(env, mdp)
visualize_grid(env, ql)