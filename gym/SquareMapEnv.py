import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans

class SquareMapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_size, num_users, num_agents, agent_velocity):
        super(SquareMapEnv, self).__init__()

        self.map_size = map_size
        self.num_users = num_users
        self.num_agents = num_agents
        self.agent_velocity = agent_velocity

        self.directions = [0, 45, 90, 135, 180, 225, 270, 315]
        self.observation_space = spaces.Box(low=0, high=map_size, shape=(num_users + num_agents, 2), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([len(self.directions)] * self.num_agents)

        self.users = self._init_users()
        self.user_angles = self._init_user_angles()
        self.user_velocities = self._init_user_velocities()
        self.agents = self._init_agents()

        self.previous_average_min_distance = None
        self.cumulative_reward = 0

        self.viewer = None
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.user_scatter = self.ax.scatter([], [], c='blue')
        self.agent_scatter = self.ax.scatter([], [], c='red')

    def _init_users(self):
        return np.random.uniform(0, self.map_size, (self.num_users, 2))

    def _init_user_angles(self):
        return np.random.uniform(0, 360, self.num_users)

    def _init_user_velocities(self):
        return np.random.uniform(0.01, 0.05, self.num_users)

    def _init_agents(self):
        kmeans = KMeans(n_clusters=self.num_agents, n_init=10, random_state=0).fit(self.users)
        return kmeans.cluster_centers_

    def reset(self):
        self.users = self._init_users()
        self.user_angles = self._init_user_angles()
        self.user_velocities = self._init_user_velocities()
        self.agents = self._init_agents()
        self.previous_average_min_distance = None
        self.cumulative_reward = 0
        return np.vstack((self.users, self.agents))

    def step(self, action):
        self.agent_angles = np.array([self.directions[a] for a in action])

        # Update user positions with individual velocities
        user_dx = self.user_velocities * np.cos(np.radians(self.user_angles))
        user_dy = self.user_velocities * np.sin(np.radians(self.user_angles))
        new_user_positions = self.users + np.stack((user_dx, user_dy), axis=1)
        user_hit_boundary = (new_user_positions[:, 0] <= 0) | (new_user_positions[:, 0] >= self.map_size) | \
                            (new_user_positions[:, 1] <= 0) | (new_user_positions[:, 1] >= self.map_size)
        self.user_angles[user_hit_boundary] = np.random.uniform(0, 360, user_hit_boundary.sum())
        self.user_velocities[user_hit_boundary] = np.random.uniform(0.01, 0.05, user_hit_boundary.sum())
        self.users[~user_hit_boundary] = new_user_positions[~user_hit_boundary]
        self.users = np.clip(self.users, 0, self.map_size)

        # Update agent positions
        agent_dx = self.agent_velocity * np.cos(np.radians(self.agent_angles))
        agent_dy = self.agent_velocity * np.sin(np.radians(self.agent_angles))
        new_agent_positions = self.agents + np.stack((agent_dx, agent_dy), axis=1)
        agent_hit_boundary = (new_agent_positions[:, 0] <= 0) | (new_agent_positions[:, 0] >= self.map_size) | \
                             (new_agent_positions[:, 1] <= 0) | (new_agent_positions[:, 1] >= self.map_size)
        self.agents[~agent_hit_boundary] = new_agent_positions[~agent_hit_boundary]
        self.agents = np.clip(self.agents, 0, self.map_size)

        # Calculate reward based on change in average minimum distance between users and agents
        total_min_distance = 0
        for user in self.users:
            distances = np.linalg.norm(self.agents - user, axis=1)
            min_distance = np.min(distances)
            total_min_distance += min_distance

        average_min_distance = total_min_distance / self.num_users
        reward = 0

        if self.previous_average_min_distance is not None:
            distance_change = average_min_distance - self.previous_average_min_distance
            reward = -distance_change  # Reward is proportional to the negative change in distance

        # Penalize for hitting the boundary
        boundary_penalty = -10 * agent_hit_boundary.sum()
        reward += boundary_penalty

        self.cumulative_reward += reward
        self.previous_average_min_distance = average_min_distance

        # Check if cumulative reward is below -100
        if self.cumulative_reward < -100:
            done = True
        else:
            done = False

        info = {"cumulative_reward": self.cumulative_reward}

        return np.vstack((self.users, self.agents)), reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                self.ax.set_xlim(0, self.map_size)
                self.ax.set_ylim(0, self.map_size)
                self.user_scatter = self.ax.scatter(self.users[:, 0], self.users[:, 1], c='blue')
                self.agent_scatter = self.ax.scatter(self.agents[:, 0], self.agents[:, 1], c='red')
                self.fig.show()
                self.viewer = True

            self.user_scatter.set_offsets(self.users)
            self.agent_scatter.set_offsets(self.agents)
            plt.pause(0.001)  # Small pause to allow the plot to update

    def close(self):
        plt.close(self.fig)
