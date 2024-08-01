import gym
from gym import spaces
import numpy as np
import pygame

class SquareMapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_size, num_users, num_agents, user_velocity, agent_velocity, reward_threshold):
        super(SquareMapEnv, self).__init__()

        self.map_size = map_size
        self.num_users = num_users
        self.num_agents = num_agents
        self.user_velocity = user_velocity
        self.agent_velocity = agent_velocity
        self.reward_threshold = reward_threshold

        self.directions = [0, 45, 90, 135, 180, 225, 270, 315]
        self.observation_space = spaces.Box(low=0, high=map_size, shape=(num_users + num_agents, 2), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([len(self.directions)] * self.num_agents)

        self.users = self._init_users()
        self.user_angles = self._init_user_angles()
        self.agents = self._init_agents()

        self.viewer = None
        self.screen = None

    def _init_users(self):
        return np.random.uniform(0, self.map_size, (self.num_users, 2))

    def _init_user_angles(self):
        return np.random.uniform(0, 360, self.num_users)

    def _init_agents(self):
        return np.random.uniform(0, self.map_size, (self.num_agents, 2))

    def reset(self):
        self.users = self._init_users()
        self.user_angles = self._init_user_angles()
        self.agents = self._init_agents()
        return np.vstack((self.users, self.agents))

    def step(self, action):
        self.agent_angles = np.array([self.directions[a] for a in action])
        
        user_dx = self.user_velocity * np.cos(np.radians(self.user_angles))
        user_dy = self.user_velocity * np.sin(np.radians(self.user_angles))
        new_user_positions = self.users + np.stack((user_dx, user_dy), axis=1)
        user_hit_boundary = (new_user_positions[:, 0] <= 0) | (new_user_positions[:, 0] >= self.map_size) | \
                            (new_user_positions[:, 1] <= 0) | (new_user_positions[:, 1] >= self.map_size)
        self.user_angles[user_hit_boundary] = np.random.uniform(0, 360, user_hit_boundary.sum())
        self.users[~user_hit_boundary] = new_user_positions[~user_hit_boundary]
        self.users = np.clip(self.users, 0, self.map_size)

        agent_dx = self.agent_velocity * np.cos(np.radians(self.agent_angles))
        agent_dy = self.agent_velocity * np.sin(np.radians(self.agent_angles))
        new_agent_positions = self.agents + np.stack((agent_dx, agent_dy), axis=1)
        agent_hit_boundary = (new_agent_positions[:, 0] <= 0) | (new_agent_positions[:, 0] >= self.map_size) | \
                             (new_agent_positions[:, 1] <= 0) | (new_agent_positions[:, 1] >= self.map_size)
        self.agents[~agent_hit_boundary] = new_agent_positions[~agent_hit_boundary]
        self.agents = np.clip(self.agents, 0, self.map_size)

        reward = 0
        for user in self.users:
            distances = np.linalg.norm(self.agents - user, axis=1)
            if np.any(distances < self.reward_threshold):
                reward += 1
            else:
                reward -= 1

        done = False
        info = {}

        return np.vstack((self.users, self.agents)), reward, done, info

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.map_size * 10, self.map_size * 10))

        self.screen.fill((255, 255, 255))

        for user in self.users:
            pygame.draw.circle(self.screen, (0, 0, 255), (int(user[0] * 10), int(user[1] * 10)), 5)

        for agent in self.agents:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(agent[0] * 10), int(agent[1] * 10)), 5)
            pygame.draw.circle(self.screen, (255, 0, 0), (int(agent[0] * 10), int(agent[1] * 10)), int(self.reward_threshold * 10), 1)

        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

# Usage
env = SquareMapEnv(map_size=50, num_users=50, num_agents=3, user_velocity=0.01, agent_velocity=0.03, reward_threshold=3.0)
env.reset()
env.render()

# Example loop to control agent actions and see the environment
try:
    while True:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        env.render()
except KeyboardInterrupt:
    env.close()
