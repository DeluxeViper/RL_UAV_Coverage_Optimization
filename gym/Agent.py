import torch
import numpy as np
import torch.nn as nn
import gym
from collections import deque

class Replay_Buffer:
    """
    Experience Replay Buffer to store experiences
    """
    def __init__(self, size, device):
        self.device = device
        self.size = size  # size of the buffer
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.terminals = deque(maxlen=size)

    def store(self, state, action, next_state, reward, terminal):
        """
        Store experiences to their respective queues
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def sample(self, batch_size):
        """
        Sample from the buffer
        """
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=self.device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor([self.terminals[i] for i in indices], dtype=torch.bool, device=self.device)

        return states, actions, next_states, rewards, terminals

    def __len__(self):
        return len(self.terminals)

class DQN(nn.Module):
    """
    The Deep Q-Network (DQN) model
    Implement the MLP described in the assignment
    """
    def __init__(self, num_actions, feature_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(feature_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Agent:
    """
    Implementing Agent DQL Algorithm
    """
    def __init__(self, env: gym.Env, hyperparameters, device=None):
        # Some Initializations
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters

        self.num_actions = np.prod(env.action_space.nvec)  # Total number of discrete actions

        self.epsilon = 0.99
        self.loss_list = []
        self.current_loss = 0
        self.episode_counts = 0

        self.action_space  = env.action_space
        self.feature_space = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.replay_buffer = Replay_Buffer(self.hp.buffer_size, device=self.device)

        # Initiate the online and Target DQNs
        self.onlineDQN = DQN(self.num_actions, self.feature_space).to(self.device)
        self.targetDQN = DQN(self.num_actions, self.feature_space).to(self.device)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.onlineDQN.parameters(), lr=self.hp.learning_rate)

    def epsilon_greedy(self, state):
        """
        Implement epsilon-greedy policy
        """
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
                q_values = self.onlineDQN(state)
                action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(np.prod(self.action_space.nvec))
        return np.unravel_index(action, self.action_space.nvec)

    def greedy(self, state):
        """
        Implement greedy policy
        """
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
            q_values = self.onlineDQN(state)
            action = torch.argmax(q_values).item()
        return np.unravel_index(action, self.action_space.nvec)

    def apply_SGD(self, ended):
        """
        Train DQN
        ended (bool): Indicates whether the episode meets a terminal state or not. If ended,
        calculate the loss of the episode.
        """
        if len(self.replay_buffer) < self.hp.batch_size:
            return

        states, actions, next_states, rewards, terminals = self.replay_buffer.sample(self.hp.batch_size)

        # Ensure actions is a 2D tensor
        actions = actions.view(-1, 2)  # Remove the extra dimension and flatten to [batch_size, num_agents]
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)

        # Compute Q values for current states
        Q_hat = self.onlineDQN(states.view(states.size(0), -1))

        # Flatten the multi-discrete action indices
        action_indices = actions[:, 0] * len(self.action_space.nvec) + actions[:, 1]
        action_indices = action_indices.unsqueeze(1).long().to(self.device)

        # Gather the Q values for the taken actions
        Q_hat = Q_hat.gather(1, action_indices)

        with torch.no_grad():
            # Compute Q values for next states using the target network
            next_target_q_value = self.targetDQN(next_states.view(next_states.size(0), -1)).max(1)[0].unsqueeze(1)

        # Set Q values for terminal states to 0
        next_target_q_value[terminals] = 0
        y = rewards + (self.hp.discount_factor * next_target_q_value * (~terminals))

        loss = self.loss_function(Q_hat, y)

        self.current_loss += loss.item()
        self.episode_counts += 1

        if ended:
            episode_loss = self.current_loss / self.episode_counts
            self.loss_list.append(episode_loss)
            self.current_loss = 0
            self.episode_counts = 0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.onlineDQN.parameters(), 2)
        self.optimizer.step()

    def update_target(self):
        """
        Update the target network
        """
        self.targetDQN.load_state_dict(self.onlineDQN.state_dict())

    def update_epsilon(self):
        """
        Reduce epsilon by the decay factor
        """
        self.epsilon = max(0.01, self.epsilon * self.hp.epsilon_decay)

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extension
        This can be used for later test of the trained agent
        """
        torch.save(self.onlineDQN.state_dict(), path)
