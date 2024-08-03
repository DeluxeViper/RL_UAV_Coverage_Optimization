class Hyperparameters():
    def __init__(self, map_size, num_users=50, num_agents=3, agent_velocity=0.03):
        self.map_size = map_size
        self.num_users = num_users
        self.num_agents = num_agents
        self.agent_velocity = agent_velocity
        self.RL_load_path = f'./{map_size}x{map_size}/final_weights.pth'
        self.save_path = f'./{map_size}x{map_size}/final_weights'
        self.learning_rate = 1e-5  # Reduced learning rate
        self.discount_factor = 0.95
        self.batch_size = 64
        self.targetDQN_update_rate = 10
        self.num_episodes = 200  # Increased number of episodes
        self.num_test_episodes = 10
        self.epsilon_decay = 0.995
        self.buffer_size = 10000

    def change(self, map_size=None, num_users=None, num_agents=None, agent_velocity=None,
               batch_size=None, learning_rate=None, num_episodes=None, epsilon_decay=None):
        '''
        This method can change:
        map_size, num_users, num_agents, agent_velocity
        Also can change the following argument if called:
        batch_size, learning_rate, num_episodes, epsilon_decay
        '''
        if map_size is not None:
            self.map_size = map_size
            self.RL_load_path = f'./{map_size}x{map_size}/final_weights.pth'
            self.save_path = f'./{map_size}x{map_size}/final_weights'
        if num_users is not None:
            self.num_users = num_users
        if num_agents is not None:
            self.num_agents = num_agents
        if agent_velocity is not None:
            self.agent_velocity = agent_velocity
        if batch_size is not None:
            self.batch_size = batch_size
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if num_episodes is not None:
            self.num_episodes = num_episodes
        if epsilon_decay is not None:
            self.epsilon_decay = epsilon_decay
