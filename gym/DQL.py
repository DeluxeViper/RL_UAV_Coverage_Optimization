import torch
import numpy as np
import matplotlib.pyplot as plt

from Hyperparameters import Hyperparameters
from Agent import Agent
from SquareMapEnv import SquareMapEnv

class DQL:
    def __init__(self, hyperparameters: Hyperparameters, train_mode):
        if train_mode:
            render = None
        else:
            render = "human"

        # Store hyperparameters
        self.hp = hyperparameters

        # Load the custom environment with hyperparameters
        self.env = SquareMapEnv(map_size=self.hp.map_size, num_users=self.hp.num_users, num_agents=self.hp.num_agents, agent_velocity=self.hp.agent_velocity)

        # Initiate the Agent
        self.agent = Agent(env=self.env, hyperparameters=self.hp)
        self.max_steps_per_episode = 1000  # Setting maximum steps per episode to simulate 30 seconds

    def feature_representation(self, state):
        """
        Represent the feature of the state.
        For this environment, the state is already a meaningful feature representation.
        """
        return state
    
    def train(self): 
        """
        Train the DQN via DQL
        """
        
        total_steps = 0
        self.collected_rewards = []
        
        # Training loop
        for episode in range(1, self.hp.num_episodes + 1):
            # Sample a new state
            state = self.env.reset()
            ended = False
            step_size = 0
            episode_reward = 0
                                                
            while not ended and step_size < self.max_steps_per_episode:
                # Find action via epsilon greedy 
                action = self.agent.epsilon_greedy(state)
                
                # Find next state and reward
                next_state, reward, ended, info = self.env.step(action)

                # Put it into replay buffer
                self.agent.replay_buffer.store(state, action, next_state, reward, ended) 
                
                if len(self.agent.replay_buffer) > self.hp.batch_size:
                    # Use apply_SGD implementation to update the online DQN
                    self.agent.apply_SGD(ended)
                    
                    # Update target-network weights
                    if total_steps % self.hp.targetDQN_update_rate == 0:
                        # Copy the online DQN into the Target DQN
                        self.agent.update_target()

                state = next_state
                episode_reward += reward
                step_size += 1

                # Terminate if cumulative reward is below -100
                if info["cumulative_reward"] < -100:
                    ended = True
                            
            self.collected_rewards.append(episode_reward)                     
            total_steps += step_size
                                                                           
            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()
                            
            # Print Results of the Episode
            printout = (f"Episode: {episode}, "
                        f"Total Time Steps: {total_steps}, "
                        f"Trajectory Length: {step_size}, "
                        f"Sum Reward of Episode: {episode_reward:.2f}, "
                        f"Epsilon: {self.agent.epsilon:.2f}")
            print(printout)
        
        self.agent.save(self.hp.save_path + '.pth')
        self.plot_learning_curves()
                                                                    
    def play(self):  
        """
        Play with the learned policy
        You can only run it if you already have trained the DQN and saved its weights as .pth file
        """
           
        # Load the trained DQN
        self.agent.onlineDQN.load_state_dict(torch.load(self.hp.RL_load_path, map_location=torch.device(self.agent.device)))
        self.agent.onlineDQN.eval()
        
        # Playing 
        for episode in range(1, self.hp.num_test_episodes + 1):         
            state = self.env.reset()
            ended = False
            step_size = 0
            episode_reward = 0
                                                           
            while not ended and step_size < self.max_steps_per_episode:
                # Act greedy and find action
                action = self.agent.greedy(state)
                
                next_state, reward, ended, info = self.env.step(action)
                                
                state = next_state
                episode_reward += reward
                step_size += 1
                                                                                                                       
            # Print Results of Episode            
            printout = (f"Episode: {episode}, "
                        f"Steps: {step_size}, "
                        f"Sum Reward of Episode: {episode_reward:.2f}, ")
            print(printout)
        
    def plot_learning_curves(self):
        # Calculate the Moving Average over last 100 episodes
        moving_average = np.convolve(self.collected_rewards, np.ones(100) / 100, mode='valid')
        
        plt.figure()
        plt.title("Reward")
        plt.plot(self.collected_rewards, label='Reward', color='gray')
        plt.plot(moving_average, label='Moving Average', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Save the figure
        plt.savefig(f'./Reward_vs_Episode.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()
        
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_list, label='Loss', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Training Loss")
        
        # Save the figure
        plt.savefig(f'./Learning_Curve.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
