from Hyperparameters import Hyperparameters
from DQL import DQL

if __name__ == '__main__':
    map_size = 20  # Updated to match SquareMapEnv
    num_users = 20
    num_agents = 2
    agent_velocity = 0.1
    
    hyperparameters = Hyperparameters(
        map_size=map_size,
        num_users=num_users,
        num_agents=num_agents,
        agent_velocity=agent_velocity
    )
    
    train = True
    
    # Run
    drl_agent = DQL(hyperparameters, train_mode=train)  # Define the instance
    
    # Train
    if train:
        drl_agent.train()
    else:
        drl_agent.play()
