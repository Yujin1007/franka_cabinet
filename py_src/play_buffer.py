import pickle
import numpy as np
import fr3Env

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

# Load the .pkl file
pkl_rotation_file_path = "./log/expert/replay_buffer_rotation1.pkl"
pkl_force_file_path = "./log/expert/replay_buffer_force1.pkl"

with open(pkl_rotation_file_path, "rb") as file:
    replay_buffer_rotation = pickle.load(file)
    
with open(pkl_force_file_path, "rb") as file:
    replay_buffer_force = pickle.load(file)
    
HEURISTIC = 0
RL = 1
PLANNING_MODE = RL
    
env = fr3Env.cabinet_env()

states_rotation = replay_buffer_rotation.state
actions_rotation = replay_buffer_rotation.action
rewards_rotation = replay_buffer_rotation.reward
next_states_rotation = replay_buffer_rotation.next_state
not_dones_rotation = replay_buffer_rotation.not_done

states_force = replay_buffer_force.state
actions_force = replay_buffer_force.action
rewards_force = replay_buffer_force.reward
next_states_force = replay_buffer_force.next_state
not_dones_force = replay_buffer_force.not_done

def run_simulation(env, states_rotation, actions_rotation, rewards_rotation, next_states_rotation, not_dones_rotation, 
                   states_force, actions_force, rewards_force, next_states_force, not_dones_force):
    total_reward_rotation = 0
    total_reward_force = 0
    env.reset(PLANNING_MODE)
    
    for i in range(len(states_rotation)):
        state_rotation = states_rotation[i]
        action_rotation = actions_rotation[i]
        reward_rotation = rewards_rotation[i]
        next_state_rotation = next_states_rotation[i]
        not_done_rotation = not_dones_rotation[i]
        
        state_force = states_force[i]
        action_force = actions_force[i]
        reward_force = rewards_force[i]
        next_state_force = next_states_force[i]
        not_done_force = not_dones_force[i]

        env.step(action_rotation, action_force)  # Perform action in the environment

        total_reward_rotation += reward_rotation
        total_reward_force += reward_force
        
        print(not_done_force, not_done_rotation)

        if (not_done_rotation == 0) or (not_done_force == 0):
            break

    # env.close()
    return total_reward_rotation, total_reward_force

# Run simulation
total_reward_rotation, total_reward_force = run_simulation(env, states_rotation, actions_rotation, rewards_rotation, next_states_rotation, not_dones_rotation,
                                                           states_force, actions_force, rewards_force, next_states_force, not_dones_force)
print(f'Total Reward Rotation: {total_reward_rotation}, Total Reward Rotation: {total_reward_force}')