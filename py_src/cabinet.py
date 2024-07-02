import sys
sys.path.append('/home/kist/franka_cabinet')
import numpy as np
import torch
import gym
import argparse
import json
import os
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy
import fr3Env
import tools
import torch.nn as nn
import torch.nn.functional as F

''' Only for Rendering '''
import mujoco
import mujoco.viewer

''' for Tensorboard '''
import time
import psutil
import datetime
import subprocess
# import torch
import torchvision
from tensorboard import program
import webbrowser
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

HEURISTIC = 0
RL = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_time = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
writer = SummaryWriter(f"../runs/franka_cabinet/{current_time}")
# writer.add_text("Initialization", "Writer Initialized")
# time.sleep(5)
# tensorboard_cmd = ['tensorboard', '--logdir', f"../runs/franka_cabinet/{current_time}"]
# tensorboard_process = subprocess.Popen(tensorboard_cmd)
print(f"{YELLOW}[TENSORBOARD]{RESET} The data will be saved in {YELLOW}../runs/franka_cabinet/{current_time}{RESET} directory!")


tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', f"../runs/franka_cabinet/{current_time}", '--port', '6010'])
url = tb.launch()
webbrowser.open_new(url)

def main(PATH, TRAIN, OFFLINE, RENDERING):
    env = fr3Env.cabinet_env()
    env.env_rand = False
    env.rendering = RENDERING
    PLANNING_MODE = RL
    # TRAIN = True
    if OFFLINE == 0:
        IsDemo = False
    else:
        IsDemo = True
    env.train = TRAIN
    max_timesteps = 1e6
    max_episode = 1e4
    batch_size = 16
    policy_kwargs = dict(n_critics=5, n_quantiles=25)
    save_freq = 1e2
    models_dir = PATH
    pretrained_model_dir = models_dir + "3.0/" # 6.0 : 6.4 , 5: 5.7
    # pretrained_model_dir = models_dir + "10.0/" # 6.0 : 6.4 , 5: 5.7
    episode_data = []
    save_flag = False

    state_dim = env.observation_space.shape
    action_dim1 = env.rotation_action_space.shape[0]
    action_dim2 = env.force_action_space.shape[0]

    replay_buffer1 = structures.ReplayBuffer((env.len_hist,state_dim), action_dim1)
    actor1 = Actor(state_dim, action_dim1, IsDemo).to(DEVICE)

    critic1 = Critic(state_dim, action_dim1, policy_kwargs["n_quantiles"], policy_kwargs["n_critics"],IsDemo).to(DEVICE)
    critic_target1 = copy.deepcopy(critic1).to(DEVICE)

    trainer1 = Trainer(actor=actor1,
                      critic=critic1,
                      critic_target=critic_target1,
                      top_quantiles_to_drop=2,
                      discount=0.99,
                      tau=0.005,
                      target_entropy=-np.prod(env.rotation_action_space.shape).item())

    replay_buffer2 = structures.ReplayBuffer((env.len_hist,state_dim), action_dim2)
    actor2 = Actor(state_dim, action_dim2,IsDemo).to(DEVICE)

    critic2 = Critic(state_dim, action_dim2, policy_kwargs["n_quantiles"], policy_kwargs["n_critics"],IsDemo).to(DEVICE)
    critic_target2 = copy.deepcopy(critic2).to(DEVICE)

    trainer2 = Trainer(actor=actor2,
                       critic=critic2,
                       critic_target=critic_target2,
                       top_quantiles_to_drop=2,
                       discount=0.99,
                       tau=0.005,
                       target_entropy=-np.prod(env.force_action_space.shape).item())

    if IsDemo:
        print(IsDemo)
        print(OFFLINE)
        replay_buffer1_expert = structures.ReplayBuffer((env.len_hist, state_dim), action_dim1)
        replay_buffer2_expert = structures.ReplayBuffer((env.len_hist, state_dim), action_dim2)

        replay_buffer1_expert.load("./log/expert/replay_buffer_rotation.plk")
        replay_buffer2_expert.load("./log/expert/replay_buffer_force.plk")
    episode_return_rotation = 0
    episode_return_force = 0
    episode_timesteps = 0
    episode_num = 0

    if TRAIN:
        state = env.reset(PLANNING_MODE)
        pretrained_model_dir1 = pretrained_model_dir+"rotation/"
        pretrained_model_dir2 = pretrained_model_dir+"force/"
        # trainer1.load(pretrained_model_dir1)
        # trainer2.load(pretrained_model_dir2)

        actor1.train()
        actor2.train()
        
        force_gains = []
        
        for t in range(int(max_timesteps)):

            action_rotation = actor1.select_action(state)
            action_force = actor2.select_action(state)
            # print(f"{CYAN}c/action_rotation: {RESET}", action_rotation)
            # print(f"{CYAN}c/action_force:    {RESET}", action_force)
            # print(f"{CYAN}c/time_step:       {RESET}", t)

            # action_rotation = np.array([0, 0])
            # if t < 80:
            #     action_force = 0.1
            # else:
            #     action_force = 0
                
            next_state, reward_rotation, reward_force, done, _ = env.step(action_rotation, action_force)
            print(f"{CYAN}c/env.step-----    {RESET}")

            episode_timesteps += 1

            replay_buffer1.add(state, action_rotation, next_state, reward_rotation, done)
            replay_buffer2.add(state, action_force, next_state, reward_force, done)

            state = next_state
            episode_return_rotation += reward_rotation
            episode_return_force += reward_force

            episode_return_rotation_tb = torch.tensor(episode_return_rotation, dtype=torch.float32)
            episode_return_force_tb = torch.tensor(episode_return_force, dtype=torch.float32)
            # episode_action_force = torch.tensor(action_force, dtype=torch.float32)
            # episode_action_rotation = torch.tensor(action_rotation, dtype=torch.float32)
            force_gain_tb = torch.tensor(env.force_gain, dtype=torch.float32)
            force_gains.append(force_gain_tb.item())
            
            print("force_gain: ", force_gain_tb)
            
            # writer.add_scalar('Action/Rotation', episode_action_rotation, t+1)
            # writer.add_scalar('Action/Force', force_gain_tb, t+1)
            writer.add_scalar('Action/Force', force_gain_tb, episode_timesteps + 1)

            # Train agent after collecting sufficient data
            if t >= batch_size:
                if IsDemo:
                    trainer1.train_with_demo(replay_buffer1, replay_buffer1_expert, batch_size)
                    trainer2.train_with_demo(replay_buffer2, replay_buffer2_expert, batch_size)

                else:
                    trainer1.train(replay_buffer1, batch_size)
                    trainer2.train(replay_buffer2, batch_size)
            
            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                    f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
                
                # TensorBoard에 로깅
                
                writer.add_scalar('Reward/Rotation', episode_return_rotation_tb, episode_num + 1)
                writer.add_scalar('Reward/Force', episode_return_force_tb, episode_num + 1)

                # Reset environment
                state = env.reset(PLANNING_MODE)
                episode_data.append([episode_num, episode_timesteps, episode_return_rotation, episode_return_force])
                episode_return_rotation = 0
                episode_return_force = 0
                episode_timesteps = 0
                episode_num += 1
                
                force_gain_tb = 0
                force_gains = []
            
            if (env.episode_number+ 1) % save_freq == 0:
                save_flag = True
                
            if save_flag:
                path1 = models_dir + str((env.episode_number + 1) // save_freq) + "/rotation/"
                path2 = models_dir + str((env.episode_number + 1) // save_freq) + "/force/"
                os.makedirs(path1, exist_ok=True)
                os.makedirs(path2, exist_ok=True)
                if not os.path.exists(path1):
                    os.makedirs(path1)
                if not os.path.exists(path2):
                    os.makedirs(path2)
                trainer1.save(path1)
                trainer2.save(path2)

                np.save(models_dir + "reward.npy", episode_data)
                save_flag = False
        writer.close() # TensorBoard writer 종료
                
            # if i % 100 == 99:
            #     print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            #     writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
            #     running_loss = 0.0

    else:
        pretrained_model_dir1 = pretrained_model_dir+"rotation/"
        pretrained_model_dir2 = pretrained_model_dir+"force/"
        trainer1.load(pretrained_model_dir1)
        actor1.eval()
        critic1.eval()
        actor1.training = False

        trainer2.load(pretrained_model_dir2)
        actor2.eval()
        critic2.eval()
        actor2.training = False

        # reset_agent.training = False
        num_ep = 16
        force_data = []
        # env.episode_number = 3
        # df = pd.read_csv("/home/kist-robot2/Downloads/obs_real.csv")
        # states = df.to_numpy(dtype=np.float32)
        for _ in range(num_ep):
            state = env.reset(PLANNING_MODE)
            done = False
            step_cnt = 0
            episode_return_rotation = 0
            episode_return_force = 0

            while not done:

                step_cnt += 1
                action_rotation = actor1.select_action(state)
                action_force = actor2.select_action(state)

                next_state, reward_rotation, reward_force, done, _ = env.step(action_rotation, action_force)
                force_data.append(env.force)

                state = next_state
                episode_return_rotation += reward_rotation
                episode_return_force += reward_force

            # np.save("./data/torque_hybrid.npy", env.torque_data)
            # print(
            #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
            #     f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
            print(
                f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
            print("time:",env.time_done, "  contact:",env.contact_done, "  bound:",env.bound_done,
                  "  goal:", env.goal_done)
            print(env.cabinet1_angle)
            print(env.obs_omega[0])

            fig, axs = plt.subplots(3, 2, figsize=(8, 6))
            axs[0, 0].plot([sublist[0] for sublist in env.command_data])
            axs[0, 0].set_title("droll", pad=20)

            axs[0, 1].plot([sublist[1] for sublist in env.command_data])
            axs[0, 1].set_title("dpitch", pad=20)

            axs[1, 0].plot([sublist[2] for sublist in env.command_data])
            axs[1, 0].set_title("roll", pad=20)

            axs[1, 1].plot([sublist[3] for sublist in env.command_data])
            axs[1, 1].set_title("pitch", pad=20)

            axs[2, 0].plot([sublist[4] for sublist in env.command_data])
            axs[2, 0].set_title("force gain", pad=20)

            axs[2, 1].plot([sublist[5] for sublist in env.command_data])
            axs[2, 1].set_title("R force gain", pad=20)
            # plt.plot(force_data,  linestyle='-', color='b')
            # # plt.title(env.friction)
            plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="wo expert demo, first door training, ")
    parser.add_argument("--path", help="data load path", default=" ./log/0702_1/")
    parser.add_argument("--train", help="0->test,  1->train", type=int, default=1)
    parser.add_argument("--render", help="0->no rendering,  1->rendering", type=int, default=1)
    parser.add_argument("--offline", help="0->no offline data,  1->with offline data", type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['description'] = parser.description
    if args.train == 1:
        os.makedirs(args.path, exist_ok=True)
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        with open(args.path+'args_and_notes.json', 'w') as f:
            json.dump(args_dict, f, indent=4)
    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("------------------------------------")

    main(PATH=args.path, TRAIN=args.train, OFFLINE=args.offline, RENDERING=args.render)


''' Only for Rendering '''
# env = fr3Env.cabinet_env()
# env.env_rand = True

# # observation = env.reset(RL)

# with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
#     while viewer.is_running():
#         # step_start = time.time()
        
#         viewer.sync()
        
        