import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
def load_npy(file_path):
    data_npy = np.load(file_path)
    data_pd = pd.DataFrame(data_npy)
    return data_pd

def preprocess_df(data, smoothing=500, end=None):
    data.columns = ['Episode', 'Total_Steps', 'Reward_r', 'Reward_f']
    data['Cumulative_Steps'] = data['Total_Steps'].cumsum()
    
    idx = None
    
    if end is not None:
        min_num = 100000
        for i in range(len(data)):
            dif = end- data['Cumulative_Steps'][i]
            if dif >= 0:
                if dif < min_num:
                    idx = i
                    min_num = dif
            else:
                break
        if idx is not None:
            data = data[:idx]

    data['Reward_r_RollingMean'] = data['Reward_r'].rolling(window=smoothing).mean()
    data['Reward_r_RollingStd'] = data['Reward_r'].rolling(window=smoothing).std()

    data['Reward_f_RollingMean'] = data['Reward_f'].rolling(window=smoothing).mean()
    data['Reward_f_RollingStd'] = data['Reward_f'].rolling(window=smoothing).std()
    return data


def draw_plot(data1, data2, reward, label1="Ours", label2="RL", figure_number=None, obj=None):
    font_size=18
    if figure_number is not None:
        plt.figure(figure_number, figsize=(5, 4))
        ax = plt.gca()
    type=None
    if reward == "Reward_r":
        title_name = " $\mathcal{S}_R$-Policy Episode Return"
        type="redundant"
    elif reward == "Reward_f":
        title_name = "$\mathcal{S}_F$-Policy Episode Return"
        type="force"
    else:
        raise NameError
    reward_type_mean = reward +"_RollingMean"
    reward_type_std = reward +"_RollingStd"
    ax.fill_between(data1['Cumulative_Steps'],
                     data1[reward_type_mean] - data1[reward_type_std],
                     data1[reward_type_mean] + data1[reward_type_std],
                     color='r', alpha=0.1)
    ax.fill_between(data2['Cumulative_Steps'],
                    data2[reward_type_mean] - data2[reward_type_std],
                    data2[reward_type_mean] + data2[reward_type_std],
                    color='g', alpha=0.1)

    ax.plot(data1['Cumulative_Steps'], data1[reward_type_mean], label=label1, color='r')
    ax.plot(data2['Cumulative_Steps'], data2[reward_type_mean], label=label2, color='g')
    ax.set_xlabel('steps', fontsize=font_size)
    ax.set_ylabel('episode return', fontsize=font_size)

    # Set the font size for the ticks on the axes
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)

    # Add legend with custom font size
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    ax.legend(fontsize=font_size,loc='upper left')
    # ax.set_title(title_name, fontsize=14)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("/home/kist/franka_cabinet/py_src/data/"+obj+"/"+obj+"_"+type+".png")
    plt.savefig("/home/kist/franka_cabinet/py_src/data/"+obj+"/"+obj+"_"+type+".svg")

smoothing = 1000

# valve
# file_path1= '/home/kist-robot2/Franka/franka_valve_new/py_src/log/reward_graph_valve/reward.npy'
# file_path2 = '/home/kist-robot2/Franka/franka_valve_new/py_src/log/single_agent_reward_graph_valve/reward.npy'
# data1 = preprocess_df(load_npy(file_path1))
# data2 = preprocess_df(load_npy(file_path2))
# draw_plot(data1,data2,"Reward_r", figure_number=0,obj="Valve")
# draw_plot(data1,data2,"Reward_f", figure_number=1,obj="Valve")

# handle
# file_path1= '/home/kist-robot2/Franka/franka_valve_new/py_src/log/reward_graph_handle/reward.npy'
# file_path2 = '/home/kist-robot2/Franka/franka_valve_new/py_src/log/single_agent_reward_graph_handle/reward.npy'
# data1 = preprocess_df(load_npy(file_path1))
# data2 = preprocess_df(load_npy(file_path2))
# draw_plot(data1,data2,"Reward_r", figure_number=0,obj="Handle")
# draw_plot(data1,data2,"Reward_f", figure_number=1,obj="Handle")

# cabinet
file_path1="/home/kist/franka_cabinet/py_src/ ./log/0729_rerereward1/reward.npy"
file_path2="/home/kist/franka_cabinet/py_src/ ./log/singlerere/reward.npy"
# file_path1="/home/kist/franka_cabinet/py_src/ ./log/0729_rerereward/reward.npy"
# file_path2="/home/kist/franka_cabinet/py_src/ ./log/singlerere/reward.npy"
data1 = preprocess_df(load_npy(file_path1))
data2 = preprocess_df(load_npy(file_path2))
draw_plot(data1,data2,"Reward_r", figure_number=0, obj="cabinet")
draw_plot(data1,data2,"Reward_f", figure_number=1, obj="cabinet")

plt.show()