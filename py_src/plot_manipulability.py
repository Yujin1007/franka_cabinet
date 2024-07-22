import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter

def load_npy(file_path):
    data_npy = np.load(file_path)
    data_pd = pd.DataFrame(data_npy)
    return data_pd

def preprocess_df(data, smoothing=100, end=None):
    data.columns = ['qpos', 'Manipulability']
    data['Cumulative_qpos'] = data['qpos'].cumsum()
    
    idx = None
    if end is not None:
        min_num = 100000
        for i in range(len(data)):
            dif = end - data['Cumulative_qpos'][i]
            if dif >= 0:
                if dif < min_num:
                    idx = i
                    min_num = dif
            else:
                break
        data = data[:idx]
    
    # data['Reward_r_RollingMean'] = data['Reward_r'].rolling(window=smoothing).mean()
    # data['Reward_r_RollingStd'] = data['Reward_r'].rolling(window=smoothing).std()

    # data['Reward_f_RollingMean'] = data['Reward_f'].rolling(window=smoothing).mean()
    # data['Reward_f_RollingStd'] = data['Reward_f'].rolling(window=smoothing).std()
    return data

smoothing = 1000

# cabinet
file_path1="/home/kist/franka_cabinet/py_src/data/cabinet/HEURISTIC_MANIPULABILITY.npy"
file_path2="/home/kist/franka_cabinet/py_src/data/cabinet/OURS_MANIPULABILITY.npy"
data1 = preprocess_df(load_npy(file_path1),end=8e5)
data2 = preprocess_df(load_npy(file_path2),end=8e5)

# Plot the OURS_manipulability data
plt.plot(data2['qpos'],
         data2['Manipulability'],
         linestyle='-', color='r', label='Ours', linewidth=2)

# Plot the HEURISTIC_manipulability data
plt.plot(data1['qpos'],
         data1['Manipulability'],
         linestyle='-', color='b', label='Manual', linewidth=2)
ax = plt.gca()
# Set the labels for the axes with custom font size
plt.xlabel('object joint position', fontsize=20)
plt.ylabel('manipulability measure', fontsize=20)

# Set the font size for the ticks on the axes
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.tight_layout()
# Add legend with custom font size
plt.legend(fontsize=20)

plt.show()