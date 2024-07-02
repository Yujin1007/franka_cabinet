import os
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

# TensorBoard event 파일 로드
default_dir = '../runs/franka_cabinet/'
all_files = sorted(Path(default_dir).iterdir(), key=os.path.getmtime, reverse=True)
latest_folders = [str(f)[len(default_dir):] for f in all_files if f.is_dir()][:5]

print(f"{GREEN}Latest 5 created directory:{RESET}")
for i, folder in enumerate(latest_folders):
    print(f" ({i+1}) [{folder}]")
    folder_n = i

while True:
    folder_num = int(input(f"{YELLOW}Choose your directory number: {RESET}"))
    if folder_num-1 > folder_n:
        print(f"{RED}[ERROR] The directory number {folder_num} does not exist! Please choose again:){RESET}")
    else:
        print(f"[Notice] Your directory path: {default_dir}{latest_folders[folder_num-1]}")
        chosen_dir = f"{default_dir}{latest_folders[folder_num-1]}"
        break

event_files = [f for f in os.listdir(chosen_dir) if f.startswith("events.out.tfevents.")]
if not event_files:
    print(f"{RED}[ERROR] No event files found in {chosen_dir}{RESET}")
    exit()

# Use the first event file found
event_file_path = os.path.join(chosen_dir, event_files[0])
print(f"[Notice] Using event file: {event_file_path}")

event_acc = EventAccumulator(event_file_path)
event_acc.Reload()

# Scalar 데이터 추출
tags = event_acc.Tags()['scalars']
print(f"{GREEN}Available tags:{RESET}")
for i, tag in enumerate(tags):
    print(f" ({i+1}) [{tag}]")
    tag_n = i
print(f" (q) Quit")

while True:
    # 원하는 태그 선택
    tag_num = input(f"{YELLOW}Choose your tag number: {RESET}")
    if tag_num == 'q':
        print(f"[Notice] BYE!:)")
        break
    else:
        tag_num = int(tag_num)
        if tag_num-1 > tag_n:
            print(f"{RED}[ERROR] The tag number {tag_num} does not exist! Please choose again:){RESET}")
        else:
            tag = str(tags[tag_num-1])

            # 데이터 추출
            steps = []
            values = []
            for scalar_event in event_acc.Scalars(tag):
                steps.append(scalar_event.step)
                values.append(scalar_event.value)

            # 데이터 Matplotlib로 그래프로 그리기
            plt.plot(steps, values, label=tag)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(tag)
            plt.legend()
            plt.grid(True)
            print("[Notice] Successfully plotted!")
            plt.show()