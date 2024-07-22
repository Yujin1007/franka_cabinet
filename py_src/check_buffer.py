import pickle
import numpy as np

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

# Load the .pkl file
pkl_file_path = "./log/expert/replay_buffer_force1.pkl"  # 파일 경로를 여기에 입력합니다.

with open(pkl_file_path, "rb") as file:
    buffer_data = pickle.load(file)

# 데이터 타입 확인
print("Data type:", type(buffer_data))

# 데이터가 딕셔너리 형태인 경우, 키 목록과 각 키의 첫 5개 항목을 출력
if isinstance(buffer_data, dict):
    for key, value in buffer_data.items():
        print(f"Key: {key}")
        print("First 5 items:", value[:5])
elif isinstance(buffer_data, list):
    # 데이터가 리스트인 경우, 첫 5개 항목 출력
    print("First 5 items:", buffer_data[:5])
else:
    # 기타 데이터 타입의 경우, 전체 데이터를 출력 (일부 데이터만 출력되도록 설정)
    print("Data:", buffer_data)

# 주요 속성 값 출력
attributes = ['state', 'action', 'reward', 'next_state', 'not_done']
for attr in attributes:
    if hasattr(buffer_data, attr):
        value = getattr(buffer_data, attr)
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # print(f"{YELLOW}Value of {attr} (first 5 items):\n{RESET}", value[:5])
            pass
        else:
            print(f"{YELLOW}Value of {attr}:\n{RESET}", value)

# action 값만 추출하여 출력
if hasattr(buffer_data, 'action'):
    actions = getattr(buffer_data, 'action')
    if isinstance(actions, list) or isinstance(actions, np.ndarray):
        print(f"{YELLOW}Value of action (first 5 items):\n{RESET}", actions[:1000])
    else:
        print(f"{YELLOW}Value of action:\n{RESET}", actions)