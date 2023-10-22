# 环境和策略都会使用到的语言相关的 object
from pathlib import Path
import numpy as np


project_dir = Path(__file__).parent.parent
models_dir = project_dir.joinpath('models')

goal_src = 1
goal_dst = 2
policy_language_dim = 14
bert_cont_output_dim = 32


# total 用于编码 onehot embedding
total_template_list = [
        "Push the {} ball {} the {} ball",
        "Can you push the {} ball {} the {} ball",
        "Can you move the {} ball {} the {} ball",
        "Can you keep the {} ball {} the {} ball",
        "Can you help me move the {} ball {} the {} ball",
        "Can you help me keep the {} ball {} the {} ball",
        "Can you help me push the {} ball {} the {} ball",
        "Is the {} ball {} the {} ball",
        "Is there any {} ball {} the {} ball",

        "Move the {} ball {} the {} ball",
        "Keep the {} ball {} the {} ball",
        "The {} ball moves {} the {} ball",
        "The {} ball is being pushed {} the {} ball",
        "The {} ball is pushed {} the {} ball",
        "The {} ball is being moved {} the {} ball",
        "The {} ball is moved {} the {} ball",
        "The {} ball was pushed {} the {} ball",
        "The {} ball was moved {} the {} ball",
]
train_template_list = [
        "Push the {} ball {} the {} ball",                  # 训练 policy language 只用这 9 个模板
        "Can you push the {} ball {} the {} ball",
        "Can you help me push the {} ball {} the {} ball",
        "Is the {} ball {} the {} ball",
        "Is there any {} ball {} the {} ball",
        "The {} ball moves {} the {} ball",
        "The {} ball is being pushed {} the {} ball",
        "The {} ball is pushed {} the {} ball",
        "The {} ball was moved {} the {} ball",
]
test_template_list = [
        "Move the {} ball {} the {} ball",                 # 后面的类似模板用于训练不同语句的 instruction-following policy
        "Keep the {} ball {} the {} ball",
        "Can you move the {} ball {} the {} ball",
        "Can you keep the {} ball {} the {} ball",
        "Can you help me move the {} ball {} the {} ball",
        "Can you help me keep the {} ball {} the {} ball",
        "The {} ball is being moved {} the {} ball",
        "The {} ball is moved {} the {} ball",
        "The {} ball was pushed {} the {} ball",
]
error_template_list = [
        "Push the {} ball {} the {} ball",                  # 训练 policy language 只用这 9 个模板
        "Can you push the {} ball {} the {} ball",
        "Can you help me push the {} ball {} the {} ball",
        "Is the {} ball {} the {} ball",
        "Is there any {} ball {} the {} ball",
        "The {} ball moves {} the {} ball",
        "The {} ball is being pushed {} the {} ball",
        "The {} ball is pushed {} the {} ball",
        "The {} ball was moved {} the {} ball",
]

orientation_list = [
        "behind", 
        "to the left of",
        "in front of", 
        "to the right of",
]
total_orientation_list = [
        "behind", "to the behind of", "in behind of",
        "to the left of", "left", "in left of",
        "front", "to the front of", "in front of",  # 后2行最右边才没有语法错误, 用以编码 onehot
        "right", "in right of", "to the right of",
]

color_list = ['red', 'blue', 'green', 'purple', 'cyan']

color_pair2color_idx = {}
idx = 0
for i in range(len(color_list)):
        for j in range(len(color_list)):
                if j != i:
                        color_pair2color_idx[(i, j)] = idx
                        idx += 1
color_idx2color_pair = dict(zip(color_pair2color_idx.values(), color_pair2color_idx.keys()))

second_total_template_list = [
        " and push the {} ball {}",                  # 训练policy language只用前9个模板
        " and can you push the {} ball {}",
        " and can you help me push the {} ball {}",
        " and is the {} ball pushed {}",
        " and is there any {} ball pushed {}",
        " and the {} ball moves {}",
        " and the {} ball is being pushed {}",
        " and the {} ball is pushed {}",
        " and the {} ball was moved {}",

        " and move the {} ball {}",                 # 后面的类似模板用于训练不同语句的instruction-following policy
        " and keep the {} ball {}",
        " and can you move the {} ball {}",
        " and can you keep the {} ball {}",
        " and can you help me move the {} ball",
        " and can you help me keep the {} ball",
        " and the {} ball was pushed {}",
        " and the {} ball is being moved {}",
        " and the {} ball is moved {}",
]
second_train_template_list = [
        " and push the {} ball {}",
        " and can you push the {} ball {}",
        " and can you help me push the {} ball {}",
        " and is the {} ball pushed {}",
        " and is there any {} ball pushed {}",
        " and the {} ball moves {}",
        " and the {} ball is being pushed {}",
        " and the {} ball is pushed {}",
        " and the {} ball was moved {}",
]
second_test_template_list = [
        " and move the {} ball {}",
        " and keep the {} ball {}",
        " and can you move the {} ball {}",
        " and can you keep the {} ball {}",
        " and can you help me move the {} ball",
        " and can you help me keep the {} ball",
        " and the {} ball was pushed {}",
        " and the {} ball is being moved {}",
        " and the {} ball is moved {}",
]
second_error_template_list = [
        " and push the {} ball {}",
        " and can you push the {} ball {}",
        " and can you help me push the {} ball {}",
        " and is the {} ball pushed {}",
        " and is there any {} ball pushed {}",
        " and the {} ball moves {}",
        " and the {} ball is being pushed {}",
        " and the {} ball is pushed {}",
        " and the {} ball was moved {}",
]

second_orientation_list = [
        "to the back",
        "to the left",
        "to the front",
        "to the right",
]
second_total_orientation_list = [
        "to the back", "to the back", "to the back",
        "to the left", "to the left", "to the left",
        "to the front", "to the front", "to the front",
        "to the right", "to the right", "to the right",
]


# 将 dim = 6(5个球 + 1个方向) 的 goal 转换为 LangGCPEnv 中的 goal_abs(obs 的最后3维)
def goal_to_GCP_goal_abs(goal: np.ndarray):
        color_pair_0 = np.where(goal[:-1] == goal_src)[0].item()
        color_pair_1 = np.where(goal[:-1] == goal_dst)[0].item()
        
        template_idx = np.random.randint(len(train_template_list))
        orientation_idx = goal[-1]
        color_pair = (color_pair_0, color_pair_1)
        color_idx = color_pair2color_idx[color_pair]
        
        goal_abs = np.array([template_idx, orientation_idx, color_idx]).astype(int)
        
        return goal_abs


def get_best_cuda() -> int:
    import pynvml, numpy as np

    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    print("best gpu:", best_device_index)
    return best_device_index.item()
