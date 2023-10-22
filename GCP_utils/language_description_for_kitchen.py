import random
import numpy as np

from pathlib import Path
dataset_file = Path(__file__).parent.parent.joinpath('dataset').joinpath('kitchen_with_description.npy')


OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }

from utils.kitchen_descriptions import LANGUAGE_DESCRIPTION as TOTAL_LANGUAGE_DESCRIPTION
train_set_size = 40
LANGUAGE_DESCRIPTION = {}
TEST_LANGUAGE_DESCRIPTION = {}
for key, value in TOTAL_LANGUAGE_DESCRIPTION.items():
    LANGUAGE_DESCRIPTION[key] = value[:train_set_size].copy()
    TEST_LANGUAGE_DESCRIPTION[key] = value[train_set_size:].copy()

VALID_TASK_DESCRIPTION = list(TOTAL_LANGUAGE_DESCRIPTION.keys())
description_to_onehot_idx = {}
train_description_list = []
test_description_list = []
repeat_list = []
description_idx = 0
for valid_task in VALID_TASK_DESCRIPTION:
    for train_description in LANGUAGE_DESCRIPTION[valid_task]:
        train_description_list.append(train_description)
        if train_description in description_to_onehot_idx:
            repeat_list.append(train_description)
        description_to_onehot_idx[train_description] = description_idx
        description_idx = description_idx + 1
    for test_description in TEST_LANGUAGE_DESCRIPTION[valid_task]:
        test_description_list.append(test_description)
        if test_description in description_to_onehot_idx:
            repeat_list.append(test_description)
        description_to_onehot_idx[test_description] = description_idx
        description_idx = description_idx + 1
onehot_idx_to_description = dict(zip(description_to_onehot_idx.values(), description_to_onehot_idx.keys()))

BONUS_THRESH = 0.3


def judge_goal_completion(state):
    complete = []
    
    for element in OBS_ELEMENT_INDICES:
        distance = np.linalg.norm(state[OBS_ELEMENT_INDICES[element]] - OBS_ELEMENT_GOALS[element])
        if distance < BONUS_THRESH:
            complete.append(element)

    return set(complete)


def state_difference(s, s_):
    """

    :param s: 初始状态，shape 为[obs_size]
    :param s_: 结束状态，shape 为[obs_size]
    :return: 描述s到s_变化的自然语言。注意，返回的自然语言描述可能为空，主要取决于s到s_发生了多大的变化。
    """
    goal_complete_before = judge_goal_completion(s)
    goal_complete_after = judge_goal_completion(s_)

    extra_complete_goal = goal_complete_after - goal_complete_before

    res_NL = []

    for element in extra_complete_goal:
        res_NL.append(random.choice(LANGUAGE_DESCRIPTION[element]))

    if random.randint(0,1) < 0.5:
        return ";".join(res_NL)
    else:
        return " and ".join(res_NL)


if __name__ == '__main__':
    for i in range(100):
        print("\n\nstep:", i)
        print(state_difference(np.random.random(40), np.random.random(40)))
