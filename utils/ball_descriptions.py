
import os

from envs.clevr_robot_env.env import Lang2Env, dir_cnt, dir2angle_rg


import sys
sys.path.append(os.path.join(os.getcwd(), "envs"))
sys.path.append(os.path.join(os.path.join(os.getcwd(), "envs"), "clevr_robot_env"))


color_order = ["red", "blue", "green", "purple", "cyan"]

template_list = [
    "Push the {} ball {} the {} ball",              
    "Can you push the {} ball {} the {} ball",
    "Can you help me push the {} ball {} the {} ball",
    "Is the {} ball {} the {} ball",
    "Is there any {} ball {} the {} ball",
    "The {} ball moves {} the {} ball",
    "The {} ball is being pushed {} the {} ball",
    "The {} ball is pushed {} the {} ball",
    "The {} ball was moved {} the {} ball",

    "Move the {} ball {} the {} ball",             
    "Keep the {} ball {} the {} ball",
    "Can you move the {} ball {} the {} ball",
    "Can you keep the {} ball {} the {} ball",
    "Can you help me move the {} ball {} the {} ball",
    "Can you help me keep the {} ball {} the {} ball",
    "The {} ball was pushed {} the {} ball",
    "The {} ball is being moved {} the {} ball",
    "The {} ball is moved {} the {} ball",
]

orientation_list = ["behind", "to the left of", "in front of", "to the right of"]


env = Lang2Env(maximum_episode_steps=50,action_type='perfect',obs_type='order_invariant',direct_obs=True,use_subset_instruction=True,num_object=5)
calculate_description = env.calculate_description


def calculate_direction(src_xy, dst_xy):
    curr_angle = env.compute_angle(src_xy, dst_xy)
    orientation = None
    for dir in range(dir_cnt):
        angle_rg = dir2angle_rg[dir]
        if angle_rg[1] < angle_rg[0]:
            angle_flag = (angle_rg[0] <= curr_angle < 360) or (0 <= curr_angle < angle_rg[1])
        else:
            angle_flag = angle_rg[0] <= curr_angle < angle_rg[1]
        if angle_flag:
            orientation = dir
    return orientation


def get_balls_description(goal, obs, next_obs, t):
    pushed, target, orientation = calculate_description(next_obs, goal)
    description = template_list[t].format(color_order[pushed], orientation_list[orientation], color_order[target])
    return description

