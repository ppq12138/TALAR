"""Environments using kitchen and Franka robot."""
import numpy as np

from kitchen.adept_envs.utils.configurable import configurable
from kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
from kitchen.offline_env import OfflineEnv
from stable_baselines3 import LangGCPPPO
from GCP_utils.utils import models_dir

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

BONUS_THRESH = 0.3

GLOBAL_TARGET_STATE = np.zeros(30)
for element in OBS_ELEMENT_INDICES.keys():
    element_idx = OBS_ELEMENT_INDICES[element]
    element_goal = OBS_ELEMENT_GOALS[element]
    GLOBAL_TARGET_STATE[element_idx] = element_goal.copy()


@configurable(pickleable=True)
class KitchenBase(KitchenTaskRelaxV1, OfflineEnv):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True

    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        super(KitchenBase, self).__init__(**kwargs)
        OfflineEnv.__init__(
            self,
            dataset_url=dataset_url,
            ref_max_score=ref_max_score,
            ref_min_score=ref_min_score)

    def _get_task_goal(self):
        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            complete = distance < BONUS_THRESH
            if complete:
                completions.append(element)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict['bonus'] = bonus
        reward_dict['r_total'] = bonus
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        return obs, reward, done, env_info

    def render(self, mode='human'):
        # Disable rendering to speed up environment evaluation.
        return []


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']


from utils.generate_description import LANGUAGE_DESCRIPTION as TOTAL_LANGUAGE_DESCRIPTION
from GCP_utils.language_description_for_kitchen import train_description_list, test_description_list
from GCP_utils.language_description_for_kitchen import description_to_onehot_idx, onehot_idx_to_description
class KitchenLanguageV0(KitchenMicrowaveKettleLightSliderV0):
    TASK_ELEMENTS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'microwave', 'kettle']
    # TASK_ELEMENTS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']

    NEW_TASK_ELEMENTS = list(TOTAL_LANGUAGE_DESCRIPTION.keys())  
    task_to_idx = dict(zip(TASK_ELEMENTS, range(len(TASK_ELEMENTS))))
    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        self.mode = None
        self.language_goal = None
        self.language_goal_idx = None
        super(KitchenLanguageV0, self).__init__(dataset_url=dataset_url, ref_max_score=ref_max_score, ref_min_score=ref_min_score, **kwargs)
    
    def get_obs_no_goal(self) -> np.ndarray:
        return self._get_obs_no_goal()

    def _get_obs_no_goal(self) -> np.ndarray:
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        self.obs_dict['goal'] = self.goal
        
        obs_no_goal = np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp']])
        
        return obs_no_goal
    
    def _get_obs(self):
        obs_no_goal = self._get_obs_no_goal()
        
        if self.language_goal is not None:
            assert self.mode is not None
            language_goal = self.language_goal
        else:
            language_goal = np.random.choice(train_description_list)
        language_goal_idx = description_to_onehot_idx[language_goal]
        
        obs = np.r_[obs_no_goal, language_goal_idx]
        
        return obs

    def _get_task_goal(self):
        new_goal = np.zeros(self.goal.shape)
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def set_mode(self, mode: str):
        self.mode = mode

    def language_goal_to_task_set(self, language_goal: str) -> set:
        task_set = set()
        task_cnt = max(len(language_goal.split(' ; ')), len(language_goal.split(' and ')))
        for idx in range(len(self.TASK_ELEMENTS)):
            task_element = self.TASK_ELEMENTS[idx]
            new_task_element = self.NEW_TASK_ELEMENTS[idx]
            if new_task_element in language_goal:
                task_set.add(task_element)
        
        assert len(task_set) == task_cnt
        
        return task_set

    def reset_model(self):
        if self.mode is None:
            language_goal = np.random.choice(train_description_list).item()
        elif self.mode == 'train':
            language_goal = np.random.choice(train_description_list).item()
        elif self.mode == 'test':
            language_goal = np.random.choice(test_description_list).item()
        else:
            raise NotImplementedError
        language_goal_idx = description_to_onehot_idx[language_goal]
        
        assert 0 <= language_goal_idx < len(description_to_onehot_idx)
        
        self.language_goal = language_goal
        self.language_goal_idx = language_goal_idx
        
        # print(f'=' * 32 + ' Training ' + f'=' * 32)
        # for language_goal in train_description_list:
        #     print(f'    \item {language_goal}')
        # print(f'=' * 32 + ' Testing ' + f'=' * 32)
        # for language_goal in test_description_list:
        #     print(f'    \item {language_goal}')
        
        # for language_goal in train_description_list:
        #     self.language_goal_to_task_set(language_goal)
        # for language_goal in test_description_list:
        #     self.language_goal_to_task_set(language_goal)
            
        tasks_to_complete = self.language_goal_to_task_set(self.language_goal)
        self.tasks_to_complete = tasks_to_complete
        
        reset_obs = super(KitchenBase, self).reset_model()
        
        return reset_obs

    def step(self, a, b=None):
        obs, reward, done, info = super(KitchenLanguageV0, self).step(a, b)
        
        is_success = False
        is_fail = False
        if done:
            is_success = len(self.tasks_to_complete) == 0
            is_fail = len(self.tasks_to_complete) > 0

        info.update({
                'custom_info': {},
                'is_success': is_success,
                'is_fail': is_fail,
        })
        
        return obs, reward, done, info

    def render(self, mode='human'):
        return []
        # return super(KitchenBase, self).render(mode=mode)


class KitchenDemoV0(KitchenLanguageV0):
    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        self.obs_dict['goal'] = self.goal
        
        originial_obs = np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp'], self.obs_dict['goal']])
        
        goal_arr = np.zeros(len(self.TASK_ELEMENTS), dtype=np.int64)
        for task in self.tasks_to_complete:
            idx = self.task_to_idx[task]
            goal_arr[idx] = 1
        
        obs = np.r_[originial_obs, goal_arr]
        
        return obs


TASK_TO_TARGET_QP = {
    'bottom burner': np.array([-1.44183728335785, -1.76072747007054, 1.4632211364478231, -2.303325149563607, 0.5401844309524202, 2.122815253319523, 2.906521211339583, -0.0006586844128569, 0.03794845039131554]),
    'top burner':    np.array([-2.06905739559047, -1.76182566021897, 0.7596889233850814, -1.444454975858641, 0.6140481055642997, 2.117880897668754, 2.048055623969903, 0.00595780646560787, 0.00925051243007712]),
    'light switch':  np.array([-1.96052043222863, -1.75886376050430, 0.7778070947728526, -1.227939881347626, -0.101933252407512, 2.117807297433198, 1.535074663974797, 0.00320869764750263, 0.00804556505006951]),
    'slide cabinet': np.array([-1.12630370044609, -0.75072519253594, 0.8922448080396826, -2.257628872911379, 0.0968998510577811, 1.672658444626036, 1.460815224225509, -0.0005855397194995, 0.03437819100411712]),
    'microwave':     np.array([-1.00200716390155, -1.73109556692146, 1.9826565427273555, -1.602727991321340, -0.422652311314689, 1.919995270021869, 2.010248320836727, 0.04207480922004891, 0.03939296018258755]),
    'kettle':        np.array([-0.53666527766099, -1.76402197725498, 2.3582630581345545, -2.426362617523646, -0.576969361855065, 0.834533051839273, 1.533836230353666, 0.00748759385682631, 0.01871191530601525]),
}


from GCP_utils.language_description_for_kitchen import LANGUAGE_DESCRIPTION, TEST_LANGUAGE_DESCRIPTION
class KitchenLowV0(KitchenLanguageV0):
    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None,
                 d_task=None, env_type='demo', reward_type='dense', **kwargs):
        self.deterministic_task = d_task
        self.env_type = env_type
        self.reward_type = reward_type

        self.max_episode_num = 280

        self.step_cnt = None
        self.curr_task = None

        self.task_idx = 0
        self.task_sr_dict = {}
        for task in self.TASK_ELEMENTS:
            self.task_sr_dict[task] = 0.0
        
        self.prev_dist = 1
        self.completed_task_list = []
        self.target_qp = 1
        self.is_arrived = False
        
        # used by high-level env
        self.goal_abs = None
        
        super(KitchenLowV0, self).__init__(dataset_url=dataset_url, ref_max_score=ref_max_score, ref_min_score=ref_min_score, **kwargs)
    
    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        self.obs_dict['goal'] = self.goal

        # GCP Training
        if self.env_type == 'low':
            originial_obs = np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp']])
            goal_arr = np.array(self.language_goal_idx if self.language_goal_idx is not None else 0)
        # Dataset Collecting
        elif self.env_type == 'demo':
            # originial_obs = np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp'], self.obs_dict['goal']])
            originial_obs = np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp']])
            goal_arr = np.zeros(len(self.TASK_ELEMENTS))
            for task in self.tasks_to_complete:
                task_idx = self.TASK_ELEMENTS.index(task)
                goal_arr[task_idx] = 1
        else:
            raise NotImplementedError
        
        obs = np.r_[originial_obs, goal_arr]
        
        return obs

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            complete = distance < BONUS_THRESH
            if self.reward_type == 'dense':
                reward = reward - distance
            elif self.reward_type == 'shaped_dense':
                reward = (reward - distance) / np.linalg.norm(next_goal[element_idx])  # Naive dense reward
            elif self.reward_type == 'prev_curr':
                prev_dist = self.prev_dist
                curr_dist = distance
                reward = prev_dist - curr_dist
                self.prev_dist = curr_dist

            if complete:
                completions.append(element)
                reward = reward + 1
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        
        if self.reward_type == 'sparse':
            pass
        elif self.reward_type in ['dense', 'shaped_dense', 'prev_curr']:
            bonus = reward
        else:
            raise NotImplementedError

        self.is_arrived = abs(reward) > 1e-3
        if not self.is_arrived:
            qp_diff = np.linalg.norm(next_q_obs - self.target_qp) / 100  # 1e-2 is more training friendly
            # print(f'Reward: {reward}')
            # print(f'qp_diff: {qp_diff}')
            bonus = bonus - qp_diff

        reward_dict['bonus'] = bonus
        reward_dict['r_total'] = bonus
        score = bonus
        return reward_dict, score

    def define_deterministic_task(self, task: str):
        self.deterministic_task = task

    def reset_model(self):
        if self.deterministic_task is not None:
            task_idx = self.TASK_ELEMENTS.index(self.deterministic_task)
        else:
            task_idx = self.task_idx

        task = self.TASK_ELEMENTS[task_idx]
        self.curr_task = task
        self.tasks_to_complete = set([self.curr_task])
        self.target_qp = TASK_TO_TARGET_QP[self.curr_task].copy()
        self.is_arrived = False

        if self.env_type == 'low':
            new_task = self.NEW_TASK_ELEMENTS[task_idx]  
            
            if self.mode is None:
                language_goal = np.random.choice(LANGUAGE_DESCRIPTION[new_task])
            elif self.mode == 'train':
                language_goal = np.random.choice(LANGUAGE_DESCRIPTION[new_task])
            elif self.mode == 'test':
                language_goal = np.random.choice(TEST_LANGUAGE_DESCRIPTION[new_task])
            else:
                raise NotImplementedError
            language_goal_idx = description_to_onehot_idx[language_goal]
            
            assert 0 <= language_goal_idx < len(description_to_onehot_idx)
            
            self.language_goal = language_goal
            self.language_goal_idx = language_goal_idx
        
        reset_obs = super(KitchenBase, self).reset_model()

        self.step_cnt = 0
        
        next_q_obs = self.obs_dict['qp']
        next_obj_obs = self.obs_dict['obj_qp']
        next_goal = self.obs_dict['goal']
        element_idx = OBS_ELEMENT_INDICES[self.curr_task]
        idx_offset = len(next_q_obs)
        distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
        self.prev_dist = distance
        if self.prev_dist < BONUS_THRESH:
            return self.reset_model()

        step_completed_task = []
        dist_list = []
        for task in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[task]
            idx_offset = len(next_q_obs)
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                GLOBAL_TARGET_STATE[element_idx])
            dist_list.append(distance)
            if distance < BONUS_THRESH:
                step_completed_task.append(task)
        self.completed_task_list = [step_completed_task]

        self.task_idx = (self.task_idx + 1) % len(self.TASK_ELEMENTS)

        return reset_obs

    def update_sr(self, is_success: bool):
        curr_task = self.curr_task
        ratio = 0.08
        self.task_sr_dict[curr_task] = ratio * int(is_success) + (1 - ratio) * self.task_sr_dict[curr_task]
    
    def step(self, a, b=None):
        obs, reward, done, info = super(KitchenLanguageV0, self).step(a, b)

        next_q_obs = self.obs_dict['qp']
        next_obj_obs = self.obs_dict['obj_qp']
        step_completed_task = []
        dist_list = []
        for task in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[task]
            idx_offset = len(next_q_obs)
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                GLOBAL_TARGET_STATE[element_idx])
            dist_list.append(distance)
            if distance < BONUS_THRESH:
                step_completed_task.append(task)
        self.completed_task_list.append(step_completed_task)
        
        if self.step_cnt is not None:
            self.step_cnt = self.step_cnt + 1
            done = done or self.step_cnt >= self.max_episode_num

        custom_info = {}
        eval_result_dict = {}
        is_success = False
        is_fail = False
        if done:
            is_success = len(self.tasks_to_complete) == 0
            is_fail = len(self.tasks_to_complete) > 0
            self.update_sr(is_success)
            for task in self.TASK_ELEMENTS:
                custom_info[f'{task}_is_success'] = self.task_sr_dict[task]
            eval_result_dict['task_name'] = self.curr_task
            eval_result_dict['is_success'] = is_success

        info.update({
            'eval_result_dict': eval_result_dict,
            'custom_info': custom_info,
            'is_success': is_success,
            'is_fail': is_fail,
        })
        
        return obs, reward, done, info

    # Used by high-level env
    def high_step_setup(self, goal_abs: np.ndarray):
        self.goal_abs = goal_abs
        assert isinstance(self.goal_abs, np.ndarray)
    
    def high_get_obs(self, obs_no_goal: np.ndarray) -> np.ndarray:
        obs = np.r_[obs_no_goal, self.goal_abs]
        
        return obs
    
    def high_step(self, agent: LangGCPPPO, n_step: int):
        obs_no_goal = self._get_obs_no_goal()
        high_obs = self.high_get_obs(obs_no_goal)
        reward = 0
        done = False
        info = {}
        for step in np.arange(n_step):
            action = agent.predict(high_obs, deterministic=False)[0]
            obs, reward, done, info = self.step(action)
            obs_no_goal = self._get_obs_no_goal()
            high_obs = self.high_get_obs(obs_no_goal)
        
        return high_obs, reward, done, info
        

class KitchenHighNoLMV0(KitchenLowV0):
    def reset_model(self):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        
        reset_obs = super(KitchenBase, self).reset_model()
        
        return reset_obs



import gym
from gym import spaces

from GCP_utils.utils import bert_cont_output_dim
from GCP_utils.utils import get_best_cuda
class KitchenHighV0(KitchenLanguageV0):
    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None,
                 language_model_type: str = 'human',
                 low_n_step: int = 10,
                 **kwargs):
        self.low_env = gym.make('kitchen-low-v0', env_type='low', reward_type='prev_curr')
        self.language_model_type = language_model_type

        device = f'cuda:{get_best_cuda()}'
        lm_model_path = models_dir.joinpath('low').joinpath(f'kitchen_policy_ag')
        hidden_dim = 128
        output_dim = 16
        pl_dim = 16
        
        policy_epoch = 100
        lm_kwargs = {
            'device': device,
            'model_path': lm_model_path,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'policy_language_dim': pl_dim,
            'mode': 'high',
            'is_kitchen': True,
            "emb_dim": 768,
        }
        if self.language_model_type == 'human':
            lm_kwargs['epoch'] = policy_epoch
            lm_kwargs['mode'] = 'human'
        elif 'policy' in self.language_model_type:
            lm_kwargs['epoch'] = policy_epoch
        else:
            raise NotImplementedError

        policy_kwargs = {
            'features_extractor_kwargs': {
                'language_model_type': self.language_model_type,
                'language_model_kwargs': lm_kwargs,    
            }
        }
        self.low_agent = LangGCPPPO.lang_load(
                models_dir.joinpath('low').joinpath('policy_ag'),
                env=self.low_env,
                policy_kwargs=policy_kwargs,
                device=device,
                )
        self.low_n_step = low_n_step
        
        self.max_episode_num = 28
        self.step_cnt = None
        
        super(KitchenHighV0, self).__init__(dataset_url=dataset_url, ref_max_score=ref_max_score, ref_min_score=ref_min_score, **kwargs)

        self.instru_space_dim = len(description_to_onehot_idx)
        self.action_onehot_dim = 6
        if self.language_model_type == 'human':
            self.action_space = spaces.Discrete(self.instru_space_dim)
        elif 'policy' in self.language_model_type:
            self.action_indicate_list = [2, 2, 2, 2, 6, 6]  # pu_1_pred_4
            self.action_space = spaces.Discrete(np.prod(self.action_indicate_list))
        else:
            raise NotImplementedError
        
    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        self.obs_dict['goal'] = self.goal

        obs_no_goal = self.low_env.get_obs_no_goal()

        obs = obs_no_goal
        
        return obs
        
    def reset_model(self):
        self.low_env.reset()
        
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        
        reset_obs = super(KitchenBase, self).reset_model()
        
        self.step_cnt = 0
        
        return reset_obs
    
    def action2goal_abs(self, action) -> np.ndarray:
        action = np.array(action)
        if self.language_model_type == 'human':
            goal_abs = action
        elif 'policy' in self.language_model_type:
            tmp_action = action
            bool_abs_list = []
            for idx in range(4):
                tmp_abs = tmp_action % self.action_indicate_list[idx]
                tmp_action = tmp_action // self.action_indicate_list[idx]
                bool_abs_list.append(tmp_abs)
            onehot_idx_list = []
            idx = 4
            while idx < len(self.action_indicate_list):
                tmp_onehot_idx = tmp_action % self.action_indicate_list[idx]
                tmp_action = tmp_action // self.action_indicate_list[idx]
                onehot_idx_list.append(tmp_onehot_idx)
                idx += 1
            bool_abs = np.array(bool_abs_list)
            onehot_idx = np.array(onehot_idx_list)
            onehot_abs = np.zeros((onehot_idx.size, self.action_onehot_dim))
            onehot_abs[np.arange(onehot_idx.size), onehot_idx] = 1
            goal_abs = np.r_[bool_abs, onehot_abs.flatten()]
            # goal_abs = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.])  kettle
        else:
            raise NotImplementedError
        
        return goal_abs

    def step(self, a, b=None):
        if not hasattr(self, 'instru_space_dim'):  # __init__
            obs = self._get_obs()
            reward = 0
            done = False
            info = {}
        else:
            goal_abs = self.action2goal_abs(a)
            self.low_env.high_step_setup(goal_abs)
            
            low_obs, low_reward, low_done, low_info = self.low_env.high_step(self.low_agent, self.low_n_step)
            
            self.step_cnt = self.step_cnt + 1
            
            obs = self._get_obs()
            
            goal = self.obs_dict['goal'].copy()
            self.obs_dict = self.low_env.obs_dict.copy()
            self.obs_dict['goal'] = goal
            reward_dict, score = self._get_reward_n_score(self.obs_dict)
            reward = reward_dict['r_total']

            is_fail = self.step_cnt >= self.max_episode_num
            is_success = len(self.tasks_to_complete) == 0 and not is_fail
            done = is_fail or is_success
            
            info = {}
            if done:
                custom_info = {
                    'completed_cnt': len(set(self.TASK_ELEMENTS) - set(self.tasks_to_complete)),
                }
                for task in self.TASK_ELEMENTS:
                    custom_info[f'is_success_{task}'] = task not in self.tasks_to_complete
                info.update({
                        'custom_info': custom_info,
                        'is_success': is_success,
                        'is_fail': is_fail,
                })
            
        return obs, reward, done, info


class KitchenMicrowaveKettleBottomBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'bottom burner', 'light switch']
