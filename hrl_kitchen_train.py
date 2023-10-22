import gym
import copy
import kitchen
import numpy as np

from pathlib import Path
from GCP_utils.utils import models_dir
from stable_baselines3 import LangGCPPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


dataset_dir = Path(__file__).parent.parent.joinpath('dataset')

lm2bs = {
    'baseline': 64,
    'onehot': 64,
    'bert_binary': 64,
    'bert_cont': 64,
}


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Algorithm arguments')

    # utils
    parser.add_argument('--id', type=str, default='kitchen-high-v0')

    # parser.add_argument('--lm_type', type=str, default='policy_ag')
    parser.add_argument('--lm_type', type=str, default='human')
    
    parser.add_argument('--policy_epoch', type=int, default=100)

    # agent
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num', type=int, default=1)
    # parser.add_argument('--num', type=int, default=2)

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=16)

    parser.add_argument('--policy_language_dim', type=int, default=16)
    args = parser.parse_args()

    return args


def make_env(args):
    def _thunk():
        env = gym.make(args.id, language_model_type=args.lm_type)
        
        env = Monitor(env, None, allow_early_resets=True)
        
        return env

    return _thunk


def env_wrapper(args):
    envs = [
        make_env(args)
        for _ in range(args.num)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecNormalize(envs, norm_reward=True, norm_obs=False, training=False)

    return envs


from typing import List
from GCP_utils.utils import get_best_cuda


def train(
        env_list: List[VecNormalize],
        args,
):
    env = env_list[0]
    eval_env = env_list[1]
    
    num = env.num_envs
    lr = args.lr
    ns = 256
    bs = args.bs
    
    device = f'cuda:{get_best_cuda()}'
    
    prefix = args.id[:-3].replace('-', '_')
    
    model_path = models_dir.joinpath(f'kitchen_{args.lm_type}_{args.seed}')

    lm_kwargs = {
        'device': device,
        'model_path': model_path,
        'hidden_dim': args.hidden_dim,
        'output_dim': args.output_dim,
        'policy_language_dim': args.policy_language_dim,
        'mode': 'low',
        'is_kitchen': True,
        "emb_dim": 768,
    }
    if 'policy' in args.lm_type:
        lm_kwargs['epoch'] = args.policy_epoch
    elif args.lm_type == 'human':
        pass
    else:
        raise NotImplementedError
    
    kwargs = {
        'net_arch': [dict(pi=[args.hidden_dim, args.output_dim, 64, 64], vf=[args.hidden_dim, args.output_dim, 64, 64])],
    }
    policy_type = 'MlpPolicy'
    
    agent = LangGCPPPO(
        policy=policy_type,
        env=env,
        learning_rate=lr,
        n_steps=ns,
        batch_size=bs,
        tensorboard_log=f'{prefix}_baseline_train',
        device=device,
        verbose=1,
        policy_kwargs=kwargs,
        seed=args.seed,
    )

    steps = 10000000

    log_interval = 1
    save_interval = 10000000
    # save_interval = 100

    lm_type = args.lm_type
    from datetime import datetime
    train_time_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log_name = f'{train_time_start}_{args.id[:-len("-v0")].replace("-", "_")}_num_{num}_lm_{lm_type}_seed_{args.seed}'
    log_path = f'{prefix}_baseline_callback/{args.id[:-len("-v0")].replace("-", "_")}_num_{num}_lm_{lm_type}_seed_{args.seed}'
    save_path = f'{prefix}_baseline_model/{args.id[:-len("-v0")].replace("-", "_")}_num_{num}_lm_{lm_type}_seed_{args.seed}/model'
    
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList
    
    eval_interval = 10

    eval_log_path = f'{log_path}_env_eval'
    
    eval_callback = EvalCallback(
        eval_env=eval_env,
        log_path=eval_log_path,
        deterministic=False,
        eval_freq=eval_interval * agent.n_steps,
        n_eval_episodes=10 * len(env.get_attr('TASK_ELEMENTS')[0]),
        name='eval',
    )
    
    eval_callback = CallbackList([
        eval_callback,
    ])
    
    agent.lang_learn(
        total_timesteps=steps,
        log_interval=log_interval,
        tb_log_name=tb_log_name,
        callback=eval_callback,
        save_interval=save_interval,
        save_path=save_path,
    )


if __name__ == "__main__":
    args = get_args()
    
    eval_args = copy.deepcopy(args)
    eval_args.num = 1
    
    env = env_wrapper(args)
    eval_env = env_wrapper(eval_args)
    env_list = [
        env,
        eval_env,
    ]
    
    train(env_list=env_list, args=args)
