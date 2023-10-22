import copy

from envs.clevr_robot_env import HighLangGCPEnv
from stable_baselines3 import LangPPO, LangDQN, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Algorithm arguments')

    # utils
    parser.add_argument('--id', type=str, default='high')
    
    # parser.add_argument('--env_type', type=str, default='arrangement')
    parser.add_argument('--env_type', type=str, default='ordering')
    
    # parser.add_argument('--agent_name', type=str, default='ppo')
    parser.add_argument('--agent_name', type=str, default='dqn')
    
    # parser.add_argument('--lm_type', type=str, default='policy')
    parser.add_argument('--lm_type', type=str, default='human')

    # agent
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--target_ns', type=int, default=256 * 3)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--gs', type=int, default=1)
    
    # mdp
    parser.add_argument('--init_high', action='store_true')
    
    parser.add_argument('--num_obj', type=int, default=5)
    
    args = parser.parse_args()

    return args


def make_env(args):
    def _thunk():
        if args.id == 'high':
            env = HighLangGCPEnv(
                init_high=args.init_high,
                num_object=args.num_obj,
                env_type=args.env_type,
                agent_name=args.agent_name,
                language_model_type=args.lm_type,
                )
        else:
            raise NotImplementedError

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


from GCP_utils.utils import get_best_cuda
from typing import List


def train(
        env_list: List[VecNormalize],
        args,
):
    env = env_list[0]
    train_env = env_list[1]
    
    agent_name = args.agent_name
    num = env.num_envs
    lr = args.lr
    assert args.target_ns % num == 0
    ns = args.target_ns // num
    bs = args.bs
    gs = args.gs
    
    device = f'cuda:{get_best_cuda()}'
    steps = int(1e8)
    
    if agent_name == 'ppo':
        agent = LangPPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=lr,
            n_steps=ns,
            batch_size=bs,
            tensorboard_log='high_train',
            device=device,
            verbose=1,
            seed=args.seed,
        )
    elif agent_name == 'dqn':
        agent = DQN(
            policy='MlpPolicy',
            env=env,
            buffer_size=int(1e5),
            learning_starts=int(100 * env.unwrapped.get_attr('maximum_episode_steps')[0]),
            batch_size=bs,
            train_freq=4,
            gradient_steps=gs,
            target_update_interval=100,
            gamma=0.9,
            tensorboard_log='high_train',
            device=device,
            verbose=1,
            seed=args.seed,
        )
    else:
        raise NotImplementedError

    if 'ppo' in agent_name:
        log_interval = 1
        save_interval = 1
    elif 'dqn' in agent_name:
        log_interval = 10
        save_interval = 10
    else:
        raise NotImplementedError

    from datetime import datetime
    train_time_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log_name = f'{train_time_start}_{args.id}_{agent_name}_num_{num}_lr_{lr}_ns_{ns}_bs_{bs}_et_{args.env_type}_lm_{args.lm_type}_nobj_{args.num_obj}_seed_{args.seed}'
    log_path = f'high_callback/{args.id}_{agent_name}_num_{num}_lr_{lr}_ns_{ns}_bs_{bs}_et_{args.env_type}_lm_{args.lm_type}_nobj_{args.num_obj}_seed_{args.seed}'
    save_path = f'high_model/{args.id}_{agent_name}_num_{num}_lr_{lr}_ns_{ns}_bs_{bs}_et_{args.env_type}_lm_{args.lm_type}_nobj_{args.num_obj}_seed_{args.seed}/model'
    
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList
    
    eval_interval = 4
    if 'ppo' in agent_name:
        eval_freq = eval_interval * ns
    elif 'dqn' in agent_name:
        eval_freq = eval_interval * log_interval
    else:
        raise NotImplementedError
    train_log_path = f'{log_path}_env_train'
    
    train_callback = EvalCallback(
        eval_env=train_env,
        log_path=train_log_path,
        deterministic=True,
        eval_freq=eval_freq,
        n_eval_episodes=40,
        name='train',
    )
    
    eval_callback = CallbackList([
        train_callback,
    ])
    
    agent.learn(
        total_timesteps=steps,
        log_interval=log_interval,
        tb_log_name=tb_log_name,
        # callback=eval_callback,
    )
    # agent.lang_learn(
    #     total_timesteps=steps,
    #     log_interval=log_interval,
    #     tb_log_name=tb_log_name,
    #     callback=eval_callback,
    #     save_interval=save_interval,
    #     save_path=save_path,
    # )


if __name__ == "__main__":
    args = get_args()
    
    eval_args = copy.deepcopy(args)
    eval_args.num = 1
    
    train_args = copy.deepcopy(eval_args)
    
    env = env_wrapper(args)
    train_env = env_wrapper(train_args)
    
    env_list = [
        env,
        train_env,
    ]
    
    train(env_list=env_list, args=args)
