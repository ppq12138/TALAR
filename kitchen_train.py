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
    'baseline': 256,
    'onehot': 256,
    'bert_binary': 256,
    'bert_cont': 256,
}


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Algorithm arguments')

    # utils
    parser.add_argument('--id', type=str, default='kitchen-low-v0')

    # parser.add_argument('--lm_type', type=str, default='policy')
    # parser.add_argument('--lm_type', type=str, default='policy_ag')
    # parser.add_argument('--lm_type', type=str, default='onehot')
    # parser.add_argument('--lm_type', type=str, default='bert_binary')
    # parser.add_argument('--lm_type', type=str, default='bert_cont')
    parser.add_argument('--lm_type', type=str, default='baseline')
    
    parser.add_argument('--policy_epoch', type=int, default=100)
    parser.add_argument('--pretrain_path', type=str, default=None)
    # parser.add_argument('--pretrain_path', type=str, default='bert_binary')

    # agent
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num', type=int, default=2)
    # parser.add_argument('--num', type=int, default=1)


    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=16)
    parser.add_argument('--tune_flag', action="store_true", default=False)

    parser.add_argument('--policy_language_dim', type=int, default=16)

    # mdp
    parser.add_argument('--env_type', type=str, default='low')
    # parser.add_argument('--env_type', type=str, default='demo')
    
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--d_task', type=str, default=None)
    
    # parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--reward_type', type=str, default='prev_curr')
    
    args = parser.parse_args()

    return args


def make_env(args):
    def _thunk():
        env = gym.make(args.id,
                       d_task=args.d_task,
                       env_type=args.env_type,
                       reward_type=args.reward_type)
        
        env.set_mode(args.mode)
        
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
    train_env = env_list[1]
    test_env = env_list[2]
    
    num = env.num_envs
    lr = args.lr
    ns = 512
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
    if args.lm_type == 'onehot':
        pass
    elif args.lm_type in ['bert_cont', 'bert_onehot', 'bert_binary']:
        lm_kwargs.update({
            "bert_kwargs": {
                "hidden_dim": args.hidden_dim,
                "output_dim": args.output_dim,
                "tune_flag": args.tune_flag,
                "device": device,
            }})
    elif 'policy' in args.lm_type:
        lm_kwargs['epoch'] = args.policy_epoch
    elif args.lm_type in ['baseline']:
        # no multiple seed
        model_path = models_dir.joinpath(f'{args.lm_type}_0')

        lm_kwargs['model_path'] = model_path
    else:
        raise NotImplementedError
    
    if args.id == 'kitchen-low-v0':
        if args.env_type == 'demo':
            kwargs = {
                'net_arch': [dict(pi=[args.hidden_dim, args.output_dim, 64, 64], vf=[args.hidden_dim, args.output_dim, 64, 64])],
            }
            policy_type = 'MlpPolicy'
        else:
            kwargs = {
                'features_extractor_kwargs': {
                    'language_model_type': args.lm_type,
                    'language_model_kwargs': lm_kwargs,    
                }
            }
            policy_type = 'LangGCPPolicy'
    else:
        raise NotImplementedError
    
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
    pretrain_models_dir = models_dir.joinpath('pretrain')
    if args.pretrain_path is not None:
        tmp_agent = LangGCPPPO.load(pretrain_models_dir.joinpath(args.pretrain_path))
        tmp_agent.policy.features_extractor_kwargs['language_model_kwargs']['device'] = device
        tmp_agent.policy.features_extractor_kwargs['language_model_kwargs']['bert_kwargs']['device'] = device

        kwargs['features_extractor_kwargs'] = tmp_agent.policy.features_extractor_kwargs
        agent = LangGCPPPO.load(
                pretrain_models_dir.joinpath(args.pretrain_path),
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
    save_interval = 100

    lm_type = args.lm_type
    if args.pretrain_path is not None:
        lm_type = f'pretrained_{lm_type}'
    from datetime import datetime
    train_time_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log_name = f'{train_time_start}_{args.id[:-len("-v0")].replace("-", "_")}_{args.env_type}_{args.reward_type}_num_{num}_lm_{lm_type}_seed_{args.seed}_hidden_{args.hidden_dim}_output_{args.output_dim}_tuned_{args.tune_flag}'
    log_path = f'{prefix}_baseline_callback/{args.id[:-len("-v0")].replace("-", "_")}_{args.env_type}_{args.reward_type}_num_{num}_lm_{lm_type}_seed_{args.seed}_hidden_{args.hidden_dim}_output_{args.output_dim}_tuned_{args.tune_flag}'
    save_path = f'{prefix}_baseline_model/{args.id[:-len("-v0")].replace("-", "_")}_{args.env_type}_{args.reward_type}_num_{num}_lm_{lm_type}_seed_{args.seed}_hidden_{args.hidden_dim}_output_{args.output_dim}_tuned_{args.tune_flag}/model'
    
    from stable_baselines3.common.callbacks import MultiTaskEvalCallback, CallbackList
    
    eval_interval = 10

    train_log_path = f'{log_path}_env_train'
    test_log_path = f'{log_path}_env_test'
    
    train_callback = MultiTaskEvalCallback(
        eval_env=train_env,
        log_path=train_log_path,
        deterministic=False,
        # deterministic=True,
        eval_freq=eval_interval * agent.n_steps,
        n_eval_episodes=10 * len(env.get_attr('TASK_ELEMENTS')[0]),
        name='train',
    )
    test_callback = MultiTaskEvalCallback(
        eval_env=test_env,
        log_path=test_log_path,
        deterministic=False,
        # deterministic=True,
        eval_freq=eval_interval * agent.n_steps,
        n_eval_episodes=10 * len(env.get_attr('TASK_ELEMENTS')[0]),
        name='test',
    )
    
    eval_callback = CallbackList([
        train_callback,
        test_callback,
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
    # rollout_env()
    args = get_args()
    
    eval_args = copy.deepcopy(args)
    eval_args.num = 1
    
    train_args = copy.deepcopy(eval_args)
    test_args = copy.deepcopy(eval_args)
    
    train_args.mode = 'train'
    test_args.mode = 'test'
    
    env = env_wrapper(args)
    train_env = env_wrapper(train_args)
    test_env = env_wrapper(test_args)
    env_list = [
        env,
        train_env,
        test_env,
    ]
    
    train(env_list=env_list, args=args)
