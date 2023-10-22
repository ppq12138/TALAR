import copy

from GCP_utils.utils import models_dir
from envs.clevr_robot_env import LangGCPEnv
from stable_baselines3 import LangPPO, LangGCPPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Algorithm arguments')

    # utils
    parser.add_argument('--id', type=str, default='langGCP')
    
    # parser.add_argument('--lm_type', type=str, default='policy')
    parser.add_argument('--lm_type', type=str, default='policy_cont')
    # parser.add_argument('--lm_type', type=str, default='bert_binary')
    # parser.add_argument('--lm_type', type=str, default='baseline')
    
    # parser.add_argument('--policy_epoch', type=int, default=19)
    parser.add_argument('--policy_epoch', type=int, default=20)
    
    parser.add_argument('--agent_name', type=str, default='ppogcp')
    
    parser.add_argument('--pretrain_path', type=str, default=None)
    # parser.add_argument('--pretrain_path', type=str, default='bert_binary')

    # agent
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--num', type=int, default=2)
    
    parser.add_argument('--lr', type=float, default=3e-4)
    
    parser.add_argument('--bs', type=int, default=512)
    
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=32)
    parser.add_argument('--policy_language_dim', type=int, default=14)

    # mdp
    parser.add_argument('--fd', type=float, default=0.20)
    
    parser.add_argument('--num_obj', type=int, default=5)
    
    parser.add_argument('--action_type', type=str, default='perfect')
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    return args


def make_env(args):
    def _thunk():
        if args.id == 'langGCP':
            env = LangGCPEnv(maximum_episode_steps=50, action_type=args.action_type, obs_type='order_invariant',
                           direct_obs=True, use_subset_instruction=True,
                           fail_dist=args.fd,
                           language_model_type=args.lm_type,
                           mode=args.mode,
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
    test_env = env_list[2]
    error_env = env_list[3]
    
    agent_name = args.agent_name
    num = env.num_envs
    lr = args.lr
    ns = 512
    bs = args.bs
    
    device = f'cuda:{get_best_cuda()}'
    
    prefix = 'original'
    if args.pretrain_path is not None:
        prefix = 'pretrain'
    
    if agent_name == 'ppo':
        agent = LangPPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=lr,
            n_steps=ns,
            batch_size=bs,
            tensorboard_log=f'{prefix}_baseline_train',
            device=device,
            verbose=1,
            seed=args.seed,
        )
    elif agent_name == 'ppogcp':
        model_path = models_dir.joinpath(f'{args.lm_type}_{args.seed}')
        lm_kwargs = {
            'device': device,
            'model_path': model_path,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
            'policy_language_dim': args.policy_language_dim,
            'mode': 'low',
            'is_kitchen': False,
            "emb_dim": 768,
        }
        if args.lm_type == 'onehot':
            pass
        elif args.lm_type in ['bert_cont', 'bert_onehot', 'bert_binary']:
            lm_kwargs.update({
                "bert_kwargs": {
                    "hidden_dim": args.hidden_dim,
                    "output_dim": args.output_dim,
                    "tune_flag": False,
                    "device": device,
                }})
        elif 'policy' in args.lm_type:
            lm_kwargs['epoch'] = args.policy_epoch
        elif args.lm_type in ['baseline']:
            model_path = models_dir.joinpath(f'{args.lm_type}')
            lm_kwargs['model_path'] = model_path
        else:
            raise NotImplementedError
        
        kwargs = {
            'features_extractor_kwargs': {
                'language_model_type': args.lm_type,
                'language_model_kwargs': lm_kwargs,    
            }
        }
        
        agent = LangGCPPPO(
            policy='LangGCPPolicy',
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
            agent = LangGCPPPO.lang_load(
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
    else:
        raise NotImplementedError

    # steps = 100000000
    steps = 2500000

    if 'ppo' in agent_name:
        log_interval = 1
        save_interval = 1
        save_interval = 100000000
    else:
        raise NotImplementedError

    if args.pretrain_path is not None:
        log_interval = 1
        save_interval = 10000000

    from datetime import datetime
    train_time_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log_name = f'{train_time_start}_{args.id}_{agent_name}_num_{num}_lr_{lr}_ns_{ns}_bs_{bs}_fd_{args.fd}_lm_{args.lm_type}_nobj_{args.num_obj}_seed_{args.seed}_hidden_{args.hidden_dim}_output_{args.output_dim}'
    log_path = f'{prefix}_baseline_callback/{args.id}_{agent_name}_num_{num}_lr_{lr}_ns_{ns}_bs_{bs}_fd_{args.fd}_lm_{args.lm_type}_nobj_{args.num_obj}_seed_{args.seed}_hidden_{args.hidden_dim}_output_{args.output_dim}'
    save_path = f'{prefix}_baseline_model/{args.id}_{agent_name}_num_{num}_lr_{lr}_ns_{ns}_bs_{bs}_fd_{args.fd}_lm_{args.lm_type}_nobj_{args.num_obj}_seed_{args.seed}_hidden_{args.hidden_dim}_output_{args.output_dim}/model'
    
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList
    
    eval_interval = 4

    save_interval = 50
    eval_interval = 50

    train_log_path = f'{log_path}_env_train'
    test_log_path = f'{log_path}_env_test'
    error_log_path = f'{log_path}_env_error'
    
    train_callback = EvalCallback(
        eval_env=train_env,
        log_path=train_log_path,
        deterministic=False,
        eval_freq=eval_interval * agent.n_steps,
        n_eval_episodes=40,
        name='train',
    )
    test_callback = EvalCallback(
        eval_env=test_env,
        log_path=test_log_path,
        deterministic=False,
        eval_freq=eval_interval * agent.n_steps,
        n_eval_episodes=40,
        name='test',
    )
    error_callback = EvalCallback(
        eval_env=error_env,
        log_path=error_log_path,
        deterministic=False,
        eval_freq=eval_interval * agent.n_steps,
        n_eval_episodes=40,
        name='error',
    )
    
    eval_callback = CallbackList([
        train_callback,
        test_callback,
        error_callback,
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
    
    train_args = copy.deepcopy(eval_args)
    test_args = copy.deepcopy(eval_args)
    error_args = copy.deepcopy(eval_args)
    
    train_args.mode = 'train'
    test_args.mode = 'test'
    error_args.mode = 'error'
    
    env = env_wrapper(args)
    train_env = env_wrapper(train_args)
    test_env = env_wrapper(test_args)
    error_env = env_wrapper(error_args)
    env_list = [
        env,
        train_env,
        test_env,
        error_env,
    ]    
    
    train(env_list=env_list, args=args)
    
