import os
import sys
sys.path.append(os.path.join(os.getcwd(), "envs"))
sys.path.append(os.path.join(os.path.join(os.getcwd(), "envs"), "clevr_robot_env"))

import argparse
import datetime
import torch

from algorithms.demonstration import TrajectoryDemonstration
from algorithms.translation.sentence_encoder import BertEncoder
from algorithms.translation.translator import VAETranslationTrainer
from utils.model_utils import load_bert_based_policy_language_model
from utils.utils import  setup_seeds, get_best_cuda




def get_args():
    parser = argparse.ArgumentParser()

    # training config
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--epoch', type=int, default=101)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument("--debug", default=False, action="store_true")

    # load task language model
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--cpt_epoch', default=200, type=int)

    # network params
    parser.add_argument('--language_model', type=str, default="bert")
    parser.add_argument("--bert_dim", default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--num_grammar', default=40, type=int)
    parser.add_argument('--test_ratio', default=0.0512, type=float)
    parser.add_argument("--latent", default=64, type=int)
    parser.add_argument("--kld", default=0.01, type=float)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument("--sample_middle_state", default=True, action="store_false")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    setup_seeds(args.seed)
    max_traj_len = 280
    log_dir = None
    demonstration_path = "dataset/kitchen_demo.npy"

    demonstration = TrajectoryDemonstration()
    demonstration.load(demonstration_path)
    
    device = torch.device(f"cuda:{get_best_cuda()}") if args.cuda else torch.device("cpu")

    # load pretrained policy language model
    policy_language_model = load_bert_based_policy_language_model(args.path, False, device=device, epoch=args.cpt_epoch)

    # setup natural language model
    if args.language_model == "bert":

        language_model = BertEncoder(
            hidden_dim=args.hidden_dim,
            output_dim=args.bert_dim, 
            device=device,
            n_layer=args.n_layer
        )
    else: 
        raise NotImplementedError
    
    trainer = VAETranslationTrainer(
        demonstration=demonstration,
        policy_language_model=policy_language_model,
        nlp_model=language_model,

        human_l_dim=language_model.dim,
        policy_l_dim=policy_language_model.dim,
        latent_dim=args.latent,
        hidden_dim=args.hidden_dim,
        loss_type="mse",

        lr=args.lr,
        kld_coef=args.kld,
        num_epoch=args.epoch,
        batch_size=args.batch_size,
        trajectory_length=max_traj_len,
        device=device,
        log_dir=log_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        demonstration_path=demonstration_path,
        sample_middle_state=args.sample_middle_state,
        test_set_ratio=args.test_ratio,
        partial_num_description=1,
        num_grammar=args.num_grammar
    )
    trainer.train()



if __name__ == "__main__":
    main()