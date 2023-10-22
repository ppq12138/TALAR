import os
import sys
import argparse
import datetime

import torch

from algorithms.translation.sentence_encoder import BertEncoder
from utils.utils import setup_seeds, get_best_cuda
from algorithms.demonstration import TrajectoryDemonstration
from algorithms.generation.mlm_language_abstraction import RelationModel, LanguageAbstractionTrainer



def get_args():
    parser = argparse.ArgumentParser()

    # training config
    parser.add_argument('--cuda', action="store_true", default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=501)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=1)

    # task language config
    parser.add_argument('--tl_model', type=str, default="relation")
    parser.add_argument('--latent', type=int, default=16)

    # relation model config
    parser.add_argument('--num_obj', type=int, default=5)
    parser.add_argument('--num_pu', type=int, default=2)
    parser.add_argument('--num_pred', type=int, default=4)
    parser.add_argument('--num_variable', type=int, default=2)
    parser.add_argument('--reparameterize', action="store_false", default=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)

    # training parameters  
    parser.add_argument('--n_depth', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sample_middle_state', default=True, action="store_false")
    parser.add_argument('--language_model', default="bert", type=str)
    parser.add_argument('--using_middle_embed', default=True, action="store_false")
    parser.add_argument('--bert_dim', default=32, type=int)
    parser.add_argument('--num_grammar', default=9, type=int)
    parser.add_argument('--test_ratio', default=0.0512, type=float)
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--sample_from_predefined_set', default=False, action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    setup_seeds(args.seed)
    log_dir = None

    # load dataset
    demonstration_path = "/home/yangxy/workspace/language_rl/dataset/demo5_2023-01-13_with_symbol_description.npy"
    print(f"Load demonstration from {demonstration_path}")
    
    demonstration = TrajectoryDemonstration()
    demonstration.load(demonstration_path)
    obsSize = demonstration.fields_attrs["observations"]["shape"][-1]
    device = torch.device(f"cuda:{get_best_cuda()}") if args.cuda else torch.device("cpu")

    print(f"Choose task model: {args.tl_model}")
    if args.tl_model == "relation":
        policy_language_model = RelationModel(
            input_dim=obsSize,
            num_obj=args.num_obj,
            num_variable=args.num_variable,
            num_pu=args.num_pu,
            num_pred=args.num_pred,
            hidden_dim=args.hidden_dim,
            reparameterize=args.reparameterize,
            device=device
        )
    else:
        raise NotImplementedError

    if args.language_model == "bert":
        language_model = BertEncoder(
            output_dim=args.bert_dim, device=device,
            hidden_dim=args.hidden_dim
        )
    else: 
        raise NotImplementedError
    
    
    trainer = LanguageAbstractionTrainer(
        demonstration=demonstration,
        policy_language_model=policy_language_model,
        language_model=language_model,
        obs_dim=obsSize,
        device=device,
        log_dir=log_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        lr=args.lr,
        num_epoch=args.epoch,
        batch_size=args.batch_size,
        sample_middle_state=args.sample_middle_state,
        using_middle_embed=args.using_middle_embed,
        demonstration_path=demonstration_path,
        num_grammar=args.num_grammar,
        test_set_ratio=args.test_ratio,
        sample_from_predefined_set=args.sample_from_predefined_set,
        auto_regressive=True,
        n_layer=args.n_layer,
        env="ball"
    )
    trainer.train()



if __name__ == "__main__":
    main()