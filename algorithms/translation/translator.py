import os
import json
import os.path
import datetime
import copy
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from .base import BaseTrainer, BaseModel
from utils.kitchen_descriptions import LANGUAGE_DESCRIPTION, id2key
from utils.ball_descriptions import get_balls_description, template_list



LOG_MAX = 1E6
LOG_MIN = 1E-8

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20



class VAETranslator(BaseModel):
    def __init__(
        self, 
        human_l_dim=16, 
        policy_l_dim=40, 
        lantent_dim=16, 
        hidden_dim=256, 
        loss_type="bce", 
        seperate_obj_pred_loss=False, 
        num_pu=4, 
        num_pred=4, 
        object_loss_coef=0.1,
        device=torch.device("cpu"),
        latent_type="binary"
    ) -> None:
        """
        seperate_obj_pred_loss: if true, compute predicate logits loss and obejct loss seperatly
        """
        super(VAETranslator, self).__init__()

        self.human_l_dim = human_l_dim
        self.policy_l_dim = policy_l_dim
        self.latent_dim = lantent_dim
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type
        self.seperate_obj_pred_loss = seperate_obj_pred_loss
        self.num_pu = num_pu
        self.num_pred = num_pred
        self.object_loss_coef = object_loss_coef
        self.latent_type = latent_type
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()

        
        self.encoder = nn.Sequential(
            nn.Linear(human_l_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, lantent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(lantent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, policy_l_dim)
        )

        if self.latent_type == "binary":
            self.decoder_act = nn.Sigmoid()

        if loss_type == "mse":    
            self.criterion = nn.MSELoss()
        if self.seperate_obj_pred_loss:
            self.ce_loss = nn.CrossEntropyLoss()

        self.apply(self.weights_init)


    def forward(self, human_l, policy_l):
        batch_size = policy_l.shape[0]

        posterior = self.encoder(human_l)
        mean, log_var = torch.chunk(posterior, 2, dim=1)
        posterior = self.reparamterize(mean, log_var)
        kld_loss = self.compute_kld_loss(mean, log_var)

        recon = self.decoder(posterior)

        if self.seperate_obj_pred_loss:
            predicate_part = policy_l.view(batch_size, self.num_pu, -1)[:, :, :self.num_pred].reshape(batch_size, -1)
            obj_part = policy_l.view(batch_size, self.num_pu, -1)[:, :, self.num_pred:].reshape(batch_size * self.num_pu * 2, -1)
            obj_label = obj_part.argmax(1)

            recon_predicate_part = recon.view(batch_size, self.num_pu, -1)[:, :, :self.num_pred].reshape(batch_size, -1)
            recon_obj_part = recon.view(batch_size, self.num_pu, -1)[:, :, self.num_pred:].reshape(batch_size * self.num_pu * 2, -1)

            if self.loss_type == "mse":
                recon_predicate_part = self.decoder_act(recon_predicate_part)
                pred_loss = self.criterion(recon_predicate_part, predicate_part)
            else:
                pred_loss = self.criterion(recon_predicate_part.view(-1, 1), predicate_part.view(-1, 1))
            obj_loss = self.ce_loss(recon_obj_part, obj_label)
            return {"recon_loss": pred_loss + self.object_loss_coef * obj_loss, "kld_loss": kld_loss, "loss/pred_loss": pred_loss, "loss/obj_loss": obj_loss}
        else:
            if self.latent_type == "binary":
                recon = self.decoder_act(recon)

            if self.loss_type == "mse":
                recon_loss = self.criterion(recon, policy_l)
            else:
                recon_loss = self.criterion(recon.view(-1, 1), policy_l.view(-1, 1))

            return {"recon_loss": recon_loss, "kld_loss": kld_loss}    
    

    def encode(self, human_l, reparameterize=False):

        posterior = self.encoder(human_l)
        mean, std = torch.chunk(posterior, 2, dim=1)
        if reparameterize:
            posterior = self.reparamterize(mean, std)
        else:
            posterior = mean
        recon = self.decoder(mean)
        batch_size = recon.shape[0]
        if self.seperate_obj_pred_loss:
            recon_predicate_part = recon.view(batch_size, self.num_pu, -1)[:, :, :self.num_pred].reshape(batch_size, -1)
            recon_obj_part = recon.view(batch_size, self.num_pu, -1)[:, :, self.num_pred:].reshape(batch_size, -1)
            if self.loss_type == "mse":
                recon_predicate_part = self.decoder_act(recon_predicate_part)
            recon = torch.cat([recon_predicate_part, recon_obj_part], dim=-1).view(batch_size, -1)
        else:
            if self.latent_type == "binary":
                recon = self.decoder_act(recon)

        return recon
    

    @staticmethod
    def reparamterize(mu, log_var):
        std = torch.exp(0.5 *log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


    @staticmethod
    def compute_kld_loss(mean, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)


    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
    

    def get_hyperparameters(self):
        return dict(
            human_l_dim=self.human_l_dim, 
            policy_l_dim=self.policy_l_dim, 
            lantent_dim=self.latent_dim, 
            hidden_dim=self.hidden_dim, 
            loss_type=self.loss_type, 
            seperate_obj_pred_loss=self.seperate_obj_pred_loss, 
            num_pu=self.num_pu, 
            num_pred=self.num_pred, 
            object_loss_coef=self.object_loss_coef,
            latent_type=self.latent_type
        )



class VAETranslationTrainer(BaseTrainer):
    def __init__(
        self, 
        # translator parameter
        human_l_dim=1, 
        policy_l_dim=1, 
        latent_dim=16, 
        hidden_dim=256, 
        loss_type="bce", 
        seperate_obj_pred_loss=False,
        object_loss_coef=0.05,
        # training parameter
        demonstration=None, 
        lr=3e-4,
        joint_training=False, 
        translator_loss_coef=0.1, 
        kld_coef=0.1, 
        num_epoch=int(3e5), 
        batch_size=256, 
        trajectory_length=50, 
        device=torch.device("cpu"), 
        log_dir=None, log_interval=1, 
        save_interval=1000,
        # other model parameter
        policy_language_model=None, 
        nlp_model=None, 
        # record for load checkpoint
        demonstration_path=None, 
        sample_middle_state=False,
        partial_num_description=9,
        num_grammar=1,
        test_set_ratio=0.0512,
        env="kitchen"
    ) -> None:
        """
        joint training: if true, do not load policy language model's network 
                        and backpropogate translator loss to policy language model 
        """

        self.human_l_dim = human_l_dim
        self.policy_l_dim = policy_l_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type
        self.seperate_obj_pred_loss = seperate_obj_pred_loss
        self.object_loss_coef = object_loss_coef
        self.env = env

        self.demonstration = demonstration
        self.lr = lr
        self.joint_training = joint_training
        self.translator_loss_coef = translator_loss_coef
        self.kld_coef = kld_coef

        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.device = device

        self.demonstration_path = demonstration_path
        self.sample_middle_state = sample_middle_state

        self.policy_language_model = policy_language_model
        self.policy_language_model
        self.nlp_model = nlp_model
        self.nlp_model.to(device)

        self.encoder_optim = torch.optim.Adam(self.nlp_model.parameters(), lr=lr)

        latent_type = "binary"
        if hasattr(self.policy_language_model, "latent_type"):
            if self.policy_language_model.latent_type == "continuous":
                latent_type = "continous"

        self.translator = VAETranslator(
            human_l_dim=human_l_dim, policy_l_dim=policy_l_dim, lantent_dim=latent_dim, 
            loss_type=loss_type,
            seperate_obj_pred_loss=self.seperate_obj_pred_loss, object_loss_coef=self.object_loss_coef,
            num_pu=self.policy_language_model.num_pu, num_pred=self.policy_language_model.num_pred,
            hidden_dim=self.hidden_dim, device=self.device, latent_type=latent_type
            )
        self.translator.to(device)
        self.optim = torch.optim.Adam(self.translator.parameters(), lr=lr)

        if self.joint_training:
            self.policy_optim = torch.optim.Adam(self.policy_language_model.parameters(), lr=lr)
        
        self.partial_num_description = partial_num_description
        self.num_grammar = num_grammar
        self.log_dir = f"results/train/VAETranslator/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_t{self.partial_num_description}_g{self.num_grammar}_{loss_type}_hu{human_l_dim}_pol{policy_l_dim}_" \
                       f"l{latent_dim}_lr{self.lr}_kl_{kld_coef}_sep{int(self.seperate_obj_pred_loss)}_" \
                       f"objcoef{self.object_loss_coef}_joint{int(self.joint_training)}_" \
                       f"trancoef_{self.translator_loss_coef}" if log_dir is None else log_dir
        self.test_set_ratio = test_set_ratio

    def train(self):
        self.writer = SummaryWriter(self.log_dir)

        # split train set and test set
        total_idx = np.arange(self.demonstration.sampleSize)
        np.random.shuffle(total_idx)
        num_test = int(len(total_idx) * self.test_set_ratio)
        test_idx = total_idx[:num_test]
        train_idx = total_idx[num_test:]
        train_size = len(train_idx)

        print(f"TOTAL idx: {len(total_idx)}， num_test: {num_test}, len_test: {len(test_idx)}")
        
        for epoch in range(self.num_epoch):
            for indices in self.get_batch_indices(train_idx, self.batch_size, shuffle=True):
                batch_data = self.preprocess_batch(indices)
                obs, next_obs, sentences = batch_data["obs"], batch_data["next_obs"], batch_data["sentences"]
            
                if not self.joint_training:
                    with torch.no_grad():
                        policy_language = self.policy_language_model.encode(obs.to(self.device), next_obs.to(self.device))
                    
                    human_language = self.nlp_model(sentences)  # TODO
                    
                    results = self.translator(human_language, policy_language)
                    loss = self.kld_coef * results["kld_loss"] + results["recon_loss"]
                else:
                    # policy language model loss
                    policy_results, policy_language = self.policy_language_model.compute_loss_and_return_embedding(obs, next_obs)
                    
                    # translator loss
                    human_language = self.nlp_model(sentences)
                    results = self.translator(human_language, policy_language)
                    
                    loss = self.translator_loss_coef * (self.kld_coef * results["kld_loss"] + results["recon_loss"]) + policy_results["recon_loss"]

                if self.joint_training:
                    self.policy_optim.zero_grad()
                self.encoder_optim.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.encoder_optim.step()

                
                if self.joint_training:
                    self.policy_optim.step()
                    results.update({
                        "loss/policy_recon_loss": policy_results["recon_loss"]
                    })

            # eval
            traj_kld_loss, traj_recon_loss, idx = 0., 0., 0
            for indices in self.get_batch_indices(test_idx, self.batch_size, shuffle=False):
                batch_data = self.preprocess_batch(indices)
                obs, next_obs, sentences = batch_data["obs"], batch_data["next_obs"], batch_data["sentences"]
                # forward
                with torch.no_grad():
                    policy_language = self.policy_language_model.encode(obs.to(self.device), next_obs.to(self.device))
                    human_language = self.nlp_model(sentences) 
                    eval_results = self.translator(human_language, policy_language)
                traj_kld_loss += eval_results["kld_loss"].item()
                traj_recon_loss += eval_results["recon_loss"].item()
                idx += 1
            traj_kld_loss = traj_kld_loss if idx == 0 else traj_kld_loss / idx
            traj_recon_loss = traj_recon_loss if idx == 0 else traj_recon_loss / idx
            results.update({"eval/traj_kld_loss": torch.Tensor([traj_kld_loss])})
            results.update({"eval/traj_recon_loss": torch.Tensor([traj_recon_loss])})

            traj_kld_loss, traj_recon_loss, idx = 0., 0., 0
            for indices in self.get_batch_indices(test_idx, self.batch_size, shuffle=False, mode="test"):
                batch_data = self.preprocess_batch(indices)
                obs, next_obs, sentences = batch_data["obs"], batch_data["next_obs"], batch_data["sentences"]
                # forward
                with torch.no_grad():
                    policy_language = self.policy_language_model.encode(obs.to(self.device), next_obs.to(self.device))
                    human_language = self.nlp_model(sentences) 
                    eval_results = self.translator(human_language, policy_language)
                traj_kld_loss += eval_results["kld_loss"].item()
                traj_recon_loss += eval_results["recon_loss"].item()
                idx += 1
            traj_kld_loss = traj_kld_loss if idx == 0 else traj_kld_loss / idx
            traj_recon_loss = traj_recon_loss if idx == 0 else traj_recon_loss / idx
            results.update({"eval/template_kld_loss": torch.Tensor([traj_kld_loss])})
            results.update({"eval/template_recon_loss": torch.Tensor([traj_recon_loss])})

            train_description_set = list(LANGUAGE_DESCRIPTION.values())
            num_goals, total_des = len(train_description_set), len(train_description_set[0])
            train_description_set = np.array(train_description_set)
            test_description_set = (train_description_set[:, self.num_grammar:]).reshape(-1)
            train_description_set = (train_description_set[:, :self.num_grammar]).reshape(-1)
            with torch.no_grad():
                train_tl = self.encode(train_description_set.tolist())
                test_tl = self.encode(test_description_set.tolist())
                total_tl = torch.cat([train_tl, test_tl], dim=0)
            train_std = (train_tl.reshape(num_goals, self.num_grammar, -1).std(1)).mean()
            test_std = (test_tl.reshape(num_goals, total_des - self.num_grammar, -1).std(1)).mean()
            total_std = (total_tl.reshape(num_goals, total_des, -1).std(1)).mean()
            results.update({"eval/train_std": train_std})
            results.update({"eval/test_std": test_std})
            results.update({"eval/total_std": total_std})

            if epoch % self.log_interval == 0:
                if self.joint_training:
                    results.update({
                        "loss/policy_recon_loss": policy_results["recon_loss"]
                    })
                for k in results.keys():
                    self.writer.add_scalar(k, results[k].item(), epoch)
                    print(f"[Epoch {epoch}] {k}: {results[k].item()}")

            if epoch % self.save_interval == 0:
                self.save_checkpoint(self.log_dir, epoch)
    

    def get_batch_indices(self, total_indices, batch_size, shuffle=False, mode='train'):
        shuffle_indices = copy.deepcopy(total_indices)
        if mode == 'train':
            shuffle_indices = list(itertools.product(shuffle_indices, np.arange(self.num_grammar)))
        else:
            if self.env == "kitchen":
                num_total_description = len(LANGUAGE_DESCRIPTION[id2key[0]])
            elif self.env == "ball":
                num_total_description = len(template_list)
            else:
                raise NotImplementedError
            shuffle_indices = list(itertools.product(shuffle_indices, np.arange(self.num_grammar, num_total_description)))

        if shuffle:
            random.shuffle(shuffle_indices)

        start, end = 0, batch_size
        while end < len(total_indices):
            yield shuffle_indices[start:min(end, len(total_indices))]

            start += batch_size
            end += batch_size


    def preprocess_batch(self, indices):
        description_indices = np.array(indices)[:, 1]
        indices = np.array(indices)[:, 0]
        obs = torch.from_numpy(self.demonstration.fields["observations"][indices]).to(self.device).float()
        next_obs = torch.from_numpy(self.demonstration.fields["next_observations"][indices]).to(self.device).float()
        valids = torch.from_numpy(self.demonstration.fields["valid"][indices])
        goals = self.demonstration.fields["goals"][indices].astype(np.int16)

        batch_ids = np.arange(len(obs))
        traj_lengths = (valids.sum(1) - 1).reshape(-1).long()
        next_obs = next_obs[batch_ids, traj_lengths]
        
        sentences = []
        if self.sample_middle_state:
            if self.env == "kitchen":
                tmp_valids = valids
                tmp_obs = torch.zeros_like(next_obs)
                for i in range(len(obs)):
                    tmp_idx = tmp_valids[i].sum()  
                    tmp_idx = np.random.choice(int(tmp_idx))
                    tmp_obs[i] = obs[i][tmp_idx]
                    descriptions = LANGUAGE_DESCRIPTION[id2key[goals[i].item()]][description_indices[i]]
                    sentences.append(descriptions)
                obs = tmp_obs
            elif self.env == "ball":
                tmp_valids = valids
                tmp_obs = torch.zeros_like(next_obs)
                for i in range(len(obs)):
                    tmp_idx = tmp_valids[i].sum()  
                    tmp_idx = np.random.choice(int(tmp_idx))
                    tmp_obs[i] = obs[i][tmp_idx]
                    descriptions = get_balls_description(goals[i], obs[i][tmp_idx].cpu().numpy(), next_obs[i].cpu().numpy(), description_indices[i])
                    sentences.append(descriptions)
                obs = tmp_obs
            else:
                raise NotImplementedError
        return {
            "obs": obs, "next_obs": next_obs, "sentences": sentences
        }


    def encode(self, sentences):
        with torch.no_grad():
            human_language = self.nlp_model(sentences)
            policy_language = self.translator.encode(human_l=human_language)
        
        try:
            if self.policy_language_model.latent_type != "continuous":
                policy_language[policy_language>=0.5] = 1
                policy_language[policy_language<0.5] = 0
        except:
            policy_language[policy_language>=0.5] = 1
            policy_language[policy_language<0.5] = 0
        return policy_language
    

    def save_checkpoint(self, path, epoch=None):
        self.policy_language_model.save_checkpoint(path, epoch)
        self.nlp_model.save_checkpoint(path, epoch)
        self.translator.save_checkpoint(path, epoch)

        path = os.path.join(path, "models")
        if not os.path.exists(path):
            os.makedirs(path)

        if epoch is None:
            cpt_path = os.path.join(path, f"{self.__class__.__name__}.jar")
        else:
            cpt_path = os.path.join(path, f"{self.__class__.__name__}_{epoch}.jar")

        checkpoint = {
            "name": self.__class__.__name__,
            "nlp_model_optim": self.encoder_optim.state_dict(),
            "tranlator_optim": self.optim.state_dict(),
            "parameters": self.get_hyperparameters()
        }
        if self.joint_training:
            checkpoint.update({
                "policy_optim": self.policy_optim.state_dict()
            })
        torch.save(checkpoint, os.path.join(cpt_path))

        param = json.dumps(self.get_hyperparameters())
        param_path = os.path.join(path, f"{self.__class__.__name__}Param.json")
        with open(param_path, 'w', encoding='utf8') as f:
            f.write(param)

        print(f"Save checkpoint at {cpt_path}. Save hyper parameters at {param_path}.")
    

    def load_checkpoint(self, path, epoch):

        self.policy_language_model.load_checkpoint(path, epoch)
        self.nlp_model.load_checkpoint(path, epoch)
        self.translator.load_checkpoint(path, epoch)
        self.translator.to(self.device)

        if epoch is None:
            cpt_path = os.path.join(path, "models", f"{self.__class__.__name__}.jar")
        else:
            cpt_path = os.path.join(path, "models", f"{self.__class__.__name__}_{epoch}.jar")
        checkpoint = torch.load(cpt_path, map_location=self.device) 
        self.encoder_optim.load_state_dict(checkpoint["nlp_model_optim"])
        self.optim.load_state_dict(checkpoint["tranlator_optim"])
        if self.joint_training:
            self.policy_optim.load_state_dict(checkpoint["policy_optim"])

        print("Load checkpoint successfully.")
    
    
    def get_hyperparameters(self):
        return dict(
            # translator parameter
            human_l_dim=self.human_l_dim, 
            policy_l_dim=self.policy_l_dim, 
            latent_dim=self.latent_dim, 
            hidden_dim=self.hidden_dim, 
            loss_type=self.loss_type, 
            seperate_obj_pred_loss=self.seperate_obj_pred_loss,
            object_loss_coef=self.object_loss_coef,
            # training parameter
            lr=self.lr,
            joint_training=self.joint_training, 
            translator_loss_coef=self.translator_loss_coef, 
            kld_coef=self.kld_coef, 
            num_epoch=self.num_epoch, 
            batch_size=self.batch_size, 
            trajectory_length=self.trajectory_length, 
            log_interval=self.log_interval, 
            save_interval=self.save_interval,
            # record for load checkpoint
            demonstration_path=self.demonstration_path, 
            sample_middle_state=self.sample_middle_state,
            partial_num_description=self.partial_num_description,
            num_grammar=self.num_grammar,
            test_set_ratio=self.test_set_ratio,
            env=self.env
        )
