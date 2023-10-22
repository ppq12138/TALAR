import os
import copy
import datetime
import json
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from algorithms.demonstration import BaseDemonstration
from .base import BaseModel, BaseTrainer
from utils.kitchen_descriptions import LANGUAGE_DESCRIPTION, id2key
from utils.ball_descriptions import get_balls_description, template_list


LOG_MAX = 1E6
LOG_MIN = 1E-8

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

predefined_set = ["behind", "back", "left", "front", "right", "red", "blue", "green", "purple", "cyan"]



class RelationModel(BaseModel):
    def __init__(
        self, 
        input_dim=10,
        num_obj=5, 
        num_variable=2, 
        num_pu=4, 
        num_pred=4, 
        hidden_dim=128, 
        reparameterize=True, 
        device=torch.device("cpu"), 
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.num_obj = num_obj
        self.num_variable = num_variable
        self.num_pu = num_pu
        self.num_pred = num_pred
        self.device = device
        self.hidden_dim = hidden_dim
        self.reparameterize = reparameterize
        
        self.attention_network = nn.Sequential(
            nn.Linear(self.input_dim * 2 , self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_obj * self.num_pu * self.num_variable),
        )

        predicate_input_dim = 2 * self.input_dim + self.num_obj * self.num_variable

        self.predicate_networks= nn.ModuleList([nn.Sequential(
            nn.Linear(predicate_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        ) for _ in range(self.num_pred)])

        decoder_input_suffix_dim = self.num_pu * self.num_obj * self.num_variable
        
        self.dim = self.num_pu * self.num_pred + decoder_input_suffix_dim

        self.apply(self.weights_init)
        self.mse_loss = nn.MSELoss()
    

    def forward(self, obs, next_obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float().to(self.device)
            next_obs = torch.from_numpy(next_obs).float().to(self.device)

        latent = self.encode(obs, next_obs)
        return latent
        

    def encode(self, obs, next_obs, split_predicate_and_objects=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).to(self.device).contiguous().float()
            next_obs = torch.from_numpy(next_obs).to(self.device).contiguous().float()
        
        # calculate attentioned object indices
        batch_size = obs.shape[0]
        attention = self.attention_network(torch.cat([obs, next_obs], dim=-1))  # batch_size x num_objects x num_pu x num_variable
        attention = attention.view(batch_size, self.num_pu, self.num_variable, -1)
        attention = F.gumbel_softmax(attention, -2)
        if self.reparameterize:
            hard_code = torch.zeros_like(attention.view(batch_size * self.num_pu * self.num_variable, -1))
            idx = torch.max(attention.view(batch_size * self.num_pu * self.num_variable, -1), -1)[1]   # TODO
            hard_code = hard_code.scatter_(1, idx.view(-1, 1), 1).view(batch_size, self.num_pu, self.num_variable, -1) # TODO
            attention = attention + (hard_code - attention).detach()  # TODO
        attention_one_hot = attention # TODO
        attention = attention.repeat_interleave(2, -1).view(batch_size * self.num_pu, 2, -1)
        
        # calculate the outputs of predicate networks
        one_hot_pred_list = []
        repeat_obs = obs.repeat_interleave(self.num_pu, 0)
        repeat_next_obs = next_obs.repeat_interleave(self.num_pu, 0)
        for pred in range(self.num_pred):
            pred_input = torch.cat([repeat_obs, repeat_next_obs, attention_one_hot.view(batch_size * self.num_pu, -1)], dim=-1)
            pred_outputs = self.predicate_networks[pred](pred_input)
            one_hot_pred = F.gumbel_softmax(pred_outputs, dim=1)
            if self.reparameterize:
                sample_pred = torch.eye(2, dtype=one_hot_pred.dtype, device=one_hot_pred.device)[torch.max(one_hot_pred, dim=-1)[1]][:, 0]
                one_hot_pred = sample_pred.detach() + one_hot_pred[:, 0] - (one_hot_pred.detach())[:, 0]
            else:
                one_hot_pred = one_hot_pred[:, 0]
            one_hot_pred_list.append(one_hot_pred.view(batch_size, self.num_pu, -1))
        
        one_hot_outputs = torch.cat(one_hot_pred_list, dim=-1)
        if split_predicate_and_objects:
            return attention_one_hot, one_hot_outputs
        
        one_hot_outputs = torch.cat([one_hot_outputs, attention_one_hot.view(batch_size, self.num_pu, -1)], -1)
        return one_hot_outputs.view(batch_size, -1)


    def get_hyperparameters(self):
        return dict(
            input_dim=self.input_dim,
            num_variable=self.num_variable, 
            num_pu=self.num_pu, 
            num_pred=self.num_pred, 
            hidden_dim=self.hidden_dim, 
            reparameterize=self.reparameterize,
            num_obj=self.num_obj
        )


class LanguageAbstractModel(BaseModel):
    def __init__(
        self,
        policy_language_dim=10,
        language_dim=768,
        language_model=None,
        policy_language_model=None,
        vocab_size=None,
        hidden_dim=128,
        mask_num=2,                 
        device=torch.device("cpu"),
        supervised_learning=False,
        using_middle_embed=False,
        sample_from_predefined_set=False,
        auto_regressive=False,      
        n_layer=3
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.policy_language_model = policy_language_model
        
        self.policy_language_dim = policy_language_dim
        self.language_dim = language_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.auto_regressive = auto_regressive
        self.mask_num = mask_num if not auto_regressive else 1
        self.device = device
        self.supervised_learning = supervised_learning
        self.using_middle_embed = using_middle_embed
        self.sample_from_predefined_set = sample_from_predefined_set

        mlp_input_dim = self.language_dim if self.supervised_learning else self.policy_language_dim + self.language_dim
        mlp_output_dim = self.vocab_size if self.using_middle_embed else  self.vocab_size * self.mask_num
        if n_layer == 3:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, mlp_output_dim),
            )  
        else:
            self.mlp = nn.Linear(mlp_input_dim, mlp_output_dim)
        
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.weights_init)


    def forward(self, obs, next_obs, sentences, eval=False):
        policy_language = self.policy_language_model(obs, next_obs)
        if self.using_middle_embed:
            masked_language_embedding, _, pred_words_idx = self.language_model(sentences, mask=True, eval=eval)
            masked_language_embedding = masked_language_embedding.view(-1, self.language_dim)
            policy_language = policy_language.repeat_interleave(pred_words_idx.shape[1], 0)
        else:
            raise NotImplementedError

        if self.supervised_learning:
            pred_input = masked_language_embedding
        else:
            pred_input = torch.cat([policy_language, masked_language_embedding], dim=-1)

        pred_output = self.mlp(pred_input).view(-1, self.vocab_size)
        mlm_loss = self.criterion(pred_output, pred_words_idx.long().to(pred_output.device).reshape(-1))
        return {
            "loss/mlm_loss": mlm_loss,
            "latent/mean": policy_language.float().mean(), 
            "latent/std_over_sample": policy_language.std(1).mean(),
            "latent/std_over_batch": policy_language.std(0).mean()
        }


    def save_checkpoint(self, path, epoch=None):
        self.policy_language_model.save_checkpoint(path, epoch)
        self.language_model.save_checkpoint(path, epoch)
        return super().save_checkpoint(path, epoch)
    

    def load_checkpoint(self, path, epoch):
        self.policy_language_model.load_checkpoint(path, epoch)
        self.language_model.load_checkpoint(path, epoch)
        return super().load_checkpoint(path, epoch)

    
    def get_hyperparameters(self):
        return dict(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            mask_num=self.mask_num,
            policy_language_dim=self.policy_language_dim,
            language_dim=self.language_dim,
            supervised_learning=self.supervised_learning,
            sample_from_predefined_set=self.sample_from_predefined_set,
            auto_regressive=self.auto_regressive,
            using_middle_embed=self.using_middle_embed
        )



class LanguageAbstractionTrainer(BaseTrainer):
    def __init__(
        self,
        demonstration: BaseDemonstration=None, 
        policy_language_model=None,
        language_model=None,
        obs_dim=10, 
        trajectory_length=50,             
        device=torch.device("cpu"),         
        lr=3E-4, 
        log_interval=1000, 
        save_interval=10000, 
        num_epoch=1000000, 
        batch_size=256, 
        log_dir=None,
        sample_middle_state=False,    
        supervised_learning=False,        
        using_middle_embed=False,       
        demonstration_path=None,
        partial_num_description=1,
        num_grammar=1,
        test_set_ratio=0.0512,
        sample_from_predefined_set=False,
        auto_regressive=False,
        n_layer=3,
        env="kitchen"                       # kitchen or ball
    ) -> None:

        self.demonstration = demonstration 
        self.policy_language_model = policy_language_model
        self.policy_language_model.to(device)
        self.language_model = language_model
        self.language_model.to(device)
        self.obs_dim = obs_dim
        self.trajectory_length = trajectory_length
        self.device = device     
        self.lr = lr
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.demonstration_path = demonstration_path
        self.supervised_learning = supervised_learning
        self.sample_middle_state = sample_middle_state
        self.using_middle_embed = using_middle_embed
        self.sample_from_predefined_set = sample_from_predefined_set
        self.auto_regressive = auto_regressive
        self.env = env

        self.language_abstract_model = LanguageAbstractModel(
            language_model=self.language_model,
            policy_language_model=self.policy_language_model,
            vocab_size=self.language_model.vocab_size,
            policy_language_dim=self.policy_language_model.dim,
            language_dim=self.language_model.dim,
            device=self.device,
            supervised_learning=self.supervised_learning,
            using_middle_embed=using_middle_embed,
            sample_from_predefined_set=self.sample_from_predefined_set,
            auto_regressive=self.auto_regressive,
            n_layer=n_layer
        )
        self.language_abstract_model.to(device)

        self.optim = torch.optim.Adam(self.language_abstract_model.parameters(), lr=self.lr)
        self.policy_language_optim = torch.optim.Adam(self.policy_language_model.parameters(), lr=self.lr)
        try:
            self.language_optim = torch.optim.Adam(self.language_model.parameters(), lr=self.lr)
        except:
            pass

        num_balls = self.policy_language_model.num_obj if hasattr(self.policy_language_model, "num_obj") else 0
        self.partial_num_description = partial_num_description
        self.num_grammar = num_grammar
        self.log_dir = f"results/train/{self.__class__.__name__}/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"{type(self.language_model).__name__}_{type(self.policy_language_model).__name__}_ball{num_balls}_t{self.partial_num_description}_g{self.num_grammar}_midemb{int(self.using_middle_embed)}_su{int(self.supervised_learning)}_arg{self.policy_language_model.num_variable}_pu{self.policy_language_model.num_pu}_pred{self.policy_language_model.num_pred}_middle{self.sample_middle_state}_lr{self.lr}" if log_dir is None else log_dir
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

        print(f"Total indices {len(total_idx)}, num_test: {num_test}, num_train: {train_size}")

        for epoch in range(self.num_epoch):
            for indices in self.get_batch_indices(train_idx, self.batch_size, shuffle=True):
                batch_data = self.preprocess_batch(indices)
                obs, next_obs, sentences = batch_data["obs"], batch_data["next_obs"], batch_data["sentences"]
            
                # forward
                results = self.language_abstract_model(obs.to(self.device), next_obs.to(self.device), sentences)

                # backward training
                loss = results["loss/mlm_loss"]
                self.optim.zero_grad()
                self.policy_language_optim.zero_grad()
                try:
                    self.language_optim.zero_grad()
                except:
                    pass
                loss.backward()
                self.optim.step()
                self.policy_language_optim.step()
                try:
                    self.language_optim.step()
                except:
                    pass

            # eval 
            traj_loss, idx = 0., 0
            for indices in self.get_batch_indices(test_idx, self.batch_size, shuffle=False):
                batch_data = self.preprocess_batch(indices)
                obs, next_obs, sentences = batch_data["obs"], batch_data["next_obs"], batch_data["sentences"]
                # forward
                with torch.no_grad():
                    eval_results = self.language_abstract_model(obs.to(self.device), next_obs.to(self.device), sentences, eval=True)
                traj_loss += eval_results["loss/mlm_loss"].item()
                idx += 1
            traj_loss = traj_loss if idx == 0 else traj_loss / idx
            results.update({"eval/traj_mlm_loss": torch.Tensor([traj_loss])})

            traj_loss, idx = 0., 0
            for indices in self.get_batch_indices(test_idx, self.batch_size, shuffle=False, mode="test"):
                batch_data = self.preprocess_batch(indices)
                obs, next_obs, sentences = batch_data["obs"], batch_data["next_obs"], batch_data["sentences"]
                # forward
                with torch.no_grad():
                    eval_results = self.language_abstract_model(obs.to(self.device), next_obs.to(self.device), sentences, eval=True)
                traj_loss += eval_results["loss/mlm_loss"].item()
                idx += 1
            traj_loss = traj_loss if idx == 0 else traj_loss / idx
            results.update({"eval/template_mlm_loss": torch.Tensor([traj_loss])})

            # log and save checkpoint
            if epoch % self.log_interval == 0:
                print(f"[Epoch {epoch}]", '-' * 40)
                for k in results.keys():
                    self.writer.add_scalar(k, results[k].item(), epoch)
                    print(f"{k}: {results[k].item()}")

            if epoch % self.save_interval == 0 or epoch == self.num_epoch - 1:
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

        if len(total_indices) <= batch_size:
            yield shuffle_indices

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
            elif self.env == "ball":
                tmp_valids = valids
                tmp_obs = torch.zeros_like(next_obs)
                for i in range(len(obs)):
                    tmp_idx = tmp_valids[i].sum() 
                    tmp_idx = np.random.choice(int(tmp_idx))
                    tmp_obs[i] = obs[i][tmp_idx]
                    descriptions = get_balls_description(goals[i], obs[i][tmp_idx], next_obs[i], description_indices[i])
                    sentences.append(descriptions)
                obs = tmp_obs
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return {
            "obs": obs, "next_obs": next_obs, "sentences": sentences
        }


    def save_checkpoint(self, path, epoch=None):
        self.language_abstract_model.save_checkpoint(path, epoch)

        path = os.path.join(path, "models")
        if not os.path.exists(path):
            os.makedirs(path)
        
        if epoch is None:
            cpt_path = os.path.join(path, f"{self.__class__.__name__}.jar")
        else:
            cpt_path = os.path.join(path, f"{self.__class__.__name__}_{epoch}.jar")

        checkpoint = {
            "name": f"{self.__class__.__name__}",
            "optim": self.optim.state_dict(),
            "policy_language_optim": self.policy_language_optim.state_dict(),
            "parameters": self.get_hyperparameters()
        }
        torch.save(checkpoint, cpt_path)

        param = json.dumps(self.get_hyperparameters())
        param_path = os.path.join(path, f"{self.__class__.__name__}Param.json")
        with open(param_path, 'w', encoding='utf8') as f:
            f.write(param)

        print(f"Save checkpoint at {cpt_path}. Save hyper parameters at {param_path}.")
    

    def load_checkpoint(self, path, epoch=None):
        self.language_abstract_model.load_checkpoint(path, epoch)
        if epoch is None:
            cpt_path = os.path.join(path, "models", f"{self.__class__.__name__}.jar")
        else:
            cpt_path = os.path.join(path, "models", f"{self.__class__.__name__}_{epoch}.jar")
        
        checkpoint = torch.load(cpt_path,  map_location=self.device)
        self.optim.load_state_dict(checkpoint["optim"])
        self.policy_language_optim.load_state_dict(checkpoint["policy_language_optim"])
        self.language_optim.load_state_dict(checkpoint["language_optim"])
        print(f"Load checkpoint from {path}. Epoch {epoch}")
    
    
    def get_hyperparameters(self):
        return dict(
            obs_dim=self.obs_dim, 
            trajectory_length=self.trajectory_length,               
            lr=self.lr, 
            log_interval=self.log_interval, 
            save_interval=self.save_interval, 
            num_epoch=self.num_epoch, 
            batch_size=self.batch_size, 
            sample_middle_state=self.sample_middle_state,
            supervised_learning=self.supervised_learning,
            using_middle_embed=self.using_middle_embed,
            demonstration_path=self.demonstration_path,
            partial_num_description=self.partial_num_description,
            num_grammar=self.num_grammar,
            test_set_ratio=self.test_set_ratio,
            sample_from_predefined_set=self.sample_from_predefined_set,
            auto_regressive=self.auto_regressive,
            env=self.env
        )