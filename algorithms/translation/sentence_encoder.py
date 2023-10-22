import os
import os.path
import json
import datetime
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel
from .base import BaseModel
from GCP_utils.utils import models_dir
BERT_MODEL_PATH = models_dir.joinpath('bert-base-uncased')
GPT2_MODEL_PATH = models_dir.joinpath('gpt2_small')

 

class BertEncoder(BaseModel):
    def __init__(
        self, hidden_dim=128, output_dim=16, device=torch.device("cpu"), n_layer=2, return_bert=False
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.device = device
        self.return_bert = return_bert

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.model_config = BertConfig.from_pretrained(BERT_MODEL_PATH)
        self.model = [BertModel.from_pretrained(BERT_MODEL_PATH, config=self.model_config)]
        self.model[0].to(device)
        self.bert_dim = 768
        self.vocab_size = self.tokenizer.vocab_size
        self.dim = self.output_dim
        if n_layer == 2:
            self.mlp = nn.Sequential(
                nn.Linear(self.bert_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        else:
            self.mlp = nn.Linear(self.bert_dim, self.output_dim)
                
        self.mlp.to(self.device)
    
    def vocab2id(self, word):
        return self.tokenizer(word)["input_ids"][1]

    def forward(self, text, mask=False, eval=False):
        if isinstance(text, np.ndarray):
            if len(text.shape) > 1:
                text = text.reshape(-1)
            text = text.tolist()
        with torch.no_grad():
            token =  self.tokenizer(text, return_tensors='pt',  padding=True, truncation=True)
        
        pred_labels, pred_positions = None, None
        if mask:
            token, pred_positions, pred_labels = self.mask_sentences(token, eval=eval)

        with torch.no_grad():
            token = token.to(self.device)
            out = self.model[0](**token)

        if pred_positions is None:
            emb = out[0][:, 0, :]
        else:
            batch_size, num_pred_positions = pred_positions.shape
            pred_positions = torch.from_numpy(pred_positions.reshape(-1)).long()
            batch_id = torch.arange(batch_size)
            batch_id = torch.repeat_interleave(batch_id, num_pred_positions)
            
            emb = out[0][batch_id, pred_positions]
            emb = emb.reshape(batch_size, num_pred_positions, -1)
            pred_positions = pred_positions.reshape(-1, 1).long()
            pred_labels = pred_labels.reshape(-1, 1).long()
        
        if self.return_bert:
            return emb

        output = self.mlp(emb)
        if pred_positions is None:
            return output
            
        return output, pred_positions, pred_labels
    
    def mask_sentences(self, tokens, eval=False):
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        batch_size = len(input_ids)
        lengths = attention_mask.sum(1).reshape(-1, 1)

        batch_ids = np.arange(batch_size)
        if eval:
            pred_positions = np.ones(batch_size).astype(np.int16)
        else:
            pred_positions = np.random.randint(1, lengths, (batch_size, 1)).reshape(-1)
        pred_labels = input_ids[batch_ids, pred_positions]
        
        # modify mask
        for b in range(batch_size):
            pred_pos = pred_positions[b]
            tokens["attention_mask"][b][pred_pos+1:] = 0
            tokens["input_ids"][b][pred_pos] = 7308
            tokens["input_ids"][b][pred_pos+1:] = 0
        return tokens, pred_positions.reshape(-1, 1), pred_labels.reshape(-1, 1)

    def get_hyperparameters(self):
        return dict(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_layer=self.n_layer
        )



class FinetunedBertEncoder(BaseModel):
    def __init__(
        self, hidden_dim=128, output_dim=16, device=torch.device("cpu"), tune_flag=False
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.tune_flag = tune_flag

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.model_config = BertConfig.from_pretrained(BERT_MODEL_PATH)
        self.model = BertModel.from_pretrained(BERT_MODEL_PATH, config=self.model_config)
        self.model.to(device)
        self.bert_dim = 768
        self.vocab_size = self.tokenizer.vocab_size
        self.dim = self.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.bert_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.mlp.to(self.device)
    
    def vocab2id(self, word):
        return self.tokenizer(word)["input_ids"][1]

    def forward(self, text, pred_positions=None):
        if isinstance(text, np.ndarray):
            text = text.tolist()
        
        if self.tune_flag:
            token =  self.tokenizer(text, return_tensors='pt',  padding=True, truncation=True)
            token = token.to(self.device)
            out = self.model(**token)
        else:
            with torch.no_grad():
                token =  self.tokenizer(text, return_tensors='pt',  padding=True, truncation=True)
                token = token.to(self.device)
                out = self.model(**token)
            
        if pred_positions is None:
            emb = out[0][:, 0, :]
        else:
            batch_size = pred_positions.shape[0]
            num_pred_positions = pred_positions.shape[1]
            pred_positions = torch.from_numpy(pred_positions.reshape(-1)).long() + 1  
            batch_id = torch.arange(batch_size)
            batch_id = torch.repeat_interleave(batch_id, num_pred_positions)
            
            emb = out[0][batch_id, pred_positions]
            emb = emb.reshape(batch_size, num_pred_positions, -1)
            
        output = self.mlp(emb)
        return output

    def bert_forward(self, text, pred_positions=None):
        if isinstance(text, np.ndarray):
            text = text.tolist()
        
        if self.tune_flag:
            token =  self.tokenizer(text, return_tensors='pt',  padding=True, truncation=True)
            token = token.to(self.device)
            out = self.model(**token)
        else:
            with torch.no_grad():
                token =  self.tokenizer(text, return_tensors='pt',  padding=True, truncation=True)
                token = token.to(self.device)
                out = self.model(**token)
            
        if pred_positions is None:
            emb = out[0][:, 0, :]
        else:
            batch_size = pred_positions.shape[0]
            num_pred_positions = pred_positions.shape[1]
            pred_positions = torch.from_numpy(pred_positions.reshape(-1)).long() + 1   
            batch_id = torch.arange(batch_size)
            batch_id = torch.repeat_interleave(batch_id, num_pred_positions)
            
            emb = out[0][batch_id, pred_positions]
            emb = emb.reshape(batch_size, num_pred_positions, -1)
            
        return emb
    
    def get_hyperparameters(self):
        return dict(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            tune_flag=self.tune_flag
        )