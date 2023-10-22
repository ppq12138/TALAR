import os.path
import json
import datetime
import numpy as np
import torch
import torch.nn as nn



class BaseModel(nn.Module):


    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


    def save_checkpoint(self, path, epoch=None):

        path = os.path.join(path, "models")
        if not os.path.exists(path):
            os.makedirs(path)
        
        if epoch is None:
            cpt_path = os.path.join(path, f"{self.__class__.__name__}.jar")
        else:
            cpt_path = os.path.join(path, f"{self.__class__.__name__}_{epoch}.jar")
        
        checkpoint = {
            "name": self.__class__.__name__,
            "model": self.state_dict(),
            "parameters": self.get_hyperparameters(),
        }
        torch.save(checkpoint, cpt_path)

        param = json.dumps(self.get_hyperparameters())
        param_path = os.path.join(path, f"{self.__class__.__name__}Param.json")
        with open(param_path, 'w', encoding='utf8') as f:
            f.write(param)

        print(f"Save checkpoint at {cpt_path}. Save hyper parameters at {param_path}.")


    def load_checkpoint(self, path, epoch=None):
        if epoch is None:
            checkpoint = torch.load(os.path.join(path, "models", f"{self.__class__.__name__}.jar"), map_location=self.device) 
        else:
            try:
                checkpoint = torch.load(os.path.join(path, "models", f"{self.__class__.__name__}.jar"), map_location=self.device) 
            except:
                checkpoint = torch.load(os.path.join(path, "models", f"{self.__class__.__name__}_{epoch}.jar"), map_location=self.device) 
        
        self.load_state_dict(checkpoint["model"])
        self.to(self.device)


    def get_hyperparameters(self) -> dict:
        raise NotImplementedError  



class BaseTrainer:

    def train(self):
        raise NotImplementedError
    
    def save_checkpoint(self, path):
        raise NotImplementedError


    def load_checkpoint(self, path):
        raise NotImplementedError   
    
