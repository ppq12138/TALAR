import os
import json
import datetime
import torch
import numpy as np
import sys
import os
import sys


from transformers import BertModel, BertConfig, BertTokenizer
from GCP_utils.utils import models_dir
BERT_MODEL_PATH = models_dir.joinpath('bert-base-uncased')

from algorithms.translation.sentence_encoder import  BertEncoder, FinetunedBertEncoder
from algorithms.translation.translator import VAETranslationTrainer


sys.path.append(os.path.join(os.getcwd(), "envs"))
sys.path.append(os.path.join(os.path.join(os.getcwd(), "envs"), "clevr_robot_env"))



def load_bert_based_policy_language_model(path, return_encoder=True, device=torch.device("cpu"), epoch=None):    
    from algorithms.generation.mlm_language_abstraction import RelationModel

    try:
        
        with open(os.path.join(path, "models", "RelationModelParam.json"), "r", encoding="utf-8") as f:
            policy_language_param = json.load(f)
        policy_language_model = RelationModel(device=device, **policy_language_param)
    except:
        raise NotImplementedError
    policy_language_model.load_checkpoint(path, epoch)

    if return_encoder:
        return policy_language_model.encode
    return policy_language_model

    
def load_language_abstract_translator(path, device=torch.device("cpu"), epoch=None):
    try:
        with open(os.path.join(path, "models", "BertEncoderParam.json"), "r", encoding="utf-8") as f:
            language_param = json.load(f)
        language_model = BertEncoder(device=device, **language_param)
    except:
        
        with open(os.path.join(path, "models", "FinetunedBertEncoderParam.json"), "r", encoding="utf-8") as f:
            language_param = json.load(f)
        language_model = FinetunedBertEncoder(device=device, **language_param)
    
    with open(os.path.join(path, "models", "VAETranslationTrainerParam.json"), "r", encoding="utf-8") as f:
        trainer_param = json.load(f)

    policy_language_model = load_bert_based_policy_language_model(path, device=device, epoch=epoch, return_encoder=False)
    
    demonstration = None

    trainer = VAETranslationTrainer(
        log_dir="results/test/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        demonstration=demonstration, 
        policy_language_model=policy_language_model, 
        nlp_model=language_model,
        device=device,
        **trainer_param
    )
    trainer.load_checkpoint(path, epoch)
    return trainer.encode


def load_bert_model(device=torch.device("cpu")):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    model_config = BertConfig.from_pretrained(BERT_MODEL_PATH)
    model = BertModel.from_pretrained(BERT_MODEL_PATH, config=model_config)
    model.to(device)

    def encode(text):
        if isinstance(text, np.ndarray):
            text = text.tolist()

        with torch.no_grad():
            token =  tokenizer(text, return_tensors='pt',  padding=True, truncation=True)
            token = token.to(device)
            out = model(**token)
            emb = out[0][:, 0, :]
        return emb
    return encode


def load_mlp_translator():
    pass


def load_baseline_model(path, device=torch.device("cpu"), kitchen=True):
    pass

