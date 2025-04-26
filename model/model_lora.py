import os
import random
import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class LoRA(nn.Module):
    def __init__(self,d, k, r):
        super().__init__()
        self.B = nn.Parameter(torch.zeros((d,r)))
        self.A = nn.Parameter((torch.randn((r,k)))) 
        torch.nn.init.normal_(self.A, mean= 0.0, std= 0.02) # weight initialization i.e divide by sqrt(in_features)
        
        
    def forward(self,x):
        return torch.matmul(x,torch.matmul(self.B, self.A))


def apply_lora(model,r):
    for name, module in model.named_modules():
        
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:

            lora = LoRA(module.weight.shape[0], module.weight.shape[1], 8)
            original_forward = module.forward
            
            def lora_forward(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
        
#             module.forward = lambda x: original_forward(x) + lora(x)
            module.forward = lora_forward
    
            setattr(module, 'lora', lora)


def save_lora(model, output_path):
    lora_parameters = {}
    for (k,v) in model.state_dict().items():
        if 'lora' in k:
            lora_parameters[k] = v
            print(k, v)
            
    torch.save(lora_parameters, output_path)

def load_lora(model, model_path):
    lora_parameters = torch.load(model_path, map_location=model.device)
    for k,v in model.state_dict().items():
        for lk, lv in lora_parameters.items():
            if k == lk:
                v = lv

