import random

import torch
import torch.nn as nn
import numpy as np

class TokenMasker(nn.Module):
    
    def __init__(self, mask_token=None, mask_prob=0.6, range_start=None, range_end=None):
        super().__init__()
        self.mask_token = mask_token
        self.range = [range_start,range_end]
        self.mask_prob = mask_prob

    def forward(self, tokens):
        tokens = tokens.clone() ### important, must have
        tokens, labels = self.perform_mask(tokens)
        return tokens, labels

    
    def perform_mask(self, tokens):
        
        tokens = np.array(tokens.cpu().numpy())

        ### generate indicator first:
        mask_indicator = np.zeros(tokens.shape, dtype=np.int64)
        for i in range(len(mask_indicator)):
            while all(mask_indicator[i] == 0):
                for j in range(1, len(mask_indicator[0])):
                    if tokens[i][j]!=0 and random.random() < self.mask_prob:
                        mask_indicator[i][j] = 1
                        
        labels = -np.ones(tokens.shape, dtype=np.int64)
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                
                if mask_indicator[i][j] == 1 :
                    src_token = tokens[i][j]
                    prob = random.random()   #### e-6 too much time
                    if prob < 0.8:
                        tokens[i][j] = self.mask_token  ### e-6 have no idea why too much 
                    elif prob < 0.9: 
                        tokens[i][j] = random.choice(list(range(*self.range)))   
                    #tokens[i][j] = self.mask_token
                    labels[i][j] = src_token

        # labels = -np.ones(tokens.shape, dtype=np.int64)
        # for i in range(tokens.shape[0]):
        #     for j in range(tokens.shape[1]):                
        #         if mask_indicator[i][j] == 1 :
        #             labels[i][j] = tokens[i][j]
        #             tokens[i][j] = self.mask_token  ### e-6 have no idea why too much                         
                    

        tokens =torch.from_numpy(tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        
        return tokens, labels