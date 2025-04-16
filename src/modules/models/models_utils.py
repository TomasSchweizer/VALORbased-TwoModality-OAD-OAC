import torch
import torch.nn as nn
import  torch.nn.functional as F

class L2NormalizationLayer(nn.Module):
    def __init__(self, normalized_shape):
        super(L2NormalizationLayer, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, x):
        nomalization_dim = torch.where(torch.tensor(x.shape)==self.normalized_shape)[0][-1]

        return F.normalize(x, p=2, dim=nomalization_dim)