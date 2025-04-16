import torch
import torch.nn as nn

class ContrastiveHead(nn.Module):

    def __init__(self, 
                 embed_dims,
                 contra_dims,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.contra_dims = contra_dims
        self.contra_proj=  nn.Linear(self.embed_dims, self.contra_dims, bias=False)
        self.weighted_proj = nn.Sequential(
            nn.Linear(self.contra_dims, self.contra_dims),
            nn.GELU(),
            nn.Linear(self.contra_dims, 1),
        )

    def forward_contra_proj(self, global_features):
        
        return self.contra_proj(global_features)
    
    def forward_weighted_proj(self, contra_features):
        
        return self.weighted_proj(contra_features)