import math

import torch
import torch.nn as nn
from mmengine.model.weight_init import constant_init, trunc_normal_init

class ConvAdapter(nn.Module):

    def __init__(
        self,
        embed_dims,
        scaling_factor,
        kernel_size = 3,
        dilation = 1,     
    ):
        
        super().__init__()

        self.inner_dims = int(embed_dims * scaling_factor)

        self.down_proj = nn.Linear(embed_dims, self.inner_dims )
        trunc_normal_init(self.down_proj, std=0.02, bias=0.0)


        self.activation = nn.GELU()

        # Depthwise convolution
        self.dwconv = nn.Conv1d(
            self.inner_dims ,
            self.inner_dims ,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=self.inner_dims, #! Why self.inner_dims instead of 1
        )
        trunc_normal_init(self.dwconv, mean=0.0, std=math.sqrt(2.0 / self.inner_dims ), bias=0.0)

        # FeedForward via conv1d
        self.conv = nn.Conv1d(self.inner_dims , self.inner_dims , 1)
        trunc_normal_init(self.conv, mean=0.0, std=math.sqrt(2.0 / self.inner_dims ), bias=0.0)

        self.up_proj = nn.Linear(self.inner_dims , embed_dims)
        constant_init(self.up_proj, 0.0)
        
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
            x: Tensor (b, t, hp, wp, d)
        """

        b, t, hp, wp, _ = x.shape
        

        # Down-projection and activation
        x = self.down_proj(x)
        x = self.activation(x)
        
        # Save for residual
        res = x

        # Reshape for depthwise convolution over t
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(b*hp*wp, self.inner_dims, t) # [b*hp*wp,d_inner,t]

        # Depthwise convolution
        x = self.dwconv(x) # [b*hp*wp,d_inner,t]
        x = self.conv(x) # [b*hp*wp,d_inner,t]      

        # Reshape and permute back
        x = x.reshape(b, hp, wp, self.inner_dims, t) # [b,hp,wp,d_inner,t]
        x = x.permute(0, 4, 1, 2, 3) # [b,t,hp,wp,d_inner]

        # skip connection 
        x = res + x

        # Up-projection 
        x = self.up_proj(x) 
        
        x = x * self.gamma
        return x # scaling


class VideoLadderAdapter(nn.Module):
    
    def __init__(
        self,
        depth,
        conv_adapter,
    ):        
        super().__init__()
        
        self.depth = depth
        conv_adapter_cfg = conv_adapter
        
        self.ladder_adapter = nn.ModuleList([])

        for _ in range(self.depth):

            self.ladder_adapter.append(
                ConvAdapter(
                    **conv_adapter_cfg
                )
            )
    
    def forward(self, feat_map_each_layer):
        """
            feat_map_each_layer: List[Tensor (b, t, hp, hw, d)]
        """

        output = feat_map_each_layer[-1] # This is the output of the last layer of the frozen video encoder

        for layer_id in range(self.depth):
            output = output + self.ladder_adapter[layer_id](feat_map_each_layer[layer_id])

        return output