import hydra

import torch
import torch.nn as nn


class BaseHead(nn.Module):

    def __init__(
            self,
            name,
            num_layers,
            activation,
            output_activation,
            output_norm,
            bias,
            embed_dim,
            expansion_factor,
            output_dim
    ):

        super().__init__()
        
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.expanded_dim = int(self.embed_dim * expansion_factor)

        self.head = nn.ModuleList()
        if name == "linear":        

            if num_layers == 1:
                self.head.append(nn.Linear(self.embed_dim, self.output_dim, bias=bias))            
            elif num_layers > 1:                
                for layer_idx in range(num_layers):
                    if layer_idx == 0:
                        self.head.append(
                            nn.Linear(self.embed_dim, 
                                       self.expanded_dim,
                                      bias=bias)
                            )
                    elif layer_idx == num_layers-1:
                        self.head.append(
                            nn.Linear(self.expanded_dim, 
                                      self.output_dim, 
                                      bias=bias)
                            )
                    else:
                        self.head.append(
                            nn.Linear(self.expanded_dim, 
                                      self.expanded_dim, 
                                      bias=bias)
                            )
                self.head = nn.Sequential(*self.head)

        elif name == "nonelinear":
            if num_layers == 1:
                self.head.append(nn.Linear(self.embed_dim, self.output_dim, bias=bias))
                if output_activation != None:
                    self.head.append(hydra.utils.instantiate(output_activation)) 
                else:
                    raise ValueError(f"No output_activation: {output_activation} is used. This is just a linear layer!")
            elif num_layers > 1:                
                for layer_idx in range(num_layers):
                    if layer_idx == 0:
                        self.head.append(
                            nn.Linear(self.embed_dim, 
                                      self.expanded_dim,
                                      bias=bias)
                            )
                        if activation != None:
                            self.head.append(hydra.utils.instantiate(activation))
                        else:
                            raise ValueError(f"No activation: {output_activation} is used. This is just a linear layer!")
                    elif layer_idx == num_layers-1:
                        self.head.append(
                            nn.Linear(self.expanded_dim, 
                                      self.output_dim, 
                                      bias=bias)
                            )
                        self.head.append(hydra.utils.instantiate(output_activation)) if output_activation != None else None
                    else:
                        self.head.append(
                            nn.Linear(self.expanded_dim, 
                                      self.expanded_dim, 
                                      bias=bias)
                            )
                        if activation != None:
                            self.head.append(hydra.utils.instantiate(activation))  
            else:
                raise ValueError(f"Only num_layers larger than 0 are possible. Given num_layers: {num_layers}")  
        else:
            raise ValueError(f"{name} is not a BaseFeaturehead. Use name: identity or linear or non_linear.")    

        self.head.append(hydra.utils.instantiate(output_norm, normalized_shape=self.output_dim)) if output_norm != None else None
        self.head = nn.Sequential(*self.head)   


    def forward(self, outputs):
        outputs = self.head(outputs)
        return outputs



class Conv1D_Head(nn.Module):
    
    def __init__(
            self,
            kernel_size,
            num_classes,
    ):

        super().__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0.0
        self.padding = self.kernel_size - (self.kernel_size//2 + 1)
        self.num_classes = num_classes


        self.conv1d_layer_in = nn.Conv1d(in_channels = 1,
                                 out_channels = num_classes,
                                 kernel_size = self.kernel_size,
                                 padding = self.padding)
        
        self.conv1d_layer_middle = nn.Conv1d(in_channels = num_classes,
                                            out_channels = num_classes,
                                            kernel_size = self.kernel_size,
                                            padding = self.padding)
        self.conv1d_layer_out = nn.Conv1d(in_channels = num_classes,
                                            out_channels = num_classes,
                                            kernel_size = self.kernel_size,
                                            padding = self.padding)
        
        self.relu = nn.ReLU()
        
        


    def forward(self, outputs):

        # outputs commes in as b,l,d -> we want for conv1d b*l, 1, d
        b,l,d = outputs.shape
        outputs = outputs.reshape(b*l, 1, d)
        
        outputs = self.conv1d_layer_in(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv1d_layer_middle(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv1d_layer_out(outputs)
        
        # reshape back to batch formulation
        outputs = outputs.reshape(b, l, d, self.num_classes)

        return outputs