import torch
import torch.nn as nn

from src.modules.models.mamba.mamba_architectures import MambaDecoder

class VideoMambaDecoder(nn.Module):
    
    def __init__(
        self,
        adapter,
        base,
        head,
    ):
        super().__init__()        
              
        adapter_cfg = adapter
        self.encoder_embed_dims = adapter_cfg.encoder_embed_dims
        self.decoder_embed_dims = adapter_cfg.decoder_embed_dims
        self.adapter = nn.Linear(self.encoder_embed_dims, self.decoder_embed_dims, bias=False)        

        base_cfg = base
        base_cfg.embed_dim = self.decoder_embed_dims 
        self.base = MambaDecoder(**base_cfg)
        self.num_layers = self.base.num_layers

        head_cfg = head
        self.num_classes = head_cfg.num_classes
        self.head = nn.Linear(self.decoder_embed_dims, self.num_classes, bias=True)
                              
    def forward(self, feat_map_conv_pooled, inference_params):
        """
        feat_map_conv_pooled: (1, b, d)
        """
        adapted_feat_map_conv_pooled = self.adapter(feat_map_conv_pooled)
        longer_temporal_features = self.base(adapted_feat_map_conv_pooled, inference_params)
        logits = self.head(longer_temporal_features)

        return longer_temporal_features, logits 
                