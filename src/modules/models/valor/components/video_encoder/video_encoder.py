from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from src.modules.models.valor.components.video_encoder.vit_VideoMAEv2 import VisionTransformer
from src.modules.models.valor.components.video_encoder.video_adapters import VideoLadderAdapter
from src.modules.models.valor.components.video_encoder.video_conv_poolers import VideoFeatureMap2DConvPooler, VideoFeatureMap3DConvPooler 


class VideoMAEv2VideoLadderAdapterVideoFeatureMapConvPooler(nn.Module):

    def __init__(
        self,
        video_backbone,
        video_ladder_adapter,
        video_feature_map_conv_pooler,        
    ):
        super().__init__()

        video_backbone_cfg = video_backbone
        video_ladder_adapter_cfg = video_ladder_adapter
        video_feature_map_conv_pooler_cfg = video_feature_map_conv_pooler

        if  video_backbone_cfg.pretrained:
            backbone_checkpoint_path = Path(video_backbone_cfg.pretrained_checkpoints_dir) / video_backbone_cfg.pretrained_checkpoint_name
            backbone_checkpoint = torch.load(backbone_checkpoint_path, map_location = 'cpu')
            video_backbone_cfg_instantiate = OmegaConf.to_container(video_backbone_cfg.instantiate)
            self.video_backbone = VisionTransformer(**video_backbone_cfg_instantiate) #!MODEL CLASS
            missing_keys, unexpected_keys = self.video_backbone.load_state_dict(backbone_checkpoint, strict=True)
            if missing_keys != [] or unexpected_keys != []:
                print(f"Unexpected keys {unexpected_keys}")
                print(f"Missing_keys  {missing_keys}")
        else:
            self.video_backbone = VisionTransformer(**video_backbone_cfg_instantiate)  #!MODEL CLASS
        self.embed_dims = self.video_backbone.embed_dims
        
        if not video_backbone_cfg.unfreeze_backbone:
            self.video_backbone.requires_grad_(False)


        video_ladder_adapter_cfg.depth = video_backbone_cfg.instantiate.depth
        video_ladder_adapter_cfg.conv_adapter.embed_dims = video_backbone_cfg.instantiate.embed_dims
        
        # TODO check if this is the problem
        #self.video_ladder_adapter = VideoLadderAdapter(**video_ladder_adapter_cfg) #!MODEL CLASS
        #self.video_ladder_adapter = nn.Identity()

        video_feature_map_conv_pooler_cfg.embed_dims = video_backbone_cfg.instantiate.embed_dims
        #self.video_feature_map_3Dconv_pooler = VideoFeatureMap3DConvPooler(**video_feature_map_conv_pooler_cfg)
        #self.video_feature_map_2Dconv_pooler = VideoFeatureMap2DConvPooler(**video_feature_map_conv_pooler_cfg)
        #self.video_feature_map_2Dconv_pooler = nn.Identity()

    def forward(self, x):
        """
        x: (b, c, t*2, h, w)
        """

        feat_map_each_layer = self.video_backbone(x) # feat_map_each_layer: List[Tensor(b, t, hp, hw, d)]
        #feat_map_adapted = self.video_ladder_adapter(feat_map_each_layer) # Tensor(b, t, hp, hw, d)
        # TODO check if this is the problem
        #feat_map_adapted = feat_map_adapted[-1]

        #feat_map_conv_pooled = self.video_feature_map_3Dconv_pooler(feat_map_adapted)[None, :,:] # Tensor(1, b, d)
        #feat_map_conv_pooled, feat_map_conv_pooled_mean = self.video_feature_map_2Dconv_pooler(feat_map_adapted) # Tensor(b, t, d)
        feat_map_adapted = None
        feat_map_conv_pooled = None
        feat_map_conv_pooled_mean = feat_map_each_layer[None, :,:] 
        return (feat_map_adapted, feat_map_conv_pooled, feat_map_conv_pooled_mean)