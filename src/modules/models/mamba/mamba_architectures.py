import math
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from src.modules.models.mamba.components.mamba_blocks import MambaBlocks

class VideoMAEv2BackboneTransformerDecoderAdapterMambaBaseHead(nn.Module):
    
    def __init__(
        self,
        backbone, 
        adapter,
        base,
        head,
        weights_init,
        data_info,
        criterions,
        metrics,
    ):

        super().__init__()

        self.backbone_cfg = backbone
        if  self.backbone_cfg.pretrained:
            backbone_checkpoint_path = Path(self.backbone_cfg.pretrained_checkpoints_dir) / self.backbone_cfg.pretrained_checkpoint_name
            self.backbone_checkpoint = torch.load(backbone_checkpoint_path, map_location = 'cpu')
            self.backbone_cfg.instantiate = OmegaConf.to_container(self.backbone_cfg.instantiate)
            self.backbone = hydra.utils.instantiate(self.backbone_cfg.instantiate, _recursive_=False, _convert_="partial")
            self.backbone.load_state_dict(self.backbone_checkpoint, strict=True)
        else:
            self.backbone = hydra.utils.instantiate(self.backbone_cfg.instantiate, _recursive_=False)
        self.embed_dim = self.backbone.embed_dims
        if not self.backbone_cfg.unfreeze_backbone:
            self.backbone.requires_grad_(False)

        self.adapter_cfg = adapter
        self.adapter_cfg.decoder.decoder_layer["d_model"] = self.embed_dim
        self.adapter = hydra.utils.instantiate(self.adapter_cfg, _recursive_=False) 

        self.base_cfg = base
        self.base_cfg["embed_dim"] = self.embed_dim
        self.mamba_decoder = hydra.utils.instantiate(self.base_cfg, _recursive_=False)

        self.head_cfg = head
        self.head_cfg["embed_dim"] = self.embed_dim
        self.head = hydra.utils.instantiate(self.head_cfg, _recursive_=False)

        self.weights_init_cfg = weights_init
        self._weights_init()
        
        self.data_info = data_info

        # Criterions
        self.criterions_cfg = criterions
        
        self.losses = nn.ModuleDict()
        self.losses_cfgs = {}
        for loss, loss_cfg in self.criterions_cfg.losses.items():              
            self.losses[loss] = hydra.utils.instantiate(loss_cfg.instantiate)
            self.losses_cfgs[loss] =  loss_cfg


        self.train_metrics_cfg = metrics.train
        self.train_metrics = nn.ModuleDict()
        self.train_metrics_cfgs = {}
        for metric, metric_cfg in self.train_metrics_cfg.items():
            metric_name = f"train_{metric}"
            self.train_metrics[metric_name] = hydra.utils.instantiate(metric_cfg.instantiate)
            self.train_metrics_cfgs[metric_name] = metric_cfg


        self.valid_metrics_cfg = metrics.valid
        self.valid_metrics = nn.ModuleDict()
        self.valid_metrics_cfgs = {}
        for metric, metric_cfg in self.valid_metrics_cfg.items():
            metric_name = f"valid_{metric}"
            self.valid_metrics[metric_name] = hydra.utils.instantiate(metric_cfg.instantiate)
            self.valid_metrics_cfgs[metric_name] = metric_cfg

    def _weights_init(self):
      
        n_backbone_layers = self.backbone_cfg.instantiate.depth
        
        adapter_weigths_init_cfg = self.weights_init_cfg.adapter_weights_init
        for name, module in self.adapter.named_modules():
            if isinstance(module, nn.Linear):
                hydra.utils.call(adapter_weigths_init_cfg.linear.weight, tensor=module.weight)
                if module.bias is not None:
                    hydra.utils.call(adapter_weigths_init_cfg.linear.bias, tensor=module.bias)             
            elif isinstance(module, nn.LayerNorm):                
                hydra.utils.call(adapter_weigths_init_cfg.layer_norm.weight, tensor=module.weight)
                if module.bias is not None:
                    hydra.utils.call(adapter_weigths_init_cfg.layer_norm.bias, tensor=module.bias)
        
        if adapter_weigths_init_cfg.rescale_prenorm_residual:
            for p_name, p in self.adapter.named_parameters():
                if len(p_name.split(".")) > 2:
                    layer_idx = int(p_name.split(".")[2])
                    p_name = ".".join(p_name.split(".")[3:6])
                if p_name in adapter_weigths_init_cfg.linear.names:
                    hydra.utils.call(adapter_weigths_init_cfg.linear.weight, tensor=p)
                    with torch.no_grad():
                        layer_idx = n_backbone_layers + layer_idx + 1 
                        p /= math.sqrt(adapter_weigths_init_cfg.n_residuals_per_layer * layer_idx)

        n_backbone_adap_layers = n_backbone_layers + self.adapter_cfg.decoder.num_layers

        mamba_decoder_weigths_init_cfg = self.weights_init_cfg.mamba_decoder_weigths_init
        for name, module in self.mamba_decoder.named_modules():
            # Not accessed because no bias is used in linear layers of mamba decoder at the moment
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        hydra.utils.call(mamba_decoder_weigths_init_cfg.linear.bias, tensor=module.bias)     
            # rescales out projection linear layer for each layer. Making them smaller the later the layer. 
            if mamba_decoder_weigths_init_cfg.rescale_prenorm_residual:
                for p_name, p in module.named_parameters():
                    if p_name in mamba_decoder_weigths_init_cfg.linear.names:
                        hydra.utils.call(mamba_decoder_weigths_init_cfg.linear.weight, tensor=p)
                        with torch.no_grad():
                            layer_idx = n_backbone_adap_layers + module.layer_idx + 1 
                            p /= math.sqrt(mamba_decoder_weigths_init_cfg.n_residuals_per_layer * layer_idx)

        n_backbone_adap_base_layers = n_backbone_adap_layers + self.mamba_decoder.num_layers

        # Init head weigths
        head_weigths_init_cfg = self.weights_init_cfg.head_weigths_init
        for name, module in self.head.named_modules():
             if isinstance(module, nn.Linear):
                hydra.utils.call(head_weigths_init_cfg.linear.weight, tensor=module.weight)
                layer_idx = n_backbone_adap_base_layers 
                module.weight.data = module.weight.data / math.sqrt(mamba_decoder_weigths_init_cfg.n_residuals_per_layer * layer_idx)
                if module.bias is not None:
                    hydra.utils.call(head_weigths_init_cfg.linear.bias, tensor=module.bias)

    def forward(self, batch, inference_params):
        """
        batch_images: l, c, t, h, w         
        """

        batch_clips = batch["clips"]
        b,l,c,t,h,w = batch_clips.shape
        d = self.embed_dim
        t_half = t//2
        hp = h // self.backbone_cfg.instantiate.patch_size
        wp = w // self.backbone_cfg.instantiate.patch_size
        n_tokens = t_half*hp*wp

        features = self.backbone(batch_clips[0,:,:,:,:,:]) # l, d, t//2, h//p, w//p
        features = torch.reshape(features, (l, d, n_tokens))
        features = torch.permute(features, dims=(0,2,1))
        
        adapted_features = self.adapter(features) # l, num_queries, d
        nq = adapted_features.shape[1]
        adapted_features = torch.reshape(adapted_features, (1,l*nq,d)) # 1, l*nq, d

        representation = self.mamba_decoder(adapted_features, inference_params)
        representation = representation[:,(nq-1)::nq,:]

        logits = self.head(representation)

        batch_outputs = {}
        batch_outputs["logits"] = logits # b, l, num_classes        

        return batch_outputs   
      

    def calculate_loss(self, batch_outputs, batch_targets, phase):

        loss = torch.zeros((1), device=batch_targets["onehot"].device)
        for loss_name, loss_func in self.losses.items():            
            loss_cfg = self.losses_cfgs[loss_name]
            
            if loss_cfg.form == "flattened":
                batch_outputs, batch_targets = _flatten_batch_dim_outputs_and_targets(batch_outputs, batch_targets)
                outputs_for_loss, targets_for_loss = _prepare_outputs_and_targets(batch_outputs, batch_targets, loss_cfg, self.data_info, )
                                
                if not outputs_for_loss.numel() == 0 and not targets_for_loss.numel() == 0:
                    loss_weight = loss_cfg.weight
                    loss = loss + loss_weight * loss_func(outputs_for_loss, targets_for_loss) 
                
                if torch.is_tensor(loss):
                    if torch.isnan(loss):
                        raise RuntimeError(f"loss is :{loss}")
            
            elif loss_cfg.form == "batch":
                
                batch_outputs_for_loss, batch_targets_for_loss = _prepare_batch_outputs_and_targets(batch_outputs, batch_targets, loss_cfg, self.data_info)
                
                if not batch_outputs_for_loss.numel() == 0 and not batch_targets_for_loss.numel() == 0:
                    loss_weight = loss_cfg.weight
                    loss = loss + loss_weight * loss_func(batch_outputs_for_loss, batch_targets_for_loss) 
            
            else:
                raise ValueError(f"Loss form <{loss_cfg.form}> isn't implemented or is not set!")

        return loss
    
    def compute_metrics_in_train_step(self, batch_outputs, batch_targets):

        train_metrics_step = {}
        for metric_name, metric in self.train_metrics.items():
            if metric_name not in ["train_mAP", "train_action_mAP", "train_ConfusionMatrix"]:
                metric_cfg = self.train_metrics_cfgs[metric_name]    
                if metric_cfg.form == "flattened":        
                    batch_outputs, batch_targets = _flatten_batch_dim_outputs_and_targets(batch_outputs, batch_targets)
                    outputs_for_metric, targets_for_metric = _prepare_outputs_and_targets(batch_outputs, batch_targets, metric_cfg, self.data_info)
                    if not outputs_for_metric.numel() == 0 and not targets_for_metric.numel() == 0:

                        if metric_cfg.get("apply_softmax", False):
                            outputs_for_metric = F.softmax(outputs_for_metric, dim=1)
                        elif metric_cfg.get("apply_sigmoid", False):
                            outputs_for_metric = F.sigmoid(outputs_for_metric)
                        
                        # TODO delete later
                        # outputs_for_metric = outputs_for_metric[-1]
                        # targets_for_metric = targets_for_metric[-1]

                        # metric_value = metric(outputs_for_metric[None].detach(), targets_for_metric[None].detach())
                        metric_value = metric(outputs_for_metric.detach(), targets_for_metric.detach())

                        if metric_value.numel() == 1:
                            wandb_name = "/".join(metric_name.split("_", 1))
                            train_metrics_step[wandb_name] = metric_value
                        
                        elif metric_value.numel() == self.data_info.used_num_classes:
                            for class_idx, class_name in enumerate(self.data_info.used_class_names):
                                wandb_name = "".join(metric_name.split("_", 1)) + "/" + class_name
                                train_metrics_step[wandb_name] = metric_value[class_idx]
                        
                        elif metric_value.numel() == self.data_info.used_num_classes_without:
                            for class_idx, class_name in enumerate(self.data_info.used_class_names_without):
                                wandb_name = "".join(metric_name.split("_", 1)) + "/" + class_name
                                train_metrics_step[wandb_name] = metric_value[class_idx]         

                elif metric_cfg.form == "batch":
                    raise ValueError(f"Metric form {metric_cfg.form} isn't implemented yet!")
                else:
                    raise ValueError(f"Metric form <{metric_cfg.form}> isn't implemented or is not set!")

        return train_metrics_step
        
    def compute_metrics_in_valid_step(self, batch_outputs, batch_targets):

        valid_metrics_step = {}
        for metric_name, metric in self.valid_metrics.items():
            if metric_name not in ["valid_mAP", "valid_action_mAP", "valid_ConfusionMatrix"]:
                metric_cfg = self.valid_metrics_cfgs[metric_name]    
                if metric_cfg.form == "flattened":        
                    batch_outputs, batch_targets = _flatten_batch_dim_outputs_and_targets(batch_outputs, batch_targets)
                    outputs_for_metric, targets_for_metric = _prepare_outputs_and_targets(batch_outputs, batch_targets, metric_cfg, self.data_info)
                    if not outputs_for_metric.numel() == 0 and not targets_for_metric.numel() == 0:

                        if metric_cfg.get("apply_softmax", False):
                            outputs_for_metric = F.softmax(outputs_for_metric, dim=1)
                        elif metric_cfg.get("apply_sigmoid", False):
                            outputs_for_metric = F.sigmoid(outputs_for_metric)
                        
                        metric_value = metric(outputs_for_metric.detach(), targets_for_metric.detach())
                        
                        if metric_value.numel() == 1:                        
                            wandb_name = "/".join(metric_name.split("_", 1))
                            valid_metrics_step[wandb_name] = metric_value
                        
                        elif metric_value.numel() == self.data_info.used_num_classes:
                            for class_idx, class_name in enumerate(self.data_info.used_class_names):
                                wandb_name = "".join(metric_name.split("_", 1)) + "/" + class_name
                                valid_metrics_step[wandb_name] = metric_value[class_idx]
                        
                        elif metric_value.numel() == self.data_info.used_num_classes_without:
                            for class_idx, class_name in enumerate(self.data_info.used_class_names_without):
                                wandb_name = "".join(metric_name.split("_", 1)) + "/" + class_name
                                valid_metrics_step[wandb_name] = metric_value[class_idx]         

                elif metric_cfg.form == "batch":
                    raise ValueError(f"Metric form {metric_cfg.form} isn't implemented yet!")
                else:
                    raise ValueError(f"Metric form <{metric_cfg.form}> isn't implemented or is not set!")

        return valid_metrics_step

    def compute_metrics_on_train_epoch_end(self):

        train_metrics_epoch = {}

        train_metric = self.train_metrics["train_mAP"].compute()        
        
        if isinstance(train_metric, dict): 
            train_metrics_epoch["train/mAP"] = train_metric["mAP"]
            for name, value in train_metric["per_class_AP"].items():                
                wandb_name = "/".join(("train"+name).split("_", 1))
                train_metrics_epoch[wandb_name] = value  
        else:
            train_metrics_epoch["train/mAP"] = train_metric

        if "train_ConfusionMatrix" in self.train_metrics.keys():
            self.train_metrics["train_ConfusionMatrix"].compute(phase="train")


        return train_metrics_epoch
    
    def compute_metrics_on_valid_epoch_end(self):

        valid_metrics_epoch = {}

        valid_metric = self.valid_metrics["valid_mAP"].compute()
     
        if isinstance(valid_metric, dict): 
            valid_metrics_epoch["valid/mAP"] = valid_metric["mAP"]
            for name, value in valid_metric["per_class_AP"].items():                
                wandb_name = "/".join(("valid"+name).split("_", 1))
                valid_metrics_epoch[wandb_name] = value  
        else:
            valid_metrics_epoch["valid/mAP"] = valid_metric   

        if "valid_ConfusionMatrix" in self.valid_metrics.keys():
            self.valid_metrics["valid_ConfusionMatrix"].compute(phase="valid")
        
        return valid_metrics_epoch

    def update_train_metrics_states(self, batch_outputs, batch_targets):
        
        for metric_name, metric in self.train_metrics.items():
            if metric_name in ["train_mAP", "train_action_mAP", "train_ConfusionMatrix"]:
                metric_cfg = self.train_metrics_cfgs[metric_name]    
                if metric_cfg.form == "flattened":        
                    batch_outputs, batch_targets = _flatten_batch_dim_outputs_and_targets(batch_outputs, batch_targets)
                    outputs_for_metric, targets_for_metric = _prepare_outputs_and_targets(batch_outputs, batch_targets, metric_cfg, self.data_info)
                    if not outputs_for_metric.numel() == 0 and not targets_for_metric.numel() == 0:

                        if metric_cfg.get("apply_softmax", False):
                            outputs_for_metric = F.softmax(outputs_for_metric, dim=1)
                        elif metric_cfg.get("apply_sigmoid", False):
                            outputs_for_metric = F.sigmoid(outputs_for_metric)
                        
                        # TODO delete later
                        # outputs_for_metric = outputs_for_metric[-1]
                        # targets_for_metric = targets_for_metric[-1]
                        metric.update(outputs_for_metric.detach(), targets_for_metric.detach())   

    def update_valid_metrics_states(self, batch_outputs, batch_targets):

        for metric_name, metric in self.valid_metrics.items():
            if metric_name in ["valid_mAP", "valid_action_mAP", "valid_ConfusionMatrix"]:
                metric_cfg = self.valid_metrics_cfgs[metric_name]    
                if metric_cfg.form == "flattened":        
                    batch_outputs, batch_targets = _flatten_batch_dim_outputs_and_targets(batch_outputs, batch_targets)
                    outputs_for_metric, targets_for_metric = _prepare_outputs_and_targets(batch_outputs, batch_targets, metric_cfg, self.data_info)
                    if not outputs_for_metric.numel() == 0 and not targets_for_metric.numel() == 0:

                        if metric_cfg.get("apply_softmax", False):
                            outputs_for_metric = F.softmax(outputs_for_metric, dim=1)
                        elif metric_cfg.get("apply_sigmoid", False):
                            outputs_for_metric = F.sigmoid(outputs_for_metric)
                        
                        metric.update(outputs_for_metric.detach(), targets_for_metric.detach())
    
    def reset_train_metrics_states(self):
        
        self.train_metrics["train_mAP"].reset()
        if "train_ConfusionMatrix" in self.train_metrics.keys():
            self.train_metrics["train_ConfusionMatrix"].reset()

    def reset_valid_metrics_states(self):
        
        self.valid_metrics["valid_mAP"].reset()
        if "valid_ConfusionMatrix" in self.valid_metrics.keys():
            self.valid_metrics["valid_ConfusionMatrix"].reset()

def _flatten_batch_dim_outputs_and_targets(batch_outputs, batch_targets):
    
    if len(batch_targets["onehot"].shape) > 2:
        batch_targets["onehot"] = torch.flatten(batch_targets["onehot"], start_dim=0, end_dim=1) # b*l, num_classes

    if torch.is_tensor(batch_outputs.get("logits", False)):
        if len(batch_outputs["logits"].shape) > 2:
            batch_outputs["logits"] = torch.flatten(batch_outputs["logits"], start_dim=0, end_dim=1) # b*l, used_num_classes 

    return batch_outputs, batch_targets

def _prepare_outputs_and_targets(outputs, targets, prepare_cfg, data_info):

    form = prepare_cfg.form
    outputs_cfg = prepare_cfg.outputs
    targets_cfg = prepare_cfg.targets

    outputs_key = outputs_cfg.type
    targets_key = targets_cfg.type
    applied_to_frames = prepare_cfg.applied_to_frames
    
    prepare_function_name = "_".join((form, outputs_key, targets_key, applied_to_frames))

    if prepare_function_name == "flattened_logits_labels_allframes":
        return _flattened_logits_labels_allframes(outputs, targets, outputs_key, data_info) 

def _flattened_logits_labels_allframes(outputs, targets, outputs_key, data_info):
    
    labels = torch.argmax(targets["onehot"], dim=1) # b*l
    logits = outputs[outputs_key] # b*l, used_num_classes

    return logits, labels

def _prepare_batch_outputs_and_targets(batch_outputs, batch_targets, prepare_cfg, data_info):

    form = prepare_cfg.form
    outputs_cfg = prepare_cfg.outputs
    targets_cfg = prepare_cfg.targets

    outputs_key = outputs_cfg.type
    targets_key = targets_cfg.type
    applied_to_frames = prepare_cfg.applied_to_frames
    
    prepare_function_name = "_".join((form, outputs_key, targets_key, applied_to_frames))

    if prepare_function_name == "batch_logits_labels_allframes":
        return _batch_logits_labels_allframes(batch_outputs, batch_targets, outputs_key, data_info) 

def _batch_logits_labels_allframes(batch_outputs, batch_targets, outputs_key, data_info):
    
    batch_labels = torch.argmax(batch_targets["onehot"], dim=2) # b, l
    batch_logits = batch_outputs[outputs_key] # b, l, used_num_classes

    return batch_logits, batch_labels

class MambaDecoder(nn.Module):
    
    def __init__(
        self, 
        num_layers,
        embed_dim,
        d_state_expansion_factor,
        d_conv,
        dt_rank, 
        block_expansion_factor,
        fused_add_norm=False, 
        residual_in_fp32=True
    ):
        
        super().__init__()

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.mamba_blocks = MambaBlocks(self.num_layers, 
                                        self.embed_dim, 
                                        d_state_expansion_factor, 
                                        d_conv, 
                                        dt_rank,
                                        block_expansion_factor,
                                        fused_add_norm,
                                        residual_in_fp32)


    def forward(self, x, inference_params=None):

        y = self.mamba_blocks(x, inference_params=inference_params)

        return y