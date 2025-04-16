from pathlib import Path

import hydra
from dataclasses import dataclass, field

import torch
from pytorch_lightning import LightningModule

# class MulitGPUSingleOptimModule(LightningModule):
    
#     def __init__(
#         self,
#         model,
#     ):
    
#         super().__init__()

#         self.model_cfg = model
#         self.optimizer_cfg = self.model_cfg.pop("optimizer", None) 
#         self.scheduler_cfg = self.model_cfg.pop("scheduler", None)   
        
#         self.model = hydra.utils.instantiate(self.model_cfg,  _recursive_=False)   

#         self.save_hyperparameters()

#     def on_train_start(self) -> None:
        
#         self.model.reset_train_metrics_states()
#         self.model.reset_valid_metrics_states()


#     def training_step(self, batch, batch_idx):

#         batch_targets = {}
#         batch_outputs = {}

#         batch_targets["onehot"] = batch.pop("targets") # b, l, num_classes        

#         batch_outputs = self.model(batch, inference_params=None)

#         train_loss = self.model.calculate_loss(batch_outputs, batch_targets, phase="train")
#         if train_loss != None:
#             self.log("loss/train", train_loss, prog_bar=True, on_step=True, on_epoch=True)

#         # Only for special epoch based metric MeanAveragePrecision
#         self.model.update_train_metrics_states(batch_outputs, batch_targets)
   

#         # For accuracy metric
#         train_metrics_step = self.model.compute_metrics_in_train_step(batch_outputs, batch_targets)
#         train_metrics_step = {name: metric.detach().item() for name, metric in train_metrics_step.items()}  
#         self.log_dict(train_metrics_step, prog_bar=False, on_step=False, on_epoch=True)

#         return {"loss": train_loss}
    
#     # TODO add on train epoch end to calculate mAP
#     def on_train_epoch_end(self):
#         train_metrics_epoch = self.model.compute_metrics_on_train_epoch_end()
#         self.log_dict(train_metrics_epoch, prog_bar=False)
#         self.model.reset_train_metrics_states()

#     def on_validation_start(self):
#         pass
    
#     def validation_step(self, batch, batch_idx):

#         batch_targets = {}
#         batch_outputs = {}

#         batch_targets["onehot"] = batch.pop("targets") # b, l, num_classes        
#         batch_outputs = self.model(batch, inference_params=None)

#         val_loss = self.model.calculate_loss(batch_outputs, batch_targets, phase="valid")
#         if val_loss != None:
#             self.log("loss/valid", val_loss, prog_bar=True, on_step=True, on_epoch=True)

#         # Only for special epoch based metric MeanAveragePrecision
#         self.model.update_valid_metrics_states(batch_outputs, batch_targets) 

#         # For accuracy metric
#         valid_metrics_step = self.model.compute_metrics_in_valid_step(batch_outputs, batch_targets)
#         valid_metrics_step = {name: metric.detach().item() for name, metric in valid_metrics_step.items()}
#         self.log_dict(valid_metrics_step, prog_bar=False, on_step=False, on_epoch=True)    

#         return {"loss": val_loss}
    

#     def on_validation_epoch_end(self):
#         valid_metrics_epoch = self.model.compute_metrics_on_valid_epoch_end()
#         self.log_dict(valid_metrics_epoch, prog_bar=False)
#         self.model.reset_valid_metrics_states()

#     def predict_step(self, batch, batch_idx):
#         return None

#     def configure_optimizers(self):

#         optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())

#         estimated_stepping_batches = self.trainer.estimated_stepping_batches 

#         max_epochs = self.trainer.max_epochs
#         max_steps = estimated_stepping_batches

#         warm_up_epochs = self.scheduler_cfg.pop("warmup_epochs")        
#         warmup_steps =  estimated_stepping_batches // max_epochs * warm_up_epochs

#         scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer, max_steps=max_steps, warmup_steps=warmup_steps)
#         return [optimizer],  [{"scheduler": scheduler, "name": "train/lr", "interval": "step"}]
    

class MultiGPUSingleOptimModule(LightningModule):
    
    def __init__(
        self,
        model,
    ):
    
        super().__init__()

        self.model_cfg = model
        self.optimizer_cfg = self.model_cfg.pop("optimizer", None) 
        self.scheduler_cfg = self.model_cfg.pop("scheduler", None)   
        
        self.model = hydra.utils.instantiate(self.model_cfg,  _recursive_=False)   

        self.save_hyperparameters()

    def on_train_start(self) -> None:
        
        self.model.reset_train_metrics_states()
        self.model.reset_valid_metrics_states()

    def training_step(self, batch, batch_idx):

        #batch_dict = self.model.forward_training(text_tokens=batch.pop("tokens")[0,:], video_inputs=batch.pop("clips")[0,:], inference_params=None)        
        batch_dict = self.model.forward_training(text_tokens=batch.pop("tokens"), video_inputs=batch.pop("features"), inference_params=None)

        batch_dict["oad"]["oad_labels"] = batch.pop("targets") # b, l, num_classes      

        if True:
            batch_dict["contrastive_multimodal_alignment"]["oad_labels"] = batch_dict["oad"]["oad_labels"].clone()
        train_loss = self.model.calculate_loss(batch_dict, phase="train")
        if train_loss != None:
            self.log("loss/train", train_loss, prog_bar=True, on_step=True, on_epoch=True)

        # Only for special epoch based metric MeanAveragePrecision
        self.model.update_train_metrics_states(batch_dict)
   

        # For accuracy metric
        train_metrics_step = self.model.compute_metrics_in_train_step(batch_dict)
        train_metrics_step = {name: metric.detach().item() for name, metric in train_metrics_step.items()}  
        self.log_dict(train_metrics_step, prog_bar=False, on_step=False, on_epoch=True)

        return {"loss": train_loss}
    
    def on_train_epoch_end(self):    
        

        train_metrics_epoch = self.model.compute_metrics_on_train_epoch_end()
        
        #TODO delete this after creating stats for token prediction 
        train_metrics_epoch["train/masked_token_acc"] = self.model.epoch_correct_tokens / self.model.epoch_total_tokens
        
        self.log_dict(train_metrics_epoch, prog_bar=False)
        self.model.reset_train_metrics_states()

        #TODO delete this after creating stats for token prediction 
        self.model.epoch_correct_tokens = 0
        self.model.epoch_total_tokens = 0

    def on_validation_start(self):
        pass
    
    def validation_step(self, batch, batch_idx):

        #TODO maybe delete later
        if self.global_step >= 0:


            #batch_dict = self.model.forward_validation(video_inputs=batch.pop("clips")[0,:], inference_params=None)
            batch_dict = self.model.forward_validation(video_inputs=batch.pop("features"), inference_params=None, oad_labels=batch["targets"].clone(), video_name=batch.pop("video_name"), global_step=self.global_step)

            batch_dict["oad"]["oad_labels"] = batch.pop("targets") # b, l, num_classes       


            val_loss = self.model.calculate_loss(batch_dict, phase="valid")
            if val_loss != None:
                self.log("loss/valid", val_loss, prog_bar=True, on_step=True, on_epoch=True)

            # Only for special epoch based metric MeanAveragePrecision
            self.model.update_valid_metrics_states(batch_dict) 

            # For accuracy metric
            valid_metrics_step = self.model.compute_metrics_in_valid_step(batch_dict)
            valid_metrics_step = {name: metric.detach().item() for name, metric in valid_metrics_step.items()}
            self.log_dict(valid_metrics_step, prog_bar=False, on_step=False, on_epoch=True)    
        else:
            val_loss = 0.0
        return {"loss": val_loss}
    

    def on_validation_epoch_end(self):

        #TODO maybe delete later
        if self.global_step >= 0:

            valid_metrics_epoch = self.model.compute_metrics_on_valid_epoch_end()
            self.log_dict(valid_metrics_epoch, prog_bar=False)
            self.model.reset_valid_metrics_states()

    def test_step(self, batch, batch_idx):
        
        inference_params = InferenceParams(max_seqlen=1, max_batch_size=1, seqlen_offset=1, batch_size_offset=0, key_value_memory_dict={})
        batch_dict = self.model.forward_test(video_inputs=batch.pop("features"), inference_params=inference_params, oad_labels=batch["targets"].clone(), video_name=batch.pop("video_name"), global_step=self.global_step)
        batch_dict["oad"]["oad_labels"] = batch.pop("targets") # b, l, num_classes       
        self.model.update_test_metrics_states(batch_dict) 
        test_metrics_step = self.model.compute_metrics_in_test_step(batch_dict)
        test_metrics_step = {name: metric.detach().item() for name, metric in test_metrics_step.items()}
        self.log_dict(test_metrics_step, prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_dict["oad"]["oad_labels"].shape[1])   

    def on_test_epoch_end(self):
        
        test_metrics_epoch = self.model.compute_metrics_on_test_epoch_end()
        self.log_dict(test_metrics_epoch, prog_bar=False)

    def configure_optimizers(self):

        optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())

        estimated_stepping_batches = self.trainer.estimated_stepping_batches 

        max_epochs = self.trainer.max_epochs
        max_steps = estimated_stepping_batches

        warm_up_epochs = self.scheduler_cfg.pop("warmup_epochs")        
        warmup_steps =  estimated_stepping_batches // max_epochs * warm_up_epochs

        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer, max_steps=max_steps, warmup_steps=warmup_steps)
        return [optimizer],  [{"scheduler": scheduler, "name": "train/lr", "interval": "step"}]



# def build_module(module_cfg, checkpoint_path):

#     if checkpoint_path != None:
#         target_name = module_cfg._target_.split(".")[-1]     
#         if target_name == "MulitGPUSingleOptimModule":
#             module = MulitGPUSingleOptimModule.load_from_checkpoint(checkpoint_path=str(checkpoint_path))
#     else: 
#         module = hydra.utils.instantiate(module_cfg, _recursive_=False)
    
#     return module

@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample = None

    def reset(self, max_seqlen, max_batch_size, seqlen_offset=0):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = seqlen_offset
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
