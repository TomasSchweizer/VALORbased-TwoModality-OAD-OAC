import hydra
from omegaconf import DictConfig
from pathlib import Path

import wandb

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only 

def instantiate_callbacks(callbacks_cfg, logger_cfg):

    callbacks = []

    if not callbacks_cfg:
        print("No callback configs found! Skipping..")
        return callbacks
    
    for cb_name, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:

            if cb_name in ["model_checkpointer", "upload_checkpoint_artifacts"]:
                cb_conf.dirpath = Path(logger_cfg.save_dir) / "checkpoints" / logger_cfg.name 
            
            print(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log, log_freq) -> None:
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer):
        logger = trainer.logger
        logger.watch(
            model=trainer.model,
            log=self.log,
            log_freq=self.log_freq,
            log_graph=True,
        )


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, dirpath, upload_best_only = False) -> None:
        self.dirpath = dirpath
        self.upload_best_only = upload_best_only

        self.checkpoint_name = self.dirpath.name
        

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = trainer.logger
        experiment = logger.experiment
        
        ckpts = wandb.Artifact(self.checkpoint_name, type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.dirpath).rglob("*.ckpt"):
                ckpts.add_file(str(path))
        experiment.log_artifact(ckpts)


class LogValidationPredictionsCallback(Callback):
    
    def __init__(self, log_every_n_epoch, logs_per_epoch):
        
        self.log_every_n_epoch = log_every_n_epoch
        self.logs_per_epoch = logs_per_epoch

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        if trainer.current_epoch % self.log_every_n_epoch:
            outputs_flattened = outputs["outputs_flattened"]["merged_outputs_flattened"]


        print("Hi")