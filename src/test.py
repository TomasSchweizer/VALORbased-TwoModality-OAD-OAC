import warnings
warnings.filterwarnings("ignore")

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

import hydra
import wandb

import torch

from src.utils.callbacks import instantiate_callbacks
from src.utils.env import setup_env 
from src.utils.logger import setup_run

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "test.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def train(cfg):   
    
    setup_env(cfg.env)
        
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    
    print(f"Instantiating lightning model <{cfg.module._target_}>")
    module = hydra.utils.instantiate(cfg.module, _recursive_=False)

    print((f"Instantiating wandb lightning logger <{cfg.logger._target_}>"))
    setup_run(cfg.logger)
    wandb_logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)
    wandb_logger.log_hyperparams(cfg)

    callbacks = instantiate_callbacks(cfg.callbacks, cfg.logger)

    print((f"Instantiating trainer <{cfg.trainer._target_}>"))
    trainer = hydra.utils.instantiate(
            cfg.trainer, logger=wandb_logger, callbacks=callbacks
    )

    print("Starting training!")
    trainer.test(
        model=module,
        datamodule=datamodule,
        ckpt_path=cfg.checkpoint_path,
    )

if __name__ == "__main__":
    try:      
        torch.autograd.set_detect_anomaly(True) #TODO Delete after developing
        train()
    except Exception as ex:        
        if wandb.run:            
            wandb.finish(exit_code=-1)
        raise ex
    finally:
        if wandb.run:            
            wandb.finish(exit_code=0)
