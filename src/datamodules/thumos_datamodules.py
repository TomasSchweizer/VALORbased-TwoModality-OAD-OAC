from pathlib import Path

import hydra
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class ThumosImagesDataModule(LightningDataModule):

    def __init__(
        self, 
        dataset_info,
        datasets, 
        loaders,
        transforms,
    ):
        
        super().__init__()
        self.dataset_info = dataset_info
        self.datasets_cfg = datasets
        self.loaders_cfg = loaders
        self.transforms_cfg = transforms

        # Build train dataset config
        self.train_dataset_cfg = self.datasets_cfg.train_dataset
        self.train_dataset_cfg.training = True
        self.train_dataset_cfg.transforms = self.transforms_cfg.train_transforms
        self.train_dataset_cfg.dataset_info = self.dataset_info
        #self.train_dataset_cfg.fps = self.dataset_info.fps
        
        # Build valid dataset config
        if "valid_dataset" in self.datasets_cfg:
            self.valid_dataset_cfg = self.datasets_cfg.valid_dataset
            self.valid_dataset_cfg.training = False
            self.valid_dataset_cfg.transforms = self.transforms_cfg.valid_transforms
            self.valid_dataset_cfg.dataset_info = self.dataset_info
            if "video_test_0000270" in self.valid_dataset_cfg.dataset_info.valid_video_names: 
                self.valid_dataset_cfg.dataset_info.valid_video_names.remove("video_test_0000270")
            if "video_test_0001496" in self.valid_dataset_cfg.dataset_info.valid_video_names: 
                self.valid_dataset_cfg.dataset_info.valid_video_names.remove("video_test_0001496")
            #elf.valid_dataset_cfg.fps = self.dataset_info.fps

        # TODO delete after debugging
        self.train_dataset = hydra.utils.instantiate(self.train_dataset_cfg, _recursive_=False)
        self.valid_dataset = hydra.utils.instantiate(self.valid_dataset_cfg, _recursive_=False)
        self.test_dataset = None

        self.valid_dataloader_ref = None
        # self.pred_dataloader_ref = None


    def setup(self, stage):
        
        if stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.train_dataset_cfg, _recursive_=False)
            if "valid_dataset" in self.datasets_cfg:
                self.valid_dataset = hydra.utils.instantiate(self.valid_dataset_cfg, _recursive_=False)
        
        if stage == "test":
            self.test_dataset = hydra.utils.instantiate(self.valid_dataset_cfg, _recursive_=False)
        # if stage == "predict":
        #     self.valid_dataset = hydra.utils.instantiate(self.valid_dataset_cfg, _recursive_=False)

    def train_dataloader(self):
        self.train_dataset.recreate_train_dataset()
        return DataLoader(self.train_dataset, **self.loaders_cfg.get("train_loader"))

    def val_dataloader(self):
        if "valid_dataset" in self.datasets_cfg:
            if self.valid_dataloader_ref != None:
                return self.valid_dataloader_ref
            self.valid_dataloader_ref = DataLoader(self.valid_dataset, **self.loaders_cfg.get("valid_loader"))
            return self.valid_dataloader_ref
    
    def test_dataloader(self):

        return DataLoader(self.test_dataset, **self.loaders_cfg.get("valid_loader"))
         
    
    # def predict_dataloader(self):
    #     if self.pred_dataloader_ref != None:
    #         return self.pred_dataloader_ref
    #     self.pred_dataloader_ref = DataLoader(self.valid_dataset, **self.loaders_cfg.get("predict_loader"))
    #     return self.pred_dataloader_ref
