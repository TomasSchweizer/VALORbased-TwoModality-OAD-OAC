from random import shuffle
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset

class UntrimmedFullVideosFeaturesDataset(Dataset):

    def __init__(
        self,
        training,
        data_root_path,
        dataset_info,       
        random_start_idx,
        latest_start_idx_videos,
        shuffle_on_load,
        transforms,
        ):        
 
        super().__init__()

        self.training = training
        self.video_names = sorted(dataset_info.train_video_names) if training else sorted(dataset_info.valid_video_names)
        self.shuffle_on_load = shuffle_on_load

        data_root_path = Path(data_root_path)
        self.video_features_path = data_root_path / dataset_info.features_dirname
        self.target_preframe_path = data_root_path / dataset_info.target_dirname
        if not self.target_preframe_path.is_dir():
            self.target_preframe_path = data_root_path.parent / "data/thumos" / dataset_info.target_dirname #! uncomment if original targets per frame should be used.


        self.random_start_idx = random_start_idx
        self.latest_start_idx_videos = latest_start_idx_videos

        if self.shuffle_on_load:
            shuffle(self.video_names)

        self._create_features_vids()

    def _create_features_vids(self):

        self.features_vids = []

        for video_name in self.video_names:

            video_targets_path =  self.target_preframe_path / str(video_name  + ".npy")
            video_targets = np.load(video_targets_path)

            start_idx_video = np.random.choice(int(video_targets.shape[0]*self.latest_start_idx_videos), 1).item() if self.training and self.random_start_idx else 0

            self.features_vids.append([
                video_name, start_idx_video
            ])

    def __getitem__(self, idx):

        video_name, start_idx = self.features_vids[idx]
        
        video_file_name = str(video_name + ".npy")
        video_targets_path =  self.target_preframe_path / video_file_name
        video_features_path = self.video_features_path / video_file_name

        example = {}

        example['targets'] = torch.from_numpy(np.load(video_targets_path)[start_idx:,:].astype(np.float32))

        example['features'] = torch.from_numpy(np.load(video_features_path)[start_idx:,:].astype(np.float32))

        example['video_name'] = video_name

        return example 


    def recreate_train_dataset(self):
        shuffle(self.features_vids)
        self._create_features_vids()

    def __len__(self):
        return len(self.features_vids)