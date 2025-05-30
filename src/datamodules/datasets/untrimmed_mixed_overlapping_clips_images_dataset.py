from random import shuffle
from pathlib import Path

import pickle as pkl
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


#! This dataset layer loads the whole dataset memory!!
class UntrimmedMixedOverlappingClipsImagesDataset(Dataset):

    def __init__(
        self,
        training,
        data_root_path,
        load_precreated_dataset,
        save_dataset_as_precreated,
        precreated_dataset_path,
        dataset_info,       
        clip_frames,
        n_cuts,
        shuffle_on_load,
        transforms,
        ):
        
        super().__init__()

        self.training = training
        self.video_names = sorted(dataset_info.train_video_names) if training else sorted(dataset_info.valid_video_names)
        self.shuffle_on_load = shuffle_on_load

        data_root_path = Path(data_root_path)
        self.videos_imagesdirs_path = data_root_path / dataset_info.train_images_dirname if training else data_root_path / dataset_info.valid_images_dirname
        self.target_preframe_dir_path =  data_root_path / dataset_info.target_dirname
                
        self.clip_frames = clip_frames
        self.image_size = dataset_info.image_size
        self.n_cuts = n_cuts
        self.images_per_feature = dataset_info.images_per_feature
        self.sliding_window_stride = dataset_info.sliding_window_stride
        self.before_padding = self.sliding_window_stride - 1
        
        self.load_precreated_dataset = load_precreated_dataset
        self.save_dataset_as_precreated = save_dataset_as_precreated
        self.precreated_dataset_path = precreated_dataset_path

        if self.shuffle_on_load:
            shuffle(self.video_names)        

        if self.load_precreated_dataset:
            self.precreated_dataset_path = Path(self.precreated_dataset_path)
            with open(self.precreated_dataset_path / "image_clips.pkl", "rb") as image_clips_file:
                self.images_clips = pkl.load(image_clips_file)
        else:
            self._create_images_clips()

    def _create_images_clips(self):

        self.images_clips = []

        targets_all_videos = []
        swchunks_all_videos = []

        for video_name in self.video_names:

            video_targetsfile_name = str(video_name + ".npy")
            video_targets_path = self.target_preframe_dir_path / video_targetsfile_name    
            video_targets_onehot = np.load(video_targets_path).astype(np.float32) 
            n_features = video_targets_onehot.shape[0]

            video_imagesdir_name = str(video_name + ".mp4f")
            video_imagesdir_path = self.videos_imagesdirs_path / video_imagesdir_name
            video_imagefiles_paths = sorted([path for path in video_imagesdir_path.iterdir()])
            #n_frames = len(video_imagefiles_paths)
            [video_imagefiles_paths.insert(0, None) for _ in range(self.before_padding)]
            #n_frames_start_padded = len(video_imagefiles_paths)
        
            after_padding = self.images_per_feature
            [video_imagefiles_paths.append(None) for _ in range(after_padding)]
            n_frames_startend_padded = len(video_imagefiles_paths)
            swchunks = [video_imagefiles_paths[i:i+self.images_per_feature] for i in range(0, n_frames_startend_padded-(self.images_per_feature+1), self.sliding_window_stride)]
            n_chunks = len(swchunks)

            if n_chunks != n_features:
                difference = n_chunks - n_features
                if difference > 0:
                    swchunks = swchunks[:n_features]
                n_chunks = len(swchunks)
                if n_chunks != n_features:
                    print("Wrong:")
                    break 

            targets_all_videos.append(video_targets_onehot)
            swchunks_all_videos.append(swchunks) 


        targets_all_videos = np.concatenate(targets_all_videos, axis=0)
        swchunks_all_videos = np.concatenate(swchunks_all_videos, axis=0) 

        n_features_all_videos = targets_all_videos.shape[0]  
        idxs_features_all_videos = np.arange(n_features_all_videos)

        valid_cut_points_idxs = np.where(np.argmax(targets_all_videos, axis=1) == 0)[0]
        cut_points = np.random.choice(valid_cut_points_idxs, self.n_cuts).tolist()
        cut_points= sorted(cut_points)
        cut_points.insert(0,0)
        cut_points.append(idxs_features_all_videos[-1])

        targets_all_videos_mixed = []
        swchunks_all_videos_mixed = []

        for cut_idx in range(len(cut_points)-1):
            targets_all_videos_mixed.append(targets_all_videos[cut_points[cut_idx]:cut_points[cut_idx+1],:]) 
            swchunks_all_videos_mixed.append(swchunks_all_videos[cut_points[cut_idx]:cut_points[cut_idx+1],:])

        targets_all_videos = None
        swchunks_all_videos = None
        del targets_all_videos
        del swchunks_all_videos

        targets_all_videos_mixed = np.concatenate(targets_all_videos_mixed, axis=0)
        swchunks_all_videos_mixed = np.concatenate(swchunks_all_videos_mixed, axis=0)
        targets_all_videos_mixed = np.swapaxes(np.lib.stride_tricks.sliding_window_view(targets_all_videos_mixed, self.clip_frames, axis=0),1,2)
        swchunks_all_videos_mixed = np.swapaxes(np.lib.stride_tricks.sliding_window_view(swchunks_all_videos_mixed, self.clip_frames, axis=0),1,2)

        for tar, swchunk in zip(targets_all_videos_mixed, swchunks_all_videos_mixed):
            self.images_clips.append((tar,swchunk))   
        
        if self.save_dataset_as_precreated:
            self.precreated_dataset_path = Path(self.precreated_dataset_path)
            with open(self.precreated_dataset_path / "image_clips.pkl", "wb") as image_clips_file:
                pkl.dump(self.images_clips, image_clips_file)

    def __getitem__(self, idx):
        
        targets, swchunks = self.images_clips[idx]

        l = targets.shape[0]
        t = self.images_per_feature
        c,h,w = self.image_size

        example = {}

        example['targets'] = torch.from_numpy(targets.astype(np.float32))

        clips = torch.stack([torch.stack([pil_to_tensor(Image.open(image_path)) if image_path!=None else torch.zeros(c,h,w) for image_path in swchunk], dim=0) for swchunk in swchunks], dim=0)
        clips = clips.permute(0,2,1,3,4) # l, c, t, h, w 
        example['clips'] = clips.to(torch.float32)
        
        return example 


    def recreate_train_dataset(self):
        shuffle(self.video_names)
        self._create_images_clips()

    def __len__(self):
        return len(self.images_clips)