from random import shuffle
from pathlib import Path

import numpy as np
from PIL import Image
import pickle as pkl

import torch
from torch.utils.data import Dataset
#from torchvision.transforms.functional import pil_to_tensor, resize, center_crop 
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import feed_ndarray

class UntrimmedClipsImagesDataset(Dataset):

    def __init__(
        self,
        training,
        data_root_path,
        load_precreated_dataset,
        save_dataset_as_precreated,
        precreated_dataset_path,
        dataset_info,       
        clip_frames,
        shuffle_on_load,
        transforms,
        ):        
 
        super().__init__()

        self.training = training
        self.video_names = sorted(dataset_info.train_video_names) if training else sorted(dataset_info.valid_video_names)
        self.shuffle_on_load = shuffle_on_load

        data_root_path = Path(data_root_path)
        self.videos_imagesdirs_path = data_root_path / dataset_info.train_images_dirname if training else data_root_path / dataset_info.valid_images_dirname
        #self.target_preframe_dir_path =  data_root_path / dataset_info.target_dirname
        self.target_preframe_dir_path =  data_root_path.parents[1] / dataset_info.target_dirname

                
        self.clip_frames = clip_frames
        self.image_size = dataset_info.image_size
        self.images_per_feature = dataset_info.images_per_feature
        self.sliding_window_stride = dataset_info.sliding_window_stride
        self.before_padding = self.sliding_window_stride - 1
        self.train_videos_mean = dataset_info.train_videos_mean
        self.train_videos_std = dataset_info.train_videos_std

        self.load_precreated_dataset = load_precreated_dataset
        self.save_dataset_as_precreated = save_dataset_as_precreated
        self.precreated_dataset_path = precreated_dataset_path
        
        if self.shuffle_on_load:
            shuffle(self.video_names)

        if self.load_precreated_dataset:
            self.precreated_dataset_path = Path(self.precreated_dataset_path)
            file_name = f"image_clips_{'train' if self.training else 'valid'}.pkl"
            with open(self.precreated_dataset_path / file_name, "rb") as image_clips_file:
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

        num_clips = np.floor(targets_all_videos.shape[0] / self.clip_frames)
        cut_off_idx = int(num_clips * self.clip_frames)
        targets_all_videos = targets_all_videos[:cut_off_idx,:]
        swchunks_all_videos = swchunks_all_videos[:cut_off_idx,:]
        targets_all_videos = np.split(targets_all_videos, num_clips)
        swchunks_all_videos = np.split(swchunks_all_videos, num_clips)


        # num_clips = int(targets_all_videos.shape[0] / self.clip_frames)
        # targets_all_videos = np.array_split(targets_all_videos, num_clips)
        # swchunks_all_videos = np.array_split(swchunks_all_videos, num_clips)


        for tar, swchunk in zip(targets_all_videos, swchunks_all_videos):
            self.images_clips.append((tar,swchunk))   

        if self.save_dataset_as_precreated:
            self.precreated_dataset_path = Path(self.precreated_dataset_path)
            file_name = f"image_clips_{'train' if self.training else 'valid'}.pkl"
            with open(self.precreated_dataset_path / file_name, "wb") as image_clips_file:
                pkl.dump(self.images_clips, image_clips_file)
    
    
    def __getitem__(self, idx):
        
        targets, swchunks = self.images_clips[idx]

        l = targets.shape[0]
        t = self.images_per_feature
        c,h,w = self.image_size

        example = {}

        example['targets'] = torch.from_numpy(targets.astype(np.float32))

        #clips = torch.stack([torch.stack([pil_to_tensor(Image.open(image_path)) if image_path!=None else torch.zeros(c,h,w) for image_path in swchunk], dim=0) for swchunk in swchunks], dim=0)
        
        # TODO change for large dataset this is just for 

        transformations = Compose([
                                    ToTensor(),
                                    Resize([160,], antialias=True),
                                    CenterCrop([160,320]),
                                    Normalize(self.train_videos_mean, self.train_videos_std)
                                    ])
        clips = torch.stack([torch.stack([transformations(Image.open(image_path)) if image_path!=None else torch.zeros(c,160,320) for image_path in swchunk], dim=0) for swchunk in swchunks], dim=0)

        clips = clips.permute(0,2,1,3,4) # l, c, t, h, w 
        example['clips'] = clips.to(torch.float32)
        
        return example 
        

    def recreate_train_dataset(self):
        shuffle(self.video_names)
        self._create_images_clips()

    def __len__(self):
        return len(self.images_clips)


class DaliUntrimmedClipsImagesDataset(Dataset):

    def __init__(
        self,
        training,
        data_root_path,
        dataset_info,
        sequence_length,
        num_features, 
        shuffle_on_load,
        transforms,
        ):        
 
        super().__init__()

        self.training = training

        self.sequence_length = sequence_length
        self.num_features = num_features
        self.image_size = dataset_info.image_size
        self.images_per_feature = dataset_info.images_per_feature
        self.sliding_window_stride = dataset_info.sliding_window_stride
        self.before_padding = self.sliding_window_stride - 1

        data_root_path = Path(data_root_path)
        path_train_d = data_root_path / dataset_info.train_images_dirname if training else data_root_path / dataset_info.valid_images_dirname
        self.path_train_ds = sorted([path for path in path_train_d.iterdir()])

        if shuffle_on_load:
            shuffle(self.path_train_ds)

        self.ddp_device_id = 1

    def __getitem__(self, idx):
        
        @pipeline_def
        def frame_seq_pipe(path_train_d):
            frames = fn.readers.sequence(file_root=path_train_d, sequence_length=self.sequence_length) 
            return frames

        path_train_d = self.path_train_ds[idx]

        path_targets = path_train_d / "labels.pt"
        targets = torch.load(path_targets).to(dtype=torch.float32)

        l = targets.shape[0]
        t = self.images_per_feature
        c,h,w = self.image_size

        pipe = frame_seq_pipe(path_train_d=path_train_d, batch_size=self.num_features, num_threads=24, device_id=None)
        pipe.build()
        pipe_out = pipe.run()
        input_images = torch.zeros(96,16,180,320,3, dtype=torch.uint8)
        input_images = feed_ndarray(pipe_out[0].as_tensor(), input_images).permute(0,4,1,2,3).to(dtype=torch.float32)          

        example = {}
        example['targets'] = targets
        example['clips'] = input_images
        
        return example 

    def recreate_train_dataset(self):
        shuffle(self.path_train_ds)

    def __len__(self):
        return len(self.path_train_ds)