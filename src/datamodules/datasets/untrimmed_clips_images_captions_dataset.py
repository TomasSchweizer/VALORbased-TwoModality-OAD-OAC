from random import shuffle
from pathlib import Path

import numpy as np
from PIL import Image
import pickle as pkl
from string import punctuation

import torch
from torch.utils.data import Dataset
#from torchvision.transforms.functional import pil_to_tensor, resize, center_crop 
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize

from src.modules.models.valor.components.shared_text_multimodal_encoder_decoder.bert_tokenizer import BertTokenizer

class UntrimmedClipsImagesCaptionsDataset(Dataset):

    def __init__(
        self,
        training,
        data_root_path,
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
        self.target_preframe_dir_path =  Path(data_root_path.parents[1] / dataset_info.target_dirname)
        
        self.caption_preframe_dir_path =  Path(data_root_path.parents[1] / dataset_info.captions_dirname)
        self.bert_tokenizer = BertTokenizer(data_root_path.parents[1] / "pretrained_checkpoints/bert-base-uncased-vocab.txt")
        self.cls_token = self.bert_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_token = self.bert_tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        assert self.cls_token==101
        assert self.sep_token==102

                
        self.clip_frames = clip_frames
        self.image_size = dataset_info.image_size
        self.images_per_feature = dataset_info.images_per_feature
        self.sliding_window_stride = dataset_info.sliding_window_stride
        self.before_padding = self.sliding_window_stride - 1
        self.train_videos_mean = dataset_info.train_videos_mean
        self.train_videos_std = dataset_info.train_videos_std

        
        if self.shuffle_on_load:
            shuffle(self.video_names)

        self._create_images_clips()

    def _create_images_clips(self):      


        self.images_clips = []

        targets_all_videos = []
        tokens_all_videos = []
        swchunks_all_videos = []


        for video_name in self.video_names:

            video_targetsfile_name = str(video_name + ".npy")
            video_targets_path = self.target_preframe_dir_path / video_targetsfile_name    
            video_targets_onehot = np.load(video_targets_path).astype(np.float32) 
            n_features = video_targets_onehot.shape[0]

            video_captions_name = str(video_name + ".txt")
            video_captions_path = self.caption_preframe_dir_path / video_captions_name
            self.max_len = 40
            with open(video_captions_path, "r") as f:
                captions = f.readlines()
            video_captions = [c.strip() for c in captions]
            video_captions_tokens = [self.get_tokens_from_text(vc) for vc in video_captions]

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
            tokens_all_videos.append(video_captions_tokens)
            swchunks_all_videos.append(swchunks) 

        targets_all_videos = np.concatenate(targets_all_videos, axis=0)
        tokens_all_videos = np.concatenate(tokens_all_videos, axis=0)
        swchunks_all_videos = np.concatenate(swchunks_all_videos, axis=0) 

        num_clips = np.floor(targets_all_videos.shape[0] / self.clip_frames)
        cut_off_idx = int(num_clips * self.clip_frames)
        targets_all_videos = targets_all_videos[:cut_off_idx,:]
        tokens_all_videos = tokens_all_videos[:cut_off_idx,:]
        swchunks_all_videos = swchunks_all_videos[:cut_off_idx,:]
        targets_all_videos = np.split(targets_all_videos, num_clips)
        tokens_all_videos = np.split(tokens_all_videos, num_clips)
        swchunks_all_videos = np.split(swchunks_all_videos, num_clips)



        for tar, tok, swchunk in zip(targets_all_videos, tokens_all_videos, swchunks_all_videos):
            self.images_clips.append((tar, tok, swchunk))       
    
    def __getitem__(self, idx):
        
        targets, tokens, swchunks = self.images_clips[idx]

        l = targets.shape[0]
        t = self.images_per_feature
        c,h,w = self.image_size

        example = {}

        example['targets'] = torch.from_numpy(targets.astype(np.float32))

        example['tokens'] = torch.from_numpy(tokens).to(torch.long)

        #clips = torch.stack([torch.stack([center_crop(resize(pil_to_tensor(Image.open(image_path)), [160,], antialias=True), [160,320]) if image_path!=None else torch.zeros(c,160,320) for image_path in swchunk], dim=0) for swchunk in swchunks], dim=0)
        
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


    def clean(self, text):
        """remove duplicate spaces, lower and remove punctuations """
        text = ' '.join([i for i in text.split(' ') if i != ''])
        text = text.lower()
        for i in punctuation:
            text = text.replace(i,'')
        return text
    
    def get_tokens_from_text(self, text, max_len=None):
        text = self.clean(text) 
        if self.bert_tokenizer is not None:
            tokenized_text = self.bert_tokenizer.tokenize(text)
            txt_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            bert_tokens =self.get_padded_tokens(txt_tokens, max_len)
            
            return bert_tokens

    def get_padded_tokens(self, txt_tokens, max_len=None):
        
        max_len = self.max_len if  max_len is None else max_len
        txt_tokens = txt_tokens[:max_len]

        txt_tokens = [self.cls_token] + txt_tokens + [self.sep_token]  

        txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)

        output = torch.zeros(max_len + 2, dtype=torch.long)
        output[:len(txt_tokens)] = txt_tokens
        return output

    def __len__(self):
        return len(self.images_clips)