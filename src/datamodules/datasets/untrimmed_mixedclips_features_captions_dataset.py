from random import shuffle
from string import punctuation
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset

from src.modules.models.valor.components.shared_text_multimodal_encoder_decoder.bert_tokenizer import BertTokenizer


class UntrimmedMixedClipsFeaturesCaptionsDataset(Dataset):

    def __init__(
        self,
        training,
        data_root_path,
        dataset_info,       
        clip_seconds,
        n_cuts,
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
        #self.target_preframe_path = data_root_path.parent / "data/thumos" / dataset_info.target_dirname #! uncomment if original targets per frame should be used.


        self.caption_preframe_dir_path =  data_root_path / dataset_info.captions_dirname
        self.bert_tokenizer = BertTokenizer(data_root_path.parent / "pretrained_checkpoints/bert-base-uncased-vocab.txt")
        self.max_len = dataset_info.max_captions_tokens
        self.cls_token = self.bert_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_token = self.bert_tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        assert self.cls_token==101
        assert self.sep_token==102

        self.clip_features = int(clip_seconds * dataset_info.feature_fps)
        self.n_cuts = n_cuts

        if self.shuffle_on_load:
            shuffle(self.video_names)

        self._create_clips()

    def _create_clips(self):

        self.clips = []

        targets_all_vids = []
        features_all_vids = []
        tokens_all_vids = []
        

        for video_name in self.video_names:

            video_file_name = str(video_name + ".npy")
            video_captions_name = str(video_name + ".txt")

            video_targets_path =  self.target_preframe_path / video_file_name
            video_features_path = self.video_features_path / video_file_name
            video_captions_path = self.caption_preframe_dir_path / video_captions_name            
            
            video_targets = np.load(video_targets_path).astype(np.float32)
            video_features = np.load(video_features_path).astype(np.float32)

            with open(video_captions_path, "r") as f:
                captions = f.readlines()
            video_captions = [c.strip() for c in captions]
            video_captions_tokens = np.stack([self.get_tokens_from_text(vc) for vc in video_captions], axis=0)

            targets_all_vids.append(video_targets)
            features_all_vids.append(video_features)
            tokens_all_vids.append(video_captions_tokens)

        targets_all_vids = np.concatenate(targets_all_vids)
        features_all_vids = np.concatenate(features_all_vids)
        tokens_all_vids = np.concatenate(tokens_all_vids)

        n_features_all_videos = targets_all_vids.shape[0]
        idxs_features_all_videos = np.arange(n_features_all_videos)

        valid_cut_points_idxs = np.where(np.argmax(targets_all_vids, axis=1) == 0)[0]
        if not valid_cut_points_idxs.tolist():
            cut_points = np.random.choice(idxs_features_all_videos, self.n_cuts).tolist()       
        else:
            cut_points = np.random.choice(valid_cut_points_idxs, self.n_cuts).tolist()  
        cut_points= sorted(cut_points)
        cut_points.insert(0,0)
        cut_points.append(idxs_features_all_videos[-1])

        targets_all_vids_mixed = []
        features_all_vids_mixed = []
        tokens_all_vids_mixed = []

        for cut_idx in range(len(cut_points)-1):
            targets_all_vids_mixed.append(targets_all_vids[cut_points[cut_idx]:cut_points[cut_idx+1],:]) 
            features_all_vids_mixed.append(features_all_vids[cut_points[cut_idx]:cut_points[cut_idx+1],:]) 
            tokens_all_vids_mixed.append(tokens_all_vids[cut_points[cut_idx]:cut_points[cut_idx+1],:])

        targets_all_vids = None
        features_all_vids = None
        tokens_all_vids = None 
        del targets_all_vids
        del features_all_vids
        del tokens_all_vids 

        targets_all_vids_mixed = np.concatenate(targets_all_vids_mixed, axis=0)
        features_all_vids_mixed = np.concatenate(features_all_vids_mixed, axis=0)
        tokens_all_vids_mixed = np.concatenate(tokens_all_vids_mixed, axis=0)

        self.num_clips = int(targets_all_vids_mixed.shape[0] / self.clip_features)
        cutoff_idx = self.num_clips * self.clip_features

        targets_all_vids_mixed = np.split(targets_all_vids_mixed[:cutoff_idx], self.num_clips)
        features_all_vids_mixed = np.split(features_all_vids_mixed[:cutoff_idx], self.num_clips)
        tokens_all_vids_mixed = np.split(tokens_all_vids_mixed[:cutoff_idx], self.num_clips)

        for tar, feat, tok in zip(targets_all_vids_mixed, features_all_vids_mixed, tokens_all_vids_mixed):
            self.clips.append( (torch.from_numpy(tar).to(torch.float32), torch.from_numpy(feat).to(torch.float32), torch.from_numpy(tok).to(torch.long) ) )   

    def __getitem__(self, idx):

        targets, features, tokens = self.clips[idx]
        
        example = {}

        example['targets'] =targets

        example['features'] = features

        example['tokens'] = tokens

        return example 
    
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


    def recreate_train_dataset(self):
        shuffle(self.clips)
        self._create_clips()

    def __len__(self):
        return len(self.clips)