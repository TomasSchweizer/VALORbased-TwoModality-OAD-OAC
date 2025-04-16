from random import shuffle
from string import punctuation
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset

from src.modules.models.valor.components.shared_text_multimodal_encoder_decoder.bert_tokenizer import BertTokenizer


class UntrimmedFullVideosFeaturesCaptionsDataset(Dataset):

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
            self.target_preframe_path = data_root_path.parent / "data/thumos" / dataset_info.target_dirname # TODO change back for training on all videos


        self.caption_preframe_dir_path =  data_root_path / dataset_info.captions_dirname
        self.bert_tokenizer = BertTokenizer(data_root_path.parent / "pretrained_checkpoints/bert-base-uncased-vocab.txt")
        self.max_len = dataset_info.max_captions_tokens
        self.cls_token = self.bert_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_token = self.bert_tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        assert self.cls_token==101
        assert self.sep_token==102

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
        
        video_captions_name = str(video_name + ".txt")
        video_captions_path = self.caption_preframe_dir_path / video_captions_name

        # TODO uncomment after trying to train good baseline
        with open(video_captions_path, "r") as f:
           captions = f.readlines()
        video_captions = [c.strip() for c in captions]
        video_captions_tokens = np.stack([self.get_tokens_from_text(vc) for vc in video_captions], axis=0)

        example = {}

        example['targets'] = torch.from_numpy(np.load(video_targets_path)[start_idx:,:].astype(np.float32))

        example['features'] = torch.from_numpy(np.load(video_features_path)[start_idx:,:].astype(np.float32))

        # TODO uncomment after trying to train good baseline
        example['tokens'] = torch.zeros_like(example['targets'])
        example['tokens'] = torch.from_numpy(video_captions_tokens)[start_idx:,:].to(torch.long)

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
        shuffle(self.features_vids)
        self._create_features_vids()

    def __len__(self):
        return len(self.features_vids)