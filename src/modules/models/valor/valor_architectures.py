from pathlib import Path
import math
import random

import numpy as np

import hydra
from omegaconf import OmegaConf
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.models.valor.components.video_encoder.video_encoder import VideoMAEv2VideoLadderAdapterVideoFeatureMapConvPooler
from src.modules.models.valor.components.video_decoder.video_decoder import VideoMambaDecoder
from src.modules.models.valor.components.shared_text_multimodal_encoder_decoder.shared_text_multimodal_encoder_decoder import SharedTextMultimodalBertEncoderDecoder
from src.modules.models.valor.components.contrastive_heads import ContrastiveHead

class VideoMambaDecoderSharedTextMultimodalBertEncoderDecoder(nn.Module):

    def __init__(
        self,
        video_decoder,
        shared_text_multimodal_encoder_decoder,
        contrastive_heads,
        weights_init,
        data_info,
        criterions,
        metrics,
    ):
        
        super().__init__()

        use_valor_pretrained_weights = True #! here
        freeze_valor_weights = True 


        video_decoder_cfg = video_decoder
        #video_decoder_cfg.adapter.encoder_embed_dims = 768
        self.video_decoder = VideoMambaDecoder(**video_decoder_cfg)
        self.video_decoder_embed_dims = self.video_decoder.decoder_embed_dims
        #self.video_features_position_embedding = nn.Parameter(0.02 * torch.randn(10000,self.video_decoder_embed_dims))

        #! Newly added
        self.projection_layer = nn.Sequential(nn.Linear(self.video_decoder_embed_dims , self.video_decoder_embed_dims*2),                                              
                                              nn.ReLU(),
                                              #nn.Linear(self.video_decoder_embed_dims*2, self.video_decoder_embed_dims*2),
                                              #nn.ReLU(),
                                              nn.Linear(self.video_decoder_embed_dims*2 , self.video_decoder_embed_dims))
        
        shared_text_multimodal_encoder_decoder_cfg = shared_text_multimodal_encoder_decoder
        self.shared_text_multimodal_encoder_decoder = SharedTextMultimodalBertEncoderDecoder(**shared_text_multimodal_encoder_decoder_cfg)
        #print(self.shared_text_multimodal_encoder_decoder.state_dict().keys())
        if use_valor_pretrained_weights:
            checkpoint_bert_valor_pretrained = torch.load("/home/schweizercontainer/ws_train_videomaev2_adapter_mamba_on_images/pretrained_checkpoints/bert_b_valor_l.pt")           
            missing_keys, unexpected_keys =  self.shared_text_multimodal_encoder_decoder.load_state_dict(checkpoint_bert_valor_pretrained, strict=False)

            if missing_keys != [] or unexpected_keys != []:
                print(f'Unexpected_keys in multimodal encoder: {unexpected_keys}')
                print(f'Missing_keys in multimodal encoder: {missing_keys}')
            if freeze_valor_weights:
                self.shared_text_multimodal_encoder_decoder.requires_grad_(False)

        self.multimodal_decoder_embed_dims = self.shared_text_multimodal_encoder_decoder.bert_multimodal_decoder_embed_dims
        self.text_encoder_embed_dims = self.shared_text_multimodal_encoder_decoder.bert_text_encoder_embed_dims

        contrastive_heads_cfg = contrastive_heads
        if contrastive_heads_cfg != None:
            
            contrastive_head_video_cfg = contrastive_heads_cfg.contra_head_video
            contrastive_head_video_cfg.embed_dims = self.video_decoder_embed_dims
            self.contra_head_video = ContrastiveHead(**contrastive_head_video_cfg)
            # if use_valor_pretrained_weights:
            #     checkpoint_contra_head_video_valor_pretrained = torch.load("/home/schweizercontainer/ws_train_videomaev2_adapter_mamba_on_images/pretrained_checkpoints/contra_v_valor.pt")
            #     missing_keys, unexpected_keys =  self.contra_head_video.load_state_dict(checkpoint_contra_head_video_valor_pretrained, strict=False)

            #     if missing_keys != [] or unexpected_keys != []:
            #         print(f'Unexpected_keys in multimodal encoder: {unexpected_keys}')
            #         print(f'Missing_keys in multimodal encoder: {missing_keys}')

            contrastive_head_text_cfg = contrastive_heads_cfg.contra_head_text
            contrastive_head_text_cfg.embed_dims = self.text_encoder_embed_dims
            self.contra_head_text = ContrastiveHead(**contrastive_head_text_cfg)
            # if use_valor_pretrained_weights:
            #     checkpoint_contra_head_text_valor_pretrained = torch.load("/home/schweizercontainer/ws_train_videomaev2_adapter_mamba_on_images/pretrained_checkpoints/contra_t_valor.pt")
            #     missing_keys, unexpected_keys =  self.contra_head_text.load_state_dict(checkpoint_contra_head_text_valor_pretrained, strict=False)

            #     if missing_keys != [] or unexpected_keys != []:
            #         print(f'Unexpected_keys in multimodal encoder: {unexpected_keys}')
            #         print(f'Missing_keys in multimodal encoder: {missing_keys}')


        self.weights_init_cfg = weights_init
        self._weights_init()
        
        self.data_info = data_info

        # Criterions
        self.criterions_cfg = criterions
        
        self.losses = nn.ModuleDict()
        self.losses_cfgs = {}
        for loss, loss_cfg in self.criterions_cfg.losses.items():              
            self.losses[loss] = hydra.utils.instantiate(loss_cfg.instantiate)
            self.losses_cfgs[loss] =  loss_cfg


        self.train_metrics_cfg = metrics.train
        self.train_metrics = nn.ModuleDict()
        self.train_metrics_cfgs = {}
        for metric, metric_cfg in self.train_metrics_cfg.items():
            metric_name = f"train_{metric}"
            self.train_metrics[metric_name] = hydra.utils.instantiate(metric_cfg.instantiate)
            self.train_metrics_cfgs[metric_name] = metric_cfg


        self.valid_metrics_cfg = metrics.valid
        self.valid_metrics = nn.ModuleDict()
        self.valid_metrics_cfgs = {}
        for metric, metric_cfg in self.valid_metrics_cfg.items():
            metric_name = f"valid_{metric}"
            self.valid_metrics[metric_name] = hydra.utils.instantiate(metric_cfg.instantiate)
            self.valid_metrics_cfgs[metric_name] = metric_cfg
        
        self.test_metrics_cfg = metrics.test
        self.test_metrics = nn.ModuleDict()
        self.test_metrics_cfgs = {}
        for metric, metric_cfg in self.test_metrics_cfg.items():
            metric_name = f"test_{metric}"
            self.test_metrics[metric_name] = hydra.utils.instantiate(metric_cfg.instantiate)
            self.test_metrics_cfgs[metric_name] = metric_cfg

        self.epoch_correct_tokens = 0
        self.epoch_total_tokens = 0

    def _weights_init(self):
        #TODO if training doesn't work well change initializations to none standard.
        
        self.scaling_factor = self.video_decoder.num_layers
        mamba_decoder_weigths_init_cfg = self.weights_init_cfg.mamba_decoder_weigths_init
        for name, module in self.video_decoder.named_modules():
            # Not accessed because no bias is used in linear layers of mamba decoder at the moment
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        hydra.utils.call(mamba_decoder_weigths_init_cfg.linear.bias, tensor=module.bias)     
            # rescales out projection linear layer for each layer. Making them smaller the later the layer. 
            if mamba_decoder_weigths_init_cfg.rescale_prenorm_residual:
                for p_name, p in module.named_parameters():
                    # if p_name in mamba_decoder_weigths_init_cfg.linear.names:
                    #     hydra.utils.call(mamba_decoder_weigths_init_cfg.linear.weight, tensor=p)
                    #     with torch.no_grad():
                    #         layer_idx = module.layer_idx + 1 
                    #         p /= math.sqrt(mamba_decoder_weigths_init_cfg.n_residuals_per_layer * layer_idx)
                     if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                        with torch.no_grad():
                            p /= math.sqrt(self.scaling_factor)

        head_weigths_init_cfg = self.weights_init_cfg.head_weigths_init
        for name, module in self.video_decoder.head.named_modules():
             if isinstance(module, nn.Linear):
                #hydra.utils.call(head_weigths_init_cfg.linear.weight, tensor=module.weight)
                #layer_idx = self.mamba_decoder.num_layers + self.mamba_decoder2.num_layers
                #module.weight.data = module.weight.data / math.sqrt(mamba_decoder_weigths_init_cfg.n_residuals_per_layer * layer_idx)
                if module.bias is not None:
                    hydra.utils.call(head_weigths_init_cfg.linear.bias, tensor=module.bias)



    #TODO maybe needs to be changed depending on dataset design
    def forward_training(self, text_tokens, video_inputs, inference_params=None):

        ## Video decoder
        longer_temporal_features, oad_logits = self.forward_video_decoder(feat_map_pooled = video_inputs,
                                                                        inference_params = inference_params)        

        ## Text encoder
        text_tokens = text_tokens[0,:,:]
        longer_temporal_features = longer_temporal_features[0,:,:]

        # TODO check adding before and after,also maybe after splitting
        length_longer_temporal_features = longer_temporal_features.shape[0]
        #longer_temporal_features = longer_temporal_features + self.video_features_position_embedding[:length_longer_temporal_features, :]

        text_tokens_sections, text_tokens_section_counts  = torch.unique_consecutive(text_tokens, dim=0, return_counts=True)
        section_split_idxs = torch.cumsum(text_tokens_section_counts, dim=0).to(device="cpu")
        max_features_per_caption, max_features_per_caption_idx = text_tokens_section_counts.max(dim=0)
        #! Added to make features with really long number of visual features shorter
        max_visual_features_per_captions = 1000
        if max_features_per_caption > max_visual_features_per_captions:
            sample_1000_points = sorted(random.sample(range(0,max_features_per_caption), max_visual_features_per_captions))
            longer_temporal_features_splitted = list(longer_temporal_features.tensor_split(section_split_idxs)[:-1])
            longer_temporal_features_splitted[max_features_per_caption_idx] = longer_temporal_features_splitted[max_features_per_caption_idx][sample_1000_points,:]
            max_features_per_caption = longer_temporal_features_splitted[max_features_per_caption_idx].shape[0] 
        else:
            longer_temporal_features_splitted = longer_temporal_features.tensor_split(section_split_idxs)[:-1]
        longer_temporal_features_splitted_padded = []
        cross_attention_masks = []
        for split in longer_temporal_features_splitted:
            pad_tensor = torch.zeros(1,768).to(device=split.device)
            pad_len = max_features_per_caption-split.shape[0]
            pad_mask = torch.ones((max_features_per_caption,))
            pad_mask[split.shape[0]:] = 0
            pad_mask = pad_mask.to(split.device)
            split_pad = torch.concatenate((split, pad_tensor.repeat((pad_len,1))), dim=0)
            longer_temporal_features_splitted_padded.append(split_pad)
            cross_attention_masks.append(pad_mask)
        longer_temporal_features_splitted_padded = torch.stack(longer_temporal_features_splitted_padded)
        cross_attention_masks = torch.stack(cross_attention_masks)

        assert text_tokens_sections.shape[0] == longer_temporal_features_splitted_padded.shape[0]

        #! Newly addded
        #! Also detach is added this would result that contrastive loss just changes the projection layers to create features fitting for captioning?
        longer_temporal_features_splitted_padded = self.projection_layer(longer_temporal_features_splitted_padded) # .detach()

        # print(text_tokens_sections.shape[0])
        # print(text_tokens_sections.shape[1])
        # print(longer_temporal_features_splitted_padded.shape[1])
        # print("_____")

        n_max_sections = 80
        if text_tokens_sections.shape[0] > n_max_sections:
            n_sections = text_tokens_sections.shape[0]
            n_more_than_max = n_sections - n_max_sections
            random_max_section_start_idx = np.random.choice(int(n_more_than_max), 1).item()
            text_tokens_sections = text_tokens_sections[random_max_section_start_idx:random_max_section_start_idx+n_max_sections]
            longer_temporal_features_splitted_padded = longer_temporal_features_splitted_padded[random_max_section_start_idx:random_max_section_start_idx+n_max_sections]
            cross_attention_masks = cross_attention_masks[random_max_section_start_idx:random_max_section_start_idx+n_max_sections]            
            #print(f"Video with more than a {n_max_sections} sections.")

        assert text_tokens_sections.shape[0] == longer_temporal_features_splitted_padded.shape[0]
        text_features_sections, text_features_sections_pooled = self.forward_text_encoder(text_tokens = text_tokens_sections, task_prompt=None)


        ## Contrastive loss preparations
        # Pooled video features projection and weighting for feature-/token-wise contrastive loss
        video_features_pooled_contra = self.forward_contra_proj_video(longer_temporal_features_splitted_padded)
        video_features_pooled_contra_weighted = self.forward_weighted_proj_video(video_features_pooled_contra)
        # Unpooled text features projection and weighting for feature/token-wise contrastive loss
        text_features_contra = self.forward_contra_proj_text(text_features_sections)
        text_features_contra_weighted = self.forward_weighted_proj_text(text_features_contra)
        ## Multimodal decoder

        # Captioning/MLM task
        #bs = feat_map.shape[0]
        #task_prompt_captioning = "describe the video with natural language"
        #task_prompt_captioning_tokens = self.get_task_prompts(task_prompt_captioning, bs)

        
        # TODO test if problematic, for video with really much sections randomly select the sections
        # n_max_sections = 80
        # if text_tokens_sections.shape[0] > n_max_sections:
        #     n_sections = text_tokens_sections.shape[0]
        #     n_more_than_max = n_sections - n_max_sections
        #     random_max_section_start_idx = np.random.choice(int(n_more_than_max), 1).item()
        #     less_section_text_tokens_sections = text_tokens_sections[random_max_section_start_idx:random_max_section_start_idx+n_max_sections]
        #     less_section_longer_temporal_features_splitted_padded = longer_temporal_features_splitted_padded[random_max_section_start_idx:random_max_section_start_idx+n_max_sections]
        #     less_section_cross_attention_masks = cross_attention_masks[random_max_section_start_idx:random_max_section_start_idx+n_max_sections]            
        #     print("Video with more than a hundert sections.")


        #     masked_text_tokens, labels_text_tokens = self.forward_mask_text_tokens(text_tokens=less_section_text_tokens_sections)
        #     multimodal_text_features_captioning = self.forward_multimodal_decoder(text_tokens=masked_text_tokens, 
        #                                                             task_prompt=None,
        #                                                                 video_feat=less_section_longer_temporal_features_splitted_padded,
        #                                                                 cross_attention_masks = less_section_cross_attention_masks,
        #                                                                 causal=True)
        #     logits_over_vocab_captioning = self.forward_prediction_head(multimodal_text_features_captioning)

        
        masked_text_tokens, labels_text_tokens = self.forward_mask_text_tokens(text_tokens=text_tokens_sections)
        multimodal_text_features_captioning = self.forward_multimodal_decoder(text_tokens=masked_text_tokens, 
                                                                task_prompt=None,
                                                                    video_feat=longer_temporal_features_splitted_padded,
                                                                    cross_attention_masks = cross_attention_masks,
                                                                    causal=True)
        logits_over_vocab_captioning = self.forward_prediction_head(multimodal_text_features_captioning)

        if True:
            
            self.forward_decode_tokens(logits_over_vocab_captioning, labels_text_tokens, masked_text_tokens, text_tokens_sections)

        return {
                "contrastive_multimodal_alignment": 
                    {
                        "text_tokens": text_tokens_sections, #text_tokens,
                        "text_features_contra": text_features_contra,
                        "text_features_contra_weighted": text_features_contra_weighted,
                        "video_features_pooled_contra": video_features_pooled_contra,
                        "video_features_pooled_contra_weighted": video_features_pooled_contra_weighted,
                        "video_features_mask": cross_attention_masks

                    },
                    "multimodal_captioning":
                    {
                        "logits_over_vocab": logits_over_vocab_captioning,
                        "labels_txt_tokens": labels_text_tokens,
                    },
                    "oad":
                    {
                        "oad_logits": oad_logits,
                    }
                }   
            # return (
            #             (text_tokens, text_features_contra, text_features_contra_weighted, video_features_pooled_contra, video_features_pooled_contra_weighted),
            #             (logits_over_vocab_captioning, labels_text_tokens),
            #             (longer_temporal_features, logits)
            #         )

    # def forward_training(self, text_tokens, video_inputs, inference_params=None):

    #     ## Text encoder
    #     text_tokens = text_tokens[0,:,:]
    #     text_tokens_sections, text_tokens_section_counts  = torch.unique_consecutive(text_tokens, dim=0, return_counts=True)
    #     section_split_idxs = torch.cumsum(text_tokens_section_counts, dim=0)
        

    #     text_features, text_features_pooled = self.forward_text_encoder(text_tokens = text_tokens, task_prompt=None)

    #     ## Video decoder
    #     longer_temporal_features, oad_logits = self.forward_video_decoder(feat_map_pooled = video_inputs,
    #                                                                     inference_params = inference_params)

    #     ## Contrastive loss preparations
    #     # Pooled video features projection and weighting for feature-/token-wise contrastive loss
    #     video_features_pooled_contra = self.forward_contra_proj_video(longer_temporal_features)
    #     video_features_pooled_contra_weighted = self.forward_weighted_proj_video(video_features_pooled_contra)
    #     # Unpooled text features projection and weighting for feature/token-wise contrastive loss
    #     text_features_contra = self.forward_contra_proj_text(text_features)
    #     text_features_contra_weighted = self.forward_weighted_proj_text(text_features_contra)
    #     ## Multimodal decoder

    #     # Captioning/MLM task
    #     #bs = feat_map.shape[0]
    #     #task_prompt_captioning = "describe the video with natural language"
    #     #task_prompt_captioning_tokens = self.get_task_prompts(task_prompt_captioning, bs)

    #     masked_text_tokens, labels_text_tokens = self.forward_mask_text_tokens(text_tokens=text_tokens)
    #     multimodal_text_features_captioning = self.forward_multimodal_decoder(text_tokens=masked_text_tokens, 
    #                                                                task_prompt=None,
    #                                                                 video_feat=longer_temporal_features,
    #                                                                 causal=True)
    #     logits_over_vocab_captioning = self.forward_prediction_head(multimodal_text_features_captioning)

    #     return {
    #             "contrastive_multimodal_alignment": 
    #                 {
    #                     "text_tokens": text_tokens,
    #                     "text_features_contra": text_features_contra,
    #                     "text_features_contra_weighted": text_features_contra_weighted,
    #                     "video_features_pooled_contra": video_features_pooled_contra,
    #                     "video_features_pooled_contra_weighted": video_features_pooled_contra_weighted,

    #                 },
    #                 "multimodal_captioning":
    #                 {
    #                     "logits_over_vocab": logits_over_vocab_captioning,
    #                     "labels_txt_tokens": labels_text_tokens,
    #                 },
    #                 "oad":
    #                 {
    #                     "oad_logits": oad_logits,
    #                 }
    #             }   
    #         # return (
    #         #             (text_tokens, text_features_contra, text_features_contra_weighted, video_features_pooled_contra, video_features_pooled_contra_weighted),
    #         #             (logits_over_vocab_captioning, labels_text_tokens),
    #         #             (longer_temporal_features, logits)
    #         #         )



 #TODO maybe needs to be changed depending on dataset design
    def forward_validation(self, video_inputs, inference_params=None, oad_labels=None, video_name=None, global_step=None):

       
        ## Video decoder
        longer_temporal_features, oad_logits = self.forward_video_decoder(feat_map_pooled = video_inputs,
                                                                        inference_params = inference_params)

        if (global_step % 2000) == 0:

            longer_temporal_features = longer_temporal_features[0,:,:]
            length_longer_temporal_features = longer_temporal_features.shape[0]
            #longer_temporal_features = longer_temporal_features + self.video_features_position_embedding[:length_longer_temporal_features, :]


            longer_temporal_features_splitted_padded, cross_attention_masks = self.chunk_features_based_on_oad_predictions(oad_logits[0,:,:], longer_temporal_features)
            
            #! Newly added
            longer_temporal_features_splitted_padded = self.projection_layer(longer_temporal_features_splitted_padded)

            section_captions = self.forward_autoregressive_decode_greedy(longer_temporal_features_splitted_padded, cross_attention_masks)

            predicted_ids_list = section_captions.clone().detach().cpu().tolist()
            predicted_captions_list = [self.shared_text_multimodal_encoder_decoder.bert_tokenizer.convert_ids_to_tokens(predicted_ids)  for id, predicted_ids in enumerate(predicted_ids_list)]
            predicted_captions_text = [" ".join(predicted_caption_list) for predicted_caption_list in predicted_captions_list]
            # print(predicted_captions_text[0])
            predictions = torch.argmax(oad_logits, dim=-1)[0,:]
            labels = torch.argmax(oad_labels, dim=-1)[0,:]
            action_idx = torch.where(labels!=0)[0]
            background_idx = torch.where(labels==0)[0]
            pred_labels_sections, pred_labels_section_counts  = torch.unique_consecutive(predictions, dim=0, return_counts=True)

            predicted_captions_text =  [f"{video_name}: {str(pred_labels_sections[idx].item()), str(pred_labels_section_counts[idx].item())}: {predicted_caption_text}" for idx, predicted_caption_text in enumerate(predicted_captions_text)]

            with open("caption_results.txt", mode="a") as file:
                file.writelines(predicted_captions_text)

        return  {
                    "oad":
                    {
                        "oad_logits": oad_logits,
                    }
                } 

    def forward_test(self, video_inputs, inference_params=None, oad_labels=None, video_name=None, global_step=None):
        	
        video_inputs = video_inputs[0,:,:]
        oad_logits_full_video = torch.zeros(1,video_inputs.shape[0], 22).to(video_inputs.device)
        longer_temporal_features_buffer = [] 
        
        torch.cuda.synchronize()
        start_time = time.time()
        print(f"Number of features: {video_inputs.shape[0]}")
        for feature_id, feature_video in enumerate(video_inputs):

            ## Video decoder
            longer_temporal_features, oad_logits = self.forward_video_decoder(feat_map_pooled = feature_video[None,None,:],
                                                                        inference_params = inference_params)
            
            oad_logits_full_video[:,feature_id,:] = oad_logits
            
            if feature_id == 0:
                prev_pred = torch.argmax(oad_logits[0,:,:], dim=-1)
                curr_pred = prev_pred
            else:
                curr_pred = torch.argmax(oad_logits[0,:,:], dim=-1)
            
            if curr_pred != prev_pred:
                #TODO add part to pass features and create captions delete buffer
                number_of_ltf_in_buffer = len(longer_temporal_features_buffer)
                longer_temporal_features_buffer_stacked = torch.stack(longer_temporal_features_buffer, dim=0)

                longer_temporal_features_buffer_stacked = self.projection_layer(longer_temporal_features_buffer_stacked[None,:,:])

                section_caption = self.forward_autoregressive_decode_greedy(longer_temporal_features_buffer_stacked, cross_attention_masks=None)
                predicted_ids_list = section_caption.clone().detach().cpu().tolist()
                predicted_captions_list = [self.shared_text_multimodal_encoder_decoder.bert_tokenizer.convert_ids_to_tokens(predicted_ids)  for id, predicted_ids in enumerate(predicted_ids_list)]
                predicted_captions_text = [" ".join(predicted_caption_list) for predicted_caption_list in predicted_captions_list]
                
                prev_pred = curr_pred
                longer_temporal_features_buffer = []

            longer_temporal_features_buffer.append(longer_temporal_features[0,0,:])
        
        torch.cuda.synchronize()
        end_time = time.time()
        total_time_video = end_time-start_time
        print(f'Running time: {total_time_video:.3f} seconds')

        return  {
                    "oad":
                    {
                        "oad_logits": oad_logits_full_video,
                    }
                } 

             


    def forward_video_decoder(self, feat_map_pooled, inference_params):
        
        longer_temporal_features, logits = self.video_decoder(feat_map_pooled, inference_params)

        return longer_temporal_features, logits
    
    def forward_text_encoder(self, text_tokens, task_prompt=None):

        text_features = self.shared_text_multimodal_encoder_decoder.forward_bert_text_encoder(text_tokens=text_tokens, task_prompt=task_prompt, video_feat=None, causal=False)
        text_features_pooled = text_features[:,0] # Check if correct dimension, for BERT just take the class token for pooling.

        return text_features, text_features_pooled
    
    def forward_mask_text_tokens(self, text_tokens):

        masked_text_tokens, labels_txt_tokens = self.shared_text_multimodal_encoder_decoder.token_masker(text_tokens)
        return masked_text_tokens, labels_txt_tokens
    
    def get_task_prompt(self, sentence, batch_size=0, cls_prompt=False):

        sentence = self.shared_text_multimodal_encoder_decoder.bert_tokenizer.tokenize(sentence)
        sentence = self.shared_text_multimodal_encoder_decoder.bert_tokenizer.convert_tokens_to_ids(sentence)
        task_prompt = [self.shared_text_multimodal_encoder_decoder.bos_token] + sentence + [self.shared_text_multimodal_encoder_decoder.eos_token]
        
        if not cls_prompt:
            task_prompt = torch.tensor(task_prompt).unsqueeze(0).expand(batch_size,-1).long().cuda()
        return task_prompt

    def forward_multimodal_decoder(self, text_tokens, task_prompt=None, video_feat=None, causal=True, token_type=None, cross_attention_masks=None, cache=None):
        
        multimodal_text_features = self.shared_text_multimodal_encoder_decoder.forward_bert_multimodal_decoder(text_tokens=text_tokens, task_prompt=task_prompt, video_feat=video_feat, causal=causal, token_type=token_type, cross_attention_masks=cross_attention_masks)
        return multimodal_text_features

    def forward_prediction_head(self, multimodal_text_features):
        
        logits_over_vocab = self.shared_text_multimodal_encoder_decoder.forward_bert_prediction_head(multimodal_text_features)
        return logits_over_vocab
    
    def forward_decode_tokens(self, logits_over_vocab_captioning, labels_text_tokens,  masked_text_tokens, text_tokens_sections):

        _ , predicted_ids_all_captions = torch.max(logits_over_vocab_captioning, dim=-1) 
        tokenizer = self.shared_text_multimodal_encoder_decoder.bert_tokenizer

        save_path = Path("experiments/OAD/train_thumos_tufvfcd_vufvfd_valor/captions_results")

        # Look at correct captions:
        gt_text_ids_list = text_tokens_sections.clone().detach().cpu().tolist()
       
        gt_text_ids_list_without_padding = [[gt_text_id for gt_text_id in gt_text_ids if gt_text_id!=0] for gt_text_ids in gt_text_ids_list]
        lengths_till_padding = [len(wp) for wp in gt_text_ids_list_without_padding]
        gt_captions_list_without_padding =  [tokenizer.convert_ids_to_tokens(text_ids) for text_ids in gt_text_ids_list_without_padding]
        gt_captions_text_without_padding =  [" ".join(gt_caption_list) for gt_caption_list in gt_captions_list_without_padding]

        gt_captions_list = [tokenizer.convert_ids_to_tokens(text_ids) for text_ids in gt_text_ids_list]
        gt_captions_text = [" ".join(gt_caption_list) for gt_caption_list in gt_captions_list]
        #print(gt_captions_text_without_padding[0])


        # REMOVE padding tokens 
        # masked_ids_list = masked_text_tokens.detach().cpu().tolist()
        # masked_ids_without_padding = [[masked_id for masked_id in masked_ids if masked_id!=0] for masked_ids in masked_ids_list]
        # lengths_till_padding = [len(maiwp) for maiwp in masked_ids_without_padding]
        # masked_captions_without_padding = [tokenizer.convert_ids_to_tokens(masked_ids) for masked_ids in masked_ids_without_padding]

        predicted_ids_list = predicted_ids_all_captions.clone().detach().cpu().tolist()
        predicted_captions_list_without_padding = [tokenizer.convert_ids_to_tokens(predicted_ids[:lengths_till_padding[id]])  for id, predicted_ids in enumerate(predicted_ids_list)]
        predicted_captions_text_without_padding = [" ".join(predicted_caption_list) for predicted_caption_list in predicted_captions_list_without_padding]
        #print(predicted_captions_text_without_padding[0])
        labels_text_tokens_list = labels_text_tokens.clone().detach().cpu().tolist()
        masking_idxs = [[idx for idx, labels_text_idx in enumerate(labels_text_idxs) if labels_text_idx!=-1] for labels_text_idxs in labels_text_tokens_list]

        gt_2_pred_masked_tokens_list =  [{gt_pred_tuple[0]:gt_pred_tuple[1] for id, gt_pred_tuple in enumerate(zip(tuple_captions[0], tuple_captions[1])) if id in masking_idxs[batch_id]} for batch_id, tuple_captions in enumerate(zip(gt_captions_list_without_padding, predicted_captions_list_without_padding))]
        correct_tokens = 0
        total_tokens = 0
        for g2pred_maskesd_token in gt_2_pred_masked_tokens_list:
            for gt_token, pred_token in g2pred_maskesd_token.items():
                total_tokens += 1
                if gt_token == pred_token:
                    correct_tokens += 1
        acc = correct_tokens / total_tokens
        print(f"Accuracy masked tokens: {acc:.2f}")

        self.epoch_correct_tokens += correct_tokens
        self.epoch_total_tokens += total_tokens

    def chunk_features_based_on_oad_predictions(self, oad_logits, longer_temporal_features):

        pred_labels = torch.argmax(oad_logits, dim=-1)
        
        pred_labels_sections, pred_labels_section_counts  = torch.unique_consecutive(pred_labels, dim=0, return_counts=True)
        section_split_idxs = torch.cumsum(pred_labels_section_counts, dim=0).to(device="cpu")
        max_features_per_section, max_features_per_section_idx = pred_labels_section_counts.max(dim=0)
        #! Added to make features with really long number of visual features shorter
        max_visual_features_per_sections = 1000
        if max_features_per_section > max_visual_features_per_sections:
            sample_1000_points = sorted(random.sample(range(0,max_features_per_section), max_visual_features_per_sections))
            longer_temporal_features_splitted = list(longer_temporal_features.tensor_split(section_split_idxs)[:-1])
            longer_temporal_features_splitted[max_features_per_section_idx] = longer_temporal_features_splitted[max_features_per_section_idx][sample_1000_points,:]
            max_features_per_section = longer_temporal_features_splitted[max_features_per_section_idx].shape[0] 
        else:
            longer_temporal_features_splitted = longer_temporal_features.tensor_split(section_split_idxs)[:-1]
        longer_temporal_features_splitted_padded = []
        cross_attention_masks = []
        for split in longer_temporal_features_splitted:
            pad_tensor = torch.zeros(1,768).to(device=split.device)
            pad_len = max_features_per_section-split.shape[0]
            pad_mask = torch.ones((max_features_per_section,))
            pad_mask[split.shape[0]:] = 0
            pad_mask = pad_mask.to(split.device)
            split_pad = torch.concatenate((split, pad_tensor.repeat((pad_len,1))), dim=0)
            longer_temporal_features_splitted_padded.append(split_pad)
            cross_attention_masks.append(pad_mask)
        longer_temporal_features_splitted_padded = torch.stack(longer_temporal_features_splitted_padded)
        cross_attention_masks = torch.stack(cross_attention_masks)

        return longer_temporal_features_splitted_padded, cross_attention_masks


    def forward_autoregressive_decode_greedy(self, longer_temporal_features_splitted_padded, cross_attention_masks):
        
        max_generation_len = 100 # TODO should be set via config
        n_sections = longer_temporal_features_splitted_padded.shape[0]

        section_captions = torch.zeros((n_sections, max_generation_len), dtype=torch.long).fill_(self.shared_text_multimodal_encoder_decoder.eos_token).to(longer_temporal_features_splitted_padded.device)
        unfinished_captions_flags = torch.ones(n_sections, dtype=torch.bool).to(longer_temporal_features_splitted_padded.device)

        state = None

        for token_idx in range(max_generation_len):

            if token_idx==0:
                logits = self.one_step_autoregressive_decode_greedy(longer_temporal_features_splitted_padded, cross_attention_masks, state)
            else:
                logits = self.one_step_autoregressive_decode_greedy(longer_temporal_features_splitted_padded, cross_attention_masks, state)

            _ , created_token = torch.max(logits, dim=1) #! question mark if decoding works with multiple dims

            created_token = created_token.view(-1).long() #! wierd
            unfinished_captions_flags = unfinished_captions_flags * (created_token != self.shared_text_multimodal_encoder_decoder.eos_token)
            created_token = created_token * unfinished_captions_flags.type_as(created_token) + (1 - unfinished_captions_flags.type_as(created_token)) * self.shared_text_multimodal_encoder_decoder.eos_token
            
            section_captions[:,token_idx] = created_token

            state = created_token.unsqueeze(1) if state is None else torch.cat((state,created_token.unsqueeze(1)),dim=1) #! check

            if unfinished_captions_flags.sum() == 0:
                break
        
        return section_captions

    def one_step_autoregressive_decode_greedy(self, longer_temporal_features_splitted_padded, cross_attention_masks, state):

        n_sections = longer_temporal_features_splitted_padded.shape[0]
        inserted_masked_tokens = torch.zeros(n_sections, 1 , dtype = torch.long).cuda().fill_(self.shared_text_multimodal_encoder_decoder.mask_token)
        first_bos_tokens = torch.zeros(n_sections, 1 , dtype = torch.long).cuda().fill_(self.shared_text_multimodal_encoder_decoder.bos_token)
        created_txt_tokens = torch.cat((state,inserted_masked_tokens), dim=1 ) if state is not None else inserted_masked_tokens
        created_txt_tokens = torch.cat((first_bos_tokens,created_txt_tokens), dim=1)

        multimodal_text_features_captioning = self.forward_multimodal_decoder(text_tokens=created_txt_tokens, 
                                                                task_prompt=None,
                                                                    video_feat=longer_temporal_features_splitted_padded,
                                                                    cross_attention_masks = cross_attention_masks,
                                                                    causal=True)

        logits_over_vocab_captioning = self.forward_prediction_head(multimodal_text_features_captioning)
        logits_last_token = logits_over_vocab_captioning[:,-1,:]

        return logits_last_token



    def forward_contra_proj_text(self, text_features_pooled):

        text_features_pooled_contra_proj = self.contra_head_text.forward_contra_proj(text_features_pooled)
        text_features_pooled_contra_proj = F.normalize(text_features_pooled_contra_proj, p=2, dim=-1) #! Check if correct dimension
        return text_features_pooled_contra_proj
    
    def forward_weighted_proj_text(self, text_features_pooled_contra_proj):

        text_features_pooled_contra_proj_weighted = self.contra_head_text.forward_weighted_proj(text_features_pooled_contra_proj)
        return text_features_pooled_contra_proj_weighted
    
    def forward_contra_proj_video(self, feat_map_conv_pooled): #! Two possibilities either use feat_map_conv_pooled or mamba longer_temporal_features
        
        feat_map_conv_pooled_contra_proj = self.contra_head_video.forward_contra_proj(feat_map_conv_pooled)
        feat_map_conv_pooled_contra_proj = F.normalize(feat_map_conv_pooled_contra_proj, p=2, dim=-1) #! Check if correct dimension
        return feat_map_conv_pooled_contra_proj

    def forward_weighted_proj_video(self, feat_map_conv_pooled_contra_proj):

        feat_map_conv_pooled_contra_proj_weighted = self.contra_head_text.forward_weighted_proj(feat_map_conv_pooled_contra_proj)
        return feat_map_conv_pooled_contra_proj_weighted  

    def calculate_loss(self, batch, phase):

        loss = 0.0

        if phase == "valid":
            for loss_name, loss_func in self.losses.items():
                loss_cfg = self.losses_cfgs[loss_name]
                if loss_name == "oad_loss":
                    output_name = loss_cfg.output
                    loss_weight = loss_cfg.weight
                    loss = loss + loss_weight * loss_func(**batch[output_name]) 

        else:
            for loss_name, loss_func in self.losses.items():
                loss_cfg = self.losses_cfgs[loss_name]            
                output_name = loss_cfg.output
                loss_weight = loss_cfg.weight
                
                loss = loss + loss_weight * loss_func(**batch[output_name]) 

        return loss
 
    def compute_metrics_in_train_step(self, batch):

        train_metrics_step = {}
        for metric_name, metric in self.train_metrics.items():
            if metric_name not in ["train_mAP", "train_ConfusionMatrix"]:
                metric_cfg = self.train_metrics_cfgs[metric_name]    

                metric_value = metric(*metric.prepare_preds_and_targets(**batch["oad"]))

                wandb_name = "/".join(metric_name.split("_", 1))
                train_metrics_step[wandb_name] = metric_value

        return train_metrics_step

    def compute_metrics_in_valid_step(self, batch):

        valid_metrics_step = {}
        for metric_name, metric in self.valid_metrics.items():
            if metric_name not in ["valid_mAP", "valid_ConfusionMatrix"]:
                metric_cfg = self.valid_metrics_cfgs[metric_name]  

                metric_value = metric(*metric.prepare_preds_and_targets(**batch["oad"]))

                wandb_name = "/".join(metric_name.split("_", 1))
                valid_metrics_step[wandb_name] = metric_value

        return valid_metrics_step
    
    def compute_metrics_in_test_step(self, batch):

        test_metrics_step = {}
        for metric_name, metric in self.test_metrics.items():
            if metric_name not in ["test_mAP", "test_ConfusionMatrix"]:
                metric_cfg = self.test_metrics_cfgs[metric_name]  

                metric_value = metric(*metric.prepare_preds_and_targets(**batch["oad"]))

                wandb_name = "/".join(metric_name.split("_", 1))
                test_metrics_step[wandb_name] = metric_value

        return test_metrics_step

    def compute_metrics_on_train_epoch_end(self):

        train_metrics_epoch = {}

        train_metric = self.train_metrics["train_mAP"].compute()        
        
        if isinstance(train_metric, dict): 
            train_metrics_epoch["train/mAP"] = train_metric["mAP"]
            for name, value in train_metric["per_class_AP"].items():                
                wandb_name = "/".join(("train"+name).split("_", 1))
                train_metrics_epoch[wandb_name] = value  
        else:
            train_metrics_epoch["train/mAP"] = train_metric

        if "train_ConfusionMatrix" in self.train_metrics.keys():
            self.train_metrics["train_ConfusionMatrix"].compute(phase="train")

        return train_metrics_epoch

    def compute_metrics_on_valid_epoch_end(self):

        valid_metrics_epoch = {}

        valid_metric = self.valid_metrics["valid_mAP"].compute()
     
        if isinstance(valid_metric, dict): 
            valid_metrics_epoch["valid/mAP"] = valid_metric["mAP"]
            for name, value in valid_metric["per_class_AP"].items():                
                wandb_name = "/".join(("valid"+name).split("_", 1))
                valid_metrics_epoch[wandb_name] = value  
        else:
            valid_metrics_epoch["valid/mAP"] = valid_metric   

        if "valid_ConfusionMatrix" in self.valid_metrics.keys():
            self.valid_metrics["valid_ConfusionMatrix"].compute(phase="valid")
        
        return valid_metrics_epoch
    
    def compute_metrics_on_test_epoch_end(self):

        test_metrics_epoch = {}

        test_metric = self.test_metrics["valid_mAP"].compute()
     
        if isinstance(test_metric, dict): 
            test_metrics_epoch["valid/mAP"] = test_metric["mAP"]
            for name, value in test_metric["per_class_AP"].items():                
                wandb_name = "/".join(("valid"+name).split("_", 1))
                test_metrics_epoch[wandb_name] = value  
        else:
            test_metrics_epoch["valid/mAP"] = test_metric   

        if "valid_ConfusionMatrix" in self.test_metrics.keys():
            self.test_metrics_epoch["valid_ConfusionMatrix"].compute(phase="valid")
        
        return test_metrics_epoch
    

    def update_train_metrics_states(self, batch):
        
        for metric_name, metric in self.train_metrics.items():
            if metric_name in ["train_mAP", "train_ConfusionMatrix"]:
                metric_cfg = self.train_metrics_cfgs[metric_name]   

                metric.update(**batch["oad"])   


    def update_valid_metrics_states(self, batch):

        for metric_name, metric in self.valid_metrics.items():
            if metric_name in ["valid_mAP", "valid_ConfusionMatrix"]:
                metric_cfg = self.valid_metrics_cfgs[metric_name]    

                metric.update(**batch["oad"])
    
    def update_test_metrics_states(self, batch):

        for metric_name, metric in self.test_metrics.items():
            if metric_name in ["test_mAP", "test_ConfusionMatrix"]:
                metric_cfg = self.test_metrics_cfgs[metric_name]    

                metric.update(**batch["oad"])

    def reset_train_metrics_states(self):
        
        if "train_mAP" in self.train_metrics.keys():
            self.train_metrics["train_mAP"].reset()
        if "train_ConfusionMatrix" in self.train_metrics.keys():
            self.train_metrics["train_ConfusionMatrix"].reset()

    def reset_valid_metrics_states(self):
        
        if "valid_mAP" in self.valid_metrics.keys():
            self.valid_metrics["valid_mAP"].reset()
        if "valid_ConfusionMatrix" in self.valid_metrics.keys():
            self.valid_metrics["valid_ConfusionMatrix"].reset()


#class VideoMAEv2VideoLadderAdapterVideoFeatureMapConvPoolerVideoMambaDecoderSharedTextMultimodalBertEncoderDecoder(nn.Module):

#     def __init__(
#         self,
#         video_encoder,
#         video_decoder,
#         shared_text_multimodal_encoder_decoder,
#         contrastive_heads,
#         weights_init,
#         data_info,
#         criterions,
#         metrics,
#     ):
        
#         super().__init__()

#         # Build an load video encoder from config
#         video_encoder_cfg = video_encoder
#         self.video_encoder = VideoMAEv2VideoLadderAdapterVideoFeatureMapConvPooler(**video_encoder_cfg)
#         self.video_encoder_embed_dims = self.video_encoder.embed_dims

#         video_decoder_cfg = video_decoder
#         video_decoder_cfg.adapter.encoder_embed_dims = self.video_encoder_embed_dims
#         self.video_decoder = VideoMambaDecoder(**video_decoder_cfg)
#         self.video_decoder_embed_dims = self.video_decoder.decoder_embed_dims
        
#         shared_text_multimodal_encoder_decoder_cfg = shared_text_multimodal_encoder_decoder
#         self.shared_text_multimodal_encoder_decoder = SharedTextMultimodalBertEncoderDecoder(**shared_text_multimodal_encoder_decoder_cfg)
#         self.multimodal_decoder_embed_dims = self.shared_text_multimodal_encoder_decoder.bert_multimodal_decoder_embed_dims
#         self.text_encoder_embed_dims = self.shared_text_multimodal_encoder_decoder.bert_text_encoder_embed_dims

#         contrastive_heads_cfg = contrastive_heads
#         if contrastive_heads_cfg != None:
            
#             contrastive_head_video_cfg = contrastive_heads_cfg.contra_head_video
#             contrastive_head_video_cfg.embed_dims = self.video_encoder_embed_dims
#             self.contra_head_video = ContrastiveHead(**contrastive_head_video_cfg)

#             contrastive_head_text_cfg = contrastive_heads_cfg.contra_head_text
#             contrastive_head_text_cfg.embed_dims = self.text_encoder_embed_dims
#             self.contra_head_text = ContrastiveHead(**contrastive_head_text_cfg)


#         self.weights_init_cfg = weights_init
#         self._weights_init()
        
#         self.data_info = data_info

#         # Criterions
#         self.criterions_cfg = criterions
        
#         self.losses = nn.ModuleDict()
#         self.losses_cfgs = {}
#         for loss, loss_cfg in self.criterions_cfg.losses.items():              
#             self.losses[loss] = hydra.utils.instantiate(loss_cfg.instantiate)
#             self.losses_cfgs[loss] =  loss_cfg


#         self.train_metrics_cfg = metrics.train
#         self.train_metrics = nn.ModuleDict()
#         self.train_metrics_cfgs = {}
#         for metric, metric_cfg in self.train_metrics_cfg.items():
#             metric_name = f"train_{metric}"
#             self.train_metrics[metric_name] = hydra.utils.instantiate(metric_cfg.instantiate)
#             self.train_metrics_cfgs[metric_name] = metric_cfg


#         self.valid_metrics_cfg = metrics.valid
#         self.valid_metrics = nn.ModuleDict()
#         self.valid_metrics_cfgs = {}
#         for metric, metric_cfg in self.valid_metrics_cfg.items():
#             metric_name = f"valid_{metric}"
#             self.valid_metrics[metric_name] = hydra.utils.instantiate(metric_cfg.instantiate)
#             self.valid_metrics_cfgs[metric_name] = metric_cfg


#     def _weights_init(self):
#         #TODO if training doesn't work well change initializations to none standard.
#         pass


#     #TODO maybe needs to be changed depending on dataset design
#     def forward_training(self, text_tokens, video_inputs, inference_params=None):

#         ## Video encoder
#         feat_map, feat_map_conv_pooled, feat_map_conv_pooled_mean = self.forward_video_encoder(video_inputs = video_inputs)
#         ## Text encoder
#         #text_features, text_features_pooled = self.forward_text_encoder(text_tokens = text_tokens, task_prompt=None)

#         ## Contrastive loss preparations
#         # Pooled video features projection and weighting for feature-/token-wise contrastive loss
#        # video_features_pooled_contra = self.forward_contra_proj_video(feat_map_conv_pooled)
#         #video_features_pooled_contra_weighted = self.forward_weighted_proj_video(video_features_pooled_contra)
#         # Unpooled text features projection and weighting for feature/token-wise contrastive loss
#         #text_features_contra = self.forward_contra_proj_text(text_features)
#         #text_features_contra_weighted = self.forward_weighted_proj_text(text_features_contra)

#         ## Video decoder
#         longer_temporal_features, oad_logits = self.forward_video_decoder(feat_map_conv_pooled = feat_map_conv_pooled_mean,
#                                                                         inference_params = inference_params)
#         ## Multimodal decoder

#         # Captioning/MLM task
#         #bs = feat_map.shape[0]
#         #task_prompt_captioning = "describe the video with natural language"
#         #task_prompt_captioning_tokens = self.get_task_prompts(task_prompt_captioning, bs)

#         #masked_text_tokens, labels_text_tokens = self.forward_mask_text_tokens(text_tokens=text_tokens)
#         #multimodal_text_features_captioning = self.forward_multimodal_decoder(text_tokens=masked_text_tokens, 
#         #                                                           task_prompt=None,
#         #                                                            video_feat=feat_map_conv_pooled,
#         #                                                            causal=True)
#         #logits_over_vocab_captioning = self.forward_prediction_head(multimodal_text_features_captioning)

#         return {
#                 # "contrastive_multimodal_alignment": 
#                 #     {
#                 #         "text_tokens": text_tokens,
#                 #         "text_features_contra": text_features_contra,
#                 #         "text_features_contra_weighted": text_features_contra_weighted,
#                 #         "video_features_pooled_contra": video_features_pooled_contra,
#                 #         "video_features_pooled_contra_weighted": video_features_pooled_contra_weighted,

#                 #     },
#                 #     "multimodal_captioning":
#                 #     {
#                 #         "logits_over_vocab": logits_over_vocab_captioning,
#                 #         "labels_txt_tokens": labels_text_tokens,
#                 #     },
#                     "oad":
#                     {
#                         "oad_logits": oad_logits,
#                     }
#                 }   
#             # return (
#             #             (text_tokens, text_features_contra, text_features_contra_weighted, video_features_pooled_contra, video_features_pooled_contra_weighted),
#             #             (logits_over_vocab_captioning, labels_text_tokens),
#             #             (longer_temporal_features, logits)
#             #         )


#  #TODO maybe needs to be changed depending on dataset design
#     def forward_validation(self, video_inputs, inference_params=None):

#         ## Video encoder
#         feat_map, feat_map_conv_pooled, feat_map_conv_pooled_mean = self.forward_video_encoder(video_inputs = video_inputs)
        
#         ## Video decoder
#         longer_temporal_features, oad_logits = self.forward_video_decoder(feat_map_conv_pooled = feat_map_conv_pooled_mean,
#                                                                         inference_params = inference_params)

#         return  {
#                     "oad":
#                     {
#                         "oad_logits": oad_logits,
#                     }
#                 } 


#     def forward_video_encoder(self, video_inputs):

#         feat_map, feat_map_conv_pooled, feat_map_conv_pooled_mean =  self.video_encoder(video_inputs)

#         return feat_map, feat_map_conv_pooled, feat_map_conv_pooled_mean
    
#     def forward_video_decoder(self, feat_map_conv_pooled, inference_params):
        
#         longer_temporal_features, logits = self.video_decoder(feat_map_conv_pooled, inference_params)

#         return longer_temporal_features, logits
    
#     def forward_text_encoder(self, text_tokens, task_prompt=None):

#         text_features = self.shared_text_multimodal_encoder_decoder.forward_bert_text_encoder(text_tokens=text_tokens, task_prompt=task_prompt, video_feat=None, causal=False)
#         text_features_pooled = text_features[:,0] # Check if correct dimension, for BERT just take the class token for pooling.

#         return text_features, text_features_pooled
    
#     def forward_mask_text_tokens(self, text_tokens):

#         masked_text_tokens, labels_txt_tokens = self.shared_text_multimodal_encoder_decoder.token_masker(text_tokens)
#         return masked_text_tokens, labels_txt_tokens
    
#     def get_task_prompt(self, sentence, batch_size=0, cls_prompt=False):

#         sentence = self.shared_text_multimodal_encoder_decoder.bert_tokenizer.tokenize(sentence)
#         sentence = self.shared_text_multimodal_encoder_decoder.bert_tokenizer.convert_tokens_to_ids(sentence)
#         task_prompt = [self.shared_text_multimodal_encoder_decoder.bos_token] + sentence + [self.shared_text_multimodal_encoder_decoder.eos_token]
        
#         if not cls_prompt:
#             task_prompt = torch.tensor(task_prompt).unsqueeze(0).expand(batch_size,-1).long().cuda()
#         return task_prompt

#     def forward_multimodal_decoder(self, text_tokens, task_prompt=None, video_feat=None, causal=True, token_type=None):
        
#         multimodal_text_features = self.shared_text_multimodal_encoder_decoder.forward_bert_multimodal_decoder(text_tokens=text_tokens, task_prompt=task_prompt, video_feat=video_feat, causal=causal, token_type=token_type)
#         return multimodal_text_features

#     def forward_prediction_head(self, multimodal_text_features):
        
#         logits_over_vocab = self.shared_text_multimodal_encoder_decoder.forward_bert_prediction_head(multimodal_text_features)
#         return logits_over_vocab
    
#     def forward_contra_proj_text(self, text_features_pooled):

#         text_features_pooled_contra_proj = self.contra_head_text.forward_contra_proj(text_features_pooled)
#         text_features_pooled_contra_proj = F.normalize(text_features_pooled_contra_proj, p=2, dim=-1) #! Check if correct dimension
#         return text_features_pooled_contra_proj
    
#     def forward_weighted_proj_text(self, text_features_pooled_contra_proj):

#         text_features_pooled_contra_proj_weighted = self.contra_head_text.forward_weighted_proj(text_features_pooled_contra_proj)
#         return text_features_pooled_contra_proj_weighted
    
#     def forward_contra_proj_video(self, feat_map_conv_pooled): #! Two possibilities either use feat_map_conv_pooled or mamba longer_temporal_features
        
#         feat_map_conv_pooled_contra_proj = self.contra_head_video.forward_contra_proj(feat_map_conv_pooled)
#         feat_map_conv_pooled_contra_proj = F.normalize(feat_map_conv_pooled_contra_proj, p=2, dim=-1) #! Check if correct dimension
#         return feat_map_conv_pooled_contra_proj

#     def forward_weighted_proj_video(self, feat_map_conv_pooled_contra_proj):

#         feat_map_conv_pooled_contra_proj_weighted = self.contra_head_text.forward_weighted_proj(feat_map_conv_pooled_contra_proj)
#         return feat_map_conv_pooled_contra_proj_weighted  

#     def calculate_loss(self, batch, phase):

#         loss = 0.0

#         if phase == "valid":
#             for loss_name, loss_func in self.losses.items():
#                 loss_cfg = self.losses_cfgs[loss_name]
#                 if loss_name == "oad_loss":
#                     output_name = loss_cfg.output
#                     loss_weight = loss_cfg.weight
#                     loss = loss + loss_weight * loss_func(**batch[output_name]) 

#         else:
#             for loss_name, loss_func in self.losses.items():
#                 loss_cfg = self.losses_cfgs[loss_name]            
#                 output_name = loss_cfg.output
#                 loss_weight = loss_cfg.weight
                
#                 loss = loss + loss_weight * loss_func(**batch[output_name]) 

#         return loss
 
#     def compute_metrics_in_train_step(self, batch):

#         train_metrics_step = {}
#         for metric_name, metric in self.train_metrics.items():
#             if metric_name not in ["train_mAP", "train_ConfusionMatrix"]:
#                 metric_cfg = self.train_metrics_cfgs[metric_name]    

#                 metric_value = metric(*metric.prepare_preds_and_targets(**batch["oad"]))

#                 wandb_name = "/".join(metric_name.split("_", 1))
#                 train_metrics_step[wandb_name] = metric_value

#         return train_metrics_step

#     def compute_metrics_in_valid_step(self, batch):

#         valid_metrics_step = {}
#         for metric_name, metric in self.valid_metrics.items():
#             if metric_name not in ["valid_mAP", "valid_ConfusionMatrix"]:
#                 metric_cfg = self.valid_metrics_cfgs[metric_name]  

#                 metric_value = metric(*metric.prepare_preds_and_targets(**batch["oad"]))

#                 wandb_name = "/".join(metric_name.split("_", 1))
#                 valid_metrics_step[wandb_name] = metric_value

#         return valid_metrics_step

#     def compute_metrics_on_train_epoch_end(self):

#         train_metrics_epoch = {}

#         train_metric = self.train_metrics["train_mAP"].compute()        
        
#         if isinstance(train_metric, dict): 
#             train_metrics_epoch["train/mAP"] = train_metric["mAP"]
#             for name, value in train_metric["per_class_AP"].items():                
#                 wandb_name = "/".join(("train"+name).split("_", 1))
#                 train_metrics_epoch[wandb_name] = value  
#         else:
#             train_metrics_epoch["train/mAP"] = train_metric

#         if "train_ConfusionMatrix" in self.train_metrics.keys():
#             self.train_metrics["train_ConfusionMatrix"].compute(phase="train")

#         return train_metrics_epoch

#     def compute_metrics_on_valid_epoch_end(self):

#         valid_metrics_epoch = {}

#         valid_metric = self.valid_metrics["valid_mAP"].compute()
     
#         if isinstance(valid_metric, dict): 
#             valid_metrics_epoch["valid/mAP"] = valid_metric["mAP"]
#             for name, value in valid_metric["per_class_AP"].items():                
#                 wandb_name = "/".join(("valid"+name).split("_", 1))
#                 valid_metrics_epoch[wandb_name] = value  
#         else:
#             valid_metrics_epoch["valid/mAP"] = valid_metric   

#         if "valid_ConfusionMatrix" in self.valid_metrics.keys():
#             self.valid_metrics["valid_ConfusionMatrix"].compute(phase="valid")
        
#         return valid_metrics_epoch
    

#     def update_train_metrics_states(self, batch):
        
#         for metric_name, metric in self.train_metrics.items():
#             if metric_name in ["train_mAP", "train_ConfusionMatrix"]:
#                 metric_cfg = self.train_metrics_cfgs[metric_name]   

#                 metric.update(**batch["oad"])   


#     def update_valid_metrics_states(self, batch):

#         for metric_name, metric in self.valid_metrics.items():
#             if metric_name in ["valid_mAP", "valid_ConfusionMatrix"]:
#                 metric_cfg = self.valid_metrics_cfgs[metric_name]    

#                 metric.update(**batch["oad"])

#     def reset_train_metrics_states(self):
        
#         if "train_mAP" in self.train_metrics.keys():
#             self.train_metrics["train_mAP"].reset()
#         if "train_ConfusionMatrix" in self.train_metrics.keys():
#             self.train_metrics["train_ConfusionMatrix"].reset()

#     def reset_valid_metrics_states(self):
        
#         if "valid_mAP" in self.valid_metrics.keys():
#             self.valid_metrics["valid_mAP"].reset()
#         if "valid_ConfusionMatrix" in self.valid_metrics.keys():
#             self.valid_metrics["valid_ConfusionMatrix"].reset()
#