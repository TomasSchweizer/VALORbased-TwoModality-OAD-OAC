from pathlib import Path

import torch
import torch.nn as nn

from src.modules.models.valor.components.shared_text_multimodal_encoder_decoder.bert_tokenizer import BertTokenizer
from src.modules.models.valor.components.shared_text_multimodal_encoder_decoder.masking import TokenMasker
from src.modules.models.valor.components.shared_text_multimodal_encoder_decoder.bert import BertModel
from src.modules.models.valor.components.shared_text_multimodal_encoder_decoder.bert import BertPredictionHead

class SharedTextMultimodalBertEncoderDecoder(nn.Module):
    
    def __init__(
        self,
        bert,
        bert_tokenizer,
        token_masker,
    ):
        super().__init__()
        
        bert_tokenizer_cfg = bert_tokenizer
        vocab_path = Path(bert_tokenizer_cfg.vocab_path)
        self.bert_tokenizer = BertTokenizer(vocab_path)
        self.bos_token = self.bert_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.eos_token = self.bert_tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.mask_token = self.bert_tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        token_masker_cfg = token_masker   
        token_masker_cfg.mask_token = self.mask_token    
        self.token_masker = TokenMasker(**token_masker_cfg)
        # mask_token=self.mask_token, mask_prob=self.mask_prob, range_start = 106, range_end = 30522

        bert_cfg = bert

        self.bert_multimodal_decoder_embed_dims = bert_cfg.hidden_size
        
        checkpoint_path = Path(bert_cfg.pop("checkpoint_path"))
        bert_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        bert_checkpoint = {k.replace('bert.','').replace('gamma','weight').replace('beta','bias') : v for k,v in bert_checkpoint.items()}

        self.bert_multimodal_decoder = BertModel(bert_cfg)
        missing_keys, unexpected_keys = self.bert_multimodal_decoder.load_state_dict(bert_checkpoint, strict=False)
        
        if missing_keys != [] or unexpected_keys != []:
            print(f'Unexpected_keys in multimodal encoder: {unexpected_keys}')
            print(f'Missing_keys in multimodal encoder: {missing_keys}')

        self.bert_prediction_head = BertPredictionHead(self.bert_multimodal_decoder.embeddings.word_embeddings.weight)
        bert_prediction_head_weights = {}
        bert_prediction_head_weights['dense.weight']  = bert_checkpoint['cls.predictions.transform.dense.weight']
        bert_prediction_head_weights['dense.bias']  = bert_checkpoint['cls.predictions.transform.dense.bias']
        bert_prediction_head_weights['layernorm.weight'] = bert_checkpoint['cls.predictions.transform.LayerNorm.weight' ]
        bert_prediction_head_weights['layernorm.bias'] =bert_checkpoint['cls.predictions.transform.LayerNorm.bias']
        bert_prediction_head_weights['decoder.weight'] = bert_checkpoint['cls.predictions.decoder.weight']
        bert_prediction_head_weights['decoder.bias'] = bert_checkpoint['cls.predictions.bias']

        missing_keys, unexpected_keys = self.bert_prediction_head.load_state_dict(bert_prediction_head_weights)
        
        if missing_keys != [] or unexpected_keys != []:
            print(f'Missing_keys in bert_prediction_head : {missing_keys}')
            print(f'Unexpected_keys in bert_prediction_head : {unexpected_keys}')


        # Text encoder and mulitmodal decoder share all parameters, but only multimodal encoder uses cross attention.
        self.bert_text_encoder = self.bert_multimodal_decoder 
        self.bert_text_encoder_embed_dims = bert_cfg.hidden_size


    def forward(self):
        print("Forward is not implemented.")
        pass

    def forward_bert_text_encoder(self, text_tokens, task_prompt=None, video_feat=None, causal=False):
        
        text_features = self.bert_text_encoder(tokens=text_tokens, task_prompt=task_prompt, video_feat=video_feat, causal=causal)
        return text_features

    def forward_bert_multimodal_decoder(self, text_tokens, task_prompt=None, video_feat=None, causal=True, token_type=None, cross_attention_masks=None):
        
        multimodal_text_features = self.bert_multimodal_decoder(tokens=text_tokens, task_prompt=task_prompt, video_feat=video_feat, causal=causal, token_type=token_type, cross_attention_masks=cross_attention_masks)
        return multimodal_text_features
    
    def forward_bert_prediction_head(self, multimodal_text_features):
        
        logits_over_vocab = self.bert_prediction_head(multimodal_text_features)
        return logits_over_vocab