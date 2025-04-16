import hydra

import torch
import torch.nn as nn

class TransformerDecoderAdapter(nn.Module):

    def __init__(
        self,
        decoder,
        ):

        super().__init__()
        
        self.decoder_cfg = decoder
        self.query_cfg = self.decoder_cfg.pop("query")

        self.decoder_layer_cfg = self.decoder_cfg.pop("decoder_layer")
        self.decoder_layer_cfg["dim_feedforward"] = int(self.decoder_layer_cfg.d_model * self.decoder_layer_cfg.pop("dim_feedforward_scale"))
        self.tgt_is_causal  =  self.decoder_layer_cfg.pop("tgt_is_causal",False)
        self.memory_is_causal  = self.decoder_layer_cfg.pop("memory_is_causal",False)
        use_tgt_mask  = self.decoder_layer_cfg.pop("use_tgt_mask",False)
        if use_tgt_mask:
            tgt_mask = torch.tril(torch.ones(self.query_cfg.num_queries, self.query_cfg.num_queries))
            self.register_buffer("tgt_mask", tgt_mask)
        else:
            self.tgt_mask = None

        # TODO this is hard coded maybe improve to not hard coded, but then needs to be created at each forward pass
        use_memory_mask = self.decoder_layer_cfg.pop("use_memory_mask",False)
        if use_memory_mask:
            memory_mask = torch.zeros(self.query_cfg.num_queries, 1760)
            visible_token = 1760//32*4
            for i in range(32):
                if i % 4 == 0:
                    visible_token += 1760//32*4
                memory_mask[i,:visible_token] = 1.0

            self.register_buffer("memory_mask", memory_mask)
        else:
            self.memory_mask = None


        transformer_decoder_layer = hydra.utils.instantiate(self.decoder_layer_cfg, _recursive_=False)
        self.transformer_decoder = hydra.utils.instantiate(self.decoder_cfg, decoder_layer=transformer_decoder_layer,  _recursive_=False)
        self.query_tokens = nn.Parameter(torch.zeros(1, self.query_cfg.num_queries, self.decoder_layer_cfg.d_model))

    def forward(self, x):
        
        q = self.query_tokens.repeat(len(x), 1, 1)
        out = self.transformer_decoder(tgt=q, 
                                       memory=x, 
                                       tgt_mask=self.tgt_mask, 
                                        memory_mask=self.memory_mask,
                                        tgt_is_causal=self.tgt_is_causal, 
                                        memory_is_causal=self.memory_is_causal)
        return out