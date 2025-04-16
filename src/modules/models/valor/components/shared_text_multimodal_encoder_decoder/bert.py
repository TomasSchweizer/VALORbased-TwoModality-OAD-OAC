import math
import copy

import torch
import torch.nn as nn




def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        
        super().__init__()
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.prompt_embedding = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
      
    def forward(self, input_ids, token_type = None, full_masker=False):

        if token_type  == 'prompt' or token_type is None:
            seq_length = input_ids.size(1)
            
            if token_type is None:
                if not full_masker:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                else:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                    position_ids[seq_length//2:] = position_ids[:seq_length//2] + 1

                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                token_type_ids = torch.zeros_like(input_ids)
                token_type_embeddings = self.token_type_embeddings(token_type_ids)
            
            elif token_type == 'prompt':
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                token_type_ids = torch.zeros_like(input_ids)
                token_type_embeddings = self.prompt_embedding(token_type_ids)       
            words_embeddings = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            

            embeddings = words_embeddings + position_embeddings + token_type_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)

            return embeddings
        
       
        
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, use_cache=False, cache=None, cache_first=False, layer_num=0, cache_type='unimlm'):
        mixed_query_layer = self.query(hidden_states)   ### b,n,c
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) # b,h,n,c
        key_layer = self.transpose_for_scores(mixed_key_layer)    # b,h,n,c
        value_layer = self.transpose_for_scores(mixed_value_layer) # b,h,n,c

        if use_cache:

            if not cache_first:
                key_layer = torch.cat((key_layer,cache['key'][str(layer_num)]),dim=2)
                value_layer = torch.cat((value_layer,cache['value'][str(layer_num)]),dim=2)
            if cache_type == 'unimlm':
                idx = 2
            elif cache_type == 'lm':
                idx = 1
            cache['key'][str(layer_num)] = torch.cat((key_layer[:,:,0:1],key_layer[:,:,idx:]),dim=2)  ### b,h,n,c
            cache['value'][str(layer_num)] = torch.cat((value_layer[:,:,0:1],value_layer[:,:,idx:]),dim=2)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) ## b,h,n,n
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)   ##b,h,n,c
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  ###b,n,h,c
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, cache
      

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, cross_hidden_states, cross_attention_masks):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(cross_hidden_states)
        mixed_value_layer = self.value(cross_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if cross_attention_masks is not None:
            attention_scores = attention_scores + cross_attention_masks

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer



class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertCrossOutput(nn.Module):
    def __init__(self, config):
        super(BertCrossOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.gating = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ### do not need any gates, seriously bad effect s
        #hidden_states = self.LayerNorm(self.gating * hidden_states + input_tensor)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config, attn_type):
        super(BertAttention, self).__init__()
        self.attn_type = attn_type 
        if self.attn_type == 'self':
            self.self = BertSelfAttention(config)
            self.output = BertSelfOutput(config)
        elif self.attn_type == 'cross':
            self.cross = BertCrossAttention(config)
            self.output = BertCrossOutput(config)       

    def forward(self, input_tensor, attention_mask, cross_hidden_states, use_cache=False, cache=None, cache_first=False, layer_num=0,cache_type='unimlm'):
        if self.attn_type == 'self':
            self_output, cache = self.self(input_tensor, attention_mask, use_cache=use_cache, cache=cache, cache_first=cache_first,layer_num=layer_num,cache_type=cache_type)
            attention_output = self.output(self_output, input_tensor)
        elif self.attn_type == 'cross':
            cross_output = self.cross(input_tensor, cross_hidden_states, attention_mask)
            attention_output = self.output(cross_output, input_tensor)
        return attention_output, cache


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config, attn_type='self')  
        self.cross_attn = BertAttention(config, attn_type='cross')
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        

    def forward(self, hidden_states, attention_mask, video_feat, use_cache=False, cache=None,
                                        cache_first=False, layer_num=0, cache_type='unimlm', cross_attention_masks=None):

        attention_output, cache = self.attention(hidden_states, attention_mask, cross_hidden_states=None, use_cache=use_cache, cache=cache,
                                        cache_first=cache_first, layer_num=layer_num,cache_type=cache_type)

        if video_feat is not None:
            # For multimodal decoder with cross attention
            attention_output, cache = self.cross_attn(input_tensor=attention_output, attention_mask=cross_attention_masks, cross_hidden_states=video_feat)
        
        elif video_feat is None:
            # For text encoder with just self-attention
            pass

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output,cache


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, video_feat, use_cache=False,cache=None,
                                        cache_first=False,cache_type='unimlm', cross_attention_masks=None):
        for idx, layer_module in enumerate(self.layer):

            hidden_states,cache = layer_module(hidden_states, attention_mask, video_feat, use_cache,cache,
                                    cache_first, idx, cache_type, cross_attention_masks)
            
        return hidden_states, cache


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class BertModel(nn.Module):

    def __init__(self, config):
        
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self._init_bert_weights)

    def _init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()  
        
    def forward(self, tokens, task_prompt = None, video_feat = None,  \
                             causal=False, cache = None, use_cache = False, \
                                    cache_first=False, token_type=None, cache_type='unimlm', \
                                      full_masker=False, cross_attention_masks=None):
        
        assert use_cache == False

        embedding_output = self.embeddings(tokens, token_type, full_masker)
        input_feat = embedding_output
        token_len = input_feat.shape[1]
        attention_mask_token = (tokens != 0).long()
        attention_mask = attention_mask_token


        if task_prompt is not None:            
            prompt_embedding_output = self.embeddings(task_prompt, 'prompt')
            input_feat = torch.cat((input_feat, prompt_embedding_output),dim=1)
            attention_mask_prompt = (task_prompt != 0).long()
            attention_mask = torch.cat((attention_mask, attention_mask_prompt),dim=1)

        total_len = attention_mask.shape[1]
        attention_mask = attention_mask.unsqueeze(1).expand(-1, total_len, -1).clone()
        if causal:
            if full_masker:
                attention_mask[:, : token_len//2, : token_len//2] = torch.tril(attention_mask[:, : token_len//2, : token_len//2])
                attention_mask[:, : token_len//2, token_len//2:token_len] = 0
                attention_mask[:, token_len//2:token_len, :token_len//2] = torch.tril(attention_mask[:, token_len//2:token_len, :token_len//2])
                attention_mask[:, token_len//2:token_len, token_len//2:token_len] = torch.eye(token_len//2)
                attention_mask[:, token_len:, : token_len] = 0                
            else:
                attention_mask[:, : token_len, : token_len] = torch.tril(attention_mask[:, : token_len, : token_len])
                attention_mask[:, token_len:, : token_len] = 0
        attention_mask = attention_mask.unsqueeze(1)
        #attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = attention_mask.to(dtype=torch.float32) # fp16 compatibility  ## my change
        attention_mask = (1.0 - attention_mask) * - 10000.0        

        if cross_attention_masks is not None:
            cross_attention_masks_finished = torch.zeros((cross_attention_masks.shape[0],1,cross_attention_masks.shape[1],token_len)).to(device=cross_attention_masks.device)
            for i,cross_att_mask in enumerate(cross_attention_masks):
                cross_attention_masks_finished[i,:,:,:] = cross_att_mask.repeat((token_len,1)).permute(1,0)
            cross_attention_masks_finished = cross_attention_masks_finished.to(dtype=torch.float32)
            cross_attention_masks_finished = (1.0 - cross_attention_masks_finished) * - 10000.0        
            cross_attention_masks = cross_attention_masks_finished.permute(0,1,3,2)
        
        
        sequence_output, cache = self.encoder(input_feat,
                                    attention_mask,
                                    video_feat=video_feat,
                                    use_cache=use_cache,
                                    cache=cache,
                                    cache_first=cache_first,
                                    cross_attention_masks=cross_attention_masks
                                    )

        return sequence_output
    

class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output

class BertPredictionHead(nn.Module):
    
    def __init__(self, embedding_weights):
        super().__init__()
        self.hidden_size = embedding_weights.size(1)
        self.vocab_size = embedding_weights.size(0)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = GELU()
        self.layernorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
        self.decoder.weight = embedding_weights       


    def forward(self, sequence_output):

    
        sequence_output = self.dense(sequence_output)

        sequence_output = self.activation(sequence_output)
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.decoder(sequence_output) 
        return prediction_scores