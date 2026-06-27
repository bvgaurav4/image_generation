import torch
import torch.nn as nn
from transformers import BertConfig
from tqdm.auto import tqdm

class CustomFeedForwardLayer(nn.Module):
    def __init__(self,config, rank = 64):
        super().__init__()

        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear1(x)      
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)                              # (B, S, 256)
        return x
    
class BertLayer(nn.Module):
    def __init__(self,config):
        
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True)
        
        self.cff = CustomFeedForwardLayer(config)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x,attention_mask=None):

        attention_output,_ = self.attention(x,x,x)
        x = self.norm1(x + attention_output)

        cffn_output = self.cff(x)
        x = self.norm2(x + cffn_output)
 
        return x
        
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids):

        seq_length = input_ids.size(1)

        position_ids = torch.arange(
            seq_length, device=input_ids.device
        ).unsqueeze(0)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)

        return embeddings
    

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList([
            BertLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class MyBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None,embed = False):

        x = self.embeddings(input_ids)
        

        x = self.encoder(x, attention_mask)

        if embed:
            return x[:, 0, :]  
        
        logits = self.lm_head(x)

        return logits
