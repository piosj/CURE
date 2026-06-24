import torch
import torch.nn as nn
from utils.layer import *
from torch_geometric.nn import Sequential, GCNConv
import pandas as pd
from torch import Tensor  # Necessary

class NewsEncoder(nn.Module):
    def __init__(self, cfg, pretrained_word_embedding):
        super().__init__()
        token_emb_dim = cfg.word_embedding_dim
        self.news_dim = cfg.head_num * cfg.head_dim

        self.dataset_lang = cfg.dataset_lang
        if self.dataset_lang == 'english':
            pretrain = torch.from_numpy(glove_emb).float()
            self.word_encoder = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
        else:   # Adressa
            pretrained_emb = torch.tensor(pretrained_word_embedding, dtype=torch.float)
            self.word_encoder = nn.Embedding.from_pretrained(pretrained_emb, freeze=False, padding_idx=0)

        self.category_embedding = nn.Embedding(cfg.num_categories_for_NewsEncoder + cfg.num_subcategories_for_NewsEncoder - 1,
                                               cfg.word_embedding_dim,
                                               padding_idx=0)
        
        attention_input_dim = cfg.num_filters * cfg.window_size + cfg.category_emb_dim + cfg.subcategory_emb_dim   # 300 + 100 + 100
        

        self.attention = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (MultiHeadAttention(
                attention_input_dim,
                attention_input_dim,
                attention_input_dim,
                cfg.head_num,
                25
            ), 'x, x, x, mask -> x'),

            (nn.LayerNorm(attention_input_dim), 'x -> x'),
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (AttentionPooling(
                attention_input_dim,
                cfg.attention_hidden_dim
            ), 'x, mask -> x'),

            (nn.LayerNorm(attention_input_dim), 'x -> Tensor'),
        ])        
        self.last_encoder = nn.Linear(500, 300)   # (attention_input_dim, self.news_dim); 1000 -> 500
        
        self.attetio = Sequential('x, mask', [   
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(
                token_emb_dim,
                token_emb_dim,
                token_emb_dim,
                cfg.head_num,
                cfg.head_dim
            ), 'x, x, x, mask -> x'),
            (nn.LayerNorm(self.news_dim), 'x -> x'),
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (AttentionPooling(
                self.news_dim,
                cfg.attention_hidden_dim
            ), 'x, mask -> x'),
            (nn.LayerNorm(self.news_dim), 'x -> Tensor'),  
        ])

    # --------------------------------------------------------------------- #
    # forward: attetio ➜ concat ➜ attention ➜ last_encoder                 #
    # --------------------------------------------------------------------- #
    def forward(self, title_idx, category_idx, subcategory_idx):
        device = self.word_encoder.weight.device
        
        if title_idx.dim() == 1:
            title_idx = title_idx.unsqueeze(0)              # (1, L)
        if isinstance(category_idx, int):
            category_idx = torch.tensor([category_idx], device=device)
        if isinstance(subcategory_idx, int):
            subcategory_idx = torch.tensor([subcategory_idx], device=device)
                    
        # [B, seq_len, embed_dim]
        title_embeddings = self.word_encoder(title_idx)
        mask = (title_idx != 0)   # [B, seq_len]

        # Multi-Head Self-Attention + Attention Pooling (shape: [B, news_dim])
        title_vector = self.attetio(title_embeddings, mask)  

        category_vector = self.category_embedding(category_idx)      # shape: [B, cat_dim]
        subcategory_vector = self.category_embedding(subcategory_idx)  # shape: [B, subcat_dim]

        fuse_word_emb = torch.cat([title_vector, category_vector, subcategory_vector], dim=1)  # [B, title_dim+cat_dim+subcat_dim]
        fuse_word_emb  = fuse_word_emb.unsqueeze(1)                                # (B, 1, 500)
        fuse_mask = torch.ones(fuse_word_emb.size(0), 1, device=device, dtype=torch.bool)

        attention_output = self.attention(fuse_word_emb, fuse_mask)   

        news_vector = self.last_encoder(attention_output)   # (B, 1, 300)
        news_vector = news_vector.squeeze(1)   # [B, news_dim]
        
        return news_vector