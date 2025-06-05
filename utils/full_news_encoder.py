import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general.attention.additive import AdditiveAttention
import random
import numpy as np


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_num}" if torch.cuda.is_available() else "cpu")
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        self.category_embedding = nn.Embedding(config.num_categories_for_NewsEncoder + config.num_subcategories_for_NewsEncoder - 1,   
                                               config.num_filters,
                                               padding_idx=0)
        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            1,
            config.num_filters,
            (config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.num_filters)

    def forward(self, title_idx, category_idx, subcategory_idx):
        if self.config.use_batch:
            category_vector = self.category_embedding(torch.tensor(category_idx, device=self.device).long())
            
            num_scats = self.category_embedding.num_embeddings
            assert all(0 <= sc < num_scats for sc in subcategory_idx), \
                f"max_sc={max(subcategory_idx)}, num_embeddings={num_scats}"

            subcategory_vector = self.category_embedding(torch.tensor(subcategory_idx, device=self.device).long())

            title_vector = F.dropout(self.word_embedding(title_idx),
                                    p=self.config.dropout_probability,
                                    training=self.training)
        else:
            category_vector = self.category_embedding(torch.tensor(category_idx, device=self.device).long().unsqueeze(0))

            subcategory_vector = self.category_embedding(torch.tensor(subcategory_idx, device=self.device).long().unsqueeze(0))

            title_vector = F.dropout(self.word_embedding(title_idx.unsqueeze(0)),
                                    p=self.config.dropout_probability,
                                    training=self.training)
        
        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)

        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=self.config.dropout_probability,
                                           training=self.training)

        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))

        news_vector = torch.cat(
            [category_vector, subcategory_vector, weighted_title_vector],
            dim=1)
        return news_vector
