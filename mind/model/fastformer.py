import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import math

from .PLM import PLMNR, NewsEncoder
from .layers import FastSelfAttention, AdditiveAttention

class FastUserEncoder(torch.nn.Module):
    def __init__(self, args
    ):
        super(FastUserEncoder, self).__init__()
        self.args = args
        self.user_encoder = FastSelfAttention(20, args.news_dim)
        self.news_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim
        )


    def forward(self, log_vec, log_mask):
        """
        Inputs:
            log_vec (shape): batch_size, his_len, news_dim

        Returns:
            (shape) batch_size,  news_dim
        """
        # batch_size, news_dim
        log_vec = self.user_encoder(log_vec)
        log_vec = self.news_additive_attention(log_vec)
        return log_vec


class FastFormer(PLMNR):
    def __init__(self, args):
        super(FastFormer, self).__init__(args)
        self.args = args

        self.news_encoder = NewsEncoder(args)
        self.user_encoder = FastUserEncoder(args)
        self.criterion = nn.CrossEntropyLoss()