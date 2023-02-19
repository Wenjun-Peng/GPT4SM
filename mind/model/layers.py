import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import math

class AdditiveAttention(nn.Module):
    """ AttentionPooling used to weighted aggregate news vectors
    Arg: 
        d_h: the last dimension of input
    """

    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        #       [bz, 20, seq_len, 20] x [bz, 20, 20, seq_len] -> [bz, 20, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        #       [bz, 20, seq_len, seq_len] x [bz, 20, seq_len, 20] -> [bz, 20, seq_len, 20]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, enable_gpu):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20
        self.enable_gpu = enable_gpu

        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None):
        #       Q, K, V: [bz, seq_len, 300] -> W -> [bz, seq_len, 400]-> q_s: [bz, 20, seq_len, 20]
        batch_size, seq_len, _ = Q.shape

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(
                batch_size, seq_len, seq_len
            )  #  [bz, seq_len, seq_len]
            mask = mask.unsqueeze(1).repeat(
                1, self.n_heads, 1, 1
            )  # attn_mask : [bz, 20, seq_len, seq_len]

        context, attn = ScaledDotProductAttention(self.d_k)(
            q_s, k_s, v_s, mask
        )  # [bz, 20, seq_len, 20]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )  # [bz, seq_len, 400]
        #         output = self.fc(context)
        return context  # self.layer_norm(output + residual)


class WeightedLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(WeightedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_softmax = nn.Softmax(dim=-1)(self.weight)
        return F.linear(input, weight_softmax)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features
        )

class FastSelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, **kwargs):
        super(FastSelfAttention, self).__init__()
    
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        # batch_size, seq_len, num_head * head_dim
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / math.sqrt(self.attention_head_size)

        # add attention mask
        if attention_mask is not None:
            query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        # query_weight = query_weight.unsqueeze(3).repeat(1, 1, 1, self.attention_head_size) 

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(2, 3)

        # batch_size, num_head, seq_len, head_dim
        # pooled_query_for_score = pooled_query.unsqueeze(2).repeat(1, 1, seq_len,1) 

        # batch_size, num_head, seq_len, head_dim
        key_layer = self.transpose_for_scores(mixed_key_layer)

        # batch_size, num_head, seq_len
        query_key_score = torch.matmul(key_layer, pooled_query).squeeze(-1) / math.sqrt(self.attention_head_size)
        # query_key_score = torch.sum(key_layer * pooled_query_for_score, dim=-1) / math.sqrt(self.attention_head_size)

        # add attention mask
        if attention_mask is not None:
            query_key_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        # query_key_weight = query_key_weight.unsqueeze(3).repeat(1, 1, 1, self.attention_head_size) 

        # batch_size, num_head, 1, head_dim
        # pooled_key = torch.sum(key_layer * query_key_weight, dim=2)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        # batch_size, num_head, seq_len, head_dim
        # pooled_key = pooled_key.repeat(1, 1, seq_len,1) 
        
        # batch, seq_len, num_head, seq_len
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value