import torch
from torch import nn
import torch.nn.functional as F

from .layers import AdditiveAttention, MultiHeadAttention


class TextEncoder(nn.Module):
    def __init__(self, args, embedding_matrix, enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.dropout_rate = args.dropout_rate
        pretrained_news_word_embedding = torch.from_numpy(embedding_matrix).float()
        self.word_embedding = nn.Embedding.from_pretrained(
            pretrained_news_word_embedding, freeze=False
        )

        word_embedding_dim = pretrained_news_word_embedding.shape[-1]

        self.multihead_attention = MultiHeadAttention(
            word_embedding_dim,
            args.num_attention_heads,
            args.news_dim // args.num_attention_heads,
            args.news_dim // args.num_attention_heads,
        )
        self.additive_attention = AdditiveAttention(
            args.news_dim, args.news_query_vector_dim
        )

    def forward(self, text):
        text_vector = F.dropout(
            self.word_embedding(text.long()),
            p=self.dropout_rate,
            training=self.training,
        )
        multihead_text_vector = self.multihead_attention(
            text_vector, text_vector, text_vector
        )
        multihead_text_vector = F.dropout(
            multihead_text_vector, p=self.dropout_rate, training=self.training
        )
        # batch_size, word_embedding_dim
        text_vector = self.additive_attention(multihead_text_vector)
        return text_vector


class UserEncoder(nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.multihead_attention = MultiHeadAttention(
            args.news_embedding_dim, args.num_attention_heads, 20, 20
        )
        self.additive_attention = AdditiveAttention(
            args.num_attention_heads * 20, args.query_vector_dim
        )

        self.neg_multihead_attention = MultiHeadAttention(
            args.news_embedding_dim, args.num_attention_heads, 20, 20
        )

    def forward(self, clicked_news_vecs):
        multi_clicked_vectors = self.multihead_attention(
            clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
        )
        pos_user_vector = self.additive_attention(multi_clicked_vectors)

        user_vector = pos_user_vector
        return user_vector


class NRMS(nn.Module):
    def __init__(self, args, embedding_matrix):
        self.text_encoder = TextEncoder(args, embedding_matrix)
        self.user_encoder = UserEncoder(args)

        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self, news_ids, input_ids, log_ids, log_mask, targets=None, compute_loss=True
    ):
        batch_size, npratio, word_num = candidate_news.shape
        candidate_news = candidate_news.view(-1, word_num)
        candidate_vector = self.text_encoder(candidate_news).view(
            batch_size, npratio, -1
        )

        batch_news_vec = self.news_encoder(news_ids)
        log_vec = torch.index_select(batch_news_vec, 0, log_ids).view(
            batch_size, -1, self.args.news_dim
        )
        batch_size, clicked_news_num, word_num = clicked_news.shape
        clicked_news = clicked_news.view(-1, word_num)
        clicked_news_vecs = self.text_encoder(clicked_news).view(
            batch_size, clicked_news_num, -1
        )

        user_vector = self.user_encoder(clicked_news_vecs)

        score = torch.bmm(candidate_vector, user_vector.unsqueeze(-1)).squeeze(dim=-1)

        if compute_loss:
            loss = self.criterion(score, targets)
            return loss, score
        else:
            return score
