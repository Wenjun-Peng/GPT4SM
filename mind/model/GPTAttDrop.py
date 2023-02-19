import torch
from torch import nn


from .layers import AdditiveAttention

class NewsEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.title_linear = nn.Sequential(
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.gpt_news_dim, args.gpt_news_dim),
            nn.ReLU(),
            nn.Linear(args.gpt_news_dim, args.news_dim),
        )
        self.body_linear = nn.Sequential(
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.gpt_news_dim, args.gpt_news_dim),
            nn.ReLU(),
            nn.Linear(args.gpt_news_dim, args.news_dim),
        )
        self.additive = AdditiveAttention(args.news_dim, args.news_dim//2)

    def forward(self, news_gpt_embedding, news_gpt_embedding_body):
        # print(news_gpt_embedding.shape)
        news_gpt_embedding = self.title_linear(news_gpt_embedding)
        news_gpt_embedding_body = self.body_linear(news_gpt_embedding_body)
        flag = False
        if len(news_gpt_embedding.shape) >= 3:
            # 这里train的时候一次性输入是5个，也就是一个序列
            bz, num, dim = news_gpt_embedding.shape
            flag = True

            news_gpt_embedding = news_gpt_embedding.reshape(bz*num, dim)
            news_gpt_embedding_body = news_gpt_embedding_body.reshape(bz*num, dim)
        x = torch.stack([news_gpt_embedding, news_gpt_embedding_body], dim=1)
        # print(x.shape)
        x = self.additive(x)
        if flag:
            x = x.reshape(bz, num, dim)
        return x


class UserEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.news_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim
        )

    def _process_news(
        self,
        vec,
        mask,
        additive_attention,
    ):
        vec = additive_attention(vec, mask)
        return vec

    def forward(self, log_vec, log_mask):
        """
        Returns:
            (shape) batch_size,  news_dim
        """
        # batch_size, news_dim
        log_vec = self._process_news(log_vec, log_mask, self.news_additive_attention)
        # log_vec = torch.mean(log_vec, dim=1)
        return log_vec


class BodySimpleFusionReluDropFirst(torch.nn.Module):
    """
    UniUM network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.news_encoder = NewsEncoder(args)
        self.user_encoder = UserEncoder(args)

        self.criterion = nn.CrossEntropyLoss()

    def get_user_emb(self, log_vecs, log_mask):
        return self.user_encoder(log_vecs, log_mask)

    def forward(self, batch_input, compute_loss=True):
        """
        Returns:
          click_probability: batch_size, 1 + K
        """
        # input_ids: batch, history, num_words
        batch_size = batch_input["batch_cand_title_emb"].shape[0]

        news_vec = self.news_encoder(batch_input["batch_cand_title_emb"], batch_input["batch_cand_body_emb"])
        log_vecs = self.news_encoder(batch_input["batch_click_title_emb"], batch_input["batch_click_body_emb"])

        user_vector = self.user_encoder(log_vecs, batch_input["batch_log_mask"])
        score = torch.bmm(news_vec, user_vector.unsqueeze(-1)).squeeze(dim=-1)

        if compute_loss:
            loss = self.criterion(score, batch_input["batch_labels"])
            return loss, score
        else:
            return score
