import torch
from torch import nn


from .layers import AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, title_gpt_embedding, body_gpt_embedding=None):
        return title_gpt_embedding


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


class GPTNR(torch.nn.Module):
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

    def forward(self, batch_input, compute_loss=True):
        """
        Returns:
          click_probability: batch_size, 1 + K
        """
        # input_ids: batch, history, num_words
        batch_size = batch_input["batch_cand_title_emb"].shape[0]

        news_vec = self.news_encoder(batch_input["batch_cand_title_emb"])
        log_vecs = self.news_encoder(batch_input["batch_click_title_emb"])

        user_vector = self.user_encoder(log_vecs, batch_input["batch_log_mask"])
        score = torch.bmm(news_vec, user_vector.unsqueeze(-1)).squeeze(dim=-1)

        if compute_loss:
            loss = self.criterion(score, batch_input["batch_labels"])
            return loss, score
        else:
            return score
