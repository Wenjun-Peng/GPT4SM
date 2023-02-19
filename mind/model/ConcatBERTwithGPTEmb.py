import os

import torch
from torch import nn

from model.layers import AdditiveAttention
from utils import MODEL_CLASSES
import logging

class TextEncoder(torch.nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        self.config = config_class.from_pretrained(
            args.config_name,
            output_hidden_states=True,
            num_hidden_layers=args.bert_output_layer,
        )
        self.bert_model = model_class.from_pretrained(
            args.model_name_or_path,
            config=self.config,
        )
        self.dropout_rate = args.dropout_rate
        # self.additive_attention = AdditiveAttention(
        #     self.config.hidden_size*2, args.news_query_vector_dim
        # )
        self.transform = nn.Sequential(
            nn.Linear(args.gpt_news_dim, 3072),
            nn.ReLU(),
            nn.Linear(3072, self.config.hidden_size),
        )
        # self.norm = nn.LayerNorm(self.config.hidden_size*2)

    def forward(self, input_ids, attn_masks, title_emb):
        """
        Args:
            text: Tensor(batch_size) * num_words_text * embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        title_emb = self.transform(title_emb)
        # batch_size, num_words_text
        # word_emb = self.bert_model(input_ids, attn_masks).last_hidden_state
        # repeated_title_emb = title_emb.unsqueeze(1).repeat([1, word_emb.size(1), 1])
        # word_emb = torch.cat([word_emb, repeated_title_emb], dim=-1)
        # word_emb = self.norm(word_emb)
        # text_vector = self.additive_attention(word_emb, attn_masks)
        
        pooler_output = self.bert_model(input_ids, attn_masks).pooler_output
        text_vector = torch.cat([pooler_output, title_emb], dim=-1)
        return text_vector


class NewsEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.title_encoder = TextEncoder(args)

        self.concat_dim = self.title_encoder.config.hidden_size * 2
        self.emb_transform = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim // 2),
            nn.Tanh(),
            nn.Linear(self.concat_dim // 2, args.news_dim),
        )

    def forward(
        self, title_gpt_embedding, body_gpt_embedding, title_input_ids, title_attn_masks
    ):
        title_emb = self.title_encoder(title_input_ids, title_attn_masks, title_gpt_embedding)
        news_emb = self.emb_transform(title_emb)
        return news_emb


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


class ConcatBERTwithGPTEmb(torch.nn.Module):
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

        self.fix_parameters()

    def get_user_emb(self, log_vecs, log_mask):
        return self.user_encoder(log_vecs, log_mask)

    def forward(self, batch_input, compute_loss=True):
        """
        Returns:
          click_probability: batch_size, 1 + K
        """
        batch_size = batch_input["batch_cand_title_emb"].size(0)
        cand_num = batch_input["batch_cand_title_emb"].size(1)
        hist_num = batch_input["batch_click_title_emb"].size(1)

        news_vec = self.news_encoder(
            batch_input["batch_cand_title_emb"].view(batch_size * cand_num, -1),
            batch_input["batch_cand_body_emb"].view(batch_size * cand_num, -1),
            batch_input["batch_cand_title_input_ids"].view(batch_size * cand_num, -1),
            batch_input["batch_cand_title_attn_mask"].view(batch_size * cand_num, -1),
        ).view(batch_size, cand_num, -1)
        log_vecs = self.news_encoder(
            batch_input["batch_click_title_emb"].view(batch_size * hist_num, -1),
            batch_input["batch_click_body_emb"].view(batch_size * hist_num, -1),
            batch_input["batch_click_title_input_ids"].view(batch_size * hist_num, -1),
            batch_input["batch_click_title_attn_mask"].view(batch_size * hist_num, -1),
        ).view(batch_size, hist_num, -1)

        user_vector = self.user_encoder(log_vecs, batch_input["batch_log_mask"])
        score = torch.bmm(news_vec, user_vector.unsqueeze(-1)).squeeze(dim=-1)

        if compute_loss:
            loss = self.criterion(score, batch_input["batch_labels"])
            return loss, score
        else:
            return score

    def fix_parameters(self):
        for param in self.news_encoder.title_encoder.bert_model.parameters():
            param.requires_grad = False
        
        for index, layer in enumerate(self.news_encoder.title_encoder.bert_model.encoder.layer):
            if index in self.args.bert_trainable_layer:
                logging.info(f"finetune {index} block")
                for param in layer.parameters():
                    param.requires_grad = True
