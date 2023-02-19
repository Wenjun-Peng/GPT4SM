import numpy as np

import torch
from torch.utils.data import Dataset


class BehaviorTrainCollate:
    def __init__(self, title_emb, body_emb, title_input_ids, title_attn_mask):
        self.title_emb = title_emb
        self.body_emb = body_emb
        self.title_input_ids = title_input_ids
        self.title_attn_mask = title_attn_mask

    def __call__(self, data):
        batch_cand_news, batch_click_docs, batch_log_mask, batch_labels = zip(*data)

        batch_cand_news = np.array(batch_cand_news, dtype="int32")
        batch_click_docs = np.array(batch_click_docs, dtype="int32")

        batch_cand_title_emb = torch.FloatTensor(self.title_emb[batch_cand_news])
        batch_click_title_emb = torch.FloatTensor(self.title_emb[batch_click_docs])
        batch_cand_body_emb = torch.FloatTensor(self.body_emb[batch_cand_news])
        batch_click_body_emb = torch.FloatTensor(self.body_emb[batch_click_docs])
        batch_cand_title_input_ids = torch.LongTensor(
            self.title_input_ids[batch_cand_news]
        )
        batch_cand_title_attn_mask = torch.LongTensor(
            self.title_attn_mask[batch_cand_news]
        )
        batch_click_title_input_ids = torch.LongTensor(
            self.title_input_ids[batch_click_docs]
        )
        batch_click_title_attn_mask = torch.LongTensor(
            self.title_attn_mask[batch_click_docs]
        )

        batch_log_mask = torch.FloatTensor(batch_log_mask)
        batch_labels = torch.LongTensor(batch_labels)

        batch_output = {
            "batch_cand_title_emb": batch_cand_title_emb,
            "batch_click_title_emb": batch_click_title_emb,
            "batch_cand_body_emb": batch_cand_body_emb,
            "batch_click_body_emb": batch_click_body_emb,
            "batch_cand_title_input_ids": batch_cand_title_input_ids,
            "batch_cand_title_attn_mask": batch_cand_title_attn_mask,
            "batch_click_title_input_ids": batch_click_title_input_ids,
            "batch_click_title_attn_mask": batch_click_title_attn_mask,
            "batch_log_mask": batch_log_mask,
            "batch_labels": batch_labels,
        }

        return batch_output


class BehaviorTestCollate:
    def __init__(self, news_vec):
        self.news_vec = news_vec

    def __call__(self, data):
        batch_cand_news, batch_click_docs, batch_log_mask, batch_labels = zip(*data)

        batch_cand_news_ids = []
        batch_cand_index = []
        for cnt, cand_news in enumerate(batch_cand_news):
            batch_cand_index.extend([cnt] * len(cand_news))
            batch_cand_news_ids.extend(cand_news)

        batch_cand_news_ids = np.array(batch_cand_news_ids, dtype="int32")
        batch_cand_news_emb = torch.FloatTensor(self.news_vec[batch_cand_news_ids])

        batch_click_docs = np.array(batch_click_docs, dtype="int32")
        batch_click_news_emb = torch.FloatTensor(self.news_vec[batch_click_docs])
        batch_log_mask = torch.FloatTensor(batch_log_mask)

        batch_output = {
            "batch_cand_index": batch_cand_index,
            "batch_cand_news_emb": batch_cand_news_emb,
            "batch_click_news_emb": batch_click_news_emb,
            "batch_log_mask": batch_log_mask,
            "batch_labels": batch_labels,
        }

        return batch_output


class NewsTestDataset(Dataset):
    def __init__(self, title_emb, body_emb, title_input_ids, title_attn_mask):
        self.title_emb = title_emb
        self.body_emb = body_emb
        self.title_input_ids = title_input_ids
        self.title_attn_mask = title_attn_mask

    def __len__(self):
        return self.title_emb.shape[0]

    def __getitem__(self, idx):
        return (
            self.title_emb[idx],
            self.body_emb[idx],
            self.title_input_ids[idx],
            self.title_attn_mask[idx],
        )
