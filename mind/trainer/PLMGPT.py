import numpy as np
import mlflow

import torch

from news.gpt_plm_process import PLMGPTInfo
from model import *

from data.base import BehaviorTrainDataset, BehaviorTestDataset

from data.plmgpt import BehaviorTrainCollate, BehaviorTestCollate, NewsTestDataset
from trainer.base import BaseTrainer


class PLMGPTTrainer(BaseTrainer):
    def init_news(self):
        self.news_info = PLMGPTInfo(
            self.args, self.args.news_emb_file, self.args.news_file
        )
        self.news_info.process()

        if (
            self.args.test_news_file is not None
            and self.args.test_news_file != self.args.news_file
        ):
            self.test_news_info = PLMGPTInfo(
                self.args, self.args.test_news_emb_file, self.args.test_news_file
            )
            self.test_news_info.process()
        else:
            self.test_news_info = self.news_info

    def init_dataset(self):
        self.train_ds = BehaviorTrainDataset(
            self.args, self.args.train_behavior_file, self.news_info.nid2index
        )
        self.eval_ds = BehaviorTestDataset(
            self.args, self.args.eval_behavior_file, self.news_info.nid2index
        )
        self.test_ds = BehaviorTestDataset(
            self.args, self.args.test_behavior_file, self.test_news_info.nid2index
        )

        # train eval share same news file
        self.eval_news_ds = NewsTestDataset(
            self.news_info.title_embs,
            self.news_info.body_embs,
            self.news_info.news_title,
            self.news_info.news_title_attmask,
        )
        self.test_news_ds = NewsTestDataset(
            self.test_news_info.title_embs,
            self.test_news_info.body_embs,
            self.test_news_info.news_title,
            self.test_news_info.news_title_attmask,
        )

    def init_collate_fn(self):
        self.train_collate_fn = BehaviorTrainCollate(
            self.news_info.title_embs,
            self.news_info.body_embs,
            self.news_info.news_title,
            self.news_info.news_title_attmask,
        )
        self.test_collate_cls = BehaviorTestCollate

    @torch.no_grad()
    def infer_news(self, test_news_dl, news_encoder):
        news_vecs = []
        for (
            news_title_gpt_emb,
            news_body_gpt_emb,
            news_title_input_ids,
            news_title_attn_masks,
        ) in test_news_dl:
            if self.args.enable_gpu:
                news_title_gpt_emb = news_title_gpt_emb.cuda(non_blocking=True).float()
                news_body_gpt_emb = news_body_gpt_emb.cuda(non_blocking=True).float()
                news_title_input_ids = news_title_input_ids.cuda(non_blocking=True).long()
                news_title_attn_masks = news_title_attn_masks.cuda(non_blocking=True).long()

            news_vec = (
                news_encoder(
                    news_title_gpt_emb,
                    news_body_gpt_emb,
                    news_title_input_ids,
                    news_title_attn_masks,
                )
                .detach()
                .cpu()
                .numpy()
            )
            news_vecs.append(news_vec)

        news_vecs = np.concatenate(news_vecs, axis=0)
        return news_vecs

    @torch.no_grad()
    def compute_batch_user_vec(self, user_encoder, batch_input):
        batch_user_vecs = user_encoder(
            batch_input["batch_click_news_emb"], batch_input["batch_log_mask"]
        ).detach()
        return batch_user_vecs

    @torch.no_grad()
    def compute_batch_score(self, batch_user_vecs, batch_input):
        batch_score = (
            torch.bmm(
                batch_user_vecs[batch_input["batch_cand_index"]].unsqueeze(1),
                batch_input["batch_cand_news_emb"].unsqueeze(-1),
            )
            .view(-1)
            .detach()
            .cpu()
            .numpy()
        )
        return batch_score
