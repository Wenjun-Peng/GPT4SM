import numpy as np
import mlflow

import torch

from news.gpt_plm_process import PLMGPTInfo
from model import *

from data.base import BehaviorTrainDataset, BehaviorTestDataset

from data.plmgpt import BehaviorTrainCollate, BehaviorTestCollate, NewsTestDataset
from trainer.base import BaseTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from model import *
import utils
from metrics import (
    cal_metrics,
    add_metric_dict,
    update_metric_dict,
    gather_and_print_metrics,
)
import os

class CopyGPTTrainer(BaseTrainer):
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
                )[0]
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
    
    def train(self):
        if self.args.eval_first:
            self.test(mode="eval")

        logging.info("Training...")
        if self.args.enable_ddp:
            sampler = DistributedSampler(self.train_ds, shuffle=True)
            train_dl = DataLoader(
                dataset=self.train_ds,
                sampler=sampler,
                batch_size=self.args.batch_size,
                collate_fn=self.train_collate_fn,
            )
        else:
            train_dl = DataLoader(
                dataset=self.train_ds,
                shuffle=True,
                batch_size=self.args.batch_size,
                collate_fn=self.train_collate_fn,
            )

        global_step = 0
        for ep in range(self.args.epochs):
            self.model.train()
            loss = 0.0
            accuary = 0.0
            for cnt, batch_input in enumerate(train_dl):
                global_step += 1
                batch_input['ep'] = ep
                if self.args.enable_gpu:
                    for k in batch_input:
                        if torch.is_tensor(batch_input[k]):
                            batch_input[k] = batch_input[k].cuda()
                bz_loss, y_hat = self.model(batch_input)
                loss += bz_loss.data.float()
                batch_acc = utils.acc(batch_input["batch_labels"], y_hat)
                accuary += batch_acc
                self.optimizer.zero_grad()
                bz_loss.backward()
                self.optimizer.step()

                if cnt % self.args.log_steps == 0:
                    logging.info(
                        "[{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}".format(
                            self.local_rank,
                            cnt * self.args.batch_size,
                            loss.data / (cnt + 1),
                            accuary / (cnt + 1),
                        )
                    )
                    if self.local_rank == 0:
                        mlflow.log_metric("loss", bz_loss.item(), step=global_step)
                        mlflow.log_metric("acc", batch_acc.item(), step=global_step)

            test_metrics = self.test(mode="eval")
            if self.local_rank == 0:
                for cnt, metric_name in enumerate(["auc", "mrr", "ndcg5", "ndcg10"]):
                    mlflow.log_metric(
                        f"eval_{metric_name}", test_metrics[cnt], step=global_step
                    )

                is_updated = self.update_best_metrics(test_metrics)
                if is_updated:
                    ckpt_path = os.path.join(self.args.model_dir, f"best.pt")
                    torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)
                    logging.info(f"Epoch {ep} model saved to {ckpt_path}")

        logging.info("Load best model and test on test dataset")
        self.load_model(os.path.join(self.args.model_dir, f"best.pt"))

        test_metrics = self.test(mode="test")
        if self.local_rank == 0:
            for cnt, metric_name in enumerate(["auc", "mrr", "ndcg5", "ndcg10"]):
                mlflow.log_metric(
                    f"test_{metric_name}", test_metrics[cnt], step=global_step
                )

    @torch.no_grad()
    def test(self, mode="eval"):
        self.model.eval()

        if mode == "eval":
            test_news_ds = self.eval_news_ds
            test_ds = self.eval_ds
        elif mode == "test":
            test_news_ds = self.test_news_ds
            test_ds = self.test_ds

        logging.info(f"Testing on {mode} dataset")
        test_news_dl = DataLoader(
            dataset=test_news_ds,
            batch_size=self.args.test_news_batch_size,
        )

        if self.args.enable_ddp:
            user_encoder = self.model.module.user_encoder
            news_encoder = self.model.module.news_encoder
        else:
            user_encoder = self.model.user_encoder
            news_encoder = self.model.news_encoder

        logging.info("Infer news embeddings.")
        news_vecs = self.infer_news(test_news_dl, news_encoder)

        logging.info("Testing behaviors.")
        test_collate_fn = self.test_collate_cls(news_vecs)

        if self.args.enable_ddp:
            sampler = DistributedSampler(test_ds, shuffle=False)
            test_dl = DataLoader(
                dataset=test_ds,
                sampler=sampler,
                batch_size=self.args.test_user_batch_size,
                collate_fn=test_collate_fn,
            )
        else:
            test_dl = DataLoader(
                dataset=test_ds,
                shuffle=False,
                batch_size=self.args.test_user_batch_size,
                collate_fn=test_collate_fn,
            )

        metrics_dict = add_metric_dict()
        for cnt, batch_input in enumerate(test_dl):
            if self.args.enable_gpu:
                for k in batch_input:
                    if torch.is_tensor(batch_input[k]):
                        batch_input[k] = batch_input[k].cuda()

            batch_user_vecs = self.compute_batch_user_vec(user_encoder, batch_input)
            batch_score = self.compute_batch_score(batch_user_vecs, batch_input)
                
            start = 0
            for targets in batch_input["batch_labels"]:
                if np.mean(targets) == 0 or np.mean(targets) == 1:
                    continue

                end = start + len(targets)
                score = batch_score[start:end]
                start = end
                metric_rslt = cal_metrics(score, targets)
                metrics_dict = update_metric_dict(metrics_dict, metric_rslt)

        test_metrics, _ = gather_and_print_metrics(
            metrics_dict, enable_ddp=self.args.enable_ddp
        )
        return test_metrics