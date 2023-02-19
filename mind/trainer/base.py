import os
import mlflow
import logging
import numpy as np

import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import *
import utils
from metrics import (
    cal_metrics,
    add_metric_dict,
    update_metric_dict,
    gather_and_print_metrics,
)

class BaseTrainer:
    def __init__(self, rank, args):
        self.rank = rank
        self.args = args

        self.init_ddp()
        logging.info("[-] finish init ddp.")

        self.init_model()
        logging.info("[-] finish init model.")
        self.init_optimizer()
        logging.info("[-] finish init optimizer.")
        self.init_news()
        logging.info("[-] finish init news info.")
        self.init_dataset()
        logging.info("[-] finish init dataset.")
        self.init_collate_fn()
        logging.info("[-] finish init collate_fn")
        self.init_best_metrics()
        logging.info("[-] finish init best metrics")
    
    def init_ddp(self):
        self.local_rank = self.rank % torch.cuda.device_count()

    def init_model(self):
        self.model = eval(self.args.model_name)(self.args)

        if self.args.enable_gpu:
            self.model = self.model.cuda()

        if self.args.enable_ddp:
            self.model = DDP(
                self.model, device_ids=[self.local_rank], find_unused_parameters=True
            )

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path)["model_state_dict"])

    def update_best_metrics(self, metrics):
        """Update the best metrics
        return:
            bool: whether the best metrics are updated.
        """
        if self.best_metrics is None:
            self.best_metrics = metrics
            return True
        else:
            if metrics[0] > self.best_metrics[0]:
                self.best_metrics = metrics
                return True

        return False
    
    def init_best_metrics(self):
        self.best_metrics = None

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.args.lr, amsgrad=True
        )

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

    def compute_batch_score(self, batch_user_vecs, batch_input):
        raise NotImplementedError
    
    def compute_batch_user_vec(self, user_encoder, batch_input):
        raise NotImplementedError

    def infer_news(self, test_news_dl, news_encoder):
        raise NotImplementedError

    def init_news(self):
        raise NotImplementedError

    def init_dataset(self):
        raise NotImplementedError

    def init_collate_fn(self):
        raise NotImplementedError