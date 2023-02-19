from sklearn.metrics import roc_auc_score
import numpy as np
import logging

import torch
import torch.distributed as dist

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)

def cal_metrics(score, label):
    metric_rslt = {}
    metric_rslt["AUC"] = roc_auc_score(label, score)
    metric_rslt["MRR"] = mrr_score(label, score)
    metric_rslt["nDCG5"] = ndcg_score(label, score, k=5)
    metric_rslt["nDCG10"] = ndcg_score(label, score, k=10)
    return metric_rslt

def add_metric_dict():
    metrics_dict = {}
    metrics_name = ["AUC", "MRR", "nDCG5", "nDCG10"]

    for metric_name in metrics_name:
        metrics_dict[metric_name] = []
    
    return metrics_dict

def update_metric_dict(metrics_dict, metric_rslt):
    metrics_name = ["AUC", "MRR", "nDCG5", "nDCG10"]
    for metric_name in metrics_name:
        metrics_dict[metric_name].append(
            metric_rslt[metric_name]
        )
    return metrics_dict

def get_mean(arr):
    return [np.array(i).mean() for i in arr]

@torch.no_grad()
def gather_and_print_metrics(metrics_dict, enable_ddp):
    metrics_name = ["AUC", "MRR", "nDCG5", "nDCG10"]
    arr = get_mean(
        [
            metrics_dict[metric_name]
            for metric_name in metrics_name
        ]
    )
    cnt = len(metrics_dict[metrics_name[0]])

    if enable_ddp:
        arr.append(cnt)
        arr_cnt = torch.FloatTensor(arr).cuda()

        gpu_num = torch.cuda.device_count()
        arr_cnt_list = [torch.zeros_like(arr_cnt) for _ in range(gpu_num)]

        # import pdb; pdb.set_trace()

        dist.all_gather(arr_cnt_list, arr_cnt)

        arr_cnt_list = torch.stack(arr_cnt_list, dim=0).detach().cpu()

        cnt_list = arr_cnt_list[:, -1]
        cnt = torch.sum(cnt_list).long()
        weights = cnt_list / cnt
        metrics_list = arr_cnt_list[:, :-1]
        merged_arr = torch.sum(metrics_list * weights.unsqueeze(-1), dim=0)
        merged_arr = merged_arr.detach().numpy().tolist()
    else:
        merged_arr = arr
    
    logging.info(
        "Ed: {}: {}".format(
            cnt,
            "\t".join(["{:0.2f}".format(i * 100) for i in merged_arr]),
        )
    )
    return merged_arr, cnt

def print_metrics(metrics_dict):
    metrics_name = ["AUC", "MRR", "nDCG5", "nDCG10"]
    arr = get_mean(
        [
            metrics_dict[metric_name]
            for metric_name in metrics_name
        ]
    )
    cnt = len(metrics_dict[metrics_name[0]])


    logging.info(
        "Ed: {}: {}".format(
            cnt,
            "\t".join(["{:0.2f}".format(i * 100) for i in arr]),
        )
    )
    return arr, cnt
