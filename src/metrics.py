import math
from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

def onedcg_score(row, label_col, score_col, k):
    onedcg_labels = [k if k != -1 else 0 for k in list(row[label_col])]
    onedcg_scores = list(row[score_col])
    pair = list(zip(onedcg_labels, onedcg_scores))
    pair = sorted(pair, key=lambda x: -x[1])
    idcg = sum([math.pow(0.6, i) for i in range(k)])
    dcg = sum([k[0] * 25 * math.pow(0.6, i) for i, k in enumerate(pair[:k])])
    onedcg = dcg / idcg
    return onedcg

def calc_ndcg_per_query(row, label_col, score_col):
    labels, scores = [list(row[label_col])], [list(row[score_col])]
    onedcg_at3 = onedcg_score(row, label_col, score_col, 3)
    onedcg_at5 = onedcg_score(row, label_col, score_col, 5)
    if len(labels[0]) == 1:
        return 0, 0, onedcg_at3, onedcg_at5
    
    # print(labels, scores)
    ndcg_at3 = ndcg_score(labels, scores, k=3)
    ndcg_at5 = ndcg_score(labels, scores, k=5)
    # labels = [k if k != -1 else 0 for k in labels]
    # dcg_at1 = get_dcg(scores, labels, 3)
    # idcg_at1 = get_dcg(labels, labels, 3)
    # if idcg_at1 != 0:
    #     ndcg_at1 = dcg_at1 / idcg_at1
    # else:
    #     ndcg_at1 = 0

    # dcg_at5 = get_dcg(scores, labels, 5)
    # idcg_at5 = get_dcg(labels, labels, 5)
    # if idcg_at5 != 0:
    #     ndcg_at5 = dcg_at5 / idcg_at5
    # else:
    #     ndcg_at5 = 0

    # dcg_at10 = get_dcg(scores, labels, 10)
    # idcg_at10 = get_dcg(labels, labels, 10)
    # if idcg_at10 != 0:
    #     ndcg_at10 = dcg_at10 / idcg_at10
    # else:
    #     ndcg_at10 = 0
    
    return ndcg_at3, ndcg_at5, onedcg_at3, onedcg_at5


def calc_ndcg(data, query_col="Query", label_col="Label", score_col="Score"):
    results = data.groupby([query_col]).apply(
        partial(calc_ndcg_per_query, label_col=label_col, score_col=score_col)
    )
    ndcg_at3 = []
    ndcg_at5 = []
    onedcg_at3 = []
    onedcg_at5 = []

    for r in results.tolist():
        ndcg_at3.append(r[0])
        ndcg_at5.append(r[1])
        onedcg_at3.append(r[2])
        onedcg_at5.append(r[3])
    ndcg_at3, ndcg_at5, onedcg_at3, onedcg_at5 = sum(ndcg_at3) / len(ndcg_at3), sum(ndcg_at5) / len(ndcg_at5), sum(onedcg_at3) / len(onedcg_at3), sum(onedcg_at5) / len(onedcg_at5)
    return {'ndcg_at3': ndcg_at3, 
            'ndcg_at5': ndcg_at5, 
            'onedcg_at3': onedcg_at3, 
            'onedcg_at5': onedcg_at5}
    # scores = data.groupby([query_col])['Score'].apply(list).tolist()
    # labels = data.groupby([query_col])['Label'].apply(list).tolist()
    # ndcg_at1 = ndcg_score(labels, scores, k=1)
    # ndcg_at5 = ndcg_score(labels, scores, k=5)
    # ndcg_at10 = ndcg_score(labels, scores, k=10)
    # return ndcg_at1, ndcg_at5, ndcg_at10


if __name__ == '__main__':
    data = {
        "Query": ['duihduahad', 'duihduahad', 'duihduahad', 'daafff'],
        "Label": [1., 2., 3., 0.],
        "Score": [2., 1., 4., 0.]
    }

    data4ndcg = pd.DataFrame(data)

    a, b, c = calc_ndcg(data4ndcg)


    print(a, b, c)