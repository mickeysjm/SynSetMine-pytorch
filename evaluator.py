"""
.. module:: evaluator
    :synopsis: model evaluator

.. moduleauthor:: Jiaming Shen, Ruiliang Lyu, Wenda Qiu
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import itertools
import networkx as nx


def calculate_precision_recall_f1(tp, fp, fn):
    """ Calculate precision, recall, and f1 score

    :param tp: true positive number
    :type tp: int
    :param fp: false positive number
    :type fp: int
    :param fn: false negative number
    :type fn: int
    :return: (precision, recall, f1 score)
    :rtype: tuple
    """

    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = 1.0 * tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = 1.0 * tp / (tp + fn)

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def calculate_km_matching_score(weight_nm):
    """ Calculate maximum weighted matching score

    :param weight_nm: a similarity matrix
    :type weight_nm: list
    :return: weighted matching score
    :rtype: float
    """
    x = len(weight_nm)
    y = len(weight_nm[0])
    n = max(x, y)
    NONE = -1e6
    INF = 1e9
    weight = [[NONE for j in range(n + 1)] for i in range(n + 1)]
    for i in range(x):
        for j in range(y):
            weight[i + 1][j + 1] = weight_nm[i][j]
    lx = [0. for i in range(n + 1)]
    ly = [0. for i in range(n + 1)]
    match = [-1 for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            lx[i] = max(lx[i], weight[i][j])
    for root in range(1, n + 1):
        vy = [False for i in range(n + 1)]
        slack = [INF for i in range(n + 1)]
        pre = [0 for i in range(n + 1)]
        py = 0
        match[0] = root
        while True:
            vy[py] = True
            x = match[py]
            delta = INF
            yy = 0
            for y in range(1, n + 1):
                if not vy[y]:
                    if lx[x] + ly[y] - weight[x][y] < slack[y]:
                        slack[y] = lx[x] + ly[y] - weight[x][y]
                        pre[y] = py
                    if slack[y] < delta:
                        delta = slack[y]
                        yy = y
            for y in range(n + 1):
                if vy[y]:
                    lx[match[y]] -= delta
                    ly[y] += delta
                else:
                    slack[y] -= delta
            py = yy
            if match[py] == -1: break
        while True:
            prev = pre[py]
            match[py] = match[prev]
            py = prev
            if py == 0: break
    score = 0.
    for i in range(1, n + 1):
        v = weight[match[i]][i]
        if v > NONE:
            score += v
    return score


def end2end_evaluation_matching(groundtruth, result):
    """ Evaluate the maximum weighted jaccard matching of groundtruth clustering and predicted clustering

    :param groundtruth: a list of element lists representing the ground truth clustering
    :type groundtruth: list
    :param result: a list of element lists representing the model predicted clustering
    :type result: list
    :return: best matching score
    :rtype: float
    """
    n = len(groundtruth)
    m = len(result)
    G = nx.DiGraph()
    S = n + m
    T = n + m + 1
    C = 1e8
    for i in range(n):
        for j in range(m):
            s1 = groundtruth[i]
            s2 = result[j]
            s12 = set(s1) & set(s2)
            weight = len(s12) / (len(s1) + len(s2) - len(s12))
            weight = int(weight * C)
            if weight > 0:
                G.add_edge(i, n + j, capacity=1, weight=-weight)
    for i in range(n):
        G.add_edge(S, i, capacity=1, weight=0)
    for i in range(m):
        G.add_edge(i + n, T, capacity=1, weight=0)
    mincostFlow = nx.algorithms.max_flow_min_cost(G, S, T)
    mincost = nx.cost_of_flow(G, mincostFlow) / C
    return -mincost / m


def evaluate_set_instance_prediction(model, dataset):
    """ Evaluate model on the given dataset for set-instance pair prediction task

    :param model: a trained set-instance classifier
    :type model: SSPM
    :param dataset: an ElementSet dataset with
    :type dataset: ElementSet
    :return: a dictionary of set-instance pair prediction metrics
    :rtype: dict
    """
    model.eval()

    y_true = []
    y_pred = []
    set_size = []

    # the following max_set_size and batch_size number need to be set such that one test batch can fit GPU memory
    # TODO: make this value dynamtically changeable
    max_set_size = 100
    batch_size = int(len(dataset.sip_triplets) / 2)
    for test_batch in dataset.get_test_batch(max_set_size=max_set_size, batch_size=batch_size):
        # log set size for set-size-wise error analysis
        batch_set_size = torch.sum((test_batch['set'] != 0), dim=1)
        if model.device_id != -1:
            batch_set_size = batch_set_size.to(torch.device("cpu"))
        batch_set_size = list(batch_set_size.numpy())
        set_size += batch_set_size

        # start real prediction
        mask = (test_batch['set'] != 0).float().unsqueeze(-1)
        setEmbed = model.nodeTransform(test_batch['set']) * mask
        setEmbed = model.node_pooler(setEmbed, dim=1)
        instEmbed = model.nodeTransform(test_batch['inst']).squeeze_(1)
        setScores = model.scorer(setEmbed)
        setInstSumScores = model.scorer(setEmbed + instEmbed)
        score_diff = setInstSumScores - setScores

        prediction = F.sigmoid(score_diff)
        if model.device_id != -1:
            prediction = prediction.to(torch.device("cpu"))
        cur_pred = (prediction > 0.5).squeeze().numpy()
        y_pred += list(cur_pred)

        target = test_batch['label'].float()
        loss = model.criterion(score_diff, target).item()
        if model.device_id != -1:
            target = target.to(torch.device("cpu"))
        cur_true = target.squeeze().numpy()
        y_true += list(cur_true)

    # obtain set-size-wise accuracy
    set_size2num = Counter(set_size)
    set_size2correct = defaultdict(int)
    for t, p, s in zip(y_true, y_pred, set_size):
        if t == p:
            set_size2correct[s] += 1
    set_size2accuracy = {}
    for set_size in set_size2correct:
        set_size2accuracy[set_size] = set_size2correct[set_size] / set_size2num[set_size]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    num_pred_pos = int(np.sum(y_pred))
    num_pred_neg = y_true.shape[0] - num_pred_pos
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)

    model.train()
    metrics = {"precision": precision, "recall": recall, "f1": f1, "num_pred_pos": num_pred_pos,
               "num_pred_neg": num_pred_neg, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "loss": loss,
               "accuracy": accuracy}
    return metrics


def evaluate_clustering(cls_pred, cls_true):
    """ Evaluate clustering results

    :param cls_pred: a list of element lists representing model predicted clustering
    :type cls_pred: list
    :param cls_true: a list of element lists representing the ground truth clustering
    :type cls_true: list
    :return: a dictionary of clustering evaluation metrics
    :rtype: dict
    """
    vocab_pred = set(itertools.chain(*cls_pred))
    vocab_true = set(itertools.chain(*cls_true))
    assert (vocab_pred == vocab_true), "Unmatched vocabulary during clustering evaluation"

    # Cluster number
    num_of_predict_clusters = len(cls_pred)

    # Cluster size histogram
    cluster_size2num_of_predicted_clusters = Counter([len(cluster) for cluster in cls_pred])

    # Exact cluster prediction
    pred_cluster_set = set([frozenset(cluster) for cluster in cls_pred])
    gt_cluster_set = set([frozenset(cluster) for cluster in cls_true])
    num_of_exact_set_prediction = len(pred_cluster_set.intersection(gt_cluster_set))

    # Clustering metrics
    word2rank = {}
    wordrank2gt_cluster = {}
    rank = 0
    for cid, cluster in enumerate(cls_true):
        for word in cluster:
            if word not in word2rank:
                word2rank[word] = rank
                rank += 1
            wordrank2gt_cluster[word2rank[word]] = cid
    gt_cluster_vector = [ele[1] for ele in sorted(wordrank2gt_cluster.items())]

    wordrank2pred_cluster = {}
    for cid, cluster in enumerate(cls_pred):
        for word in cluster:
            wordrank2pred_cluster[word2rank[word]] = cid
    pred_cluster_vector = [ele[1] for ele in sorted(wordrank2pred_cluster.items())]

    ARI = adjusted_rand_score(gt_cluster_vector, pred_cluster_vector)
    FMI = fowlkes_mallows_score(gt_cluster_vector, pred_cluster_vector)
    NMI = normalized_mutual_info_score(gt_cluster_vector, pred_cluster_vector)

    # Pair-based clustering metrics
    def pair_set(labels):
        S = set()
        cluster_ids = np.unique(labels)
        for cluster_id in cluster_ids:
            cluster = np.where(labels == cluster_id)[0]
            n = len(cluster)  # number of elements in this cluster
            if n >= 2:
                for i in range(n):
                    for j in range(i + 1, n):
                        S.add((cluster[i], cluster[j]))
        return S

    F_S = pair_set(gt_cluster_vector)
    F_K = pair_set(pred_cluster_vector)
    if len(F_K) == 0:
        pair_recall = 0
        pair_precision = 0
        pair_f1 = 0
    else:
        common_pairs = len(F_K & F_S)
        pair_recall = common_pairs / len(F_S)
        pair_precision = common_pairs / len(F_K)
        eps = 1e-6
        pair_f1 = 2 * pair_precision * pair_recall / (pair_precision + pair_recall + eps)

    # KM matching
    mwm_jaccard = end2end_evaluation_matching(cls_true, cls_pred)

    metrics = {"ARI": ARI, "FMI": FMI, "NMI": NMI, "pair_recall": pair_recall, "pair_precision": pair_precision,
               "pair_f1": pair_f1, "predicted_clusters": cls_pred, "num_of_predicted_clusters": num_of_predict_clusters,
               "cluster_size2num_of_predicted_clusters": cluster_size2num_of_predicted_clusters,
               "num_of_exact_set_prediction": num_of_exact_set_prediction,
               "maximum_weighted_match_jaccard": mwm_jaccard}

    return metrics
