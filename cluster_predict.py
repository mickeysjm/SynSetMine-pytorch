"""
.. module:: cluster_predict
    :synopsis: clustering by set generation algorithm

.. moduleauthor:: Jiaming Shen
"""
import torch
import numpy as np
from tqdm import tqdm


def multiple_set_single_instance_prediction(model, sets, instance, size_optimized=False):
    """ Apply the given model to predict the probabilities of adding that one instance into each of the given sets

    :param model: a trained SynSetMine model
    :type model: SSPM
    :param sets: a list of sets, each contain the element index
    :type sets: list
    :param instance: a single instance, represented by the element index
    :type instance: int
    :param size_optimized: whether to optimize the multiple-set-single-instance prediction process. If the size of each
        set in the given 'sets' varies a lot and there exists a single huge set in the given 'sets', set this parameter
        to be True
    :type size_optimized: bool
    :return:

        - scores of given sets, (batch_size, 1)
        - scores of given sets union with the instance, (batch_size, 1)
        - the probability of adding the instance into the corresponding set, (batch_size, 1)

    :rtype: tuple
    """
    if not size_optimized:  # when there exists no single big cluster, no need for complex size optimization
        return _multiple_set_single_instance_prediction(model, sets, instance)
    else:
        if len(sets) <= 10:
            return _multiple_set_single_instance_prediction(model, sets, instance)

        set_sizes = [len(ele) for ele in sets]
        tmp = sorted(enumerate(set_sizes), key=lambda x: x[1])  # (old index, set_size)
        n2o = {n: ele[0] for n, ele in enumerate(tmp)}  # new index -> old index
        o2n = {n2o[n]: n for n in n2o}  # old index -> new index
        sorted_set_sizes = [ele[1] for ele in tmp]

        # the bining method is a combination of 'sturges' and 'fd' estimators, another choice is set "bins="sturges", which generates more bins
        # c.f.: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
        _, bin_edges = np.histogram(sorted_set_sizes, bins="auto")
        inds = np.digitize(sorted_set_sizes, bin_edges)

        sorted_setScores = []
        sorted_setInstSumScores = []
        sorted_positive_prob = []
        cur_ind = inds[0]
        cur_set = [sets[tmp[0][0]]]
        for i in range(1, len(inds)):
            if inds[i] == cur_ind:
                cur_set.append(sets[tmp[i][0]])
            else:
                cur_setScores, cur_setInstSumScores, cur_positive_prob = _multiple_set_single_instance_prediction(
                    model, cur_set, instance)
                sorted_setScores += cur_setScores
                sorted_setInstSumScores += cur_setInstSumScores
                sorted_positive_prob += cur_positive_prob
                cur_ind = inds[i]
                cur_set = [sets[tmp[i][0]]]
        if len(cur_set) > 0:  # working on the last bin
            cur_setScores, cur_setInstSumScores, cur_positive_prob = _multiple_set_single_instance_prediction(
                model, cur_set, instance)
            sorted_setScores += cur_setScores
            sorted_setInstSumScores += cur_setInstSumScores
            sorted_positive_prob += cur_positive_prob

        if len(sets) != len(sorted_positive_prob):
            assert "Mismatch after binning optimization"

        setScores = []
        setInstSumScores = []
        positive_prob = []
        for o in range(len(sets)):
            setScores.append(sorted_setScores[o2n[o]])
            setInstSumScores.append(sorted_setInstSumScores[o2n[o]])
            positive_prob.append(sorted_positive_prob[o2n[o]])

        return setScores, setInstSumScores, positive_prob


def _multiple_set_single_instance_prediction(model, sets, instance):
    model.eval()

    # generate tensors
    batch_size = len(sets)
    max_set_size = max([len(ele) for ele in sets])
    batch_set_tensor = np.zeros([batch_size, max_set_size], dtype=np.int)
    for row_id, row in enumerate(sets):
        batch_set_tensor[row_id][:len(row)] = row
    batch_set_tensor = torch.from_numpy(batch_set_tensor)  # (batch_size, max_set_size)
    batch_inst_tensor = torch.tensor(instance).unsqueeze(0).expand(batch_size, 1)  # (batch_size, 1)

    batch_set_tensor = batch_set_tensor.to(model.device)
    batch_inst_tensor = batch_inst_tensor.to(model.device)

    # inference
    setScores, setInstSumScores, prediction = model.predict(batch_set_tensor, batch_inst_tensor)

    # convert to probability of each sip
    positive_prob = prediction.squeeze(-1).detach()
    positive_prob = list(positive_prob.to(torch.device("cpu")).numpy())

    setScores = setScores.squeeze(-1).detach()
    setScores = list(setScores.to(torch.device("cpu")).numpy())

    setInstSumScores = setInstSumScores.squeeze(-1).detach()
    setInstSumScores = list(setInstSumScores.to(torch.device("cpu")).numpy())

    model.train()
    return setScores, setInstSumScores, positive_prob


def set_generation(model, vocab, threshold=0.5, eid2ename=None, size_opt_clus=False, max_K=None, verbose=False):
    """ Set Generation Algorithm

    :param model: a trained set-instance classifier
    :type model: SSPM
    :param vocab: a list of elements to be clustered, each element is represented by its index
    :type vocab: list
    :param threshold: the probability threshold for determine whether to create new singleton cluster
    :type threshold: float
    :param eid2ename: a dictionary mapping element index to its corresponding (human-readable) name
    :type eid2ename: dict
    :param size_opt_clus: a flag indicating whether to optimize the multiple-set-single-instance prediction process
    :type size_opt_clus: bool
    :param max_K: maximum number of clusters, If None, we will infer this number automatically
    :type max_K: int
    :param verbose: whether to print out all intermediate results
    :type verbose: bool
    :return: a list of detected clusters
    :rtype: list
    """
    model.eval()

    clusters = []  # will be a list of lists
    candidate_pool = vocab
    if verbose:
        print("{}\t{}".format("vocab", [eid2ename[eid] for eid in vocab]))

    if verbose:
        g = tqdm(range(len(candidate_pool)), desc="Cluster prediction (aggressive one pass)...")
    else:
        g = range(len(candidate_pool))
    for i in g:
        inst = candidate_pool[i]
        if i == 0:
            cluster = [inst]
            clusters.append(cluster)
        else:
            setScores, setInstSumScores, cluster_probs = multiple_set_single_instance_prediction(
                model, clusters, inst, size_optimized=size_opt_clus
            )
            best_matching_existing_cluster_idx = -1
            best_matching_existing_cluster_prob = 0.0
            for cid, cluster_prob in enumerate(cluster_probs):
                if cluster_prob > best_matching_existing_cluster_prob:
                    best_matching_existing_cluster_prob = cluster_prob
                    best_matching_existing_cluster_idx = cid

            if verbose:
                print("Current Cluster Pool:",
                      [(cid, [eid2ename[ele] for ele in cluster]) for cid, cluster in enumerate(clusters)])
                print("-" * 20)
                print("Entity: {:<30}  best_prob = {:<8} Best-matching Cluster: {:<80} (cid={})".format(eid2ename[inst], best_matching_existing_cluster_prob, str(
                    [eid2ename[eid] for eid in clusters[best_matching_existing_cluster_idx]]), best_matching_existing_cluster_idx))

            if max_K and len(clusters) >= max_K:
                clusters[best_matching_existing_cluster_idx].append(inst)
                if verbose:
                    print("!!! Add Entity In")
            else:
                # then either add this instance to existing cluster or create a new cluster
                if best_matching_existing_cluster_prob > threshold:
                    clusters[best_matching_existing_cluster_idx].append(inst)
                    if verbose:
                        print("!!! Add Entity In")
                else:
                    new_cluster = [inst]
                    clusters.append(new_cluster)

            if verbose:
                print("-" * 120)

    model.train()
    return clusters
