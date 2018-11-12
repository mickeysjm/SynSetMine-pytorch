from collections import defaultdict
from tqdm import tqdm
import mmap
import os
import logging
import torch
from gensim.models import KeyedVectors  # used to load word2vec
import hashlib
import itertools
import json


class Metrics:

    def __init__(self):
        self.metrics = {}

    def __len__(self):
        return len(self.metrics)

    def add(self, metric_name, metric_value):
        self.metrics[metric_name] = metric_value


class Results:

    def __init__(self, filename):

        self._filename = filename

        open(self._filename, 'a+')

    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save(self, hyperparams, dev_f1='NULL', test_f1='NULL', test_precision='NULL', test_recall='NULL',
             test_avg_jaccard='NULL', test_node_precision='NULL', test_node_recall='NULL',
             test_ARI="NULL", test_FMI="NULL", test_NMI="NULL"):

        result = {'dev_f1': dev_f1,
                  'test_f1': test_f1,
                  'test_precision': test_precision,
                  'test_recall': test_recall,
                  'test_avg_jaccard': test_avg_jaccard,
                  'test_node_precision': test_node_precision,
                  'test_node_recall': test_node_recall,
                  'test_ARI': test_ARI,
                  'test_FMI': test_FMI,
                  'test_NMI': test_NMI,
                  'hash': self._hash(hyperparams)}
        result.update(hyperparams)

        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def save_metrics(self, hyperparams, metrics):

        result = metrics.metrics  # a dict
        result["hash"] = self._hash(hyperparams)
        result.update(hyperparams)
        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best(self):

        results = sorted([x for x in self],
                         key=lambda x: -x['dev_f1'])

        if results:
            return results[0]
        else:
            return None

    def __getitem__(self, hyperparams):

        params_hash = self._hash(hyperparams)

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                if datum['hash'] == params_hash:
                    del datum['hash']
                    return datum

        raise KeyError

    def __contains__(self, x):

        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                del datum['hash']

                yield datum


def save_model(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def load_model(model, load_dir, load_prefix, steps):
    model_prefix = os.path.join(load_dir, load_prefix)
    model_path = "{}_steps_{}.pt".format(model_prefix, steps)
    model.load_state_dict(torch.load(model_path))


def save_checkpoint(model, optimizer, save_dir, save_prefix, step):
    """ Save model checkpoint
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = "{}_steps_{}.pt".format(save_prefix, step)
    checkpoint = {
        "epoch": step + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, load_dir, load_prefix, step):
    """ Load model checkpoint

    Note: the output model and optimizer are on CPU and need to be explicitly moved to GPU
    c.f. https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3

    :param model:
    :param optimizer:
    :param load_dir:
    :param load_prefix:
    :param step:
    :return:
    """
    checkpoint_prefix = os.path.join(load_dir, load_prefix)
    checkpoint_path = "{}_steps_{}.pt".format(checkpoint_prefix, step)
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, start_epoch


def toGPU(optimizer, device):
    """ Move optimizer from CPU to GPU

    :param optimizer:
    :param device:
    :return:
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def check_model_consistency(args):
    """

    :param args:
    :return:
    """
    if args.use_pair_feature == 0 and not args.modelName.startswith("np_"):
        return False, "model without string pair features must has name starting with \"np_\""
    elif args.use_pair_feature == 1 and args.modelName.startswith("np_"):
        return False, "model with string pair features cannot has name starting with \"np_\""
    elif args.loss_fn == "margin_rank" and not args.modelName.endswith("s"):
        return False, "model trained with MarginRankingLoss must have the combiner that output a single " \
                      "scalar for set-instance pair (i.e., ends with Sigmoid Function)"
    elif args.loss_fn != "margin_rank" and args.modelName.endswith("s"):
        return False, "model not trained with MarginRankingLoss cannot have the combiner that output a single " \
                      "scalar for set-instance pair (i.e., ends with Sigmoid Function)"
    elif args.loss_fn == "self_margin_rank" and "_sd_" not in args.modelName:
        return False, "model trained with self MarginRankingLoss must have the combiner that " \
                      "based on score difference (sd)"
    elif args.loss_fn != "self_margin_rank" and "_sd_" in args.modelName:
        return False, "model not trained with self-based MarginRankingLoss cannot have the combiner that " \
                      "based on score difference (sd)"
    else:
        return True, ""


def myLogger(name='', logpath='./'):
    logger = logging.getLogger(name)
    if len(logger.handlers) != 0:
        print('reuse the same logger: {}'.format(name))
        return logger
    else:
        print('create new logger: {}'.format(name))
    fn = os.path.join(logpath, 'run-{}.log'.format(name))
    if os.path.exists(fn):
        print('[warning] log file {} already existed'.format(fn))
    else:
        print('saving log to {}'.format(fn))

    # following two lines are used to solve no file output problem
    # c.f. https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s-> %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', filename=fn, filemode='w')
    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s-> %(message)s',
                                  datefmt='%a %d %b %Y %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger.addHandler(console)

    return logger


def load_embedding(fi, embed_name="word2vec"):
    if embed_name == "word2vec":
        embedding = KeyedVectors.load_word2vec_format(fi)
    else:
        # TODO: allow training embedding from scratch later
        print("[ERROR] Please specify the pre-trained embedding")
        exit(-1)

    vocab_size, embed_dim = embedding.vectors.shape
    index2word = ['PADDING_IDX'] + embedding.index2word
    word2index = {word: index for index, word in enumerate(index2word)}
    return embedding, index2word, word2index, vocab_size, embed_dim


def load_raw_data(fi):
    raw_data_strings = []
    with open(fi, "r") as fin:
        for line in fin:
            raw_data_strings.append(line.strip())
    return raw_data_strings


def load_dataset(fi):
    positives = set()
    negatives = set()
    token2string = defaultdict(set)  # used for generating negative examples
    with open(fi, "r") as fin:
        for line in fin:
            line = line.strip()
            eid, synset = line.split(" ", 1)
            synset = eval(synset)
            
            for syn in synset:
                for tok in syn.split("_"):
                    token2string[tok].add((syn, eid))
            
            for pair in itertools.combinations(synset, 2):
                pair = frozenset([ele+"||"+eid for ele in pair])
                positives.add(frozenset(pair))
        
        # generate negative
        for token in tqdm(token2string, desc="Generating negative pairs ..."):
            strings = token2string[token]
            if len(strings) < 2:
                continue
            else:
                for pair in itertools.combinations(strings, 2):
                    pair = frozenset([ele[0]+"||"+ele[1] for ele in pair])
                    if pair not in positives:
                        negatives.add(pair)
    return list(positives), list(negatives)


def build_dataset(positives, negatives):
    dataset = []
    dataset.extend([(ele, 1.0) for ele in positives])
    dataset.extend([(ele, 0.0) for ele in negatives])
    return dataset


def print_args(args, interested_args="all"):
    print("\nParameters:")
    if interested_args == "all":
        for attr, value in sorted(args.__dict__.items()):
            print("\t{}={}".format(attr.upper(), value))
    else:
        for attr, value in sorted(args.__dict__.items()):
            if attr in interested_args:
                print("\t{}={}".format(attr.upper(), value))
    print('-' * 89)


def get_num_lines(file_path):
    """ Usage:
    with open(inputFile,"r") as fin:
      for line in tqdm(fin, total=get_num_lines(inputFile)):
            ...
    """
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def KM_Matching(weight_nm):
    """ Maximum weighted matching

    :param weight_nm:
    :return:
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
