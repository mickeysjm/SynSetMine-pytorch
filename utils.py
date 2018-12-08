"""
.. module:: utils
    :synopsis: utility functions

.. moduleauthor:: Jiaming Shen
"""
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
    """ A metric class wrapping all metrics

    """

    def __init__(self):
        self.metrics = {}

    def __len__(self):
        return len(self.metrics)

    def add(self, metric_name, metric_value):
        """ Add metric value for the given metric name

        :param metric_name: metric name
        :type metric_name: str
        :param metric_value: metric value
        :type metric_value:
        :return: None
        :rtype: None
        """
        self.metrics[metric_name] = metric_value


class Results:
    """ A result class for saving results to file

    :param filename: name of result saving file
    :type filename: str
    """

    def __init__(self, filename):
        self._filename = filename

        open(self._filename, 'a+')

    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save_metrics(self, hyperparams, metrics):
        """ Save model hyper-parameters and evaluation results to the file

        :param hyperparams: a dictionary of model hyper-parameters, keyed with the hyper-parameter names
        :type hyperparams: dict
        :param metrics: a Metrics object containg all model evaluation results
        :type metrics: Metrics
        :return: None
        :rtype: None
        """

        result = metrics.metrics  # a dict
        result["hash"] = self._hash(hyperparams)
        result.update(hyperparams)
        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

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
    """ Save model to file

    :param model: a trained model
    :type model: torch.nn
    :param save_dir: model save directory
    :type save_dir: str
    :param save_prefix: model snapshot prefix
    :type save_prefix: str
    :param steps: model training epoch
    :type steps: int
    :return: None
    :rtype: None
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def load_model(model, load_dir, load_prefix, steps):
    """ load model from file

    Note: You need to first initialize a model which has the same architecture/size of the model to be loaded.

    :param model: a model which has the same architecture of the model to be loaded
    :type model: torch.nn
    :param load_dir: model save directory
    :type load_dir: str
    :param load_prefix: model snapshot prefix
    :type load_prefix: str
    :param steps: model training epoch
    :type steps: int
    :return: None
    :rtype: None
    """
    model_prefix = os.path.join(load_dir, load_prefix)
    model_path = "{}_steps_{}.pt".format(model_prefix, steps)
    model.load_state_dict(torch.load(model_path))


def save_checkpoint(model, optimizer, save_dir, save_prefix, step):
    """ Save model checkpoint (including trained model, training epoch, and optimizer) to a file

    :param model: a trained model
    :type model: torch.nn
    :param optimizer: a pytorch optimizer
    :type optimizer: torch.optim
    :param save_dir: model save directory
    :type save_dir: str
    :param save_prefix: model snapshot prefix
    :type save_prefix: str
    :param step: model training epoch
    :type step: int
    :return: None
    :rtype: None
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
    """ Load model checkpoint (including trained model, training epoch, and optimizer) from a file

    Notes:

        - The loaded model and optimizer are initially on CPU and need to be explicitly moved to GPU c.f. https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3.
        - You need to first initialize a model which has the same architecture/size of the model to be loaded.

    :param model: a model which has the same architecture of the model to be loaded
    :type model: torch.nn
    :param optimizer: a pytorch optimizer
    :type optimizer: torch.optim
    :param load_dir: model save directory
    :type load_dir: str
    :param load_prefix: model snapshot prefix
    :type load_prefix: str
    :param step: model training epoch
    :type step: int
    :return: None
    :rtype: None
    """
    checkpoint_prefix = os.path.join(load_dir, load_prefix)
    checkpoint_path = "{}_steps_{}.pt".format(checkpoint_prefix, step)
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, start_epoch


def to_gpu(optimizer, device):
    """ Move optimizer from CPU to GPU

    :param optimizer: a pytorch optimizer
    :type optimizer: torch.optim
    :param device: a pytorch device, CPU or GPU
    :type device: torch.device
    :return: None
    :rtype: None
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def check_model_consistency(args):
    """ Check whether the model architecture is consistent with the loss function used

    :param args: a dictionary containing all model specifications
    :type args: dict
    :return: a flag indicating whether the model architecture is consistent with the loss function,
        if not, also return the error message
    :rtype: a tuple of (bool, str)
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


def my_logger(name='', log_path='./'):
    """ Create a python logger

    :param name: logger name
    :type name: str
    :param log_path: path for saving logs
    :type log_path: str
    :return: a logger for logging messages
    :rtype: python logger
    """
    logger = logging.getLogger(name)
    if len(logger.handlers) != 0:
        print('reuse the same logger: {}'.format(name))
        return logger
    else:
        print('create new logger: {}'.format(name))
    fn = os.path.join(log_path, 'run-{}.log'.format(name))
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
    """ Load pre-trained embedding from file

    :param fi: embedding file name
    :type fi: str
    :param embed_name: embedding format, currently only supports "word2vec" format embedding. c.f.: https://radimrehurek.com/gensim/models/keyedvectors.html
    :type embed_name: str
    :return:

        - embedding : embedding file
        - index2word: map from element index to element
        - word2index: map from element to element index
        - vocab_size: size of element pool
        - embed_dim: embedding dimension

    :rtype: (gensim.KeyedVectors, list, dict, int, int)
    """
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
    """ Load raw data from file

    :param fi: data file name
    :type fi: str
    :return: a list of raw data from file
    :rtype: list
    """
    raw_data_strings = []
    with open(fi, "r") as fin:
        for line in fin:
            raw_data_strings.append(line.strip())
    return raw_data_strings


def print_args(args, interested_args="all"):
    """ Print arguments in command line

    :param args: parsed command line argument
    :type args: Namespace
    :param interested_args: a list of interested argument names
    :type interested_args: list
    :return: None
    :rtype: None
    """
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
    r""" Return the number of lines in the file without actually reading them into the memory. Used together with
    tqdm for tracking file reading progress.

    Usage:

    .. code-block:: python

        with open(inputFile, "r") as fin:
            for line in tqdm(fin, total=get\_num\_lines(inputFile)):
                ...

    :param file_path: path of input file
    :type file_path: str
    :return: number of lines in the file
    :rtype: int
    """
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
