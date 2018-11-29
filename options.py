import argparse
import os
import torch
from utils import print_args
from datetime import datetime


def read_options():
    parser = argparse.ArgumentParser(description='supervised clustering')

    # Data parameters
    parser.add_argument('-dataset', default="NYT", type=str, help='name of the dataset')
    parser.add_argument('-data-format', default="set", type=str, choices=['set', 'sip'],
                        help='format of input training dataset [Default: set]')

    # Running mode
    parser.add_argument('-mode', default='train', type=str, choices=['train', 'eval', 'cluster_predict'],
                        help='specify model running mode, \'train\' for model training, \'eval\' for model evaluation,'
                             'and \'cluster_predict\' for model-based clustering. The later two require a trained model'
                             '[Default: train]')

    # Model parameters
    parser.add_argument('-modelName', default='np_lrlr_sd_lrlrdl', type=str,
                        help='which prediction model is used')
    parser.add_argument('-pretrained-embedding', default='embed', type=str,
                        choices=['none', 'embed', 'tfidf', 'fastText-no-subword.embed', 'fastText-with-subword.embed'],
                        help='whether to use pretrained embedding, none means training embedding from scratch')
    parser.add_argument('-embed-fine-tune', default=0, type=int,
                        help='fine tune word embedding or not, 0 means no fine tune')
    parser.add_argument('-embedSize', default=50, type=int, help='element embed size')
    parser.add_argument('-node-hiddenSize', default=250, type=int, help='hidden size used in node post_embedder')
    parser.add_argument('-combine-hiddenSize', default=500, type=int, help='hidden size used in combiner')
    parser.add_argument('-max-set-size', default=50, type=int, help='maximum size for training batch')

    # Learning options
    parser.add_argument('-batch-size', default=32, type=int, help='batch size for training')
    parser.add_argument('-lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('-loss-fn', default="self_margin_rank_bce", type=str,
                        choices=['cross_entropy', 'max_margin', 'margin_rank', 'self_margin_rank',
                                 'self_margin_rank_bce'], help='loss function used in training model')
    parser.add_argument('-margin', default=0.5, type=float, help='margin used in max_margin loss and margin_rank loss')
    parser.add_argument('-epochs', default=2000, type=int, help='number of epochs for training')
    parser.add_argument('-neg-sample-size', default=20, type=int,
                        help='number of negative samples generated for each set')
    parser.add_argument('-neg-sample-method', default="complete_random", type=str,
                        choices=["complete_random", "share_token", "mixture"], help='negative sampling method')

    # Regularization parameters
    parser.add_argument('-dropout', default=0.3, type=float, help='Dropout between layers')
    parser.add_argument('-early-stop', default=100, type=int, help='early stop epoch number')
    parser.add_argument('-eval-epoch-step', default=5, type=int, help='average number of epochs for evaluation')
    parser.add_argument('-random-seed', default=5417, type=int,
                        help='random seed used for model initialization and negative sample generation')
    parser.add_argument('-size-opt-clus', default=0, type=int,
                        help='whether conduct size optimized clustering prediction'
                             'this saves GPU memory sizes but consumes more training time)'
                             'when expecting a large number of small sets, set this option to be 0;'
                             'when expecting a small number of huge sets, set this option to be 1')
    parser.add_argument('-max-K', default=-1, type=int, help='maximum cluster number, -1 means auto-infer')
    parser.add_argument('-T', default=1.0, type=int, help='temperature scaling, 1.0 means no scaling')

    # Device options
    parser.add_argument('-device-id', default=0, type=int, help='device to use for iterate data, -1 means cpu')

    # Model saving/loading options
    parser.add_argument('-save-dir', default="./snapshots/", type=str, help='location to save models')
    parser.add_argument('-load-model', default="", type=str, help='path to loaded model')
    parser.add_argument('-snapshot', default="", type=str, help='path to model snapshot')
    parser.add_argument('-tune-result-file', default="tune_prefix", type=str, help='path to save all tuning results')

    # Other options
    parser.add_argument('-remark', default='', help='reminder of this run')

    try:
        args = parser.parse_args()
        print_args(args)
    except:
        parser.error("Unable to parse arguments")

    # Update Device information
    if args.device_id == -1:
        args.device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        args.device = torch.device("cuda:0")

    # Update Tensorboard logging
    if args.mode == "train":
        args.comment = '_{}'.format(args.remark)
    elif args.mode == "tune":
        args.comment = "_{}".format(args.tune_result_file)
    else:
        args.comment = ""

    # Model snapshot saving
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.mode == "train":
        args.save_dir = os.path.join(args.save_dir, current_time + '_' + args.remark)
    elif args.mode == "tune":
        args.save_dir = os.path.join(args.save_dir, current_time + '_' + args.tune_result_file)

    if args.max_K == -1:
        args.max_K = None

    args.size_opt_clus = (args.size_opt_clus == 1)

    return args


