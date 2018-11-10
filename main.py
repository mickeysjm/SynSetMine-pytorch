from utils import save_model, load_model, myLogger, load_embedding, load_raw_data, Results, Metrics
from model import SSPM
from dataloader import element_set
from tensorboardX import SummaryWriter
from options import read_options
import numpy as np
import torch
import random
from tqdm import tqdm
import cluster_predict
import evaluator


def run(options, train_set, dev_set, mode="train", tb_writer=None, my_logger=None):
    """ Use options to construct one model. Train the model on train_set and test its performance on dev_set

    :param options:
    :param train_set:
    :param dev_set:
    :param mode: if train, verbosely print all logs, otherwise, be silent
    :param tb_writer: if not None, write results into TensorBoard. Note: if mode="train", must provide tb_writer
    :param logger: if not None, write information into logs. Note: if mode="train", must provide logger
    :return:
    """
    model = SSPM(options)
    model = model.to(options["device"])
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=options["lr"], amsgrad=True)
    results = Metrics()

    # Training phase
    train_set.shuffle()
    train_set_size = len(train_set)
    print("train_set_size: {}".format(train_set_size))

    model.train()
    early_stop_metric_name = "FMI"  # metric used for early stop
    best_early_stop_metric = 0.0
    last_step = 0
    save_model(model, options["save_dir"], 'best', 0)  # save the initial first model

    for epoch in tqdm(range(options["epochs"]), desc="Training ..."):
        loss = 0
        epoch_samples = 0
        epoch_tn = 0
        epoch_fp = 0
        epoch_fn = 0
        epoch_tp = 0
        for train_batch in train_set.get_train_batch(max_set_size=options["max_set_size"],
                                                     neg_sample_size=options["neg_sample_size"],
                                                     neg_sample_method=options["neg_sample_method"],
                                                     batch_size=options["batch_size"]):
            train_batch["data_format"] = "sip"
            optimizer.zero_grad()
            cur_loss, tn, fp, fn, tp = model.train_step(train_batch)
            optimizer.step()

            loss += cur_loss
            epoch_tn += tn
            epoch_fp += fp
            epoch_fn += fn
            epoch_tp += tp
            epoch_samples += (tn + fp + fn + tp)

            epoch_precision, epoch_recall, epoch_f1 = evaluator.calculate_precision_recall_f1(tp=epoch_tp, fp=epoch_fp,
                                                                                              fn=epoch_fn)
        epoch_accuracy = 1.0 * (epoch_tp + epoch_tn) / epoch_samples
        loss /= epoch_samples

        my_logger.info("    train/loss (per instance): {}".format(loss))
        my_logger.info("    train/precision: {}".format(epoch_precision))
        my_logger.info("    train/recall: {}".format(epoch_recall))
        my_logger.info("    train/accuracy: {}".format(epoch_accuracy))
        my_logger.info("    train/f1: {}".format(epoch_f1))
        tb_writer.add_scalar('train/loss (per instance)', loss, epoch)
        tb_writer.add_scalar('train/precision', epoch_precision, epoch)
        tb_writer.add_scalar('train/recall', epoch_recall, epoch)
        tb_writer.add_scalar('train/accuracy', epoch_accuracy, epoch)
        tb_writer.add_scalar('train/f1', epoch_f1, epoch)

        if epoch % options["eval_epoch_step"] == 0 and epoch != 0:
            # set-instance pair prediction evaluation
            metrics = evaluator.evaluate_set_instance_prediction(model, dev_set)
            tb_writer.add_scalar('val-sip/sip-precision', metrics["precision"], epoch)
            tb_writer.add_scalar('val-sip/sip-recall', metrics["recall"], epoch)
            tb_writer.add_scalar('val-sip/sip-f1', metrics["f1"], epoch)
            tb_writer.add_scalar('val-sip/sip-loss', metrics["loss"], epoch)
            my_logger.info("    val/sip-precision: {}".format(metrics["precision"]))
            my_logger.info("    val/sip-recall: {}".format(metrics["recall"]))
            my_logger.info("    val/sip-f1: {}".format(metrics["f1"]))
            my_logger.info("    val/sip-loss: {}".format(metrics["loss"]))

            # clustering evaluation
            vocab = dev_set.vocab
            cls_pred = cluster_predict.set_generation(model, vocab, size_opt_clus=options["size_opt_clus"],
                                                      max_K=options["max_K"])
            cls_true = dev_set.positive_sets
            metrics_cls = evaluator.evaluate_clustering(cls_pred, cls_true)

            tb_writer.add_scalar('val-cluster/ARI', metrics_cls["ARI"], epoch)
            tb_writer.add_scalar('val-cluster/FMI', metrics_cls["FMI"], epoch)
            tb_writer.add_scalar('val-cluster/NMI', metrics_cls["NMI"], epoch)
            tb_writer.add_scalar('val-cluster/em', metrics_cls["num_of_exact_set_prediction"], epoch)
            tb_writer.add_scalar('val-cluster/mwm_jaccard', metrics_cls["maximum_weighted_match_jaccard"], epoch)
            tb_writer.add_scalar('val-cluster/inst_precision', metrics_cls["pair_precision"], epoch)
            tb_writer.add_scalar('val-cluster/inst_recall', metrics_cls["pair_recall"], epoch)
            tb_writer.add_scalar('val-cluster/inst_f1', metrics_cls["pair_f1"], epoch)
            tb_writer.add_scalar('val-cluster/cluster_num', metrics_cls["num_of_predicted_clusters"], epoch)
            my_logger.info("    val/ARI: {}".format(metrics_cls["ARI"]))
            my_logger.info("    val/FMI: {}".format(metrics_cls["FMI"]))
            my_logger.info("    val/NMI: {}".format(metrics_cls["NMI"]))
            my_logger.info("    val/em: {}".format(metrics_cls["num_of_exact_set_prediction"]))
            my_logger.info("    val/mwm_jaccard: {}".format(metrics_cls["maximum_weighted_match_jaccard"]))
            my_logger.info("    val/inst_precision: {}".format(metrics_cls["pair_precision"]))
            my_logger.info("    val/inst_recall: {}".format(metrics_cls["pair_recall"]))
            my_logger.info("    val/inst_f1: {}".format(metrics_cls["pair_f1"]))
            my_logger.info("    val/cluster_num: {}".format(metrics_cls["num_of_predicted_clusters"]))
            my_logger.info("    val/clus_size2num_pred_clus: {}".format(metrics_cls["cluster_size2num_of_predicted_clusters"]))

            # Early stop based on clustering results
            if metrics_cls[early_stop_metric_name] > best_early_stop_metric:
                best_early_stop_metric = metrics_cls[early_stop_metric_name]
                last_step = epoch
                save_model(model, options["save_dir"], 'best', epoch)

            my_logger.info("-" * 80)

        if epoch - last_step > options["early_stop"]:
            print("Early stop by {} steps, best {}: {}, best step: {}".format(epoch, early_stop_metric_name,
                                                                              best_early_stop_metric, last_step))
            break

        train_set.shuffle()

    my_logger.info("Final Results:")
    my_logger.info("Loading model: {}/best_steps_{}.pt".format(options["save_dir"], last_step))
    load_model(model, options["save_dir"], 'best', last_step)
    model = model.to(options["device"])

    my_logger.info("=== Set-Instance Prediction Metrics ===")
    metrics = evaluator.evaluate_set_instance_prediction(model, dev_set)
    for metric in metrics:
        my_logger.info("    {}: {}".format(metric, metrics[metric]))

    my_logger.info("=== Clustering Metrics ===")
    vocab = dev_set.vocab
    cls_pred = cluster_predict.set_generation(model, vocab, size_opt_clus=options["size_opt_clus"],
                                              max_K=options["max_K"])
    cls_true = dev_set.positive_sets
    metrics_cls = evaluator.evaluate_clustering(cls_pred, cls_true)
    for metric in metrics_cls:
        if not isinstance(metrics_cls[metric], list):
            my_logger.info("    {}: {}".format(metric, metrics_cls[metric]))

    # save all metrics
    results.add("sip-f1", metrics["f1"])
    results.add("sip-precision", metrics["precision"])
    results.add("sip-recall", metrics["recall"])
    results.add("ARI", metrics_cls["ARI"])
    results.add("FMI", metrics_cls["FMI"])
    results.add("NMI", metrics_cls["NMI"])
    results.add("pred_clus_num", metrics_cls["num_of_predicted_clusters"])
    results.add("em", metrics_cls["num_of_exact_set_prediction"])
    results.add("mwm_jaccard", metrics_cls["maximum_weighted_match_jaccard"])
    results.add("inst-precision", metrics_cls["pair_precision"])
    results.add("inst-recall", metrics_cls["pair_recall"])
    results.add("inst-f1", metrics_cls["pair_f1"])

    return results


if __name__ == '__main__':
    args = read_options()

    # Add TensorBoard Writer
    writer = SummaryWriter(log_dir=None, comment=args.comment)

    # Add Python Logger
    logger = myLogger(name='exp', logpath=writer.file_writer.get_logdir())
    logger.setLevel(0)

    # Save Parameters in TensorBoard Writer and Logger
    options = vars(args)
    logger.info("Command Line Options: {}".format(options))
    writer.add_text('Text', 'Hyper-parameters: {}'.format(options), 0)

    # Initialize random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if args.device_id != -1:
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=9)
    torch.set_num_threads(1)

    # Load embedding files and word <-> index map
    fi = "./data/{}/combined.{}".format(options["dataset"], options["pretrained_embedding"])
    embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(fi)
    logger.info("Finish loading embedding: embed_dim = {}, vocab_size = {}".format(embed_dim, vocab_size))
    options["embedding"] = embedding
    options["index2word"] = index2word
    options["word2index"] = word2index
    options["vocabSize"] = vocab_size

    # Load train_set based on different data formats
    fi = "./data/{}/train-cold.{}".format(options["dataset"], options["data_format"])
    raw_data_string = load_raw_data(fi)
    random.shuffle(raw_data_string)
    train_set_full = element_set.ElementSet("train_set_full", options["data_format"], options, raw_data_string)
    print(train_set_full)

    # Load test_set, always in set format
    fi = "./data/{}/test.set".format(args.dataset)
    raw_data_string = load_raw_data(fi)
    random.shuffle(raw_data_string)
    test_set = element_set.ElementSet("test_set", "set", options, raw_data_string)
    print(test_set)

    # Model training
    if args.mode == "train":
        # Save results
        results = Results("./results/train_{}.txt".format(args.comment))

        # Model training on train_set_full and evaluation on test_set
        metrics = run(options, train_set_full, test_set, mode="train", tb_writer=writer, my_logger=logger)

        # save to result files
        interested_hyperparameters = ["modelName", "dataset", "data_format", "pretrained_embedding", "embedSize",
                                      "node_hiddenSize", "combine_hiddenSize", "batch_size", "neg_sample_size", "lr",
                                      "dropout", "early_stop", "random_seed", "save_dir"]
        hyperparameters = {}
        for hyperparameter_name in interested_hyperparameters:
            hyperparameters[hyperparameter_name] = options[hyperparameter_name]
        results.save_metrics(hyperparameters, metrics)

    # Model prediction
    elif args.mode == "cluster_predict":
        model = SSPM(options)
        model = model.to(options["device"])
        model_path = options["snapshot"]
        model.load_state_dict(torch.load(model_path))
        vocab = test_set.vocab
        random.shuffle(vocab)
        clusters = cluster_predict.set_generation(model, vocab, threshold=0.5, eid2ename=test_set.index2word)
        for cluster in clusters:
            print([test_set.index2word[ele] for ele in cluster])

        metrics = evaluator.evaluate_clustering(clusters, test_set.positive_sets)
        for ele in sorted(metrics.items()):
            print("{}\t{}".format(ele[0], ele[1]))
