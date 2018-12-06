"""
.. module:: element_set
    :synopsis: data loader for element set 
 
.. moduleauthor:: Jiaming Shen
"""
import torch
import math
import numpy as np
import random


class ElementSet(object):
    """ Dataset Object

        :param name: dataset name
        :type name: str
        :param data_format: dataset format, either "set" or "sip"
        :type data_format: str
        :param options: dataset parameters, including two dicts mapping element to element index
        :type options: dict
        :param raw_data_strings: a list of strings representing an element set.

            - If data_format is "set", each string is of format "c0 {'d93', 'd377', 'd141', 'd63', 'd166'}".
            - If data_format is "sip", each string is of format "{'d93', 'd377'} d141 0".

        :type raw_data_strings: list
    """
    def __init__(self, name, data_format, options, raw_data_strings=None):

        self.name = name  
        self.data_format = data_format
        self.index2word = options["index2word"]
        self.word2index = options["word2index"]
        self.device = options["device"]
        self.vocab = []  # this vocab will contain only instances that appear in the above positive_set at least once
        self.max_set_size = -1  # the max_set_size in this dataset
        self.min_set_size = 1e8  # the min_set_size in this dataset
        self.avg_set_size = -1  # the avg_set_size in this dataset

        # a list of element sets
        self.positive_sets = []

        # a list of <set, instance> pairs
        self.sip_triplets = []
        self.pos_sip_cnt = -1
        self.neg_sip_cnt = -1

        # used to generate a collection of <set, instance> pair for evaluation in advance.
        self.NEG_SAMPLE_RATIO = 10  # for each positive (set, instance) pair, generate at most 10 negative pairs
        self.MAX_POS_SUB_SET_CNT = 500  # for each full set, generate at most 500 positive (set, instance) pairs

        if self.data_format == "set":
            self._initialize_set_format(raw_data_strings)
        elif self.data_format == "sip":
            self._initialize_sip_format(raw_data_strings)

        # for test set, generate sip triplets for evaluation of set-instance prediction, the negative sampling strategy,
        # negative sample size, and max set size are all prefixed
        if "test" in self.name and self.data_format == "set":
            self.sip_triplets, self.pos_sip_cnt, self.neg_sip_cnt = self._convert_set_format_to_sip_format(
                raw_sets=self.positive_sets, pos_strategy="vary_size_enumerate_with_full_set",
                neg_strategy="complete-random", neg_sample_size=10, max_set_size=50)

    def __repr__(self):
        return "<ElementSet {} (data_format = {}, vocab_size = {}, number of sets = {}, " \
               "max_set_size = {}, min_set_size = {}, avg_set_size = {}, number of set-instance pairs = {}, " \
               "positive pairs = {}, negative pairs = {})>".format(self.name, self.data_format, len(self.vocab),
                    len(self.positive_sets), self.max_set_size, self.min_set_size, self.avg_set_size,
                    len(self.sip_triplets),self.pos_sip_cnt, self.neg_sip_cnt)

    def __len__(self):
        if self.data_format == "set":
            return len(self.positive_sets)
        elif self.data_format == "sip":
            return len(self.sip_triplets)

    def _initialize_set_format(self, raw_set_strings):
        """Initialize  dataset from a collection of strings representing element sets

        :param raw_set_strings: a list of strings representing element sets
        :type raw_set_strings: list
        :return: None
        :rtype: None
        """
        set_size_sum = 0  # used to calculate self.avg_set_size
        for line in raw_set_strings:
            line = line.strip()
            eid, cls = line.split(" ", 1)
            cls = sorted(list(eval(cls)))  # sorting for reproducibility
            self.max_set_size = max(self.max_set_size, len(cls))
            self.min_set_size = min(self.min_set_size, len(cls))
            set_size_sum += len(cls)

            self.positive_sets.append(sorted([self.word2index[ele] for ele in cls]))  # sorting for reproducibility
            self.vocab.extend([self.word2index[ele] for ele in cls])

        self.avg_set_size = 1.0 * set_size_sum / len(self.positive_sets)
        self.vocab = sorted(list(set(self.vocab)))  # sorting for reproducibility

    def _initialize_sip_format(self, raw_set_instance_strings):
        """ Initialize dataset from a collection of strings representing <set, instance> pairs

        :param raw_set_instance_strings: a list of strings representing <set instance> pairs
        :type raw_set_instance_strings: list
        :return: None
        :rtype: None
        """
        for line in raw_set_instance_strings:
            line = line.strip()
            segs = line.split(" ")
            label = int(segs[-1])
            instance = self.word2index[segs[-2]]
            subset = sorted(list([self.word2index[ele] for ele in eval(" ".join(segs[:-2]))]))  # sorting for reproducibility

            self.sip_triplets.append((subset, instance, label))
            if label == 1:
                self.pos_sip_cnt += 1
            else:
                self.neg_sip_cnt += 1

            self.vocab.extend(subset)
            self.max_set_size = max(self.max_set_size, len(subset)+1)

        self.vocab = sorted(list(set(self.vocab)))  # sorting for reproducibility

    def get_train_batch(self, max_set_size=100, pos_sample_method="sample_size_random_set", neg_sample_size=1,
                        neg_sample_method="complete_random", batch_size=32):
        """ Generate one training batch of <set, instance> pairs

        :param max_set_size: maximum size of set S
        :type max_set_size: int
        :param pos_sample_method: name of positive sampling method
        :type pos_sample_method: str
        :param neg_sample_size: number of negative samples for each set
        :type neg_sample_size: int
        :param neg_sample_method: name of negative sampling method
        :type neg_sample_method: str
        :param batch_size: number of **sets** in one batch
        :type batch_size: int
        :return: a training batch containing "batch_size * (1+neg_sample_size)" <set, instance> pairs
        :rtype: dict
        """
        if self.data_format == "set":
            raw_sets = []
            for raw_set in self.positive_sets:
                raw_sets.append(raw_set)
                if len(raw_sets) % batch_size == 0:
                    sip_triplets = self._convert_set_format_to_sip_format(raw_sets=raw_sets,
                                                                          pos_strategy=pos_sample_method,
                                                                          neg_strategy=neg_sample_method,
                                                                          neg_sample_size=neg_sample_size,
                                                                          max_set_size=max_set_size)
                    batch_set = []
                    batch_inst = []
                    labels = []
                    for sip_triplet in sip_triplets:
                        batch_set.append(sip_triplet[0])
                        batch_inst.append(sip_triplet[1])
                        labels.append(sip_triplet[2])
                    batch = self._convert_sip_format_to_tensor(max_set_size, batch_set, batch_inst, labels)
                    yield batch
                    raw_sets = []

            # yield the last batch
            if len(raw_sets) != 0:
                sip_triplets = self._convert_set_format_to_sip_format(raw_sets=raw_sets, pos_strategy=pos_sample_method,
                                                                      neg_strategy=neg_sample_method,
                                                                      neg_sample_size=neg_sample_size,
                                                                      max_set_size=max_set_size)
                batch_set = []
                batch_inst = []
                labels = []
                for sip_triplet in sip_triplets:
                    batch_set.append(sip_triplet[0])
                    batch_inst.append(sip_triplet[1])
                    labels.append(sip_triplet[2])
                batch = self._convert_sip_format_to_tensor(max_set_size, batch_set, batch_inst, labels)
                yield batch

        elif self.data_format == "sip":
            batch_set = []
            batch_inst = []
            labels = []
            for sip_triplet in self.sip_triplets:
                batch_set.append(sip_triplet[0])
                batch_inst.append(sip_triplet[1])
                labels.append(sip_triplet[2])
                if len(batch_set) % (batch_size * (1+neg_sample_size)) == 0:
                    batch = self._convert_sip_format_to_tensor(max_set_size, batch_set, batch_inst, labels)
                    yield batch
                    batch_set = []
                    batch_inst = []
                    labels = []

            if len(batch_set) != 0:
                batch = self._convert_sip_format_to_tensor(max_set_size, batch_set, batch_inst, labels)
                yield batch

    def get_test_batch(self, max_set_size=5, batch_size=32):
        """ Generate one testing batch of <set, instance> pairs

        :param max_set_size: maximum size of set S
        :type max_set_size: int
        :param batch_size: number of **<set, instance> pairs** in one batch
        :type batch_size: int
        :return: a testing batch containing "batch_size" <set, instance> pairs
        :rtype: dict
        """
        batch_set = []
        batch_inst = []
        labels = []
        for idx, batch in enumerate(self.sip_triplets):
            batch_set.append(batch[0])
            batch_inst.append(batch[1])
            labels.append(batch[2])
            # convert to tensor, yield a batch, clean buffer
            if (idx+1) % batch_size == 0:
                res = self._convert_sip_format_to_tensor(max_set_size, batch_set, batch_inst, labels)
                yield res
                batch_set = []
                batch_inst = []
                labels = []

        # yield the last batch
        if (idx + 1) != len(self.sip_triplets):
            res = self._convert_sip_format_to_tensor(max_set_size, batch_set, batch_inst, labels)
            yield res

    def _shuffle(self):
        """ Shuffle dataset

        :return: None
        :rtype: None
        """
        if self.data_format == "set":
            random.shuffle(self.positive_sets)
        elif self.data_format == "sip":
            random.shuffle(self.sip_triplets)

    def _convert_set_format_to_sip_format(self, raw_sets, pos_strategy, neg_strategy, neg_sample_size=10,
                                          subset_size=5, max_set_size=50):
        """ Generate <set, instance> pairs (sip) from a collection of sets

        :param raw_sets: a list of sets
        :type raw_sets: list
        :param pos_strategy: name of positive sampling method
        :type pos_strategy: str
        :param neg_strategy: name of negative sampling method
        :type neg_strategy: str
        :param neg_sample_size: negative sampling ratio
        :type neg_sample_size: int
        :param subset_size: size of "set" in <set, instance> pairs, used only in "fix_size_repeat_set" pos_strategy
        :type subset_size: int
        :param max_set_size: maximum size of "set" in <set, instance> pairs, used only in "vary_size_enumerate" pos_strategy
        :type max_set_size: int
        :return: len(raw_sets) * (1 + neg_sample_size) sips, among which len(raw_sets) sips are positive and len(raw_sets) * neg_sample_size sips are negative
        :rtype: list

        Notes:

            - if pos_strategy is "sample_size_repeat_set", for each original set, we sample the size of "set" in sip, repeat this generated set neg_sample_size times, and pair them with each negative instance. This is the strategy to original AAAI submission.
            - if pos_strategy is "sample_size_random_set", for each original set, we sample one size of "set" in sip, and generate one set for each negative instance.
            - if pos_strategy is "fix_size_repeat_set", for each original set, we use pre-determined subset size to generate one "set" in sip, repeat this generated set neg_sample_size times, and pair them with each negative instance. This is the one used in cold-start training.
            - if pos_strategy is "vary_size_enumerate", for each original set and for each subset size less than max_set_size, we enumerate the original set and generate all possible sips. This is the one used for converting test_set in "set" format to "sip" format.
            - if pos_strategy is "vary_size_enumerate_with_full_set", it's basically same as the "vary_size_enumerate" strategy, except that it will also generate full set with only negative instances
            - if pos_strategy is "vary_size_enumerate_with_full_set_plus_group_id", it's basically same as the "vary_size_enumerate_with_full_set" strategy, expect that it will also return the group id of each sip the group id is this sip's corresponding raw set index
            - if pos_strategy is "enumerate", this is the one used for pre-generating sip triplets

        """
        if pos_strategy == "sample_size_repeat_set":
            batch_set = []
            batch_pos = []
            batch_fullset = []  # used to generate negative samples
            for raw_set in raw_sets:
                if len(raw_set) == 1:
                    batch_set.append(raw_set)
                    batch_pos.append(raw_set[0])
                    batch_fullset.append(raw_set)
                    continue

                raw_set_new = raw_set.copy()
                random.shuffle(raw_set_new)
                batch_fullset.append(raw_set_new)

                subset_size = random.randint(1, len(raw_set_new) - 1)
                subset = raw_set_new[:subset_size]
                pos_inst = raw_set_new[subset_size]
                batch_set.append(subset)
                batch_pos.append(pos_inst)

            # Randomly generate negative instances
            batch_neg = self._generate_negative_samples_within_pool(batch_fullset, neg_sample_size, remove_pos=True)

            # Convert to sip formats, notice here the subset is repeated (1+neg_sample_size) times
            sip_triplets = []
            for idx, subset in enumerate(batch_set):
                sip_triplets.append((subset, batch_pos[idx], 1))
                for neg_inst in batch_neg[idx]:
                    sip_triplets.append((subset, neg_inst, 0))

            return sip_triplets

        elif pos_strategy == "sample_size_random_set":
            batch_neg = self._generate_negative_samples_within_pool(raw_sets, neg_sample_size, remove_pos=True)
            # print("neg_sample_size: {}".format(neg_sample_size))
            # print("batch_neg:", batch_neg)
            batch_set = []
            batch_pos = []
            for raw_set in raw_sets:
                if len(raw_set) == 1:
                    batch_set.append([raw_set for _ in range(neg_sample_size+1)])
                    batch_pos.append(raw_set[0])
                    continue

                k_set = []
                raw_set_new = raw_set.copy()
                random.shuffle(raw_set_new)
                subset_size_range = min(len(raw_set_new), max_set_size)
                subset_size = random.randint(1, subset_size_range - 1)
                pos_inst = raw_set_new[0]
                start_idx = 1  # treat the first element as positive instance and skip it
                while len(k_set) != (1+neg_sample_size):
                    if start_idx + subset_size > len(raw_set_new):  # consume current pass, resample subset size
                        random.shuffle(raw_set_new)
                        subset_size = random.randint(1, subset_size_range - 1)
                        start_idx = 0
                    k_set.append(raw_set_new[start_idx: start_idx+subset_size])
                    start_idx += subset_size

                batch_set.append(k_set)
                batch_pos.append(pos_inst)

            # Convert to sip formats, notice here the subset is repeated (1+neg_sample_size) times
            sip_triplets = []
            for idx, subset_list in enumerate(batch_set):
                for idy, subset in enumerate(subset_list):
                    if idy == 0:
                        sip_triplets.append((subset, batch_pos[idx], 1))
                    else:
                        # print("idx: {}, idy:{}".format(idx, idy))
                        sip_triplets.append((subset, batch_neg[idx][idy-1], 0))

            return sip_triplets

        elif pos_strategy == "fix_size_repeat_set":
            batch_set = []
            batch_pos = []
            batch_fullset = []  # used to generate negative samples
            for raw_set in raw_sets:
                if len(raw_set) < subset_size+1:  # if we cannot sample a sip in which the "set" is of size subset_size
                    if len(raw_set) == 1:
                        batch_set.append(raw_set)
                        batch_pos.append(raw_set[0])
                        batch_fullset.append(raw_set)
                    else:
                        raw_set_new = raw_set.copy()
                        random.shuffle(raw_set_new)
                        batch_set.append(raw_set[1:])  # select maximum size of subset_size
                        batch_pos.append(raw_set[0])
                        batch_fullset.append(raw_set)
                    continue

                raw_set_new = raw_set.copy()
                random.shuffle(raw_set_new)
                batch_fullset.append(raw_set_new)

                # use given subset_size to generate sips
                subset = raw_set_new[:subset_size]
                pos_inst = raw_set_new[subset_size]
                batch_set.append(subset)
                batch_pos.append(pos_inst)

            # Randomly generate negative instances
            batch_neg = self._generate_negative_samples_within_pool(batch_fullset, neg_sample_size, remove_pos=True)

            # Convert to sip formats, notice here the subset is repeated (1+neg_sample_size) times
            sip_triplets = []
            for idx, subset in enumerate(batch_set):
                sip_triplets.append((subset, batch_pos[idx], 1))
                for neg_inst in batch_neg[idx]:
                    sip_triplets.append((subset, neg_inst, 0))

            return sip_triplets

        elif pos_strategy == "vary_size_enumerate":
            sip_triplets = []
            pos_sip_cnt_sum = 0
            neg_sip_cnt_sum = 0

            for subset_size in range(1, max_set_size+1):
                for raw_set in raw_sets:
                    if len(raw_set) < subset_size + 1:
                        continue

                    raw_set_new = raw_set.copy()
                    random.shuffle(raw_set_new)
                    batch_set = []
                    batch_pos = []
                    for _ in range(neg_sample_size+1):
                        for start_idx in range(0, len(raw_set_new)-subset_size, subset_size+1):
                            subset = raw_set_new[start_idx:start_idx+subset_size]
                            pos_inst = raw_set_new[start_idx+subset_size]
                            batch_set.append(subset)
                            batch_pos.append(pos_inst)
                        random.shuffle(raw_set_new)

                    pos_sip_cnt = int(len(batch_set) / (neg_sample_size+1))
                    pos_sip_cnt_sum += pos_sip_cnt
                    neg_sip_cnt = int(pos_sip_cnt * neg_sample_size)
                    neg_sip_cnt_sum += neg_sip_cnt

                    negative_pool = [ele for ele in self.vocab if ele not in raw_set]
                    sample_size = math.gcd(neg_sip_cnt, len(negative_pool))
                    sample_times = int(neg_sip_cnt / sample_size)

                    batch_neg = []
                    for _ in range(sample_times):
                        batch_neg.extend(random.sample(negative_pool, sample_size))

                    for idx, subset in enumerate(batch_set):
                        if idx < pos_sip_cnt:
                            pos = batch_pos[idx]
                            sip_triplets.append((subset, pos, 1))
                        else:
                            neg = batch_neg[idx-pos_sip_cnt]
                            sip_triplets.append((subset, neg, 0))

            return sip_triplets, pos_sip_cnt_sum, neg_sip_cnt_sum

        elif pos_strategy == "vary_size_enumerate_with_full_set":
            sip_triplets = []
            pos_sip_cnt_sum = 0
            neg_sip_cnt_sum = 0

            for subset_size in range(1, max_set_size+1):
                for raw_set in raw_sets:
                    if len(raw_set) < subset_size:
                        continue
                    raw_set_new = raw_set.copy()
                    random.shuffle(raw_set_new)
                    batch_set = []
                    batch_pos = []
                    if len(raw_set) == subset_size:  # put the entire full set
                        for _ in range(neg_sample_size+1):
                            batch_set.append(raw_set)
                            batch_pos.append(random.choice(raw_set))
                    else:
                        for _ in range(neg_sample_size+1):
                            for start_idx in range(0, len(raw_set_new)-subset_size, subset_size+1):
                                subset = raw_set_new[start_idx:start_idx+subset_size]
                                pos_inst = raw_set_new[start_idx+subset_size]
                                batch_set.append(subset)
                                batch_pos.append(pos_inst)
                            random.shuffle(raw_set_new)

                    pos_sip_cnt = int(len(batch_set) / (neg_sample_size+1))
                    pos_sip_cnt_sum += pos_sip_cnt
                    neg_sip_cnt = int(pos_sip_cnt * neg_sample_size)
                    neg_sip_cnt_sum += neg_sip_cnt

                    negative_pool = [ele for ele in self.vocab if ele not in raw_set]
                    sample_size = math.gcd(neg_sip_cnt, len(negative_pool))
                    sample_times = int(neg_sip_cnt / sample_size)

                    batch_neg = []
                    for _ in range(sample_times):
                        batch_neg.extend(random.sample(negative_pool, sample_size))

                    for idx, subset in enumerate(batch_set):
                        if idx < pos_sip_cnt:
                            pos = batch_pos[idx]
                            sip_triplets.append((subset, pos, 1))
                        else:
                            neg = batch_neg[idx-pos_sip_cnt]
                            sip_triplets.append((subset, neg, 0))

            return sip_triplets, pos_sip_cnt_sum, neg_sip_cnt_sum

        elif pos_strategy == "vary_size_enumerate_with_full_set_plus_group_id":
            sip_triplets = []
            pos_sip_cnt_sum = 0
            neg_sip_cnt_sum = 0
            groups = []

            for subset_size in range(1, max_set_size+1):
                for group_id, raw_set in enumerate(raw_sets):
                    if len(raw_set) < subset_size:
                        continue
                    raw_set_new = raw_set.copy()
                    random.shuffle(raw_set_new)
                    batch_set = []
                    batch_pos = []
                    if len(raw_set) == subset_size:  # put the entire full set
                        for _ in range(neg_sample_size+1):
                            batch_set.append(raw_set)
                            batch_pos.append(random.choice(raw_set))
                    else:
                        for _ in range(neg_sample_size+1):
                            for start_idx in range(0, len(raw_set_new)-subset_size, subset_size+1):
                                subset = raw_set_new[start_idx:start_idx+subset_size]
                                pos_inst = raw_set_new[start_idx+subset_size]
                                batch_set.append(subset)
                                batch_pos.append(pos_inst)
                            random.shuffle(raw_set_new)

                    pos_sip_cnt = int(len(batch_set) / (neg_sample_size+1))
                    pos_sip_cnt_sum += pos_sip_cnt
                    neg_sip_cnt = int(pos_sip_cnt * neg_sample_size)
                    neg_sip_cnt_sum += neg_sip_cnt

                    negative_pool = [ele for ele in self.vocab if ele not in raw_set]
                    sample_size = math.gcd(neg_sip_cnt, len(negative_pool))
                    sample_times = int(neg_sip_cnt / sample_size)

                    batch_neg = []
                    for _ in range(sample_times):
                        batch_neg.extend(random.sample(negative_pool, sample_size))

                    for idx, subset in enumerate(batch_set):
                        if idx < pos_sip_cnt:
                            pos = batch_pos[idx]
                            sip_triplets.append((subset, pos, 1))
                            groups.append(group_id)
                        else:
                            neg = batch_neg[idx-pos_sip_cnt]
                            sip_triplets.append((subset, neg, 0))
                            groups.append(group_id)

            return sip_triplets, pos_sip_cnt_sum, neg_sip_cnt_sum, groups

        elif pos_strategy == "enumerate":
            sip_triplets = []
            pos_sip_cnt_sum = 0
            neg_sip_cnt_sum = 0

            for r in range(1, max_set_size + 1):
                for positive_full_set in raw_sets:
                    if len(positive_full_set) < r + 1:  # unable to sample a set of size r
                        continue
                    negative_pool = [ele for ele in self.vocab if ele not in positive_full_set]

                    subsets = []  # cache subsets
                    pos_insts = []  # cache positive instance
                    for _ in range(neg_sample_size + 1):
                        for start_idx in range(0, len(positive_full_set) - r, (r + 1)):
                            subset = positive_full_set[start_idx:start_idx + r]
                            pos_inst = positive_full_set[start_idx + r]
                            subsets.append(subset)
                            pos_insts.append(pos_inst)
                        random.shuffle(positive_full_set)

                    pos_pairs_cnt = int(len(subsets) / (neg_sample_size + 1))
                    neg_pairs_cnt = int(pos_pairs_cnt * neg_sample_size)
                    pos_sip_cnt_sum += pos_pairs_cnt
                    neg_sip_cnt_sum += neg_pairs_cnt

                    sample_size = math.gcd(neg_pairs_cnt, len(negative_pool))
                    sample_times = int(neg_pairs_cnt / sample_size)
                    neg_insts = []
                    for _ in range(sample_times):
                        neg_insts.extend(random.sample(negative_pool, sample_size))

                    for idx, subset in enumerate(subsets):
                        if idx < pos_pairs_cnt:
                            pos = pos_insts[idx]
                            sip_triplets.append([subset, pos, 1])
                        else:
                            neg = neg_insts[idx - pos_pairs_cnt]
                            sip_triplets.append([subset, neg, 0])

            return sip_triplets, pos_sip_cnt_sum, neg_sip_cnt_sum

    def _convert_sip_format_to_tensor(self, max_set_size, batch_set, batch_inst, labels):
        """ Generate tensors for <set, instance> pairs

        :param max_set_size: maximum size of "set" in <set, instance> pairs
        :type max_set_size: int
        :param batch_set: a list of "sets" in <set, instance> pairs
        :type batch_set: list
        :param batch_inst: a list of "instances" in <set, instance> pairs
        :type batch_inst: list
        :param labels: a list of labels for each above <set, instance> pair
        :type labels: list
        :return: a dict of pytorch tensors representing <set, instance> pairs with their corresponding labels
        :rtype: dict
        """
        batch_size = len(batch_set)
        batch_set_tensor = np.zeros([batch_size, max_set_size], dtype=np.int)
        for row_id, row in enumerate(batch_set):
            if len(row) > max_set_size:
                batch_set_tensor[row_id][:] = row[:max_set_size]
            else:
                batch_set_tensor[row_id][:len(row)] = row

        batch_set_tensor = torch.from_numpy(batch_set_tensor)  # (batch_size, max_set_size)
        batch_inst_tensor = torch.tensor(batch_inst)  # (batch_size, )
        batch_inst_tensor.unsqueeze_(1)  # (batch_size, 1)

        label_tensor = torch.tensor(labels).unsqueeze(1)

        return {'set': batch_set_tensor.to(self.device), 'inst': batch_inst_tensor.to(self.device),
                'label': label_tensor.to(self.device)}

    def _generate_negative_samples_within_pool(self, positive_sets, neg_sample_size, remove_pos=True):
        """ Generate negative samples from vocabulary

        :param positive_sets: a list of positive sets
        :type positive_sets: list
        :param neg_sample_size: negative sampling ratio
        :type neg_sample_size: int
        :param remove_pos: whether to remove instances in positive sets from the vocabulary
        :type remove_pos: bool
        :return: a list of negative sets
        :rtype: list
        """
        batch_neg = []
        for positive_set in positive_sets:
            if remove_pos:
                sample_pool = [ele for ele in self.vocab if ele not in positive_set]
            else:
                sample_pool = self.vocab

            if neg_sample_size <= len(sample_pool):
                neg = random.sample(sample_pool, neg_sample_size)
            else:
                repeat_time = int(neg_sample_size / len(sample_pool))
                neg = sample_pool.copy() * repeat_time
                remaining_num = neg_sample_size - len(sample_pool) * repeat_time
                neg += random.sample(sample_pool, remaining_num)

            batch_neg.append(neg)
        return batch_neg
