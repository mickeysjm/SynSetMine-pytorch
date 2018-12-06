import torch
import torch.nn as nn
from math import floor


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, in_tensor):
        return in_tensor


# module to split at a given point and sum
class SplitSum(nn.Module):
    def __init__(self, splitSize):
        super(SplitSum, self).__init__()
        self.splitSize = splitSize

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstHalf = inTensor[:, :self.splitSize]
            secondHalf = inTensor[:, self.splitSize:]
        else:
            firstHalf = inTensor[:, :, :self.splitSize]
            secondHalf = inTensor[:, :, self.splitSize:]
        return firstHalf + secondHalf


# module to split at a given point and max
class SplitMax(nn.Module):
    def __init__(self, splitSize):
        super(SplitMax, self).__init__()
        self.splitSize = splitSize  # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstHalf = inTensor[:, :self.splitSize]
            secondHalf = inTensor[:, self.splitSize:]
        else:
            firstHalf = inTensor[:, :, :self.splitSize]
            secondHalf = inTensor[:, :, self.splitSize:]
        numDims = firstHalf.dim()
        concat = torch.cat((firstHalf.unsqueeze(numDims), secondHalf.unsqueeze(numDims)), numDims)
        maxPool = torch.max(concat, numDims)[0]
        return maxPool


# module to split at two given points, sum the first and the third parts, and then concat with second part
class SplitSumConcat(nn.Module):
    def __init__(self, splitSize1, splitSize2):
        super(SplitSumConcat, self).__init__()
        self.splitSize1 = splitSize1
        self.splitSize2 = splitSize2

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstPart = inTensor[:, :self.splitSize1]
            secondPart = inTensor[:, self.splitSize1:(self.splitSize1 + self.splitSize2)]
            thirdPart = inTensor[:, (self.splitSize1 + self.splitSize2):]
        else:
            firstPart = inTensor[:, :, :self.splitSize1]
            secondPart = inTensor[:, :, self.splitSize1:(self.splitSize1 + self.splitSize2)]
            thirdPart = inTensor[:, :, (self.splitSize1 + self.splitSize2):]

        return torch.cat([firstPart + thirdPart, secondPart], dim=-1)


# module to split at two given points, max the first and the third parts, and then concat with second part
class SplitMaxConcat(nn.Module):
    def __init__(self, splitSize1, splitSize2):
        super(SplitMaxConcat, self).__init__()
        self.splitSize1 = splitSize1
        self.splitSize2 = splitSize2

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstPart = inTensor[:, :self.splitSize1]
            secondPart = inTensor[:, self.splitSize1:(self.splitSize1 + self.splitSize2)]
            thirdPart = inTensor[:, (self.splitSize1 + self.splitSize2):]
        else:
            firstPart = inTensor[:, :, :self.splitSize1]
            secondPart = inTensor[:, :, self.splitSize1:(self.splitSize1 + self.splitSize2)]
            thirdPart = inTensor[:, :, (self.splitSize1 + self.splitSize2):]
        numDims = firstPart.dim()
        concat = torch.cat((firstPart.unsqueeze(numDims), thirdPart.unsqueeze(numDims)), numDims)
        maxPool = torch.max(concat, numDims)[0]

        return torch.cat([maxPool, secondPart], dim=-1)


def select_model(params):
    """ Select model architecture based on params

    :param params: a dictionary contains model architecture name and model sizes
    :type params: dict
    :return: a dictionary contains all model components
    :rtype: dict
    """
    node_embedder = nn.Embedding(params['vocabSize']+1, params['embedSize'])  # in paper, referred as "Embedding Layer"
    node_postEmbedder = Identity()  # in the paper, referred as "Embedding Transformer"
    node_pooler= torch.sum  # in the paper, this is fixed to be "sum", but you can replace it with mean/max/min function
    scorer = Identity()  # in the paper, this is referred as "Post Transformer"

    # following four modules are not discussed and used in paper
    edge_embedder = Identity()
    edge_postEmbedder = Identity()
    edge_pooler = torch.sum
    combiner = Identity()

    if params['modelName'] == 'np_lrlr_concat_lrldrl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            nn.Linear(2*params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize']/2)),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize']/2), 2)
        )
    elif params['modelName'] == 'np_lrlr_sum_lrldrl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitSum(params['node_hiddenSize']),
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize']/2)),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize']/2), 2)
        )
    elif params['modelName'] == 'np_lrlr_max_lrldrl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitMax(params['node_hiddenSize']),
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize']/2)),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize']/2), 2)
        )
    elif params['modelName'] == 'np_lrlr_sum_lrldrls':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        combiner = nn.Sequential(
            SplitSum(params['node_hiddenSize']),
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1, bias=False),
            nn.Sigmoid()
        )
    elif params['modelName'] == 'np_lrlr_max_lrldrls':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        combiner = nn.Sequential(
            SplitMax(params['node_hiddenSize']),
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
            nn.Sigmoid()
        )
    elif params['modelName'] == 'np_lrlr_sd_lrlrdl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        scorer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
        )
    elif params['modelName'] == 'np_lrlr_msd_lrlrdl':
        node_pooler = torch.mean

        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        scorer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
        )
    elif params['modelName'] == 'np_szlrlr_sd_lrlrdl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize']+1, params['embedSize']+1, bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize']+1, params['node_hiddenSize']),
            nn.ReLU()
        )

        scorer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
        )
    elif params['modelName'] == 'np_lrlr_sd_szlrlrdl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        scorer = nn.Sequential(
            nn.Linear(params['node_hiddenSize']+1, params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
        )
    elif params['modelName'] == 'np_id_sd_lrlrdl':  # no embedding transformation
        node_postEmbedder = Identity()
        scorer = nn.Sequential(
            nn.Linear(params['embedSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
        )
    elif params['modelName'] == 'np_id_sd_lrdl':  # no embedding transformation
        node_postEmbedder = Identity()
        scorer = nn.Sequential(
            nn.Linear(params['embedSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(params['combine_hiddenSize'], 1),
        )
    elif params['modelName'] == 'np_lrlr_sd_dl':  # no combining layer
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        scorer = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.Linear(params['node_hiddenSize'], 1),
        )
    elif params['modelName'] == 'np_lrlr_sd_lrlrdls':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        scorer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
            nn.Sigmoid()
        )
    elif params['modelName'] == 'np_lrlr_sd_lrlrdlt':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        scorer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1, bias=False),
            nn.Tanh()
        )
    elif params['modelName'] == 'np_ltlt_sd_lrlrdlt':
        print("Using model: {}".format(params['modelName']))
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.Tanh(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.Tanh()
        )

        scorer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1, bias=False),
            nn.Tanh()
        )
    elif params['modelName'] == 'np_lsls_sd_lrlrdlt':
        print("Using model: {}".format(params['modelName']))
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.Sigmoid(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.Sigmoid()
        )

        scorer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1, bias=False),
            nn.Tanh()
        )
    elif params['modelName'] == 'np_none_sd_lrlrdlt':
        scorer = nn.Sequential(
            nn.Linear(params['embedSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']), nn.Linear(floor(params['combine_hiddenSize'] / 2), 1, bias=False),
            nn.Tanh()
        )
    elif params['modelName'] == 'np_lrlr_sd_lrlrlrdrl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        scorer = nn.Sequential(
            nn.Linear(params['node_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), floor(params['combine_hiddenSize'] / 4)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize'] / 4), 1),
        )
    elif params['modelName'] == 'lrlr-dlrlr_sum_lrldrl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['edge_hiddenSize'], params['edge_hiddenSize']),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitSumConcat(params['node_hiddenSize'], params['edge_hiddenSize']),
            nn.Linear(params['node_hiddenSize'] + params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 2),
        )
    elif params['modelName'] == 'lrlr-dlrlr_max_lrldrl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['edge_hiddenSize'], params['edge_hiddenSize']),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitMaxConcat(params['node_hiddenSize'], params['edge_hiddenSize']),
            nn.Linear(params['node_hiddenSize'] + params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 2),
        )
    elif params['modelName'] == 'lrlr-dlrlr_sum_lrldrls':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['edge_hiddenSize'], params['edge_hiddenSize']),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitSumConcat(params['node_hiddenSize'], params['edge_hiddenSize']),
            nn.Linear(params['node_hiddenSize'] + params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
            nn.Sigmoid()
        )
    elif params['modelName'] == 'lrlr-dlrlr_max_lrldrls':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['embedSize'], bias=False),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['edge_hiddenSize'], params['edge_hiddenSize']),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitMaxConcat(params['node_hiddenSize'], params['edge_hiddenSize']),
            nn.Linear(params['node_hiddenSize'] + params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize'] / 2)),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize'] / 2), 1),
            nn.Sigmoid()
        )
    elif params['modelName'] == 'drlr-drlr_concat_ldrl':
        node_postEmbedder = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU()
        )

        combiner = nn.Sequential(
            nn.Linear(2*params['node_hiddenSize']+params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.Dropout(params['dropout']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], 2)
        )
    elif params['modelName'] == "lr-lr_concat_lrdl":
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU()
        )

        combiner = nn.Sequential(
            nn.Linear(2 * params['node_hiddenSize'] + params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(params['combine_hiddenSize'], 2)
        )
    elif params['modelName'] == "lr-lr_concat_lrlrdl":
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU()
        )

        combiner = nn.Sequential(
            nn.Linear(2 * params['node_hiddenSize'] + params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize']/2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize']/2), 2)
        )
    elif params['modelName'] == "lr-lr_concat_ltltdl":
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU()
        )

        combiner = nn.Sequential(
            nn.Linear(2 * params['node_hiddenSize'] + params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.Tanh(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize']/2)),
            nn.Tanh(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize']/2), 2)
        )
    elif params['modelName'] == "lr-lr_concat_lrlrlrdl":
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU()
        )

        combiner = nn.Sequential(
            nn.Linear(2 * params['node_hiddenSize'] + params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize']/2)),
            nn.ReLU(),
            nn.Linear(floor(params['combine_hiddenSize']/2), floor(params['combine_hiddenSize']/4)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize']/4), 2),
        )
    elif params['modelName'] == "lr-lr_concat_ltltltdl":
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU()
        )

        edge_postEmbedder = nn.Sequential(
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU()
        )

        combiner = nn.Sequential(
            nn.Linear(2 * params['node_hiddenSize'] + params['edge_hiddenSize'], params['combine_hiddenSize']),
            nn.Tanh(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize']/2)),
            nn.Tanh(),
            nn.Linear(floor(params['combine_hiddenSize']/2), floor(params['combine_hiddenSize']/4)),
            nn.Tanh(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize']/4), 2),
        )
    elif params['modelName'] == 'lrlr-lrlr_concat_lrlrdl':
        node_postEmbedder = nn.Sequential(
            nn.Linear(params['embedSize'], params['node_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['node_hiddenSize'], floor(params['node_hiddenSize']/2)),
            nn.ReLU(),
        )

        edge_postEmbedder = nn.Sequential(
            nn.Linear(params['string_pair_feature_dimension'], params['edge_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['edge_hiddenSize'], floor(params['edge_hiddenSize']/2)),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            nn.Linear(2 * floor(params['node_hiddenSize'] / 2) + floor(params['edge_hiddenSize'] / 2),
                      params['combine_hiddenSize']),
            nn.ReLU(),
            nn.Linear(params['combine_hiddenSize'], floor(params['combine_hiddenSize']/2)),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(floor(params['combine_hiddenSize']/2), 2)
        )

    content = {'node_embedder': node_embedder, 'node_postEmbedder': node_postEmbedder, 'node_pooler': node_pooler,
               'edge_embedder': edge_embedder, 'edge_postEmbedder': edge_postEmbedder, 'edge_pooler': edge_pooler,
               'combiner': combiner, 'scorer': scorer}

    return content
