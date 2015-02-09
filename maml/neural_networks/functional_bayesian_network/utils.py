__author__ = 'epyzerknapp'

import numpy as np


def normalize_targets(targets):
    """
    This utility function normalizes the targets to have zero mean and unit standard deviation
    :param targets: The targets to be normalized
    :return: normed_targets : a numpy array of the normalized targets
    :return: stdevs : a numpy array of the standard deviations of the targets
    :return: means : a numpy array of the means of the targets
    """

    means = np.mean(Y,axis=0)
    stds = np.std(Y,axis=0)
    normed_targets = (targets - means) / stds
    return normed_targets, stds, means

def normalize_inputs(inputs):
    """
    This utility function normalizes the inputs to have zero mean and unit standard deviation
    :param inputs: The inputs to be normalized
    :return: normed_inputs : a numpy array of the normalized targets
    :return: stdevs : a numpy array of the standard deviations of the targets
    :return: means : a numpy array of the means of the targets
    """
    stds = np.std(inputs, 0)
    stds[stds == 0] = 1
    means = np.mean(inputs, 0)
    normed_inputs = (inputs - np.full(inputs.shape, means)) / np.full(inputs.shape, stds)