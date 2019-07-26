import numpy as np


def recall(targets, probabilities, thre=0.5):
    """

    :param probabilities: numpy array, values from 0 to 1
    :param targets: numpy array, values 0 or 1
    :param thre: float number value from 0 to 1
    :return: float, value from 0 to 1
    """
    condition_positive = len(targets[targets == 1])
    mask = np.array(probabilities >= thre)
    true_positive = len(targets[mask][targets[mask] == 1])
    return true_positive / condition_positive


def fall_out(targets, probabilities,  thre=0.5):
    """

    :param probabilities: numpy array, values from 0 to 1
    :param targets: numpy array, values 0 or 1
    :param thre: float number value from 0 to 1
    :return: float, value from 0 to 1
    """
    condition_negative = len(targets[targets == 0])
    mask = np.array(probabilities >= thre)
    false_positive = len(targets[mask][targets[mask] == 0])
    return false_positive / condition_negative


def precision(targets, probabilities, thre=0.5):
    """

    :param probabilities: numpy array, values from 0 to 1
    :param targets: numpy array, values 0 or 1
    :param thre: float number value from 0 to 1
    :return: float, value from 0 to 1
    """
    mask = np.array(probabilities >= thre)
    true_positive = len(targets[mask][targets[mask] == 1])
    pred_condition_positive = len(probabilities[probabilities >= thre])
    return true_positive / pred_condition_positive


def my_f1_score(targets, probabilities, thresh=0.5):
    prec = precision(targets, probabilities, thre=thresh)
    rec = recall(targets, probabilities, thre=thresh)
    return 2 * prec * rec / (prec + rec)
