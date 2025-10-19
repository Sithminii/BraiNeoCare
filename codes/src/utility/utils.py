import numpy as np
import os
import random
import warnings

warnings.simplefilter("ignore", category=FutureWarning)


def get_ID(string):
    '''
    Extracts numerical part from a string
    '''
    string_list = list(string)
    idx=''
    for i in string_list:
        if i.isdigit():
            idx+=i
    return int(idx)



def prob_to_binary(array, threshold=0.5):
    '''
    Converts probabilities into binary values; 1 if prob >= threshold else 0

    inputs:
        array - array of probabilities
        threshold - threshold for converting a probability to a binary values, 1 or 0
    output:
        array of 1 and 0 s
    '''
 
    bin_array = (array >= threshold).astype(np.uint8)
    return bin_array



def compute_metrics(conf_mat):
    '''
    Computes accuracy, sensitivity, specificity and f1-score from a confusion matrix

    inputs:
        conf_mat - 2d confusion matrix
    output:
        accuracy, sensitivity, specificity and f1-score
    '''
    TN, FP, FN, TP = conf_mat

    acc = (TP+TN)/(TP+TN+FP+FN)
    sen = TP / (TP+FN)
    spec = TN/(TN+FP)
    f1_score = 2*TP/(2*TP + FN + FP)

    return acc, sen, spec, f1_score
