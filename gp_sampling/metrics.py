#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def fuzzy_precall(cps, cp_estimation, fuzzy = 5):
    true_positives = []
    false_positives = []
    false_negatives = []
    
    for ce in cp_estimation:
        matching = [c in list(range(ce - fuzzy, ce + fuzzy)) for c in cps]
        if any(matching):
            true_positives.append(ce)
        else:
            false_positives.append(ce)
        
    for cp in cps:
        matching = [cp in list(range(ce - fuzzy, ce + fuzzy)) for ce in cp_estimation]
        if not any(matching):
            false_negatives.append(cp)
        
    if (len(true_positives) + len(false_positives)) == 0:
        precision = 1.0
    else:
        precision = len(true_positives)/(len(true_positives) + len(false_positives))
        
    if (len(true_positives) + len(false_negatives)) == 0:
        recall = 1.0
    else:
        recall = len(true_positives)/(len(true_positives) + len(false_negatives))
        
    return precision, recall
    
def a12(lst1,lst2,rev=True):
    """
    Python version of non-parametrwaic hypothesis testing using Vargha and Delaney's A12 statistic.

    Implementation borrowed from Tim Menzies at
    https://gist.github.com/timm/5630491

    :param lst1: sample of nums
    :param lst2: sample of nums
    :param rev:
    :return:
    """
    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if x==y:
                same += 1
            elif rev and x > y:
                more += 1
            elif not rev and x < y:
                more += 1
    return (more + 0.5*same) / (len(lst1)*len(lst2))
