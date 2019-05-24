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
        matching = [c in range(ce - fuzzy, ce + fuzzy) for c in cps]
        if any(matching):
            true_positives.append(ce)
        else:
            false_positives.append(ce)
        
    for cp in cps:
        matching = [cp in range(ce - fuzzy, ce + fuzzy) for ce in cp_estimation]
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
    