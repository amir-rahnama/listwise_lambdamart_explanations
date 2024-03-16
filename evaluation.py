import numpy as np
from scipy.stats import rankdata
import pandas as pd
import pickle
import lightgbm
import os
from lime import lime_tabular
import sys
sys.path.append('..')
from lirme_v2 import LIRME
from sklearn.utils import shuffle
from scipy.stats import spearmanr, kendalltau
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('..')
from get_exp_new import lime_exp, shap_exp, random_exp, grad_exp
from greedy_score import Greedy
from sklearn.metrics import ndcg_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns

def min_max(v):
    return (v - v.min()) / (v.max() - v.min())


def get_fidelity_ltr(q_exps, doc_values, pred_fn):
    doc_values_p = doc_values.copy()
    prod = np.dot(doc_values, q_exps.flatten())
    output = pred_fn(doc_values_p)
    return kendalltau(prod, output).statistic


def validity_completeness(exp_val, doc_values, pred_fn, eval_key):
    instances_explained = doc_values.copy()
    base_pred_list = pred_fn(instances_explained)
    cutoff = np.linspace(0.05, 0.45, 10)
    
    total_feat = instances_explained.shape[1]
    all_pred_diff = []
    
    for cut in cutoff:
        top_k = int(np.floor(cut * total_feat))
        #print(cut, total_feat)
    
        if eval_key == 'completeness':
            feat_selected = np.abs(exp_val).argsort()[-top_k:][::-1]
        elif eval_key == 'validity': 
            feat_selected = np.abs(exp_val).argsort()[:top_k]
        else:
            raise Exception('choose either completeness or validity')
        #print('feat_selected', feat_selected)
        instances_explained[:, feat_selected] = np.mean(instances_explained[:, feat_selected], axis=0)
    
        new_pred_list = pred_fn(instances_explained)
        rank_diff = kendalltau(new_pred_list, base_pred_list).correlation
    
        base_rank_ = len(base_pred_list) - rankdata(base_pred_list).astype(int)
        new_rank_ = len(new_pred_list) - rankdata(new_pred_list).astype(int)
    
        all_pred_diff.append(np.mean(np.abs(new_rank_ - base_rank_)) / instances_explained.shape[0])
    
    return all_pred_diff


def get_infidelity(q_exps, doc_values, pred_fn, background, top_k_percent=0.1):
    top_k = int(top_k_percent * background.shape[1])

    doc_values_p = doc_values.copy()

    q_exp_val = q_exps.flatten()
    top_features =  np.argsort(np.abs(q_exp_val))[-top_k:][::-1]
    doc_values_p[:, top_features]  =  np.mean(background[:, top_features], axis=0)

    prod = np.dot(doc_values_p, q_exp_val)
    
    output_0 = pred_fn(doc_values)
    output_1 = pred_fn(doc_values_p)
    output = np.abs(output_1 - output_0)
    
    return kendalltau(prod, output).statistic

def get_explain_ndcg(q_exps, doc_values, pred_fn):
        doc_values_p = doc_values.copy()
        q_exp_val = q_exps.flatten()
        
        prod = np.dot(doc_values, q_exp_val)        
        output = min_max(pred_fn(doc_values_p))
        
        return ndcg_score([output], [prod], k=10)

def get_dpff(q_exps, doc_values, model):
    dpff_val = model.feature_importance('split')
    output = kendalltau(np.abs(q_exps), dpff_val).statistic
    return output

def get_auc(exp):
    cutoffs = np.linspace(0.05, 0.45, 10)
    auc = {}
    temp = np.array(exp).mean(axis=1)
    auc_ = 0
    for k in range(1, len(cutoffs) - 1):
        x = cutoffs[k] - cutoffs[k - 1]
        y = temp[k] + temp[k-1]
        auc_ += y / ( 2 * x)
    return auc_

def summarize(eval):
    eval_summary = {}
    for measure in eval.keys():
        eval_summary[measure] = {}
        for exp_name in eval[measure].keys():
            eval_summary[measure][exp_name] = {}
            if measure in ['validity', 'completeness']: 
                eval_summary[measure][exp_name] = get_auc(eval[measure][exp_name])
            else: 
                eval_summary[measure][exp_name] = np.nanmean(eval[measure][exp_name]) 
    return eval_summary