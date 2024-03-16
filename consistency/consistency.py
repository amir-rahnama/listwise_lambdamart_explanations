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
from sklearn.metrics import ndcg_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
from listwise_exp import pmi_exp, rpi_exp, grad_q_exp, greedy_exp, random_q_exp
from ranklime import RankLIME

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Foo')
    parser.add_argument('--d', '-dataset', dest='dataset', required=True, help='Output file name')
    args = parser.parse_args()
    
    DATASET_NAME = args.dataset

    if not DATASET_NAME in ['mq2008', 'web10k', 'yahoo']:
        raise Exception('Dataset must be mq2008, web10k, yahoo')
    
    BASE_PATH = os.getcwd()
    path = f'../data/'
    background_dict = pickle.load( open( f"{path}/{DATASET_NAME}_background_v3.p", "rb" ) )
    test_sample = pickle.load( open( f"{path}/{DATASET_NAME}_test_sample_v2.p", "rb" ) )
    
    model_path = f'../models/lmart_{DATASET_NAME}_v2.txt'
    ranker = lightgbm.Booster(model_file=f'{model_path}')

    background = []
    for q in background_dict:
        background.extend(background_dict[q])
    background = np.array(background).reshape(-1, test_sample[list(test_sample.keys())[0]].shape[1])

    lirme = LIRME(background)

    np.random.seed(10)

    e_sample_sizes = [500, 1000, 1500, 2000, 3000, 4000, 5000]

    p_exps = {}
    pointwise_exp = ['lime', 'shap', 'lirme', 'exs_score', 'exs_top_k_binary', 'exs_top_k_rank']

    for sample_size in e_sample_sizes:
        for key in list(test_sample.keys())[:10]:
            print(key)
            p_exps[key] = {}
            
            for e in pointwise_exp:
                p_exps[key][e] = []
            
            q_docs = test_sample[key]
            pred_q_docs = ranker.predict(q_docs)
                
            for i, instance in enumerate(q_docs): 
                #print(i, instance)
                p_exps[key]['lime'].append(lime_exp(instance, ranker, background, sample_size=sample_size))
                p_exps[key]['shap'].append(shap_exp(instance, background, ranker, sample_size=sample_size))
        
                exp_lirme = lirme.explain(instance, ranker.predict, pred_q_docs, 
                            sur_type='ridge', label_type='regression', 
                            instance_idx=i, top_rank=5, sample_size=sample_size)
                p_exps[key]['lirme'].append(exp_lirme)
        
                exp_exs_top_k_rank = lirme.explain(instance, ranker.predict, pred_q_docs, 
                            sur_type='svm', label_type='top_k_rank', 
                            instance_idx=i, top_rank=5, sample_size=sample_size)
                p_exps[key]['exs_top_k_rank'].append(exp_exs_top_k_rank)
                
                exp_exs_top_k_binary = lirme.explain(instance, ranker.predict, pred_q_docs, 
                            sur_type='svm', label_type='top_k_binary', 
                            instance_idx=i, top_rank=5, sample_size=sample_size)
                p_exps[key]['exs_top_k_binary'].append(exp_exs_top_k_binary)
        
                exp_exs_score = lirme.explain(instance, ranker.predict, pred_q_docs, 
                            sur_type='svm', label_type='score', 
                            instance_idx=i, top_rank=5, sample_size=sample_size)
                p_exps[key]['exs_score'].append(exp_exs_score)

        pickle.dump(p_exps, open( f"./result/{DATASET_NAME}_consistency_pointwise_exps_{sample_size}.p", "wb" ) )

    listwise_exp = ['rank_lime']
    
    rlime = RankLIME(background)

    #listwise_sample_sizes = [100, 200, 400, 500, 700, 1000, 1500, 2000, 3000]
    listwise_sample_sizes = [400, 500, 700, 1000, 1500, 2000, 3000]

    for sample_size in listwise_sample_sizes:
        print(sample_size)
        l_exps = {}         
        for key in test_sample.keys():
            doc_values = test_sample[key]
            
            l_exps[key] = {}
            for e in listwise_exp:
                l_exps[key][e] = {}
            
            l_exps[key]['rank_lime'] = rlime.explain(doc_values, ranker.predict, [], sample_size=sample_size)
            
        pickle.dump(l_exps, open( f"./result/{DATASET_NAME}_consistency_listwise_exp_{sample_size}.p", "wb" ) )

