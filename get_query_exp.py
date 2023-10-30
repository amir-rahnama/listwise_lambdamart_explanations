import pickle
from scipy import stats
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances 
from sklearn.metrics import roc_auc_score, jaccard_score, ndcg_score
from scipy.spatial.distance import dice
import sys
#sys.path.append('../../..')
#sys.path.append('..')
sys.path.append('.')
sys.path.append('./exp/')
from data.get_data import get_data

import lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import  pandas as pd
from scipy.stats import spearmanr
import sklearn
#from query_decision_path import get_dp_query 
from sklearn.preprocessing import normalize

np.random.seed(10)


def grad_exp(instances, model):
    epsilon = 0.01
    total_instances = instances.shape[0]
    grad = np.zeros(instances.shape[1])
    
    for j in range(instances.shape[1]):
        h = np.zeros(instances.shape[1])
        h[j] = epsilon
        df_en1 = model.predict((instances + h).reshape(total_instances, -1))[0] 
        df_en2 =  model.predict((instances - h).reshape(total_instances, -1))[0]
        grad[j] = df_en1 - df_en2/ 2 * epsilon
    
    return grad #/ np.sum(grad)
    
def rpi_exp(instances, model, trials=3):
    fi = np.zeros(instances.shape[1])
    
    for j in range(instances.shape[1]):
        instances_copy = instances.copy()
        base_pred = ranker.predict(instances_copy)
        sps = []
        
        for i in range(trials):
            instances_copy[:, j] = np.random.shuffle(instances_copy[:,j])
            new_pred = ranker.predict(instances_copy)
            sps.append(spearmanr(new_pred, base_pred).correlation)
        fi[j] = np.nanmean(sps)
    
    return fi / np.sum(fi)


def pmi_exp(instances, model, trials=3):
    fi = np.zeros(instances.shape[1])
    
    for j in range(instances.shape[1]):
        instances_copy = instances.copy()
        base_pred = ranker.predict(instances_copy)
        sps = []
        
        for i in range(trials):
            instances_copy[:, j] = np.random.shuffle(instances_copy[:,j])
            new_pred = ranker.predict(instances_copy)
            sps.append(np.mean(np.abs(new_pred - base_pred)))
        
        fi[j] = np.mean(sps)
    
    fi /= np.sum(fi)
    
    return fi

def binarize_top(exps, type_importance='most'):
    top_k = int(np.floor(0.2 * exps.shape[1]))
    
    binary_exp = np.zeros((exps.shape[0], exps.shape[1]))
    for i in range(len(exps)): 
        if type_importance == 'most':
            ranked_first = np.argsort(np.abs(exps[i]))[-top_k:][::-1] 
        elif type_importance == 'least': 
            ranked_first = np.argsort(np.abs(exps[i]))[:top_k]
        else: 
            raise
        binary_exp[i, ranked_first] = 1
    return binary_exp

sub_keys = ['lime', 'shap', 'lirme', 'exs_v1', 'exs_v2', 'random']

exps_yahoo = pickle.load(open( "./exp/save/exps_yahoo_200_1675497465.696698.p", "rb" ) )
exps_web10k = pickle.load(open( "./exp/save/exps_web10k_200_1675464327.034055.p", "rb" ) )
exps_mq2008 = pickle.load(open( "./exp/save/exps_mq2008_200_1675505826.393946.p", "rb" ) )
exps_all = [exps_mq2008, exps_web10k, exps_yahoo]
dataset_names = ['mq2008', 'web10k', 'yahoo']

def get_queries(dataset_name):
    if dataset_name == 'mq2008':
        quries_selected = ['qid:14717', 'qid:11069', 'qid:19782', 'qid:19875', 'qid:19503',
                           'qid:15956', 'qid:18411', 'qid:18995', 'qid:14427', 'qid:15860',
                           'qid:10943', 'qid:13120', 'qid:10032', 'qid:15860', 'qid:13083',
                           'qid:19503', 'qid:15251', 'qid:11639', 'qid:10915', 'qid:19353',
                           'qid:10943', 'qid:15168', 'qid:15251', 'qid:14805', 'qid:11437',
                           'qid:18995', 'qid:16454', 'qid:12237', 'qid:13388', 'qid:14322']
    elif dataset_name == 'web10k':
        quries_selected = ['qid:11653', 'qid:7093', 'qid:13093', 
                           'qid:10558', 'qid:6523','qid:12718', 'qid:22318', 
                           'qid:17848', 'qid:6148', 'qid:24088']
    elif dataset_name == 'yahoo':
        quries_selected = [24085., 25140., 25870., 28031., 28974., 26276., 29725., 28617.,
                               24116., 25865., 29759., 27204., 28587., 27549., 23228.]
    return quries_selected

#Fixing random problem
for i in range(len(exps_all)):
    random_exps = exps_all[i]['random']
    for j in range(len(random_exps)):
        exps_all[i]['random'][j] =  exps_all[i]['random'][j].flatten()
        
'''  
all_idx_list = []

for j in range(len(dataset_names)):
        d_name = dataset_names[j]
        print('getting all_idx_list for ', d_name)

        X_train, y_train, qids_train, X_valid, y_valid, qids_valid, all_feat_names, test_q_info = get_data(d_name, training=False)
        
        total_features = X_valid.shape[1]
        
        quries_selected = get_queries(d_name)

        doc_per_query = []
        for i in range(len(quries_selected)):
            q_sel = quries_selected[i]
            doc_per_query.append(len(np.argwhere(test_q_info == q_sel).flatten()))
        doc_per_query = np.array(doc_per_query)

        idx_list = np.zeros(len(doc_per_query))
        
        for i in range(len(doc_per_query)):
            idx_list[i] = idx_list[i-1] + doc_per_query[i]
        idx_list = np.array(idx_list)
        idx_list = np.insert(idx_list, 0, 0).astype(int)
    
        all_idx_list.append(idx_list)

     
all_instances = {}
all_q_explanations = {}
all_scores = {}
        
for j in range(len(dataset_names)):
    ranker = lightgbm.Booster(model_file='./model/save/lmart_{}.txt'.format(dataset_names[j]))
    d_name = dataset_names[j]
    all_instances[d_name] = []
    all_q_explanations[d_name] = {}
    all_scores[d_name] = []

    X_train, y_train, qids_train, X_valid, y_valid, qids_valid, all_feat_names, test_q_info = get_data(d_name, training=False)
    
    idx_list = all_idx_list[j]
    
    for i in range(1, len(idx_list)):
        all_instances[d_name].append(X_valid[idx_list[i-1]:idx_list[i]])
        all_scores[d_name].append(ranker.predict(X_valid[idx_list[i-1]:idx_list[i]]))
        
    #for k in exps_all[j].keys():
    for k in sub_keys:
        all_q_explanations[d_name][k] = []
        for l in range(1, len(idx_list)):
            all_q_explanations[d_name][k].append(exps_all[j][k][idx_list[l-1]:idx_list[l]])    

d_info = {'all_instances': all_instances, 
          'all_scores': all_scores, 
          'all_q_explanations': all_q_explanations}

pickle.dump( d_info, open( "./exp/save/d_info.p", "wb" ) )
'''

d_info = pickle.load( open( "./exp/save/d_info.p", "rb" ) )
all_instances = d_info['all_instances']
all_scores = d_info['all_scores']
all_q_explanations = d_info['all_q_explanations']

query_exp = {}

for j in range(len(dataset_names)):
    print('getting query_exp rpi, pmi, grad for ', dataset_names[j])
    d_name = dataset_names[j]
    query_exp[d_name] = {}
    ranker = lightgbm.Booster(model_file='./model/save/lmart_{}.txt'.format(dataset_names[j]))
    all_i = all_instances[d_name]
    rpi_exp_vals = []
    pmi_exp_vals = []
    grad_exp_vals = []
    
    for i in range(len(all_i)):
    #for i in range(2):
        grad_exp_vals.append(grad_exp(all_i[i], ranker))
        rpi_exp_vals.append(rpi_exp(all_i[i], ranker))
        pmi_exp_vals.append(pmi_exp(all_i[i], ranker))
    
    query_exp[d_name]['rpi'] = rpi_exp_vals
    query_exp[d_name]['pmi'] = pmi_exp_vals
    query_exp[d_name]['grad'] = grad_exp_vals


for i in range(len(dataset_names)):
    d_name = dataset_names[i]
    for j in range(len(sub_keys)):    
        query_exp[d_name][sub_keys[j]] = {}
        tmp = []
        for q_id in range(len(all_scores[d_name])):
            q_scores = all_scores[d_name][q_id]
            q_ranks = np.argsort(q_scores)[::-1]
            
            if len(q_scores) < 10: 
                cutoff = len(q_scores)
            else: 
                cutoff = int(np.floor(0.2 * len(q_scores)))
                
            top_ranked_instance_idx = q_ranks[0:cutoff]
            #low_ranked_instance_idx = q_ranks[-5:]
            query_exp_ = np.array(all_q_explanations[d_name][sub_keys[j]][q_id])[top_ranked_instance_idx]
            #normed_query_exp_ = normalize(query_exp_, axis=0)
            
            for l in range(query_exp_.shape[0]):
                query_exp_[l] /= np.sum(query_exp_[l] + 0.0001)
            
            q_exp_ = np.mean(query_exp_, axis=0)
            tmp.append(q_exp_)
        
        query_exp[d_name][sub_keys[j]] = tmp

#pickle.dump( query_exp, open( "./exp/save/query_exp_all_v5_top.p", "wb" ) )
#pickle.dump( query_exp, open( "./exp/save/query_exp_rank_lime.p", "wb" ) )