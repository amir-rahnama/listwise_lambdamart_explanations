import re
import shap
from lime import lime_tabular
import lightgbm
import pandas as pd
import numpy as np
#from ltr_exps_new import LIRME
#from lirme import LIRME
from lirme_v2 import LIRME

import sys
sys.path.append('../..')
import pickle
#from data.get_data import get_data
import time
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time


def find_feature_idx(feature_rule):
    res = re.findall(r'feature_id_(\d+)', feature_rule)
    if len(res) > 0:
        return int(res[0])

def transform_lime_exp(exp, features):
    transform_exp = np.zeros(len(features))

    for i in range(len(exp)):
        feature_idx = np.array(find_feature_idx(exp[i][0]))
        transform_exp[feature_idx] = exp[i][1]
    
    return transform_exp


def find_feature_idx(feature_rule):
    res = re.findall(r'feature_id_(\d+)', feature_rule)
    if len(res) > 0:
        return int(res[0])

    
def transform_lime_exp(exp, features):
    transform_exp = np.zeros(len(features))

    for i in range(len(exp)):
        feature_idx = np.array(find_feature_idx(exp[i][0]))
        transform_exp[feature_idx] = exp[i][1]
    
    return transform_exp


def shap_exp(instances, train_data, model, sample_size):
    sample_size_background = 100
    s = shap.sample(train_data, sample_size_background)
    explainer = shap.KernelExplainer(model.predict, s)
    shap_exps = explainer.shap_values(instances, nsamples=sample_size)
    
    return shap_exps

def random_exp(size):
    return np.random.dirichlet(np.ones(size), size=1).flatten()

def lime_exp(instance, model, train_data, sample_size):
    feature_names = ['feature_id_' + str(i) for i in np.arange(train_data.shape[1])]
    lime_exp = lime_tabular.LimeTabularExplainer(train_data[:2000], 
                                             kernel_width=3, verbose=False, mode='regression',
                                             feature_names=feature_names)

    exp = lime_exp.explain_instance(instance, model.predict, num_features=train_data.shape[1],  
                                    num_samples=sample_size)
    lime_e = exp.as_list()
    lime_e_trans = transform_lime_exp(lime_e, feature_names)
    
    return lime_e_trans

def grad_exp(instance, model):
    epsilon = 0.01

    grad = np.zeros(instance.shape[0])
    
    for j in range(instance.shape[0]):
        h = np.zeros(instance.shape[0])
        h[j] = epsilon
        df_en1 = model.predict((instance + h).reshape(1, -1))[0] 
        df_en2 =  model.predict((instance - h).reshape(1, -1))[0]
        grad[j] = np.abs(df_en1 - df_en2)/ 2 * epsilon
    
    #grad = grad #/ np.sum(grad)
    
    return grad


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EXP LMART",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset_name", choices=['yahoo', 'mq2008', 'web10k'], help="name of the dataset")

    args = parser.parse_args()
    config = vars(args)

    print(config)
    total_instances = 200

    dataset_name = config['dataset_name']
    
    if dataset_name == 'mq2008':
        e_sample_size = 2000
        quries_selected = ['qid:14717', 'qid:11069', 'qid:19782', 'qid:19875', 'qid:19503',
                           'qid:15956', 'qid:18411', 'qid:18995', 'qid:14427', 'qid:15860',
                           'qid:10943', 'qid:13120', 'qid:10032', 'qid:15860', 'qid:13083',
                           'qid:19503', 'qid:15251', 'qid:11639', 'qid:10915', 'qid:19353',
                           'qid:10943', 'qid:15168', 'qid:15251', 'qid:14805', 'qid:11437',
                           'qid:18995', 'qid:16454', 'qid:12237', 'qid:13388', 'qid:14322']
    elif dataset_name == 'web10k':
        e_sample_size = 4000
        quries_selected = ['qid:11653', 'qid:7093', 'qid:13093', 
                           'qid:10558', 'qid:6523','qid:12718', 'qid:22318', 
                           'qid:17848', 'qid:6148', 'qid:24088']
    elif dataset_name == 'yahoo':
        e_sample_size = 5000
        quries_selected = [24085., 25140., 25870., 28031., 28974., 26276., 29725., 28617.,
                               24116., 25865., 29759., 27204., 28587., 27549., 23228.]
    else:
        raise Exception("You need to pass in the dataset name correctly")
        
    ranker = lightgbm.Booster(model_file='./model/save/lmart_{}.txt'.format(dataset_name))
    ranker.params['objective'] = 'binary'

    st = time.time()

    X_train, y_train, qids_train, X_valid, y_valid, qids_valid, all_feat_names, test_q_info = get_data(dataset_name, training=False)
    
    scaler = StandardScaler(with_mean=False)
    valid_scaled = scaler.fit_transform(X_valid)
    valid_scaled = np.nan_to_num(valid_scaled)
    
    train_scaled = scaler.fit_transform(X_train)
    train_scaled = np.nan_to_num(train_scaled)
    
    et = time.time()
    
    print('Loaded Data: {} seconds'.format(et - st))
    
    all_docs = []
    
    for q_s in quries_selected:
        print(q_s)
        docs_for_query = np.argwhere(test_q_info == q_s).flatten()
        all_docs.extend(docs_for_query)
    print(test_q_info)
    print(len(all_docs))
    sys.exit()
    lirme = LIRME(valid_scaled[:1000])
    
    #exps = {'lime': [], 'shap': [], 'lirme': [], 'exs_v1': [],  'exs_v2': [], 'grad': [], 'random': []}
    exps = {'rank_lime': []}
    
    for i in all_docs:
        print('getting Explanations for Instance ', i)
        #time_s = time.time()
        all_related_docs = np.argwhere(test_q_info[i] == test_q_info).flatten()
        all_related_docs_preds = ranker.predict(valid_scaled[all_related_docs])
        
        exp_rank_lime = lirme.explain(valid_scaled[i], ranker.predict, all_related_docs_preds, 
                    sur_type='rank_lime', label_type='regression', 
                    instance_idx=i, top_rank=5, sample_size=e_sample_size)

        #time_f = time.time()
        #time_exps['rank_lime'].append(time_f - time_s)
        exps['rank_lime'].append(exp_rank_lime)
    
        if i >= 5 and i % 5 == 0:
            pickle.dump(exps, open( "./exp/save/exps_{}_{}_tmp_rank_lime.p".format(dataset_name, total_instances), "wb" ) )
    
    pickle.dump(exps, open( "./exp/save/exps_{}_{}_rank_lime.p".format(dataset_name, total_instances, datetime.now().timestamp()), "wb" ) )
    
    '''exps = {'lime': [], 'shap': [], 'lirme': [], 'exs_v1': [],  'exs_v2': [], 'grad': [], 'random': []}
    
    time_exps = {'lime': [], 'shap': [], 'lirme': [], 'exs_v1': [], 'exs_v2': [], 'grad': [],  'random': []}
    

    for i in all_docs:
        print('getting Explanations for Instance ', i)
        time_s = time.time()
        exps['lime'].append(lime_exp(valid_scaled[i], ranker, train_scaled, sample_size=e_sample_size))
        time_f = time.time()
        time_exps['lime'].append(time_f - time_s)
        
        time_s = time.time()
        exps['shap'].append(shap_exp(valid_scaled[i], train_scaled, ranker, sample_size=e_sample_size))
        time_f = time.time()
        time_exps['shap'].append(time_f - time_s)

        all_related_docs = np.argwhere(test_q_info[i] == test_q_info).flatten()
        all_related_docs_preds = ranker.predict(valid_scaled[all_related_docs])
       
        time_s = time.time()
        exp_lirme = lirme.explain(valid_scaled[i], ranker.predict, all_related_docs_preds, 
                    sur_type='ridge', label_type='regression', 
                    instance_idx=i, top_rank=5, sample_size=e_sample_size)
        time_f = time.time()
        time_exps['lirme'].append(time_f - time_s)
        exps['lirme'].append(exp_lirme)
        
        time_s = time.time()
        gradient_exp = grad_exp(valid_scaled[i], ranker)
        time_f = time.time()
        time_exps['grad'].append(time_f - time_s)
        exps['grad'].append(gradient_exp)
        
        
        time_s = time.time()
        exp_exs_v1 = lirme.explain(valid_scaled[i], ranker.predict, all_related_docs_preds, 
                    sur_type='svm', label_type='top_k_binary', 
                    instance_idx=i, top_rank=5, sample_size=e_sample_size)
        time_f = time.time()
        time_exps['exs_v1'].append(time_f - time_s)
        exps['exs_v1'].append(exp_exs_v1)
        
        time_s = time.time()
        exp_exs_v2 = lirme.explain(valid_scaled[i], ranker.predict, all_related_docs_preds, 
                    sur_type='svm', label_type='score', 
                    instance_idx=i, top_rank=5, sample_size=e_sample_size)
        time_f = time.time()
        time_exps['exs_v2'].append(time_f - time_s)
        exps['exs_v2'].append(exp_exs_v2)
        
        exps['random'].append(random_exp(valid_scaled.shape[1]))

        if i >= 5 and i % 5 == 0:
            pickle.dump(exps, open( "./exp/save/exps_{}_{}_tmp.p".format(dataset_name, total_instances), "wb" ) )
            pickle.dump(time_exps, open( "./exp/save/time_{}_{}_tmp.p".format(dataset_name, total_instances), "wb" ) )

    pickle.dump(exps, open( "./exp/save/exps_{}_{}_{}.p".format(dataset_name, total_instances, datetime.now().timestamp()), "wb" ) )
    pickle.dump(time_exps, open( "./exp/save/time_{}_{}.p".format(dataset_name, datetime.now().timestamp()), "wb" ) )'''
