### Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
import lightgbm
import pickle
from get_exp_new import lime_exp, shap_exp, random_exp, grad_exp
from lirme_v2 import LIRME


dataset_name = 'yahoo'
print('Getting Explanations for Dataset', dataset_name)

ranker = lightgbm.Booster(model_file='./models/lmart_{}.txt'.format(dataset_name))
ranker.params['objective'] = 'binary'

background_dict = pickle.load( open( "./data/{}_background.p".format(dataset_name), "rb" ) )
test = pickle.load( open( "./data/{}_test_sample.p".format(dataset_name), "rb" ) )


key_val = list(test.keys())[0]
background = []
for q in background_dict:
    background.extend(background_dict[q])
background = np.array(background).reshape(-1, test[key_val].shape[1])


lirme = LIRME(background)
e_sample_size = 4000

p_exps = {}
pointwise_exp = ['lime', 'shap', 'lirme', 'exs_v1', 'exs_v2', 'grad_d', 'random_d']

for key in test.keys():
    print('Document explanations for query', key)
    p_exps[key] = {}
    for e in pointwise_exp:
        p_exps[key][e] = []
    q_docs = test[key]
    pred_q_docs = ranker.predict(q_docs)

    for i in range(len(q_docs)): 
        print('Document explanations for instance', i)
        instance = q_docs[i]
        p_exps[key]['lime'].append(lime_exp(instance, ranker, background, sample_size=e_sample_size))
        p_exps[key]['shap'].append(shap_exp(instance, background, ranker, sample_size=e_sample_size))


        exp_lirme = lirme.explain(instance, ranker.predict, pred_q_docs, 
                    sur_type='ridge', label_type='regression', 
                    instance_idx=i, top_rank=5, sample_size=e_sample_size)
        p_exps[key]['lirme'].append(exp_lirme)

        gradient_exp = grad_exp(instance, ranker)
        p_exps[key]['grad_d'].append(gradient_exp)

        exp_exs_v1 = lirme.explain(instance, ranker.predict, pred_q_docs, 
                    sur_type='svm', label_type='top_k_binary', 
                    instance_idx=i, top_rank=5, sample_size=e_sample_size)
        p_exps[key]['exs_v1'].append(exp_exs_v1)

        exp_exs_v2 = lirme.explain(instance, ranker.predict, pred_q_docs, 
                    sur_type='svm', label_type='score', 
                    instance_idx=i, top_rank=5, sample_size=e_sample_size)
        p_exps[key]['exs_v2'].append(exp_exs_v2)

        p_exps[key]['random_d'].append(random_exp(instance.shape[0]))

pickle.dump(p_exps, open( "./exps/{}_pointwise_exps_v1.p".format(dataset_name), "wb" ) )


rank_lime_sample_size = {
 'mq2008': 200,  
 'web10k': 500,
  'yahoo': 500
}

listwise_exp = ['pmi', 'rpi', 'grad', 'rank_lime', 'greedy_score', 'random']
from ranklime import RankLIME

from listwise_exp import pmi_exp, rpi_exp, grad_q_exp, greedy_exp, random_q_exp

rlime = RankLIME(background)

l_exps = {}
for key in test.keys():
    print('query explanations for query', key)
    doc_values = test[key]
    l_exps[key] = {}
    for e in listwise_exp:
        l_exps[key][e] = {}
    
    l_exps[key]['pmi'] = pmi_exp(doc_values, ranker)
    l_exps[key]['rpi'] = rpi_exp(doc_values, ranker)
    l_exps[key]['grad'] = grad_q_exp(doc_values, ranker)
    l_exps[key]['rank_lime'] = rlime.explain(doc_values, ranker.predict, [], sample_size=rank_lime_sample_size[dataset_name])
    l_exps[key]['greedy_score'] = greedy_exp(doc_values, ranker.predict)
    l_exps[key]['random'] = random_q_exp(doc_values)

pickle.dump(l_exps, open( "./exps/{}_listwise_exps_v1.p".format(dataset_name), "wb" ) )



