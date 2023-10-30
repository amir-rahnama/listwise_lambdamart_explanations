import pickle
import numpy as np
import pandas as pd
import time
import logging 
import sys
#from scipy.stats import spearmanr,pearsonr
import lightgbm
from scipy.stats import rankdata
from sklearn.preprocessing import normalize

class Greedy:
    #def __init__(self, data, pred_fn):
        #per_query = data[:, 1:]
        #mask_values = np.mean(per_query, axis=0)
        #self.mask_values = np.insert(mask_values, 0, np.nan)
        #self.pred_fn = pred_fn
        #self.base_pred = lmart.predict(data)
        #self.base_rank = rankdata([-1 * i for i in self.base_pred]).astype(int)
        
    def find_con_pair(self, base_rank, base_pred, new_pred, new_rank):
        epsilon = 1e-3
        con_pairs = np.zeros((base_pred.shape[0], base_pred.shape[0]))
        for i in range(base_pred.shape[0] - 1):
            for j in range(i + 1, base_pred.shape[0] - 1):
                condi = (base_pred[i] - new_pred[i] > base_pred[j] - new_pred[j])
                if condi:
                    w_p = new_rank[i] - base_rank[i] + new_rank[j] - base_rank[j]
                    diff = new_pred[i] - base_pred[i] + new_pred[j] - base_pred[j]
                    con_pairs[i, j] = diff * w_p
        
        con_pairs = con_pairs / (np.linalg.norm(con_pairs) + epsilon)

        return con_pairs
    

    def get_score_rank(self, instance, feature_to_ex, pred_fn):
        #not_in_exp_feat = np.setxor1d(all_feat, exp_feat_set)
        instsance_ = instance.copy()
        instsance_[:, feature_to_ex] = np.mean(instsance_[:, feature_to_ex], axis=0)
        new_pred = pred_fn(instsance_)
        new_rank = rankdata([-1 * i for i in new_pred]).astype(int)

        return new_pred, new_rank 


    def greedy_cover(self, instance, all_feat, pred_fn, exp_size = 9):
        not_selected = all_feat
        selected = []
        base_pred = pred_fn(instance)
        base_rank = rankdata([-1 * i for i in base_pred]).astype(int)

        exp = []
        imp = []
        
        while len(selected) < exp_size:
            u_vals = {}
            for a_f in not_selected:
                new_pred, new_rank = self.get_score_rank(instance, a_f, pred_fn)
                c_p = self.find_con_pair(base_rank, base_pred, new_pred, new_rank)
                u_vals[a_f] = np.sum(c_p)

            u_keys = list(u_vals.keys())
            u_vals_values = list(u_vals.values())

            max_u_fea_id = np.argmax(u_vals_values)

            max_u_fea_id = np.argmax(u_vals_values)
            sel_f = u_keys[max_u_fea_id]
            imp_val = u_vals_values[max_u_fea_id]

            exp.append(sel_f)
            imp.append(imp_val)
            
            base_pred = new_pred
            base_rank = new_rank

            selected.append(sel_f)
            not_selected = np.setxor1d(all_feat, selected)


        g_exp = np.zeros(instance.shape[1])

        for j in exp:
            g_exp[j] = imp[j]
        
        #g_exp = g_exp / np.sum(g_exp)

        return g_exp

if __name__ == "__main__":
    
    '''base_path = '/Users/amirrahnama/code/ltr_explainability'


    lmart = LambdaMART()
    lmart.load('{}/models/trained/lambdamart_model_1.lmart'.format(base_path))

    num_feature_names = np.array(['tf_body', 'tf_anchor', 'tf_title', 'tf_url', 'tf_all_document', 'idf_body', 
                    'idf_anchor', 'idf_title', 'idf_url', 'idf_all_document', 'tfidf_body', 'tfidf_anchor',
                    'tfidf_title', 'tf_idf_url', 'tfidf_all_document', 'dl_body' , 'dl_anchor', 'dl_title', 'dl_url', 
                    'dl_all_document', 'okapi_body', 'okapi_anchor', 'okapi_title', 'okapi_url','okapi_all_document', 
                    'lmirabs_body', 'lmirabs_anchor', 'lmirabs_title', 'lmirabs_url', 'lmirabs_all_document', 'lmirdir_body', 'lmirdir_anchor',
                    'lmirdir_title', 'lmirdir_url', 'lmirdir_document', 'lmirjm_body', 'lmirjm_anchor', 'lmirjm_title', 'lmirjm_url', 
                    'lmirjm_all_document', 'pagerank', 'inlink_number', 'outlink_number', 'num_slash_in_url', 'length_url', 'num_child'])

    validation = pickle.load( open( '{}/MQ2008/validation.pickle'.format(base_path), 'rb' ) )
    train = pickle.load( open( '{}/MQ2008/train.pickle'.format(base_path), 'rb' ) )
    test = pickle.load( open( '{}/MQ2008/test.pickle'.format(base_path), 'rb' ) )
    data = pd.concat([train, validation])

    all_feature_names = np.array(list(data.columns))
    y_train = data['rank_num']
    y_test = test['rank_num']

    x_train  = data.values[:, 1:-1]
    x_test = test.values[:, 1:-1]
    all_feature_names = all_feature_names[1:-1]

    columns_zero = np.where((x_train != 0).any(axis=0) == False)[0]
    x_train = np.delete(x_train, columns_zero, axis=1)
    x_test = np.delete(x_test, columns_zero, axis=1)
    num_feature_names  = np.delete(num_feature_names, columns_zero)

    x_train_transformed = x_train.copy()
    x_test_transformed = x_test.copy()

    scaler = StandardScaler()
    scaler.fit(x_train_transformed[:, 1:])
    x_train_transformed[:, 1:] = scaler.transform(x_train_transformed[:, 1:])
    x_test_transformed[:, 1:] = scaler.transform(x_test_transformed[:, 1:])

    all_unique_queries = np.unique(x_test_transformed[:, 0])
    
    query_id = 0'''
    
    
    greedy_exp = {}
    
    for dataset_name in ['mq2008', 'web10k', 'yahoo']:
        greedy_exp[dataset_name] = []
        d_info = pickle.load( open( "./exp/save/d_info.p", "rb" ) )
        all_instances = d_info['all_instances'][dataset_name]
        ranker = lightgbm.Booster(model_file='./model/save/lmart_{}.txt'.format(dataset_name))

        for i in range(len(all_instances)):
            per_query = all_instances[i]

            greedy_ = Greedy()
            base_pred = ranker.predict(per_query)
            base_rank = rankdata([-1 * i for i in per_query]).astype(int)
            g_exp_, g_imp_ = greedy_.greedy_cover(per_query, [1], [0], np.arange(1, per_query.shape[1]), ranker.predict, base_rank, base_pred, k=20)

            g_exp = np.zeros(per_query.shape[1])
            for j in range(len(g_exp_)):
                g_exp[j] = g_imp_[j]
            g_exp = g_exp / np.sum(g_exp)
            
            greedy_exp[dataset_name].append(g_exp)
            
    pickle.dump(greedy_exp, open( "./exp/save/query_greedy_score_v1.p", "wb" ) )
