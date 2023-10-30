import re
import shap
from lime import lime_tabular
import lightgbm
import pandas as pd
import numpy as np
import sys
sys.path.append('../..')
sys.path.append('./exp/')
#from data.get_data import get_data
import pickle
from scipy.stats import truncnorm, norm
import time
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

import numpy as np
from scipy.stats import truncnorm
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from utils import get_bxi, compute_mu_sigma_tilde, get_training_data_stats
import collections
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from scipy.special import expit
from scipy.special import softmax


epsilon = np.finfo(float).eps 

def custom_loss_function(y_true, y_pred, *args, **kwargs):
    total = len(y_true)
    loss_val = 0
    
    for s in range(1, total - 1):
        for i in range(total):
            if  y_true[i] < y_true[s]:
                loss_val += expit(y_true[i] < y_true[s])
    
    return loss_val

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def listnet_loss(y_true, y_pred):
    P_y_i = softmax(y_true)
    P_z_i = softmax(y_pred)
    loss = - np.sum(P_y_i * np.log(P_z_i + epsilon))
    
    
    print(loss)
    return loss

class RankLIME:
    def __init__(self, data):
        self.data = data
        #self.kernel_width = np.sqrt(data.shape[1]) * .75
        self.kernel_width = np.sqrt(data.shape[1]) * .75
        self.training_data = data
        self.feature_names = np.arange(self.training_data.shape[1])
        self.categorical_features = list(range(self.training_data.shape[1]))
        self.to_discretize = self.categorical_features
        self.stats()
        self.scaler = StandardScaler(with_mean=False)
        self.scaler.fit(data)
        self.random_state = 10
        self.cov = np.cov(data.T)

    def kernel(self, d, kernel_width=None):
        if kernel_width: 
            k_width = 0.3
        else:
            k_width = self.kernel_width
            
        return np.exp(-(d ** 2) / (k_width **2))
    
    def get_bins(self, data):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature], [25, 50, 75]))
            bins.append(qts)
        return bins

    def discretize(self, data):
        ret = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                ret[feature] = int(self.lambdas[feature](ret[feature]))
            else:
                ret[:, feature] = self.lambdas[feature](
                    ret[:, feature]).astype(int)
        return ret

    def get_undiscretize_values(self, feature, values):
        mins = np.array(self.mins_all[feature])[values]
        maxs = np.array(self.maxs_all[feature])[values]

        means = np.array(self.means_all[feature])[values]
        stds = np.array(self.stds_all[feature])[values]

        minz = (mins - means) / stds
        maxz = (maxs - means) / stds
        min_max_unequal = (minz != maxz)

        ret = minz
        ret[np.where(min_max_unequal)] = truncnorm.rvs(
            minz[min_max_unequal],
            maxz[min_max_unequal],
            loc=means[min_max_unequal],
            scale=stds[min_max_unequal],
            random_state=self.random_state
        )
        return ret

    def undiscretize(self, data):
        ret = data.copy()
        for feature in self.means_all:
            if len(data.shape) == 1:
                ret[feature] = self.get_undiscretize_values(
                    feature, ret[feature].astype(int).reshape(-1, 1)
                )
            else:
                ret[:, feature] = self.get_undiscretize_values(
                    feature, ret[:, feature].astype(int)
                )
        return ret
        
        
    def stats(self):
        self.lambdas = {}
        self.names = {}
        self.mins_all = {}
        self.maxs_all = {}
        self.means_all = {}
        self.stds_all = {}
        self.feature_values = {}
        self.feature_frequencies = {}
        
        bins = self.get_bins(self.training_data)
        bins = [np.unique(x) for x in bins]

        for feature, qts in zip(self.to_discretize, bins):
            n_bins = qts.shape[0]  # Actually number of borders (= #bins-1)
            boundaries = np.min(self.training_data[:, feature]), np.max(self.training_data[:, feature])
            name = self.feature_names[feature]

            self.names[feature] = ['%s <= %.2f' % (name, qts[0])]
            for i in range(n_bins - 1):
                self.names[feature].append('%.2f < %s <= %.2f' %
                                           (qts[i], name, qts[i + 1]))
            self.names[feature].append('%s > %.2f' % (name, qts[n_bins - 1]))

            self.lambdas[feature] = lambda x, qts=qts: np.searchsorted(qts, x)
            discretized = self.lambdas[feature](self.training_data[:, feature])

            self.means_all[feature] = []
            self.stds_all[feature] = []
            for x in range(n_bins + 1):
                selection = self.training_data[discretized == x, feature]
                mean = 0 if len(selection) == 0 else np.mean(selection)
                self.means_all[feature].append(mean)
                std = 0 if len(selection) == 0 else np.std(selection)
                std += 0.00000000001
                self.stds_all[feature].append(std)
            self.mins_all[feature] = [boundaries[0]] + qts.tolist()
            self.maxs_all[feature] = qts.tolist() + [boundaries[1]]
        
        discretized_training_data = self.discretize(self.training_data)
        
        for feature in self.categorical_features:
            column = discretized_training_data[:, feature]

            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                             float(sum(frequencies)))
            
    
    def get_exp_vals(self, samples, labels, sample_weights):
        #custom_loss = make_scorer(custom_loss_function, greater_is_better=False)
        custom_loss = make_scorer(listnet_loss, greater_is_better=True)
        
        if np.sum(sample_weights) == 0:
            sample_weights = np.repeat((1/samples.shape[1]), samples.shape[1])
        print('sum of weights', np.sum(sample_weights))
        
        param_grid = {'alpha':[1]}
        
        split_num = 2
        

        
        ridge_pairwise = GridSearchCV(Ridge(), param_grid, scoring=custom_loss, cv=split_num, verbose=3)
        ridge_pairwise.fit(samples, labels, sample_weight=sample_weights)
        exp = ridge_pairwise.best_estimator_.coef_
  
        return exp
        
    def quantile_sampling(self, instance, num_samples=1):         
        data = np.random.multivariate_normal(np.zeros(instance.shape[0]), self.cov, num_samples)

        data = np.array(data)
        data = data * self.scaler.scale_ + instance
        
        data_row = instance
        num_cols = self.training_data.shape[1]
        data = np.zeros((num_samples, num_cols))
        first_row = self.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        
        for column in self.categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = np.random.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        
        inverse[1:] = self.undiscretize(inverse[1:])
        inverse[0] = data_row
        
        return data.flatten(), inverse.flatten()
           
    def explain(self, instance, pred_fun, query_doc_preds, instance_idx=None, top_rank=5, sample_size=100):

        #data, inverse = self.quantile_sampling(instance)
        #data = []
        #inverse = []
        
        #all_samples = []
        #all_inverse = []
        
        '''
        for i in range(instance.shape[0]):
            temp_ = self.quantile_sampling(instance[i])
            data.append(temp_[0])
            inverse.append(temp_[1])'''
        
        if (type(instance)) == pd.core.frame.DataFrame:
            instance = instance.values
        
        datas = []
        labelss = []
        sample_weightss = []
        
       
        
        for k in range(sample_size):
            number_feat_selected = np.random.choice(np.arange(0, self.data.shape[1]), replace=False)
            feat_selected = np.random.choice(np.arange(0, self.data.shape[1]), number_feat_selected, replace=False)
    
            new_samples = np.zeros(instance.shape)

            for j in feat_selected:
                new_samples[:, j] = np.random.normal(0, np.std(self.data[:, j]), size=instance.shape[0])
                
                #new_samples[:, j] = np.random.normal(0, 1, size=instance.shape[0])
            
            d = new_samples * self.scaler.scale_ + instance
            #d = new_samples  + instance
            
            labels = pred_fun(d)

            distances = np.linalg.norm(d - instance, axis=1)
            
            #distances /= np.sum(distances)
            sample_weights = self.kernel(distances, 1)
            
            count_nan = np.count_nonzero(np.isnan(sample_weights))
            if count_nan > 0:
                print(distances)

            sample_weightss.append(sample_weights)

            datas.append(d)
            labelss.append(labels)
        
        
        data = np.array(datas).reshape(-1, self.data.shape[1])
        labels = np.array(labelss).flatten()
        sample_weight = np.array(sample_weightss).flatten()
                
        exp = self.get_exp_vals(data, labels, sample_weight)
        
        return exp
    
if __name__ == '__main__':
   
    '''dataset_names = ['yahoo', 'web10k', 'mq2008']
    d_info = pickle.load( open( "./exp/save/d_info.p", "rb" ) )
    all_instances = d_info['all_instances']

    rlime_exps = {}
    
    for dataset_name in dataset_names:
        print(dataset_name)
        rlime_exps[dataset_name] = []
        X_train, y_train, qids_train, X_valid, y_valid, qids_valid, all_feat_names, test_q_info = get_data(dataset_name, training=False)
        
        ranker = lightgbm.Booster(model_file='./model/save/lmart_{}.txt'.format(dataset_name))
        
        scaler = StandardScaler(with_mean=False)
        valid_scaled = scaler.fit_transform(X_valid)
        valid_scaled = np.nan_to_num(valid_scaled)

        train_scaled = scaler.fit_transform(X_train)
        train_scaled = np.nan_to_num(train_scaled)

        mean_val = np.mean(train_scaled, axis=0)
        mean_cov = np.cov(train_scaled.T)

        rlime = RankLIME(valid_scaled[:1000])

        for i in range(len(all_instances[dataset_name])):               
        #for i in range(1):               
            q_data = np.array(all_instances[dataset_name][i])
            q_data = scaler.transform(q_data)
            q_data = np.nan_to_num(q_data)

            all_related_docs = q_data
            all_related_docs_preds = ranker.predict(all_related_docs)

            exp_rank_lime = rlime.explain(q_data, ranker.predict, all_related_docs_preds)
            rlime_exps[dataset_name].append(exp_rank_lime)
    pickle.dump(rlime_exps, open( "./exp/save/query_ranklime_exp_all.p", "wb" ) )'''
    
    
    
    
    
    
    
    