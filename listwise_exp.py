
import numpy as np
from scipy.stats import spearmanr
from greedy_score import Greedy
from sklearn.utils import shuffle

def grad_q_exp(instances, model):
    epsilon = 0.01
    total_instances = instances.shape[0]
    grad = np.zeros(instances.shape[1])
    
    for j in range(instances.shape[1]):
        h = np.zeros(instances.shape[1])
        h[j] = epsilon
        df_en1 = model.predict((instances + h).reshape(total_instances, -1))[0] 
        df_en2 =  model.predict((instances - h).reshape(total_instances, -1))[0]
        grad[j] = np.abs(df_en1 - df_en2)/ 2 * epsilon
    
    return grad #/ np.sum(grad)
    
def rpi_exp(instances, model, trials=10):
    fi = np.zeros(instances.shape[1])
    
    for j in range(instances.shape[1]):
        instances_copy = instances.copy()
        base_pred = model.predict(instances_copy)
        sps = []
        
        for i in range(trials):
            instances_copy[:, j] = np.random.shuffle(instances_copy[:,j])
            new_pred = model.predict(instances_copy)
            sps.append(np.abs(spearmanr(new_pred, base_pred).correlation))
        fi[j] = np.nanmean(sps)
    
    return fi / np.sum(fi)


def pmi_exp(instances, model, trials=10):
    fi = np.zeros(instances.shape[1])
    
    for j in range(instances.shape[1]):
        instances_copy = instances.copy()
        base_pred = model.predict(instances_copy)
        sps = []
        
        for i in range(trials):
            instances_copy[:, j] = shuffle(instances_copy[:,j])
            new_pred = model.predict(instances_copy)
            sps.append(np.mean(np.abs(new_pred - base_pred)))
        
        fi[j] = np.mean(sps)
    
    fi /= np.sum(fi)
    
    return fi

def greedy_exp(doc_values, pred_fn):
    size = doc_values.shape[1]
    greedy_ = Greedy()
    
    g_exp = greedy_.greedy_cover(doc_values, np.arange(0, doc_values.shape[1]),
                                 pred_fn, size)

    return g_exp

def random_q_exp(instances):
    return np.random.dirichlet(np.ones(instances.shape[1])).flatten()

