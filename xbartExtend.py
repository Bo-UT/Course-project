#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator
from xbart import XBART

# add get_params() and set_params() to XBART
class xbartExtendRegressor(BaseEstimator):
    def __init__(self,num_trees: int = 100, num_sweeps: int = 40, n_min: int = 1,
                      num_cutpoints: int = 100, alpha: float = 0.95, beta: float = 1.25, tau = "auto",
                      burnin: int = 15, mtry = "auto", max_depth_num: int = 250,
                      kap: float = 16.0, s: float = 4.0, verbose: bool = False,
                      parallel: bool = False, seed: int = 0, model: str = "Normal",
                      no_split_penality = "auto", sample_weights_flag: bool = True, num_classes = 1):
        self.num_trees = num_trees
        self.num_sweeps = num_sweeps
        self.n_min = n_min
        self.num_cutpoints = num_cutpoints
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.burnin = burnin
        self.mtry = mtry
        self.max_depth_num = max_depth_num
        self.kap = kap
        self.s = s
        self.verbose = verbose
        self.parallel = parallel
        self.seed = seed
        self.model = model
        self.no_split_penality = no_split_penality
        self.sample_weights_flag = sample_weights_flag
        self.num_classes = num_classes 
        #take the estimator
        self.clfs = {}
        
    def fit(self,X_dtrain, y_dtrain,cat_num=7): #total crash last 7 are binary, total fatal crash last 8 are binary
        # num_cat_var, number of categorical vairalbes, need put in the last num_cat_var columns in X_dtrain 
        self.X = copy.copy(X_dtrain)
        self.y = copy.copy(y_dtrain)
        model = XBART(**super().get_params())
        model.fit(self.X, self.y,cat_num)
        self.clfs['model'] = model
        return self
    
    def predict(self, X_dtest):
        return self.clfs['model'].predict(X_dtest)   
              

#Binary classifier
class xbartExtendClassifier(BaseEstimator):
    def __init__(self,num_trees: int = 100, num_sweeps: int = 40, n_min: int = 1,
                      num_cutpoints: int = 100, alpha: float = 0.95, beta: float = 1.25, tau = "auto",
                      burnin: int = 15, mtry = "auto", max_depth_num: int = 250,
                      kap: float = 16.0, s: float = 4.0, verbose: bool = False,
                      parallel: bool = False, seed: int = 0, model: str = "Normal",
                      no_split_penality = "auto", sample_weights_flag: bool = True, num_classes = 2):
        self.num_trees = num_trees
        self.num_sweeps = num_sweeps
        self.n_min = n_min
        self.num_cutpoints = num_cutpoints
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.burnin = burnin
        self.mtry = mtry
        self.max_depth_num = max_depth_num
        self.kap = kap
        self.s = s
        self.verbose = verbose
        self.parallel = parallel
        self.seed = seed
        self.model = model
        self.no_split_penality = no_split_penality
        self.sample_weights_flag = sample_weights_flag
        self.num_classes = num_classes 
        #take the estimator
        self.clfs = {}
        
    def fit(self,X_dtrain, y_dtrain, num_cat_var = 0):
        # num_cat_var, number of categorical vairalbes, need put in the last num_cat_var columns in X_dtrain 
        copy_y = copy.copy(y_dtrain)
        #XBART only accept y encoded in {-1,1}, so convert y in {0,1} to {-1,1}
        copy_y = (copy_y>0)*2-1
        model_fit = XBART(**super().get_params())
        model_fit.fit(X_dtrain, copy_y,num_cat_var)
        self.clfs['model'] = model_fit
        return self
    
    def predict_proba(self,X_dtest):
        pred_y = self.clfs['model'].predict(X_dtest,return_mean = False)
        pd1 = pd.DataFrame(pred_y).apply(lambda x: np.sum(x>0)/self.num_sweeps, axis=1)
        pd0 = 1-pd1
        pd_np = pd.concat([pd0,pd1], axis=1).to_numpy()
        return pd_np
        
    def predict(self, X_dtest):
        return np.argmax(self.predict_proba(X_dtest), axis=1)
                 


