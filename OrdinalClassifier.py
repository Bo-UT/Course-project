#!/usr/bin/env python
# coding: utf-8

# In[1]:



from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import copy

from sklearn.ensemble import RandomForestClassifier
from xbartExtend import xbartExtendClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[2]:


''' Adapted from 
https://github.com/garyongguanjie/Ordinal-Classifier/blob/master/OrdClass.py'''

class RandomForestOrdinalClassifier(BaseEstimator):
    """"
    Helper class that solves ordinal classification (classes that have an order to them eg cold,warm,hot)
    """
    def __init__(self,
                   n_estimators=100,
                   criterion="gini",
                   max_depth=None,
                   min_samples_split=2,
                   min_samples_leaf=1,
                   min_weight_fraction_leaf=0.,
                   max_features="auto",
                   max_leaf_nodes=None,
                   min_impurity_decrease=0.,
                   min_impurity_split=None,
                   bootstrap=True,
                   oob_score=False,
                   n_jobs=None,
                   random_state=None,
                   verbose=0,
                   warm_start=False,
                   class_weight=None,
                   ccp_alpha=0.0,
                   max_samples=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        
        """
        y needs to be a number that start from 0 and increments by 1
        classifier object needs to be able to return a probability
        """
        self.clfs = []
    
    def fit(self,X,y):
        no_of_classifiers = np.max(y) #since y starts from 0
        clf = RandomForestClassifier(**super().get_params())
        for i in range(no_of_classifiers):
          # make a copy of y because we want to change the values of y
            copy_y = np.copy(y)
          # make a binary classification here
            copy_y[copy_y<=i] = 0
            copy_y[copy_y>i] = 1
            classifier = copy.deepcopy(clf)
            classifier.fit(X,copy_y)
            self.clfs.append(classifier)
        return self
  
    def predict_proba(self,test):
        prob_list = []
        final_prob = []
        length = len(self.clfs)
        for clf in self.clfs:
            # Take probablity for class "1"
            prob_list.append(clf.predict_proba(test)[:,1])
        #Implement method from https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf   
        for i in range(length+1):
            if i == 0:
                final_prob.append(1-prob_list[i])
            elif i == length:
                final_prob.append(prob_list[i-1])
            else:
                final_prob.append(prob_list[i-1]-prob_list[i])
        answer = np.array(final_prob).transpose()
        self.final_prob= answer
        return answer
  
    def predict(self,test):
        self.predict_proba(test)
        #Return the class with highest probablity
        return np.argmax(self.final_prob,axis=1)
  
    def score(self,X,y,sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


# In[ ]:


#XGBoost
class XGBOrdinalClassifier(BaseEstimator):
    """"
    Helper class that solves ordinal classification (classes that have an order to them eg cold,warm,hot)
    """
    def __init__(self,
                 #Parameters from XGBClassifier
                 objective="binary:logistic", # binary classification 
                 # Parameters from XGBModel
                 max_depth=6, max_leaves=0,max_bin=256,learning_rate=0.3, n_estimators=100,
                 verbosity=1, booster='gbtree', tree_method='auto', 
                 n_jobs=None, gamma=0,
                 min_child_weight=1, max_delta_step=0, subsample=1,
                 colsample_bytree=1, colsample_bylevel=1,
                 colsample_bynode=1, reg_alpha=0, reg_lambda=1,
                 scale_pos_weight=1, base_score=0.5, random_state=None,
                 missing=np.nan, num_parallel_tree=1,
                 monotone_constraints=None, interaction_constraints=None,
                 importance_type="gain"):
        
        self.objective = objective
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin=max_bin
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        
        
        """
        y needs to be a number that start from 0 and increments by 1
        classifier object needs to be able to return a probability
        """
        self.clfs = []

    def fit(self,X,y,**fit):
        no_of_classifiers = np.max(y) #since y starts from 0
        clf = XGBClassifier(**super().get_params())
        for i in range(no_of_classifiers):
          # make a copy of y because we want to change the values of y
            copy_y = np.copy(y)
          # make a binary classification here
            copy_y[copy_y<=i] = 0
            copy_y[copy_y>i] = 1
            classifier = copy.deepcopy(clf)
            classifier.fit(X,copy_y,**fit)
            self.clfs.append(classifier)
        return self
  
    def predict_proba(self,test):
        prob_list = []
        final_prob = []
        length = len(self.clfs)
        for clf in self.clfs:
            # Take probablity for class "1"
            prob_list.append(clf.predict_proba(test)[:,1])
        #Implement method from https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf   
        for i in range(length+1):
            if i == 0:
                final_prob.append(1-prob_list[i])
            elif i == length:
                final_prob.append(prob_list[i-1])
            else:
                final_prob.append(prob_list[i-1]-prob_list[i])
        answer = np.array(final_prob).transpose()
        self.final_prob= answer
        return answer
  
    def predict(self,test):
        self.predict_proba(test)
        #Return the class with highest probablity
        return np.argmax(self.final_prob,axis=1)
  
    def score(self,X,y,sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
  


# In[ ]:


# lighGBM
class LGBMOrdinalClassifier(BaseEstimator):
    """"
    Helper class that solves ordinal classification (classes that have an order to them eg cold,warm,hot)
    """
    def __init__(self,boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000,
                 objective='binary', class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
                 subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1, silent=True,
                 importance_type='split'):
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate=learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.class_weight = class_weight
        self.min_split_gain =min_split_gain
        self.min_child_weight=min_child_weight
        self.min_child_samples=min_child_samples
        self.subsample=subsample
        self.subsample_freq=subsample_freq
        self.colsample_bytree=colsample_bytree
        self.reg_alpha=reg_alpha
        self.reg_lambda=reg_lambda
        self.random_state=random_state
        self.n_jobs = n_jobs
        self.silent=silent
        self.importance_type=importance_type
        
        """
        y needs to be a number that start from 0 and increments by 1
        classifier object needs to be able to return a probability
        """
        self.clfs = []
    
    def fit(self,X,y,**fit):
        no_of_classifiers = np.max(y) #since y starts from 0
        clf = LGBMClassifier(**super().get_params())
        for i in range(no_of_classifiers):
          # make a copy of y because we want to change the values of y
            copy_y = np.copy(y)
          # make a binary classification here
            copy_y[copy_y<=i] = 0
            copy_y[copy_y>i] = 1
            classifier = copy.deepcopy(clf)
            classifier.fit(X,copy_y,**fit)
            self.clfs.append(classifier)
        return self
    
    def predict_proba(self,test):
        prob_list = []
        final_prob = []
        length = len(self.clfs)
        for clf in self.clfs:
            # Take probablity for class "1"
            prob_list.append(clf.predict_proba(test)[:,1])
        #Implement method from https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf   
        for i in range(length+1):
            if i == 0:
                final_prob.append(1-prob_list[i])
            elif i == length:
                final_prob.append(prob_list[i-1])
            else:
                final_prob.append(prob_list[i-1]-prob_list[i])
        answer = np.array(final_prob).transpose()
        self.final_prob= answer
        return answer
  
    def predict(self,test):
        self.predict_proba(test)
        #Return the class with highest probablity
        return np.argmax(self.final_prob,axis=1)
  
    def score(self,X,y,sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


# In[ ]:


# XBART classifier
class xbartOrdinalClassifier(BaseEstimator):

    def __init__(self,num_trees: int = 100, num_sweeps: int = 50, n_min: int = 1,
                    num_cutpoints: int = 100, alpha: float = 0.95, beta: float = 1.25, tau = "auto",
                    burnin: int = 15, mtry = "auto", max_depth_num: int = 250,
                    kap: float = 16.0, s: float = 4.0, verbose: bool = False,
                    parallel: bool = False, seed: int = 0, model: str = "Normal",
                    no_split_penality = "auto", sample_weights_flag: bool = True, num_classes = 2): # binary classification

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
        self.clfs = []
  
    def fit(self,X,y, num_cat_var=0):
        no_of_classifiers = np.max(y) #since y starts from 0
        clf = xbartExtendClassifier(**super().get_params())
        for i in range(no_of_classifiers):
          # make a copy of y because we want to change the values of y
            copy_y = np.copy(y)
          # make a binary classification here
            copy_y[copy_y<=i] = 0
            copy_y[copy_y>i] = 1
            classifier = copy.deepcopy(clf)
            classifier.fit(X,copy_y,num_cat_var)
            self.clfs.append(classifier)
        return self

    def predict_proba(self,test):
        prob_list = []
        final_prob = []
        length = len(self.clfs)
        for clf in self.clfs:
            # Take probablity for class "1"
            prob_list.append(clf.predict_proba(test)[:,1])
        #Implement method from https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf   
        for i in range(length+1):
            if i == 0:
                final_prob.append(1-prob_list[i])
            elif i == length:
                final_prob.append(prob_list[i-1])
            else:
                final_prob.append(prob_list[i-1]-prob_list[i])
        answer = np.array(final_prob).transpose()
        self.final_prob= answer
        return answer
    
    def predict(self,test):
        self.predict_proba(test)
        #Return the class with highest probablity
        return np.argmax(self.final_prob,axis=1)
    
    def score(self,X,y,sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

