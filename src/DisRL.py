import pandas as pd 
import numpy as np
import math
from itertools import chain, combinations
import itertools
from numpy.random import random
from bisect import bisect_left
from random import sample
from scipy.stats.distributions import poisson, gamma, beta, bernoulli, binom
import time
import operator
from collections import Counter, defaultdict, deque
from scipy.sparse import csc_matrix

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from multiprocessing import Pool
from copy import copy, deepcopy
import random as rd
import sys
import time
from math import sqrt
import re
import cProfile
import faulthandler
import matplotlib.pyplot as plt
from tqdm import tqdm

from itertools import product

from deepdiff import DeepDiff

faulthandler.enable()

class DisRL(BaseEstimator):
    def __init__(self, Niteration, c1=0, c2=0, max_depth=3, ntrees = 128, minsupp = 5, mode='SA',tempfactor=.05):
        self.trainX = None
        self.trainY = None
        self.Niteration = Niteration
        self.nobs = None
           
        # initialze after first run of generate_rules
        # format (mu, idx of indices of data in stratum with assigned mu)        
        self.regrl = {} # keyend by strata mu, valued by strata rules
        self.bestobj = float("inf") # best objective value found so far. We aim to minimize the objvecti
        self.bestMSE = float("inf")
        self.bestregrl = []
        self.initial_pool_size = 5 # initial number of rules

        self.cart_obj = 0

        self.enableprefix = True # WAS TRUE
        self.enablebound = True # WAS TRUE
        self.enablemsesort = False
        
        self.newrule = 0
        
        self.tempfactor = tempfactor
        self.cart_init = True

        self.c1 = c1
        self.c2 = c2
        self.max_depth=max_depth
        self.ntrees = ntrees
        self.minsupp = minsupp
        self.mode = mode

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {'Niteration':self.Niteration,
                'c1':self.c1, 
                'c2':self.c2, 
                'max_depth':self.max_depth,
                'ntrees':self.ntrees,
                'minsupp':self.minsupp,
                'mode':self.mode
        }
                
    def stratify(self,regrl):
        a = regrl[len(regrl)-1]['drule'] == []
        leftover = set(list(range(self.nobs)))
        for s in range(len(regrl)-1):
            z = np.sum(self.RMatrix[:,regrl[s]['drule']],axis = 1)>0
            regrl[s]['indices'] = list(leftover.intersection(np.where(z)[0]))
                             
            leftover = leftover.difference(regrl[s]['indices'])
        regrl[len(regrl)-1]['indices'] = list(leftover)
        
        if len(regrl[len(regrl)-1]['indices']) ==0:
            regrl.pop(len(regrl)-1)
            regrl[len(regrl)-1]['drule'] = []
    
        for s in range(len(regrl)):
            if len(regrl[s]['indices']) > 0:
                regrl[s]['mu'] = np.mean(self.trainY[regrl[s]['indices']])
            else:
                regrl[s]['mu'] = np.mean(self.trainY)
            
    def clean_regrl(self,regrl_new,print_message = False):
        s = 0
        flag = False
        
        while s < len(regrl_new)-1:
            if len(regrl_new[s]['indices'])==0 or len(regrl_new[s]['drule'])==0 or len(regrl_new[s]['indices']) < .01 * self.nobs:
                regrl_new.pop(s)
                self.stratify(regrl_new)
                flag = True
            else:
                s = s + 1
        if flag:
            self.stratify(regrl_new)

            
        # inoke theorem 1
       
        if len(regrl_new) > 2:
            while True and self.enablebound:
                checktune = []
                for i in range(len(regrl_new)-1):
                    A = self.c2*self.nobs
                    B = .0000001 + len(regrl_new[i]['indices'])
                    C = len(regrl_new[i+1]['indices'])+.000000001
                    checktune.append(np.abs(regrl_new[i]['mu'] - regrl_new[i+1]['mu']) <= math.sqrt(self.c2*(1./B + 1./C)))
               
                if True in checktune:
                    x = checktune.index(True)
                    if print_message:
                        print('found strata {} and {} to merge'.format(x,x+1))
                    if regrl_new[x]['drule'] != [] and regrl_new[x+1]['drule'] != []:
                        regrl_new[x]['drule'] = regrl_new[x]['drule'] + regrl_new[x+1]['drule']
                        regrl_new.pop(x+1)
                        self.stratify(regrl_new)
                    else:
                        if len(regrl_new) > 1:
                            regrl_new.pop(x)
                    self.stratify(regrl_new)
                else:
                    break

        activct = 0
        # invoke theorem 3
        tune = True
       
        while len(regrl_new) > 2:
            supplens = [len(regrl_new[s]['indices']) for s in range(len(regrl_new)-1)]
            removeidx = next((i for i in range(len(regrl_new)-1) if supplens[i] < self.stratcvbd),None)
            if removeidx == None:
                break
            else:
                activct += 1
                regrl_new.pop(removeidx)
                
                self.stratify(regrl_new)
                
        # invoke corr 3
        activct = 0
        tune = True
        while tune:
            supplens = []
            for i in range(len(regrl_new)-1):
                stratacov = set(regrl_new[i]['indices'])
                eff_supps = [len(set(self.RMatrix[:,regrl_new[i]['drule'][j]]).intersection(stratacov)) for j in range(len(regrl_new[i]['drule']))]
                supplens.append(eff_supps)

            tune = False
            for eslist in supplens:
                idx = next((j for j in range(len(eslist)) if eslist[j] < self.isetcvbd),None)
                if idx != None:
                    activct += 1
                    if len(regrl_new[i]['drule']) > 1:
                        
                        regrl_new[i]['drule'].pop(idx)
                        self.stratify(regrl_new)
                        tune = True
                        break
                    
        
            if tune == False:
                break
     
    def generate_rules(self, supp,maxlen,forest_size=50):
        # must binarize data first       
        self.maxlen = maxlen
        self.minsupp = supp
        self.forest_size=forest_size
        
        print("starting RF rule generation...")
        tps = time.time()
        pool = Pool(processes=maxlen)
        self.dic = pool.map(self.rule_miner_randomforest,list(range(2,maxlen+1,1)))
        tpool = time.time()

        RMatrix = np.zeros((self.nobs,0))
        rules = []
        trmats = time.time()
        for l in range(maxlen-1):
            RMatrix = np.concatenate((RMatrix,self.dic[l][1]),axis = 1)
            rules.extend(self.dic[l][0])
        trmate = time.time()

        trules = time.time()
        rule_mu = []
        rule_std = []
        for i,rule in enumerate(rules):
            rule_mu.append(np.mean(self.trainY[np.where(RMatrix[:,i])[0]]))
            rule_std.append(np.var(self.trainY[np.where(RMatrix[:,i])[0]]))
        trulee = time.time()
        
        tsubset = time.time()
        min_idx = np.where(np.sum(RMatrix,axis = 0)>RMatrix.shape[0]*supp/100.0)[0]
        l = np.sum(RMatrix, axis=0)
        self.rule_mu = np.array(rule_mu)[min_idx]
        self.rule_std = np.array(rule_std)[min_idx]
        self.RMatrix = RMatrix[:,min_idx]
        self.rules = [rule for i,rule in enumerate(rules) if i in min_idx]


        if self.cart_init:
            self.get_cart_bd()
        else:
            self.cart_regrl = []
            self.cart_MSE = -1
        pool.close()

        if len(self.rules) == 0:
            raise ValueError('No rules generated!')
        
        return self.rules
            
    def rule_miner_randomforest(self,length):
        # uncomment for large data
        #clf = RandomForestRegressor(n_estimators=self.forest_size,max_samples=10000,max_depth=length,n_jobs=-1)
        clf = RandomForestRegressor(n_estimators=self.forest_size,max_depth=length,n_jobs=-1)
        clf.fit(self.trainX, self.trainY)
        return self.extract_rules_rf(clf)

    def extract_rules_rf(self,clf):
        RMatrix = np.zeros((self.nobs,0))
        rules = []
        for n in range(len(clf.estimators_)):
            x = get_lineage(clf.estimators_[n],self.trainX.columns)
            leaves = clf.estimators_[n].apply(self.trainX)
            leave_id = np.unique(leaves)
            leave_id_index = {}
            for i,lid in enumerate(leave_id):
                leave_id_index[lid] = i
            Z = np.zeros((self.nobs,len(leave_id)))

            Z[list(range(self.nobs)),[leave_id_index[leave] for leave in leaves]] = int(1)
            RMatrix = np.concatenate((RMatrix,Z),axis = 1)
            rules.extend(x.values())
        return rules, RMatrix

    # use self.RMatrix
    # arg (z, self.RMatrix)
    # construct DisRL with k - 1 leaves
    def get_cart_bd(self):
        # Get cart 
        cart_begin = time.time()
        clf = DecisionTreeRegressor(max_depth=self.max_depth,criterion='mse')
        clf.fit(self.trainX, self.trainY)
        cart_end = time.time()
        print('Time to generate CART: ', cart_end - cart_begin)

        # Get RMatrix and rules
        rules = []
        x = get_lineage(clf, self.trainX.columns)
        leaves = clf.apply(self.trainX)
        leave_id = np.unique(leaves)
        leave_id_index = {}
        for i,lid in enumerate(leave_id):
            leave_id_index[lid] = i
        Z = np.zeros((self.nobs, len(leave_id)))

        Z[list(range(self.nobs)), [leave_id_index[leave] for leave in leaves]] = int(1)
        self.RMatrix = np.concatenate((self.RMatrix,Z),axis=1)
        rules.extend(x.values())

        # add rules to corpus
        print("Initial CART model #rules: ", len(rules))
        # check if rules intersection self.rules nontrivial
        self.rules = self.rules + rules

        self.rule_mu =[]
        self.rule_std = []
        for i,rule in enumerate(self.rules):
            self.rule_mu.append(np.mean(self.trainY[np.where(self.RMatrix[:,i])]))
            self.rule_std.append(np.var(self.trainY[np.where(self.RMatrix[:,i])[0]]))

        self.rule_mu = np.array(self.rule_mu)
        self.rule_std = np.array(self.rule_std)
        
        # use all but one leaf
        regrl_cart = [{'drule':[x]} for x in np.arange(0,len(rules)-1)]
        regrl_cart.append({'drule':[]})
        
        self.stratify(regrl_cart)
        self.clean_regrl(regrl_cart)
        
        # get obj
        loss_breakdown_cart =  self.compute_loss(regrl_cart)
        k = len(regrl_cart)
        kmse_cart = sum([x[0] for i,x in enumerate(loss_breakdown_cart) if i <k])
        obj_cart = sum(sum(np.array(loss_breakdown_cart)))

        self.cart_obj = obj_cart
        self.cart_regrl = regrl_cart # maybe use this as fallback if cannot find good soln in 50% iter
        self.cart_MSE = kmse_cart
        
    # Auxiliary function used in SA_Pattern based to display current strata state
    def print_strata(self,regrl):
        print("Number of strata: ", len(regrl))
        print("Rules in the strata: ")
        for s,drule in enumerate(regrl):
            print("[",s,"]: Y = ", drule['mu'], "is predicted by:", drule['drule'], "supp=",len(drule['indices']))
    
    def print_strata_n_breakdown(self,regrl,breakdown):
        print("Number of strata: ", len(regrl))
        print("Rules in the strata: ")
        for s,drule in enumerate(regrl):
            print("[",s,"]: Y =", drule['mu'], "is predicted by:", drule['drule'], "supp=",len(drule['indices']),"loss=", breakdown[s][0])


    def fit(self, X, y):
        #X,y = check_X_y(X,y)
        self.trainX = X
        self.trainY = np.array(y)
        self.nobs = len(self.trainY)
        self.yrange = np.ptp(self.trainY)
        self.stratcvbd = (self.nobs * self.c2) / self.yrange**2
        self.isetcvbd = (self.nobs * self.c1) / self.yrange**2 


        self.generate_rules(self.minsupp, self.max_depth, self.ntrees)

        self.train()

        return self

    def train(self,weights = None,print_message=False,init = None):
        self.prefix = []
        self.bestobj_t = []
        self.obj_t =[]
        print('Using c1 = ', self.c1, 'c2 = ', self.c2, flush=True)
        if print_message:
            print('Searching for an optimal solution...')

        T0 = self.yrange * self.tempfactor
        trainMSEs = []
        elapsed_time = []
        # initialize with a random pattern set
        if init == None:
            #INITIALIZE STRATA HERE
            regrl_curr = [] # disrl is a list of dictionaries, each dictionary contains the rules, mu, and indices of instances in the stratum
            #randomly pick 5 rules
            
            init_rule_indices = sample(list(range(len(self.rules))),min(len(self.rules),self.initial_pool_size))
            init_rule_mu = self.rule_mu[init_rule_indices]
            try:
                [init_rule_indices,init_rule_mu]=[list(x) for x in zip(*sorted(zip(init_rule_indices,init_rule_mu),key = lambda t:t[1]))]
            except Exception as e:
                print(e)
                print('Failed with the following initial rule set size: ', len(self.rules))
                return None

            #sort rules according to mu
            regrl_curr = [{'drule':[x]} for x in init_rule_indices]
            regrl_curr.append({'drule':[]})
        else:
            print('Taking the init model')
            regrl_curr = deepcopy(init)

        # assign instances to strata
        self.stratify(regrl_curr)

        self.clean_regrl(regrl_curr, print_message)
                
        # Compute the costs of each strata
        loss_breakdown_curr=  self.compute_loss(regrl_curr)
        
        k = len(regrl_curr)
        kmse_curr = sum([x[0] for i,x in enumerate(loss_breakdown_curr) if i <k])
        self.loss_breakdown_curr = loss_breakdown_curr
        obj_curr = sum(sum(np.array(loss_breakdown_curr)))
        if self.cart_obj != 0 and self.cart_init:
            print('obj_curr, cart_obj: ', obj_curr, self.cart_obj)
            self.bestobj = min(obj_curr, self.cart_obj)
            self.bestregrl = self.cart_regrl
            self.bestmodel_MSE = self.cart_MSE
        else:
            self.bestobj  = obj_curr
            self.bestmodel_MSE = np.sum([x[0] for x in loss_breakdown_curr])
            self.bestregrl = regrl_curr
        # Begin Stochastic Local Search
        loss_breakdown_best = loss_breakdown_curr
        for it in tqdm(range(self.Niteration)):

            T = T0 / (1 + math.log(1 + it))#**(1 - it/self.Niteration)
            regrl_new = deepcopy(regrl_curr)

            if print_message:
                print('\n============== iter = {} =============='.format(it))
                print('T0,it,1-it/nit,T',T0,it,(1-it/self.Niteration),T)
                print('before proposing')
                self.print_strata_n_breakdown(regrl_new,loss_breakdown_curr)

            k, move = self.propose(regrl_new,loss_breakdown_curr,print_message = print_message)
            
            self.stratify(regrl_new)
            self.clean_regrl(regrl_new,print_message)

            loss_breakdown_new =  self.compute_loss(regrl_new)
            if print_message:
                print('\n after stratify and clean:')
                self.print_strata_n_breakdown(regrl_new,loss_breakdown_new)

            try:
                k = min(k,len(regrl_new))
                self.prefix.append(sum([len(regrl_new[s]['indices']) for s in range(k)]))
                kmse_new = sum([x[0] for i,x in enumerate(loss_breakdown_new) if i <k])/(0.00001+sum([len(regrl_new[s]['indices']) for s in range(k)]))
            except Exception:
                print("Fatal: failed to compute MSE of prefix!")
                print(k, len(regrl_new))
                sys.exit()
            
            obj_new = np.sum(np.array(loss_breakdown_new))

            try:
                if obj_new < obj_curr:
                    alpha1 = 1
                else:
                    alpha1 = math.exp(float(-obj_new +obj_curr)/T)
            except Exception:
                print('alpha1 overflow')
                alpha1 = float('inf')
            alpha2 = kmse_new < kmse_curr            
            
            MSE_new = np.sum([x[0] for x in loss_breakdown_new])
            testErr = 0
            
            if it % 50 == 0 and print_message:
                print ("iter = : ", it)
                print('pt_new = {}, pt_curr = {}, pt_min = {}, T = {}, alpha1 = {},alpha2={}'.format(obj_new,obj_curr,self.bestobj,T,alpha1,alpha2 ))
                print('break down = {}'.format(loss_breakdown_new))
                self.print_strata(regrl_new)
                print('\n')

            if print_message == True:
                print('\npt_new = {}, pt_curr = {}, pt_min = {}, T = {}, alpha1 = {}, alpha2 = {}'.format(obj_new,obj_curr,self.bestobj,T,alpha1,alpha2 ))
                print('break down = {}'.format(loss_breakdown_new))
                print('\n')

            mus = [np.float(regrl_new[s]['mu']) for s in range(len(regrl_new))]
            
            if obj_new < self.bestobj and not np.isnan(np.array(mus)).any():
                self.bestobj = obj_new
                self.bestregrl = deepcopy(regrl_new)
                self.loss_breakdown_best = loss_breakdown_new
                loss_breakdown_best = loss_breakdown_new
                self.bestmodel_MSE = MSE_new
                self.bestobj_t.append(obj_new)
                
            if print_message:
                print("lb: ", loss_breakdown_new)
                print("strat size: ", len(regrl_new))
                print("mse: ", MSE_new)
                print('\n*** found a new sol *** at iter = ', it)
                if MSE_new == 0:
                    print('===============0 at iter =====================', it)
                self.print_strata(regrl_new)
                print('\npt_new = {}, pt_curr = {}, pt_min = {}, T = {}, alpha1 = {}, alpha2 = {}'.format(obj_new,obj_curr,self.bestobj,T,alpha1, alpha2 ))
                self.print_strata_n_breakdown(regrl_new,loss_breakdown_new)

                
            if MSE_new < self.bestMSE:
                self.bestMSE = MSE_new
                self.bestMSE_regrl = deepcopy(regrl_new)

            defaultmu = regrl_new[len(regrl_new)-1]['mu']
            defaultrset = regrl_new[len(regrl_new)-1]['drule']
            mus = [np.float(regrl_new[s]['mu']) for s in range(len(regrl_new))]
            
            # accept proposal with probability governed by temperature/error, making sure model is not degenerate
            if (random() <= alpha1) and not np.isnan(np.array(mus)).any() and len(regrl_new) != 0 and len(mus) == len(set(mus)) and len(defaultrset) == 0:
                
                if print_message:
                    print('=== accept the proposal ==== ')
                    print('had prob <= alpha = ', alpha1, 'new/old obj: ', obj_new, obj_curr)
                    if obj_new < obj_curr:
                        print('new ones was better')

                obj_curr,loss_breakdown_curr, regrl_curr, kmse_curr= obj_new,loss_breakdown_new, deepcopy(regrl_new), kmse_new
                self.obj_t.append(obj_new)

            it=it+1
        if True:#print_message:
            print("Lowest attained MSE is: ", self.bestMSE, ' using ', it, ' iters')
            print("Final achieved objective value is: ", self.bestobj)
            print("Number of strata: ", len(self.bestregrl))
            print("Number of rules: ", sum([len(self.bestregrl[s]['drule']) for s in range(len(self.bestregrl)-1)]))
            print("df.shape: ", self.trainX.shape)
                
        return self.bestregrl
        
        
    #@profile
    def propose(self, regrl, loss_breakdown, print_message = False):
        tstart = time.time()
        self.loss_breakdown = loss_breakdown
        self.regrl = regrl
        # calculate costs for each strata
        # then find minimal k such that strata up to k are so bad you MUST optimize them
        costs_strata = [sum(row) for row in loss_breakdown]
        activct = 0
        if self.enableprefix and len(regrl) > 1:
            cumsum = np.cumsum(np.array(costs_strata))
            k = len(regrl)
            for i, x in enumerate(cumsum):
                if x >= self.bestobj:
                    k = i
                    break

            if print_message:
                print("limited to strata: ", k, " of ", len(regrl), " strata")
            
        else:
            k = len(costs_strata)

        # limit to the worst strata
        # repeatedly attempt to choose a move and avoid edge cases
        # here s and j are given as in the conference paper
        valid_move = False
        
        count = 0
        if len(regrl)==0:
            move = ['insert', 0]
        else:
            self.k = k
            self.costs_strata = costs_strata
            try:
                if k > 0:
                    s = np.random.choice(np.arange(k), p = np.ravel(costs_strata[:k])/sum(costs_strata[:k]))
                else:
                    s = 0
            except Exception:
                s = 0
            self.s = s

            if s not in range(len(regrl)):
                s = 0

            while valid_move == False:
                prob_arr = np.ravel(costs_strata[:k])/sum(costs_strata[:k])
                try:
                    s = np.random.choice(np.arange(k), p = prob_arr)
                    j = np.random.choice(3, p = np.array(loss_breakdown[s])/sum(loss_breakdown[s]))
                except:
                    s = 0
                    
                    if len(regrl) >= 2:
                        s = len(regrl)-2
                    else:
                        s = 0
                    j = np.random.choice(3, p = [1/3,1/3,1/3])
                self.s = s
                self.regrl = regrl
                self.k = k
                count += 1
                if j == 1: # reduce the size of the rules in stratum s
                    if len(regrl) > 1:
                        valid_move = True
                        move = ['cut', s]
                elif j == 2: # reduce the number of strata
                    if len(regrl)>=2 :
                        valid_move = True
                        move = ['merge', s]
                else:
                    action = np.random.choice(5,p = [1.0/8,1.0/8,1.0/8,1.0/8,1.0/2])
                    
                    if action == 0:
                        if len(regrl[s]['drule']) >= 1:
                            valid_move = True
                            move = ['cut',s]
                    elif action == 1:
                        if s > 0 and regrl[s-1]['drule'] != [] and len(regrl) > 1 and s -1 != len(regrl)-1:
                            move = ['add',s-1]
                            valid_move = True 
                        elif regrl[s]['drule'] != [] and s != len(regrl)-1:
                            move = ['add', s]
                            valid_move = True           
                    elif action == 2:
                        if len(regrl[s]['drule']) > 1 and regrl[s]['drule'] != []:
                            valid_move = True
                            move = ['split',s]
                    elif action == 3:
                        if len(regrl) >= 2 and s< len(regrl)-1 and len(regrl[s]['drule']) >= 1 and regrl[s]['drule'] != []:
                            move = ['replace',s]
                            valid_move = True
                    else:
                        if s != len(regrl) - 1 or len(regrl) == 1:
                            move = ['insert', s]
                            valid_move = True
                            if valid_move and len(regrl) == 1 and print_message:
                                print('chose insert for single rule model')

                covered = []
                if valid_move == True:
                    
                    for k in range(s):
                        covered = covered + regrl[k]['indices']
                    if len(covered) == self.nobs and move[0] in ['insert', 'replace', 'add']:
                        valid_move = False
                    
                if count >= 10000 and print_message:
                    print('STUCK')
                    print('len regrl = {}'.format(len(regrl)))
                    self.print_strata(regrl)
                
                if count >10010:
                    self.print_strata(regrl)
                    raise Exception('COULD NOT FIND A MOVE!!!')
                    #sys.exit()
                    
        self.excecute_move(move,regrl,print_message)
        
        if print_message:
            print('***{} ***'.format(move))

        return k, move[0]


    def cut(self, regrl, s, print_message=False, randomcutprob=.7):
        self.regrl = regrl
        if len(regrl[s]['drule'])>1:
            if random() > randomcutprob:
                regrl[s]['drule'].pop(sample(list(range(len(regrl[s]['drule']))),1)[0])
                self.stratify(regrl)
            else:
                if print_message:
                    print('cut with heuristic')
                
                weights = abs(np.array(regrl[s]['mu'],dtype=np.float64) - np.float64(self.rule_mu[regrl[s]['drule']]))#,dtype=np.float64))
                
                if 0 in list(weights):
                    weights = np.array([.000000001 + w for w in weights])
                
                weights = weights / np.sum(weights)
                weights = np.array(weights, dtype = np.float64)
                idx = np.array(range(len(regrl[s]['drule'])),dtype=np.intp)
                choice = np.random.choice(idx, p =weights)
                regrl[s]['drule'].pop(choice)
                self.stratify(regrl)
        else:
            if len(regrl) > 1:
                regrl.pop(s)
            self.stratify(regrl)

    def add(self,regrl,s, print_message=False,randomcutprob=.7):
        """ add """
        tads = time.time()
        if random()>randomcutprob and regrl[s]['drule'] != []:
            ct = 0
            while True:
                ct = ct + 1
                newrule = sample(list(np.arange(len(self.rules))), 1)[0]

                if ct > 10000:
                    print("STUCK IN ADD")
                    print('Breaking...')
                    break
                if newrule not in regrl[s]['drule']:
                    regrl[s]['drule'].append(newrule)
                    break
                     
        else:
            covered = []
            
            for k in range(s+1):
                covered = covered + regrl[k]['indices'] 

            leftover = np.array([int(i) for i in range(self.nobs) if i not in covered])
            
            if regrl[s]['drule'] != [] and len(leftover) > 0 and self.enablemsesort:
                try:
                    tmeanmu1 = time.time()
                    Y_meanmu1 =(regrl[s]['mu']*len(regrl[s]['indices'])+ np.dot(self.trainY[leftover].T,self.RMatrix[leftover,:]))/(len(regrl[s]['indices']) + np.sum(self.RMatrix[leftover],axis = 0))
                except Exception:
                    print('meanmuproblem: ', leftover, s)
                        
                mse = np.zeros(len(Y_meanmu1))
                K =np.array([self.trainY[leftover[clinalg.cwhere(self.RMatrix[leftover,i])]] for i in range(self.RMatrix.shape[1])])
                
                if self.trainY[regrl[s]['indices']].ndim == 2:
                    print("============ndim error for Y")
                # k is empty
                if K.ndim == 2 or K.size == 0:
                    mse = clinalg.cmse2(float(regrl[s]['mu']),
                                       self.trainY[regrl[s]['indices']],
                                       self.trainY[leftover].T,
                                       self.RMatrix[leftover],
                                       Y_meanmu1)
                
                else:
                    lens = np.array([len(k) for k in K])
                    mse = clinalg.cmse1(float(regrl[s]['mu']),self.trainY[regrl[s]['indices']],self.trainY[leftover].T,self.RMatrix[leftover],K,Y_meanmu1,lens)
                    
                choices = mse.argsort()[:10]
                newrule = sample(list(choices),1)[0]
                self.newrule = newrule

            elif regrl[s]['drule'] != [] and len(leftover) > 0 and not self.enablemsesort:
                newrule = sample(list(np.arange(len(self.rules))), 1)[0]
                self.newrule = newrule

                
            if self.newrule>=len(self.rules):
                print('here is an error and choices =', choices)
                    
            if self.newrule not in regrl[s]['drule'] and regrl[s]['drule'] != []:
                regrl[s]['drule'].append(self.newrule)
                
                    
    def replace(self,regrl,s,print_message,randomcutprob):
        # first cut
        treps = time.time()
        if random() > randomcutprob:
            regrl[s]['drule'].pop(sample(list(range(len(regrl[s]['drule']))),1)[0])

        else:
            weights = abs(np.array(regrl[s]['mu'],dtype=np.float64) - np.array(self.rule_mu[regrl[s]['drule']],dtype=np.float64))

            weights = np.array(weights,dtype=np.float64)
            
            if 0 in list(weights):
                weights = np.array([.00000000001 + w for w in weights])
            weights = weights / np.sum(weights)
            
            a = len(regrl[s]['drule'])
            
            if a > 1:
                choice = np.random.choice(range(a), p = weights)
            else:
                choice = 0
            regrl[s]['drule'].pop(choice)      
        # add
        self.stratify(regrl)
        covered = []
        for k in range(s):
            covered = covered + regrl[k]['indices'] 

        leftover = np.array([i for i in range(self.nobs) if i not in covered])
        if len(leftover) > 0 and self.enablemsesort:
            try:
                Y_meanmu1 =(regrl[s]['mu']*len(regrl[s]['indices'])+ np.dot(self.trainY[leftover].T,self.RMatrix[leftover,:]))/(len(regrl[s]['indices']) + np.sum(self.RMatrix[leftover],axis = 0))
            except Exception:
                print('meanmu prob: ', leftover, s)
               
            
            mse = np.zeros(len(Y_meanmu1))
            K =np.array([self.trainY[leftover[clinalg.cwhere(self.RMatrix[leftover,i])]] for i in range(self.RMatrix.shape[1])])
        
            # k is empty
            if K.ndim == 2 or K.size == 0:
                mse = clinalg.cmse2(float(regrl[s]['mu']),
                                       self.trainY[regrl[s]['indices']],
                                       self.trainY[leftover].T,
                                       self.RMatrix[leftover],
                                       Y_meanmu1)
                
            else:
                lens = np.array([len(k) for k in K])
                mse = clinalg.cmse1(float(regrl[s]['mu']),self.trainY[regrl[s]['indices']],self.trainY[leftover].T,                                      self.RMatrix[leftover],K,Y_meanmu1,lens)
            
            choices = mse.argsort()[:1]
            newrule = sample(list(choices),1)[0]
            self.newrule = newrule

            
        elif regrl[s]['drule'] != [] and len(leftover) > 0 and not self.enablemsesort:
            newrule = sample(list(np.arange(len(self.rules))), 1)[0]
            self.newrule = newrule
        if self.newrule>=len(self.rules):
            print('here is an error and choices =', choices)
        if self.newrule not in regrl[s]['drule'] and regrl[s]['drule'] != []:
            regrl[s]['drule'].append(self.newrule)

                    
    def split(self, regrl, s, print_message, randomcutprob):
        if s > 1 and s<len(regrl)-1 and regrl[s]['drule'] != []:
            rules = np.array(regrl[s]['drule'])
            muranks = [(r, self.rule_mu[r]) for r in rules]
            muranks = sorted(muranks, key = lambda x: x[1])
            cut = sample(list(range(1,len(rules))),1)[0]
            regrl[s]['drule'] = list(rules[:cut])
            regrl.insert(s+1, {'drule':list(rules[cut:])})
                
    def merge(self,regrl,s,print_message,randomcutprob):
        ver = 0
            
        if print_message:
            print('performing random merge with stratum', s)
        
        if s==len(regrl)-1:
            ver = 0
            regrl.pop(s-1)
            
        elif (s == 0  or (random() < 0.5 and s< len(regrl)-1)) and regrl[s + 1]['drule'] != [] and regrl[s]['drule'] != []:
            # merge with lower stratum
            ver = 1
            regrl[s]['drule'] = regrl[s]['drule'] + regrl[s+1]['drule']
            regrl.pop(s+1)
        else:
            
            if s > 0 and regrl[s]['drule'] != [] and regrl[s-1]['drule'] != []:   
                ver = 2
                if regrl[s]['drule'] == [] or regrl[s-1]['drule'] == []:
                    print("------------------------------------ERRANT MERGE")
                regrl[s]['drule'] = regrl[s]['drule'] + regrl[s-1]['drule']
                regrl.pop(s-1)
            
        self.stratify(regrl)

    #@profile
    def insert(self,regrl,s,print_message,randomcutprob): 
        newrule = sample(list(np.arange(len(self.rules))), 1)[0]
        self.newrule = newrule
        if regrl[s]['drule'] != []:
            regrl.insert(s,{'drule':[newrule]})
            

    def excecute_move(self,move,regrl,print_message = False,randomcutprob = .7):#, q):
        s = move[1]
        self.s = s
        
        if move[0]=='cut':
            self.cut(regrl,s,False,randomcutprob)
            
        elif move[0] == 'add': 
            self.add(regrl,s,False,randomcutprob)
            
        elif move[0] == 'replace':
            self.replace(regrl,s,False,randomcutprob)
            
        elif move[0] == 'split':
            self.split(regrl,s,False,randomcutprob)
            
        elif move[0] == 'merge':
            self.merge(regrl,s,False,randomcutprob)
            
        elif move[0] == 'insert':
            self.insert(regrl,s,False,randomcutprob)
            
    def compute_loss(self, regrl): 
        loss_breakdown = np.array([[np.sum((regrl[s]['mu']-self.trainY[regrl[s]['indices']])**2)/self.nobs, self.c1*len(regrl[s]['drule']), self.c2] for s in range(len(regrl))])
        loss_breakdown = [[np.asscalar(np.asarray(it)) for it in x] for x in loss_breakdown]
       
        return loss_breakdown
    
    def predict(self,X):
        Z = np.zeros((X.shape[0],len(self.regrl)))
        for s in range(len(self.regrl)-1):
            for irule in self.regrl[s]['drule']:
                w = np.zeros(X.shape[0])
                for cond in self.rules[irule]:
                    condition = re.split('[>=]|<',cond)
                    
                    if len(condition) == 2:
                        
                        w = w + np.array(X[condition[0]]<float(condition[1])).astype(int)

                    else:
                        
                        w = w + np.array(X[condition[0]]>=float(condition[2])).astype(int)

                Z[:,s] = Z[:,s] + (w==len(self.rules[irule])).astype(int)
            Z[:,s]= np.array(Z[:,s]>0).astype(int)
        Z[:,len(self.regrl)-1] = 1
        
        ypred = np.zeros(X.shape[0])
        
        for s in range(len(self.regrl)-1,-1,-1):
            ypred[np.where(Z[:,s])[0]] = self.regrl[s]['mu']
        
        return ypred

    def printmodel(self, regrl):
        for s in range(len(regrl) - 1):
            for k, irule in enumerate(regrl[s]['drule']):
                if k > 0:
                    print('\nOR')
                if k == 0:
                    print()
                print('IF', end=' ')
                for l, cond in enumerate(self.rules[irule]):
                    if l < len(self.rules[irule]) - 1:
                        print(str(cond) + ' AND ', end = ' ')
                    else:
                        print(str(cond), end = ' ')

            print('\nTHEN Y = ' + str(round(regrl[s]['mu'],2)), end = ' ')
            print('with supp = ' + str(len(regrl[s]['indices'])))

        print('\nELSE Y = ' + str(round(regrl[len(regrl) - 1]['mu'],2)), end = ' ')
        print('with supp = ' + str(len(regrl[len(regrl) - 1]['indices'])))

    def writemodel(self, regrl, filename, nstrat, nrules, MAE, RRMSE, c1, c2):
        fdf = open(filename, "a+")
        print('open file')
        fdf.write("\n\n\n")
        fdf.write('c1, c2: ' + str(c1) + ' ' + str(c2) + '\n')
        for s in range(len(regrl) - 1):
            for k, irule in enumerate(regrl[s]['drule']):
                if k > 0:
                    fdf.write('\nOR\n')
                if k == 0:
                    fdf.write('\n')
                fdf.write('IF ')
                for l, cond in enumerate(self.rules[irule]):

                    if l < len(self.rules[irule]) - 1:
                        fdf.write(str(cond) + " AND ")
                    else:
                        fdf.write(str(cond))

            fdf.write("\nTHEN Y = $" + str(regrl[s]['mu']))
            fdf.write(" with supp = " + str(len(regrl[s]['indices'])) + "\n")

        fdf.write('ELSE Y = $' + str(regrl[len(regrl) - 1]['mu']))
        fdf.write(" with supp = " + str(len(regrl[len(regrl) - 1]['indices'])) +
                  "\n")
        fdf.write('NSTRAT: ' + str(nstrat) + ' NRULES: ' + str(nrules) +
                  ' MAE: ' + str(MAE) + ' RRMSE: ' + str(RRMSE))
        print('end show model')




def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        return 0

def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def getConfusion(Yhat,Y):
    if len(Yhat)!=len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y),np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y)-FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP,FP,TN,FN

def get_lineage(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = '<'
        else:
            parent = np.where(right == child)[0].item()
            split = '>='
        lineage.append(features[parent]+split+str(threshold[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    rules = {}
    newrule = []
    for child in idx:
        for node in recurse(left, right, child):
            if type(node)!=str:
                rules[node] = newrule
                newrule = []
            else:
                newrule.append(node)

    return rules

