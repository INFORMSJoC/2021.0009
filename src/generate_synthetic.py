from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
#from dataprocessor2 import *
#from neds_proc import *
import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pickle
import time
import csv

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from RegRL import * 
from sklearn.datasets import make_regression
from sklearn.preprocessing import normalize
import sys
#from plot_utils import *
from symreggen import *
from tqdm import tqdm

def rrmse(y, y_pred):
    return np.sqrt(1 - r2_score(y, y_pred))

#matplotlib.use('TKAgg')
runs = 5#30
n_sample =10000# 20000
n_features = 10#200
noise_pct = 0
noise_magnitude = 0
print('n_sample should be 10k')
print('n_sample: ', n_sample, flush=True)
pickleres = []

iters = 10000

rrmses_cart = np.zeros(iters)
rrmses_nocart = np.zeros(iters)

ks = [5]

#for i,n in enumerate([n_sample]):
for cart_init in [True, False]:
    trainVec = []
    testVec = []
    rrmses = []
    timeVec = []
    for k in ks:#tqdm(range(1,runs)):
        print('n = ', n_sample, ' run/seed = ', k,flush=True)

        np.random.seed(k)
        m = ''
        for i in range(1,n_features+1):
            c = str(np.random.randint(1,10)) + '*x' + str(i)
            c += np.random.choice(['+', '-'], p = [0.5, 0.5])
            m+= c
        m = m[:-1]
        print('m = ', m)


        x = gen_regression_symbolic(m=m,
                                    n_samples=n_sample, 
                                    n_features=n_features,
                                    noise_magnitude=noise_magnitude,
                                    noise_pct=noise_pct
        )
        np.random.shuffle(x)
        print('done gen data!', flush=True)
        try:
            
            split = int(np.floor(.8*n_sample))
            training, test = x[:split,:], x[split:,:]

            # save the data
            training = pd.DataFrame(data=training, index=range(training.shape[0]),
                                    columns = ['x'+str(k) for k in range(training.shape[1]-1)]+['Y'])
            test = pd.DataFrame(data=test, index=range(test.shape[0]),
                                columns = ['x'+str(k) for k in range(test.shape[1]-1)]+['Y'])

            training.to_csv('synth/synthetic'+str(n_sample)+'_train'+str(k)+'.csv')
            test.to_csv('synth/synthetic'+str(n_sample)+'_test'+str(k)+'.csv')
            #print('SKIPP REST',flush=True)
            #continue
        except Exception as e:
            print(e)
            pickle.dump(x, open('synth/data20k_'+str(k)+'.p','wb'))
            print('dumped instead at ', k)
            continue

        trainy = training['Y']
        trainX = training.drop('Y',axis=1)
        testy = test['Y']
        testX = test.drop('Y',axis=1)


        '''

        trainy = np.array(training['Y'])#training[:,-1]).ravel()

        print(np.mean(trainy))

        trainX = np.array(training.drop('Y',axis=1))#[:,:-1])

        testy = np.array(test['Y'])#test[:,-1]).ravel()
        testX = np.array(test.drop('Y',axis=1))#test[:,:-1])

         #trainy = pd.DataFrame(data=trainy,index=range(trainy.shape[0]),
         #                      columns = range(1)).flatten()
        trainy = pd.Series(trainy)
        trainX = pd.DataFrame(data=trainX,index=range(trainX.shape[0]),
                                  columns = [str(k) for k in range(trainX.shape[1])])

        #testy = pd.DataFrame(data=testy,index=range(testy.shape[0]),
        #                     columns = range(1)).flatten()
        testy = pd.Series(testy)
        testX = pd.DataFrame(data=testX,index=range(testX.shape[0]),
                                 columns = [str(k) for k in range(testX.shape[1])])

        '''
        print(trainy.shape, testy.shape)
        supp = 5#15
        treedepth = 4
        forest_size = 128
        #iters = 10000#500
        start_time = time.time()
        temp = .5#.005#np.random.uniform(1,10) * (10**(-4))
        #model = DRL(trainX, trainy, 'randomforest', testX, testy, True, True,False, temp)
        model = REGRL(trainX, trainy, temp, cart_init)
        c1= 0#.00000002#.0001#np.random.uniform(0,.0000005)
        c2= 0#.00000001#.001#np.random.uniform(0,.00005)
        model.set_parameters(c1, c2)
        gen_start = time.time()
        try:
            model.generate_rules(supp=supp, maxlen=treedepth,forest_size=forest_size)
        except Exception as e:
            print(e)
            continue
        gen_time = time.time()
        RegRL, trainRRMSEs = model.train(Niteration=iters,print_message=False)

        
        xs = np.array(list(range(iters)))

        plt.clf()
        #save / plot data
        if cart_init:
            rrmses_cart += trainRRMSEs
            plt.plot(xs, trainRRMSEs,c='b')
            plt.title('cart init run ' + str(k))
            plt.show()
            plt.savefig('cart'+str(k)+'.png')
            data = {'x': xs, 'y':trainRRMSEs}
            pickle.dump(data, open('cart_k='+str(k)+'.p', 'wb'))
        else:
            rrmses_nocart += trainRRMSEs
            plt.plot(xs, trainRRMSEs)
            plt.title('no cart init run ' + str(k))
            plt.show()
            plt.savefig('no_cart'+str(k)+'.png')
            data = {'x': xs, 'y':trainRRMSEs}
            pickle.dump(data, open('nocart_k='+str(k)+'.p', 'wb'))

        train_time = time.time()
        timeVec.append(train_time - start_time)

        y_pred_test = model.predict(RegRL,testX)
        pred_time = time.time()
        #print("total time: ", pred_time - start_time)
        #print("rule generation time: ", gen_time - gen_start)
        #print("train time: ", train_time - gen_time)
        #print("prediction time: ", pred_time - train_time)
        try:
            drl_pred_err = mean_squared_error(y_pred_test, testy)
            print('rrmse: ', rrmse(testy, y_pred_test))
            rrmses.append(drl_pred_err / np.var(testy))#rrmse(testy, y_pred_test))
        except Exception:
            print("MSE ERROR")

        testVec.append(drl_pred_err)
        #print(y_pred_test[:10])
        #print(testy[:10])

        initmse = drl_pred_err
        #print("===================INITMSE: ", drl_pred_err)
        nstrat = len(RegRL)
        nrules = sum([len(RegRL[s]['drule']) for s in range(len(RegRL)-1)])
        #print('nstrat nrule', nstrat, nrules)
     
    # below is bugged?
    #print('\nruns,samples,features,noise mag prop,noise prop: ', runs, n_sample, n_features,noise_magnitude, noise_pct)
    #print('Avg MSE: ', np.average(testVec))
    #print('Avg RRMSE: ', np.average(rrmses))


rrmses_cart = rrmses_cart / runs
rrmses_nocart = rrmses_nocart / runs

plt.plot(list(range(len(rrmses_cart))), rrmses_cart,c='b', label='CART init')
plt.plot(list(range(len(rrmses_nocart))), rrmses_nocart,c='r',label='No CART init')
plt.legend()
plt.title('conv')
plt.show()
plt.savefig('convtest22.png')
