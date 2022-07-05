import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pandas
from sklearn.model_selection import train_test_split
from itertools import product
from scipy.stats import pearsonr
import random

def sample_formulation(seq, k, kmers):
    base = 'ATCG'
    for i in range(len(seq)):
        if seq[i] == 'N':
            seq = seq[:i] + base[random.randint(0, 3)] + seq[i+1:]

    for i in range(len(kmers)):
        kmers[i] = ''.join(kmers[i])
    kmer_in_seq = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmer_in_seq.append(kmer)
    cnt = []
    for i in kmers:
        cnt.append(kmer_in_seq.count(i))
    feature_vector = [str(c/len(kmer_in_seq)) for c in cnt]
    return feature_vector

def k_mer(k, filename):
    base = 'ATCG'
    kmers = list(product(base, repeat=k))
    with open(filename) as fa:
        lines = fa.readlines()
    seq = []
    for i in range(len(lines)):
        if i%2 == 1:
            seq.append(lines[i].strip().upper())
    feature_vectors = []
    for j in seq:
        #print(sample_formulation(j, k, kmers))
        #print(len(feature_vectors))
        feature_vectors.append(sample_formulation(j, k, kmers))

    filename = filename.split('.')
    feature_vectors = pandas.DataFrame(feature_vectors)
    feature_vectors.to_csv(filename[0] + '_' + str(k) + 'mer' + '.npy', index=False, header=False, encoding='gbk',
                           float_format='%.4', sep=" ")


def read_file(dataset, filename):
    with open(filename,'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            num = line.split(' ')
            dataset.append(np.asarray(num, dtype=float))
    file.close()
    return dataset

def read_dataset(seq_filename, y_filename):
    x = []
    with open(seq_filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            num = line.split(' ')
            x.append(np.asarray(num, dtype=float))
    file.close()
    y = pd.read_csv(y_filename)
    y = np.asarray(y['Log'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=23, test_size=0.2)
    print(len(x_train))
    print(len(x_test))
    return x_train, y_train, x_test, y_test

#网格寻优
def GridSearch(x_train, y_train, model, params):

    model.fit(x_train, y_train)
    model = GridSearchCV(estimator=model, param_grid=params, cv=10,n_jobs=4, verbose=100)
    model.fit(x_train, y_train)

    result = open('dataset/result_regression.txt', 'a')
    result.writelines("Best parameters:")
    result.writelines(str(model.best_params_))
    result.writelines("Grid scores:")
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        result.writelines("%0.5f (+/-%0.05f) for %r"
                          % (mean, std * 2, params))
    result.close()

def output_result(model,x_train, y_train,x_test, y_test):

    model = model.fit(x_train, y_train)
    '''
    print(pearsonr(y_train, model.predict(x_train)))
    print(pearsonr(y_test, model.predict(x_test)))
    print(r2_score(y_train, model.predict(x_train)))
    print(r2_score(y_test, model.predict(x_test)))
    '''
    result = open('dataset\\result_regression.txt', mode='a')
    print(pearsonr(y_train, model.predict(x_train)), file=result)
    print(pearsonr(y_test, model.predict(x_test)), file=result)
    print(r2_score(y_train, model.predict(x_train)), file=result)
    print(r2_score(y_test, model.predict(x_test)), file=result)

if __name__ == '__main__':
    k_mer(5, 'dataset/regression/x-2k_sequence.fa')
    x_train, y_train, x_test, y_test = read_dataset('dataset/regression/x-2k_sequence_5mer.npy',
                                                    'dataset/regression/y.csv')
    #lasso
    params = {'alpha': [0.0000001, 0.000001, 0.000005,0.00001,0.0001,0.001,0.1,1,10]}
    model = Lasso(max_iter=1000)
    GridSearch(x_train, y_train, model, params)
    
    model = Lasso(alpha=0.000001, max_iter=2000)
    output_result(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    #Ridge
    params = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1]}
    model = Ridge()
    GridSearch(x_train, y_train, model, params )
    
    model = Ridge(alpha=0.01)
    output_result(model, x_train, y_train, x_test, y_test)

    #SVR
    model = svm.SVR()
    params = [
        {'kernel': ['linear'], 'C': [100, 1000, 5000]},
        {'kernel': ['rbf'], 'C': [100, 1000, 5000], 'gamma': [1, 0.1, 0.001]},
        {'kernel': ['ploy'], 'C': [100, 1000, 5000], 'degree': [2, 3]}
    ]
    GridSearch(x_train, y_train, model=model, params=params)

    model = svm.SVR( kernel='rbf', gamma=1, C=100, verbose=100)
    output_result(model, x_train, y_train, x_test, y_test)

    #kNN
    params = {'n_neighbors': [5, 10, 15, 20, 25,30]}
    model = KNeighborsRegressor()
    GridSearch(x_train, y_train, model, params)
    
    model = KNeighborsRegressor(n_neighbors=25)
    output_result(model, x_train, y_train, x_test, y_test)


    # RF
    params = {  'n_estimators':range(100,501,50),
        'max_depth':range(10, 51, 10),
        'max_features': range(10, 51, 10)}
    model = RandomForestRegressor(min_samples_split=2)
    GridSearch(x_train, y_train, model, params)

    model = RandomForestRegressor(n_estimators=400, max_depth=30, max_features=40, min_samples_split=2)
    output_result(model, x_train, y_train, x_test, y_test)




