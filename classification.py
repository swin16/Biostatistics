import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn import discriminant_analysis

def delete_lines(lines):
    i = 1
    x = []
    for line in lines:
        if i%2 == 0:
            x.append(line)
        i += 1
    return x

def dataset_divide(filename, train_name, test_name):
    pos_filepath = filename
    with open(pos_filepath) as pos_file:
        x = delete_lines(pos_file)
    y = [1]*len(x)
    x_train,x_test, y_train,  y_test = train_test_split(x, y, test_size=0.2)

    x_pos_train_file = open(train_name, 'wt')
    for x in x_train:
        x_pos_train_file.writelines(x)
    x_pos_train_file.close()

    x_pos_test_file = open(test_name, 'wt')
    for x in x_test:
        x_pos_test_file.writelines(x)
    x_pos_test_file.close()

def sample_formulation(seq, k, kmers):
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
        seq.append(lines[i].strip().upper())
    feature_vectors = []
    for j in seq:
        #print(len(feature_vectors))
        feature_vectors.append(sample_formulation(j, k, kmers))

    filename = filename.split('.')
    feature_vectors = pandas.DataFrame(feature_vectors)
    feature_vectors.to_csv(filename[0] + '_' + str(k) + 'mer' + '.npy', index=False, header=False, encoding='gbk', float_format='%.4', sep = " ")

def read_file(dataset, filename):
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            num = line.split(' ')
            dataset.append(np.asarray(num, dtype=float))
    file.close()
    return dataset

def read_dataset(pos_train_filename,pos_test_filename, neg_train_filename,neg_test_filename ):

    x_train = []
    x_train = read_file(x_train, pos_train_filename)
    y_train = [1] * len(x_train) + [0] * len(x_train)
    x_train = read_file(x_train, neg_train_filename)
    #print(len(x_train))
    #print(len(y_train))

    x_test = []
    x_test = read_file(x_test, pos_test_filename)
    y_test = [1] * len(x_test) + [0] * len(x_test)
    x_test = read_file(x_test, neg_test_filename)
    #print(len(x_test))
    #print(len(y_test))


    return x_train, y_train, x_test, y_test

def output_result(model,x_train, y_train,x_test, y_test):

    model.fit(x_train, y_train)
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    train_pro = model.predict_proba(x_train)
    test_pro = model.predict_proba(x_test)

    train_acc = accuracy_score(y_train, train_predict)
    train_precision = precision_score(y_train, train_predict)
    train_recall = recall_score(y_train, train_predict)
    train_f1 = f1_score(y_train, train_predict)
    train_auc = roc_auc_score(y_train, train_pro[:,1])

    test_acc = accuracy_score(y_test, test_predict)
    test_precision = precision_score(y_test, test_predict)
    test_recall = recall_score(y_test, test_predict)
    test_f1 = f1_score(y_test, test_predict)
    test_auc = roc_auc_score(y_test, test_pro[:,1])
    '''
    print("train_acc: {0:.3f}, test_acc{1:.3f}".format(train_acc, test_acc))
    print("train_pre: {0:.3f}, test_pre{1:.3f}".format(train_precision, test_precision))
    print("train_recall: {0:.3f}, test_recall{1:.3f}".format(train_recall, test_recall))
    print("train_f1: {0:.3f}, test_f1{1:.3f}".format(train_f1, test_f1))
    print("train_auc: {0:.3f}, test_auc{1:.3f}".format(train_auc, test_auc))
    '''

    result = open('dataset\\result.txt', mode='a')
    print("train_acc: {0:.3f}, test_acc{1:.3f}".format(train_acc, test_acc), file=result)
    print("train_pre: {0:.3f}, test_pre{1:.3f}".format(train_precision, test_precision), file=result)
    print("train_recall: {0:.3f}, test_recall{1:.3f}".format(train_recall, test_recall), file=result)
    print("train_f1: {0:.3f}, test_f1{1:.3f}".format(train_f1, test_f1), file=result)
    print("train_auc: {0:.3f}, test_auc{1:.3f}".format(train_auc, test_auc), file=result)

#网格寻优
def GridSearch(x_train, y_train, model, params):

    model = GridSearchCV(estimator=model, param_grid=params, cv=10, n_jobs = 4, verbose=100)
    model.fit(x_train, y_train)
    result = open('dataset/result.txt', 'a')
    result.writelines("Best parameters:")
    result.writelines(str(model.best_params_))
    result.writelines("Grid scores:")
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        result.writelines("%0.5f (+/-%0.05f) for %r"
                          % (mean, std * 2, params))
    result.close()

if __name__ == '__main__':

    dataset_divide('dataset\\pos.fa', 'dataset\\pos_train.fa', 'dataset\\pos_test.fa')
    dataset_divide('dataset\\neg.fa', 'dataset\\neg_train.fa', 'dataset\\neg_test.fa')
    
    k_mer(5, 'dataset\\pos_train.fa')
    k_mer(5, 'dataset\\pos_test.fa')
    k_mer(5, 'dataset\\neg_train.fa')
    k_mer(5, 'dataset\\neg_test.fa')

    x_train, y_train, x_test, y_test = read_dataset('dataset/pos_train_5mer.npy', 'dataset/pos_test_5mer.npy','dataset/neg_train_5mer.npy', 'dataset/neg_test_5mer.npy' )

    #SVM寻参

    model = svm.SVC(probability=True)
    params = [
        {'kernel': ['linear'], 'C': [100, 1000, 5000]},
        {'kernel': ['rbf'], 'C': [100, 1000, 5000], 'gamma': [1, 0.1, 0.001]},
        {'kernel': ['ploy'], 'C': [100, 1000, 5000], 'degree': [2, 3]}
    ]
    GridSearch(x_train, y_train, model, params)
    
    model = svm.SVC(probability=True, kernel='rbf', gamma=1, C=100, decision_function_shape='ovo', verbose=100)
    output_result(model, x_train, y_train, x_test, y_test)

    #KNN寻参
    model = KNeighborsClassifier()
    params = {'weights': ['distance'], 'n_neighbors': [i for i in range(3, 24)]}
    GridSearch(x_train, y_train, model, params)
    
    model = KNeighborsClassifier(weights='distance', n_neighbors=16)
    output_result(model, x_train, y_train, x_test, y_test)

    #RF寻参
    model = RandomForestClassifier()
    params = {  'n_estimators':range(100,601,100),
        'max_depth':range(6, 43, 6),
        'max_features': range(30, 90, 5)}
    GridSearch(x_train, y_train, model, params)
    
    model = RandomForestClassifier(n_estimators=400, max_depth=36, max_features=65, min_samples_split=2)
    output_result(model, x_train, y_train, x_test, y_test)

    #Logistic寻参
    params = {'penalty': ['l1', 'l2'],
              'C': [10, 50, 100, 150, 200, 500, 600, 800, 1000]}
    model = LogisticRegression()
    GridSearch(x_train, y_train, model, params)

    model = LogisticRegression(penalty='l2', C=600)
    output_result(model,x_train,y_train,x_test,y_test)

    #LDA
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    output_result(lda, x_train, y_train, x_test, y_test)