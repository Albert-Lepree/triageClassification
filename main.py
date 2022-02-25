
import operator
from typing import List, Dict, Union
import matplotlib.pyplot as plt
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

#from util import load_data, Classifier, Example, evaluate, remove_stop_words

def main():
    train = pd.read_csv("./data/triage/train.csv", sep='|')
    dev = pd.read_csv("./data/triage/dev.csv", sep='|')

    # count how many of each label
    label_groups = train['Label'].groupby(train['Label'])

    # calculate as a percent
    #print(label_groups.count() / len(train))

    # display most common word from each document
    #freq(train)

    # display most unique word from each document
    #tfidf(train)

    # naive bayes model
    nbMultipleRuns(train)

    #nb(train)

def freq(df):
    # changes encoding type to string?
    count = CountVectorizer()
    bag_of_words = count.fit_transform(df['Text'].values.astype('U'))

    feature_names = count.get_feature_names_out()

    #pd.set_option("display.max_rows", 202, "display.max_columns", None)
    freq = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)
    print(freq.idxmax(axis=1))

# displays most unique word from each string
def tfidf(df):
    tfidf = TfidfVectorizer()
    feature_matrix = tfidf.fit_transform(df['Text'].values.astype('U'))

    tfdif = pd.DataFrame(feature_matrix.toarray(), columns=tfidf.get_feature_names_out())

    print(tfdif.idxmax(axis=1))

accList = []

def nb(df, testSize):
    X_train, X_test, y_train, y_test = train_test_split(df['Text'],
                                                        df['Label'],
                                                        test_size=testSize,
                                                        random_state=7)

    # print('Shape of X_train: ', X_train.dtype)
    # print('Shape of X_test: ', X_test.shape)
    # print('Shape of y_train: ', y_train.shape)
    # print('Shape of y_test: ', y_test.shape)

    tfidf = TfidfVectorizer()

    X_train_tfidf = tfidf.fit_transform(X_train.values.astype('U')).toarray()
    # print('tfidf train shape: ', X_train_tfidf.shape)

    X_test_tfidf = tfidf.transform(X_test.values.astype('U')).toarray()
    # print('tfidf test shape: ', X_test_tfidf.shape)

    # model
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)

    # test results: predict data
    predicted = clf.predict(X_test_tfidf)

    from sklearn import metrics
    acc = metrics.accuracy_score(y_test, predicted)
    accList.append(acc)
    # print("accuracy is: ", acc*100)

def nbMultipleRuns(df):

    testSizes = []

    for testSize in range(1, 16):
        temp = testSize/20
        testSizes.append(temp)
        nb(df, temp)


    graphData(testSizes)


def graphData(testSizes):

    data = {'Test Sizes': testSizes, 'Accuracy': accList}

    df = pd.DataFrame(data, columns=['Test Sizes', 'Accuracy'])

    print(df)

    df.plot(x='Test Sizes', y = 'Accuracy', kind='line')
    plt.show()

if __name__=='__main__':
    main()