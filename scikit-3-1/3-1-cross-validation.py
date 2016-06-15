'''
Cross-validation: evaluating estimator performance

http://scikit-learn.org/stable/modules/cross_validation.html
'''

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import KFold

def main():
    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            iris.data, iris.target, test_size=0.4, random_state=0)

    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    print clf.score(X_test, y_test)

def cv():
    iris = datasets.load_iris()

    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_validation.cross_val_score(
            clf, iris.data, iris.target, cv=5)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    from sklearn import metrics
    scores = cross_validation.cross_val_score(
            clf, iris.data, iris.target, cv=5, scoring='f1_weighted')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def kfold():
    kf = KFold(4, n_folds=2)
    for train, test in kf:
        print("%s %s" % (train, test))

def shufflesplit():
    from sklearn.cross_validation import LabelShuffleSplit
    labels = [1, 1, 2, 2, 3, 3, 4, 4]
    slo = LabelShuffleSplit(labels, n_iter=4, test_size=0.5, random_state=0)

    for train, test in slo:
        print("%s %s" % (train, test))

if __name__ == '__main__':
    main()
    cv()
    kfold()
    shufflesplit()
