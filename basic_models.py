"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import matplotlib.pyplot as plt  # for pyplot
import numpy as np  # for matrix maths
from scipy import stats  # for t-test
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # for logistic regression
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from dataloader import load_train_dataset
from utils import get_model_performance


def run_experiment():
    # init the models for the training
    lr_model = LogisticRegression()
    naive_bayes = MultinomialNB()
    tree = DecisionTreeClassifier()

    # record the performance
    nb_performance = train_model(naive_bayes)
    lr_performance = train_model(lr_model)
    tree_performance = train_model(tree)

    # plot the performance of each models
    plt.plot(lr_performance, label="lr performance")
    plt.plot(nb_performance, label="nb performance")
    plt.plot(tree_performance, label="tree performance")
    plt.legend()

    print(lr_performance, nb_performance, tree_performance)

    # print the ttest for the machine learning based models
    lr_nb_ttest = stats.ttest_rel(lr_performance, nb_performance)
    lr_tree_ttest = stats.ttest_rel(lr_performance, tree_performance)
    nb_tree_ttest = stats.ttest_rel(nb_performance, tree_performance)
    print(lr_nb_ttest, lr_tree_ttest, nb_tree_ttest)

    plt.savefig("basic_models.png")


def train_model(base_model, n_splits=10):

    model_performance = []

    # load the train dataset
    train_dataset = load_train_dataset()
    # X,y for the training dataset
    X, y = train_dataset["pn_history"].values, np.vstack(
        train_dataset["feature_num"].values)

    # kfolds generation
    kfolds = KFold(n_splits=n_splits, shuffle=True)

    # perofrm the kfold test
    for train_set, test_set in kfolds.split(X):
        model = make_pipeline(TfidfVectorizer(),
                              MultiOutputClassifier(base_model))
        X_train, y_train = X[train_set], y[train_set]
        X_test, y_test = X[test_set], y[test_set]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_performance.append(get_model_performance(y_test, y_pred))

    # return the model perofrmance
    return model_performance


if __name__ == "__main__":
    run_experiment()  # run rxperiment
