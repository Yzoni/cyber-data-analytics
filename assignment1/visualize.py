from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
# from imblearn.over_sampling import over_sampling
# from imblearn.over_sampling import UnderSampler
# from imblearn.over_sampling import UnbalancedDataset
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import export_graphviz
from os import system


def fraud_per_feature_category(data):
    """
    Plot the percentage of fraud transactions for every feature category

    Just a test
    """
    selected_features = ['tx_variant', 'issuer_country', 'shopper_country', 'currency', 'verification', 'cvc_response',
                         'shopper_interaction']

    feature_map = defaultdict(dict)

    for row in data:
        for feature in selected_features:
            if row['fraud'] == 1:
                try:
                    feature_map[feature][row[feature]][1] += 1
                except KeyError:
                    feature_map[feature][row[feature]] = [0, 1]
            else:
                try:
                    feature_map[feature][row[feature]][0] += 1
                except KeyError:
                    feature_map[feature][row[feature]] = [1, 0]

    # To Dataframe and plot
    for k, v in feature_map.items():

        col_feature, col_feature_cat, col_ratio = [], [], []

        for k_x, v_x in v.items():
            if v_x[1] / (v_x[0] + v_x[1]) == 0:
                continue
            col_ratio.append(v_x[1] / (v_x[0] + v_x[1]))
            col_feature.append(k)
            col_feature_cat.append(k_x)

        df = pd.DataFrame(data={
            'feature': col_feature,
            k: col_feature_cat,
            '% fraud': col_ratio
        })

        sns.barplot(x=k, y='% fraud', data=df)
        plt.show()


def plot_visualizations(dataframe: pd.DataFrame):
    # TODO hoe maken we een histogram/heatmap van twee discrete waarden, aangzien bijna alle variabelen discreet zijn?

    fraud = dataframe[dataframe['fraud'] == 1]
    non_fraud = dataframe[dataframe['fraud'] == 0]

    outlier_fraction = len(fraud) / float(len(non_fraud))
    print(outlier_fraction)
    print('Cases which are Fraud:')
    print(format(len(fraud)))
    print('Cases which are NonFraud:')
    print(format(len(non_fraud)))

    pd.scatter_matrix(dataframe)
    plt.show()

    # correlation matrix
    corrmat = dataframe.corr()
    print(corrmat)
    fig = plt.figure(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.8, square=True)
    plt.show()

    # less reliable results but better for computations
    dataframe = dataframe.sample(frac=0.1, random_state=1)
    print(dataframe.shape)

    # plotting histogram of each feature
    dataframe.hist(figsize=(20, 20))
    plt.show()  # omzetten die type objecten ook naar float waardes.. meer histograms nodig van alle features... en kijk naar outliers

    # all data colums and target data column
    columns = dataframe.columns.tolist()
    target = 'fraud'

    columns = [c for c in columns if c not in ['fraud']]
    all = dataframe[columns]
    print(all.shape)

    print('our target:')
    target1 = dataframe[target]
    print(target1.shape)

    # random state
    state = 1
    # outlier detection methods
    classifiers = {
        "Isolation Forest": IsolationForest(max_samples=len(all), contamination=outlier_fraction,
                                            random_state=state),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)}

    n_outliers = len(fraud)  # fit data and tag outliers
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        if clf_name == "Local Outlier Factor":
            predictTarget = clf.fit_predict(all)
            scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(all)
        scores_pred = clf.decision_function(all)
        predictTarget = clf.predict(all)

    # Reshape prediction values to 0 for nonFraud and 1 for Fraud
    predictTarget[predictTarget == 1] = 0
    predictTarget[predictTarget == -1] = 1

    numberErrors = (predictTarget != target1).sum()

    # classification matrix
    print('{}: {}'.format(clf_name, numberErrors))
    print(accuracy_score(target1, predictTarget))
    print(classification_report(target1,
                                predictTarget))
    # dit werkt, als we als die objects kunnen omzetten in float..
    # TODO: gebruik hiervoor de "categorical_sets" die "load_data()" returned op een zelfde soort manier als in
    # TODO "create_x_y_sets()" maar dan voor de pandas dataframe

    return dataframe


def plot_roc_curve_compare(curves: list, title='Comparison of mean ROC curves'):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for idx, curve in enumerate(curves):
        fitted_classifiers = curve[0]
        set_x_test = curve[1]
        set_y_test = curve[2]
        label = curve[3]

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for clf, x_test, y_test in zip(fitted_classifiers, set_x_test, set_y_test):
            y_score = clf.predict_proba(x_test)

            fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=colors[idx],
                 label=r'%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc),
                 lw=2, alpha=.8)

    # Diagonal
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('{}.jpg'.format('_'.join(title.split(' '))))


def plot_roc_curve_kfold(fitted_classifiers: list, set_x_test: list, set_y_test: list, title='K fold ROC'):
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for clf, x_test, y_test in zip(fitted_classifiers, set_x_test, set_y_test):
        y_score = clf.predict_proba(x_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # Diagonal
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_decision_tree(clf, feature_names):
    export_graphviz(clf.estimators_[0],
                    class_names=feature_names,
                    out_file='dotfile.dot',
                    filled=True,
                    rounded=True)

    system('dot -Tpng dotfile.dot -o tree.png')
