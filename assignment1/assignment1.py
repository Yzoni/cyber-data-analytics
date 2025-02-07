import csv
from collections import defaultdict
from datetime import datetime
from pprint import pprint
import pickle

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import svm, neighbors, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingRegressor

import currency_converter
import os
from pathlib import Path
import argparse

import visualize
import util
import generate_aggregated_features

"""
POSSIBLE FEATURES
-----------------

MERGED_FEATURES = [
    'amount_same_currency_month', 'average_amount_same_merchant_month', 'average_amount_same_shopper_country_month',
    'average_daily_month', 'number_same_currency_month', 'number_same_merchant_month',
    'number_same_shopper_country_month', 'number_same_day', 'amount_same_day'
]


POST_FEATURES = [
    'country_differs'
]

RAW_FEATURES = [
    'issuer_id', 'issuer_country', 'amount', 'currency', 'shopper_country', 'ip', 'card_id',
    'shopper_interaction', 'verification', 'cvc_response', 'tx_variant', 'mail_id'
]
"""

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Constant describing the labels in the dataset that have a discrete value as string the raw data
DISCRETE_STRING_FEATURES = ['issuer_country', 'tx_variant', 'currency', 'shopper_country', 'shopper_interaction',
                            'verification', 'account_code']


def load_data(postprocess=True, use_cached=True):
    """
    Reads the raw csv data into a list of dictionaries

    :returns:
     - [{'account_code': 'MexicoAccount',
      'amount': 64900.0,
      'booking_date': datetime.datetime(2015, 8, 13, 18, 17, 46),
      'card_id': 'card255564',
      'creation_date': datetime.datetime(2015, 8, 13, 18, 17, 35),
      'currency': 'MXN',
      'cvc_response': '0',
      'fraud': 0,
      'id': '30906',
      'ip': 'ip80818',
      'issuer_country': 'MX',
      'issuer_id': 544549.0,
      'mail_id': 'email93243',
      'shopper_country': 'MX',
      'shopper_interaction': 'Ecommerce',
      'tx_variant': 'mccredit',
      'verification': 'TRUE'},
      ...]
      - 'shopper_interaction': {'Ecommerce'}, # For each key list all possible values
        'tx_variant': {'cirrus', 'electron', ...
    """

    if use_cached:
        try:
            with (ROOT / 'data.pickle').open('rb') as handle:
                print('Loading data from cache...')
                return pickle.load(handle)
        except FileNotFoundError:
            pass

    with (ROOT / 'data_for_student_case.csv').open(mode='r') as csv_file:
        reader = csv.DictReader(csv_file)

        categorical_sets = {key: set() for key in DISCRETE_STRING_FEATURES}

        data = []
        for row in reader:

            if row['simple_journal'] == 'Refused':  # Skip refused rows because of uncertainty of status
                continue

            data_row = {
                'id': int(row['txid']),
                'booking_date': datetime.strptime(row['bookingdate'], '%Y-%m-%d %H:%M:%S'),
                'issuer_country': row['issuercountrycode'] if row['issuercountrycode'] != 'NA' else None,
                'tx_variant': row['txvariantcode'],
                'issuer_id': float(row['bin']) if row['bin'].isnumeric() else None,
                'amount': float(row['amount']),
                'currency': row['currencycode'],
                'shopper_country': row['shoppercountrycode'] if row['shoppercountrycode'] != 'NA' else None,
                'shopper_interaction': row['shopperinteraction'],
                'fraud': 1 if row['simple_journal'] == 'Chargeback' else 0,
                'verification': row['cardverificationcodesupplied'] if row[
                                                                           'cardverificationcodesupplied'] != 'NA' else None,
                'cvc_response': 0 if int(row['cvcresponsecode']) > 2 else int(row['cvcresponsecode']),
                'creation_date': datetime.strptime(row['creationdate'], '%Y-%m-%d %H:%M:%S'),
                'account_code': row['accountcode'],
                'mail_id': row['mail_id'], 'ip': row['ip_id'],
                'card_id': row['card_id']

            }

            for category in categorical_sets:
                categorical_sets[category].add(data_row[category])

            data.append(data_row)

        # Create ordered list from unordered set
        for category in categorical_sets:
            categorical_sets[category] = list(categorical_sets[category])

        if postprocess:
            postprocess_data(data)

        with (ROOT / 'data.pickle').open(mode='wb') as handle:
            pickle.dump((data, categorical_sets), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return data, categorical_sets


def handle_missing_data(data: list, show=False, missing_data_strategy='remove_row'):
    """
    missing_data_count_per_column = {'issuer_country': 493, # feature name: total number missing
                                     'issuer_id': 140,
                                     'shopper_country': 482,
                                     'verification': 14717})


    missing_data_row_idxs = {167: ['issuer_country'], # IDx of row in CSV: feature that is missing
                             202: ['issuer_country'],
                             211: ['issuer_country'],
                             533: ['issuer_country'],
                             534: ['issuer_country'],
                             536: ['issuer_country'],
                             538: ['issuer_country'],
                             625: ['issuer_country'],
                             ....

    """
    missing_data_count_per_column = defaultdict(lambda: 0)
    missing_data_row_idxs = defaultdict(list)

    for row_index, row in enumerate(data):

        for key, value in row.items():
            if value is None:
                missing_data_count_per_column[key] += 1
                missing_data_row_idxs[row_index].append(key)

    if show:
        pprint(missing_data_count_per_column)
        pprint(missing_data_row_idxs)

    # Handle not available data
    if missing_data_strategy == 'remove_row':
        for idx in sorted(list(missing_data_row_idxs.keys()), reverse=True):
            del data[idx]
    elif missing_data_strategy == 'new_category':
        for k, v in missing_data_row_idxs.items():
            for column in v:
                data[k][column] = -1

    return data


def postprocess_data(data: list):
    """
    - Fix or remove missing data
    - Normalizes the value of each transaction based on currency type
    """

    data = handle_missing_data(data)

    # Normalize currency value in place
    c = currency_converter.CurrencyConverter('http://www.ecb.int/stats/eurofxref/eurofxref-hist.zip')
    for row in data:
        try:
            row['amount'] = c.convert(row['amount'], row['currency'], 'EUR', date=row['creation_date'])
        except currency_converter.RateNotFoundError:
            row['amount'] = c.convert(row['amount'], row['currency'], 'EUR')

    with open('merged.csv', mode='r') as csvfile:
        merged_data = [row for row in csv.DictReader(csvfile, delimiter=',')]

    for idx, row in enumerate(data):
        merged_row = merged_data[idx]
        for k, v in merged_row.items():
            if k == 'id':
                continue
            row[k] = v

    for row in data:
        if row['issuer_country'] != row['shopper_country']:
            row['country_differs'] = 1
        else:
            row['country_differs'] = 0

    return data


def create_x_y_sets(data: list, categorical_sets: dict, selected_features: list):
    """
    Returns separate sets for the features and labels

    Also converts non numerical features to a numeric value
    """

    print('------ Features')
    print(selected_features)
    print('------ Features')

    features = []
    labels = []
    for row in data:
        feature = []
        for x in selected_features:
            if x not in DISCRETE_STRING_FEATURES:
                feature.append(row[x])
            else:
                feature.append(util.nonnr_to_nr(categorical_sets[x], row[x]))
        features.append(feature)
        labels.append(row['fraud'])

    return np.array(features).astype(float), np.array(labels).astype(float)


def split_dataset(x, y, kfold, smote, **kwargs):
    """
    Yield cross-validation sets and optionally smote the training data
    """

    set_x_train = []
    set_x_test = []
    set_y_train = []
    set_y_test = []

    if kfold:
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(x):
            x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
            if smote:
                x_train, y_train = SMOTE(random_state=1, n_jobs=4, **kwargs).fit_sample(x[train_index], y[train_index])

            set_x_train.append(x_train)
            set_x_test.append(x_test)
            set_y_train.append(y_train)
            set_y_test.append(y_test)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        if smote:
            x_train, y_train = SMOTE(**kwargs).fit_sample(x_train, y_train)

        set_x_train.append(x_train)
        set_x_test.append(x_test)
        set_y_train.append(y_train)
        set_y_test.append(y_test)

    return set_x_train, set_x_test, set_y_train, set_y_test


def cross_validate(clf, x_test, y_test):
    y_predict = clf.predict(x_test)

    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(y_predict)):
        if y_test[i] == 1 and y_predict[i] == 1:
            TP += 1
        if y_test[i] == 0 and y_predict[i] == 1:
            FP += 1
        if y_test[i] == 1 and y_predict[i] == 0:
            FN += 1
        if y_test[i] == 0 and y_predict[i] == 0:
            TN += 1

    print('TP: {}'.format(TP))
    print('FP: {}'.format(FP))
    print('FN: {}'.format(FN))
    print('TN: {}'.format(TN))

    return TP, FP, FN, TN


def classify(x, y, kfold, smote, classifier='logistic', **kwargs):
    TP_total, FP_total, FN_total, TN_total = 0, 0, 0, 0
    fold_n = 1

    print('------ Classifier')
    print(classifier)
    print('------ Classifier')

    fitted_classifiers = []
    set_x_train, set_x_test, set_y_train, set_y_test = split_dataset(x, y, kfold, smote, **kwargs)
    for x_train, x_test, y_train, y_test in zip(set_x_train, set_x_test, set_y_train, set_y_test):
        if classifier == 'logistic':
            clf = linear_model.LogisticRegression(C=1)
        elif classifier == 'random_forest':
            clf = RandomForestClassifier(n_jobs=6)
        elif classifier == 'svm':
            clf = svm.LinearSVC()
        elif classifier == 'knn':
            clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
        elif classifier == 'mlp':
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(5, 2), random_state=1)
        elif classifier == 'gboost':
            clf = GradientBoostingRegressor()
        else:
            raise ValueError('Classifier unknown')

        clf.fit(x_train, y_train)

        print('===========')
        print('Fold {}'.format(fold_n))
        print('-----------')
        TP, FP, FN, TN = cross_validate(clf, x_test, y_test)
        print('===========')

        TP_total += TP
        FP_total += FP
        FN_total += FN
        TN_total += TN

        fitted_classifiers.append(clf)

        fold_n += 1

    print('----- Total:')
    print('TP: {}'.format(TP_total))
    print('FP: {}'.format(FP_total))
    print('FN: {}'.format(FN_total))
    print('TN: {}'.format(TN_total))

    return fitted_classifiers, set_x_test, set_y_test


def experiment_smote_parameters(features, labels):
    visualize.plot_roc_curve_compare(
        [(*classify(features, labels, kfold=10, smote=True, classifier='logistic', k_neighbors=5),
          'k_neighbors=5'),
         (*classify(features, labels, kfold=10, smote=True, classifier='logistic', k_neighbors=10),
          'k_neighbors=10'),
         (*classify(features, labels, kfold=10, smote=True, classifier='logistic', k_neighbors=15),
          'k_neighbors=15')],
        title='SMOTE compare "k_neighbours"')

    visualize.plot_roc_curve_compare(
        [(*classify(features, labels, kfold=10, smote=True, classifier='logistic', kind='regular'),
          'regular'),
         (*classify(features, labels, kfold=10, smote=True, classifier='logistic', kind='borderline1'),
          'borderline1'),
         (*classify(features, labels, kfold=10, smote=True, classifier='logistic', kind='borderline2'),
          'borderline2'),
         (*classify(features, labels, kfold=10, smote=True, classifier='logistic', kind='svm'),
          'svm')],
        title='SMOTE compare "kind"')

    visualize.plot_roc_curve_compare(
        [(*classify(features, labels, kfold=10, smote=True, classifier='logistic', m_neighbors=10),
          'm_neighbors=10'),
         (*classify(features, labels, kfold=10, smote=True, classifier='logistic', m_neighbors=20),
          'm_neighbors=20'),
         (*classify(features, labels, kfold=10, smote=True, classifier='logistic', m_neighbors=30),
          'm_neighbors=30')],
        title='SMOTE compare "m_neighbors"')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Assignment 1 - Cyber Data Analytics')
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help='Produce visualizations')
    parser.add_argument('--smote-experiment', dest='smote', action='store_true',
                        help='Run the imbalance task experiment')
    parser.add_argument('--classify-blackbox', dest='blackbox', action='store_true',
                        help='Fit and cross validate black-box algorithm')
    parser.add_argument('--classify-whitebox', dest='whitebox', action='store_true',
                        help='Fit and cross validate white-box algorithm')
    parser.add_argument('--create-derived-data', dest='derived_data', action='store_true',
                        help='Generate derived data')
    args = parser.parse_args()

    data, categorical_sets = load_data(use_cached=False)

    #################
    # Visualize task
    #################
    if args.visualize:
        visualize.fraud_per_feature_category(data)
        visualize.print_fraud_counts(data)

    #################
    # Imbalance task
    #################
    # experiment_smote_parameters(features, labels) # Experiment with SMOTE parameters
    if args.smote:
        selected_features = ['issuer_id', 'issuer_country', 'amount', 'currency', 'shopper_country',
                             'shopper_interaction', 'verification', 'cvc_response', 'tx_variant']
        features, labels = create_x_y_sets(data, categorical_sets, selected_features)
        visualize.plot_roc_curve_compare(
            [(*classify(features, labels, kfold=10, smote=False, classifier='logistic'), 'UNsmoted'),
             (*classify(features, labels, kfold=10, smote=True, classifier='logistic'), 'Smoted')])
        visualize.plot_roc_curve_compare(
            [(*classify(features, labels, kfold=10, smote=False, classifier='svm'), 'UNsmoted'),
             (*classify(features, labels, kfold=10, smote=True, classifier='svm'), 'Smoted')])
        visualize.plot_roc_curve_compare(
            [(*classify(features, labels, kfold=10, smote=False, classifier='random_forest'), 'UNsmoted'),
             (*classify(features, labels, kfold=10, smote=True, classifier='random_forest'), 'Smoted')])

    ######################
    # Classification task
    ######################
    if args.blackbox:
        selected_features = ['verification', 'issuer_country', 'amount',  'shopper_country',
                             'cvc_response', 'average_amount_same_merchant_month',
                             'average_amount_same_shopper_country_month',
                             'average_daily_month', 'number_same_currency_month', 'number_same_merchant_month',
                             'number_same_shopper_country_month', 'number_same_day', 'amount_same_day',
                             'country_differs']

        features, labels = create_x_y_sets(data, categorical_sets, selected_features)

        fitted_classifiers, set_x_test, set_y_test = classify(features, labels, kfold=True, smote=True,
                                                              classifier='logistic')

        visualize.plot_roc_curve_kfold(fitted_classifiers, set_x_test, set_y_test)
    if args.whitebox:
        selected_features = ['issuer_id', 'amount', 'shopper_interaction', 'verification',
                             'cvc_response', 'tx_variant', 'number_same_currency_month', 'number_same_merchant_month',
                             'number_same_shopper_country_month', 'number_same_day', 'amount_same_day',
                             'country_differs']
        features, labels = create_x_y_sets(data, categorical_sets, selected_features)

        random_forest_fitted_classifiers, set_x_test, set_y_test = classify(features, labels, kfold=True, smote=True,
                                                                            classifier='random_forest')
        visualize.plot_roc_curve_kfold(random_forest_fitted_classifiers, set_x_test, set_y_test)
        # visualize.plot_decision_tree(random_forest_fitted_classifiers[0], SELECTED_FEATURES)

    ###########################
    # Derived data aggregation
    ###########################
    if args.derived_data:
        generate_aggregated_features.create_all(data)
