import csv
from collections import defaultdict
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
# from imblearn.over_sampling import over_sampling
from imblearn.over_sampling import SMOTE
from sklearn import svm, neighbors, linear_model
# from imblearn.over_sampling import UnderSampler
# from imblearn.over_sampling import UnbalancedDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import visualize
import util

# Constant describing the labels in the dataset that have a discrete value as string the raw data
DISCRETE_STRING_FEATURES = ['issuer_country', 'tx_variant', 'currency', 'shopper_country', 'shopper_interaction',
                            'verification', 'account_code']


def load_data():
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

    with open('data_for_student_case.csv') as csv_file:
        reader = csv.DictReader(csv_file)

        categorical_sets = {key: set() for key in DISCRETE_STRING_FEATURES}

        data = []
        for row in reader:

            if row['simple_journal'] == 'Refused':  # Skip refused rows because of uncertainty of status
                continue

            data_row = {
                'id': row['txid'],
                'booking_date': datetime.strptime(row['bookingdate'], '%Y-%m-%d %H:%M:%S'),
                'issuer_country': row['issuercountrycode'] if row['issuercountrycode'] != 'NA' else None,
                'tx_variant': row['txvariantcode'],
                'issuer_id': float(row['bin']) if row['bin'].isnumeric() else None,
                'amount': float(row['amount']),
                'currency': row['currencycode'],
                'shopper_country': row['shoppercountrycode'] if row['shoppercountrycode'] != 'NA' else None,
                'shopper_interaction': row['shopperinteraction'],
                'fraud': 1 if row['simple_journal'] == 'Chargeback' else 0,  # TODO: SEE handle_missing_data()
                'verification': row['cardverificationcodesupplied'] if row[
                                                                           'cardverificationcodesupplied'] != 'NA' else None,
                'cvc_response': 3 if int(row['cvcresponsecode']) > 2 else row['cvcresponsecode'],
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
    currency_dict = {'SEK': 0.01 * 0.11, 'MXN': 0.01 * 0.05, 'AUD': 0.01 * 0.67, 'NZD': 0.01 * 0.61,
                     'GBP': 1.28 * 0.01}  # TODO are this all the currencies?
    for row in data:
        row['amount'] = currency_dict[row['currency']] * row['amount']

    return data


def create_x_y_sets(data: list, categorical_sets: dict):
    """
    Returns separate sets for the features and labels

    Also converts non numerical features to a numeric value
    """

    # The features that need to be selected for the feature matrix
    selected_features = ['issuer_id', 'issuer_country', 'amount', 'currency', 'shopper_country',
                         'shopper_interaction', 'verification', 'cvc_response', 'tx_variant']
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
                x_train, y_train = SMOTE(**kwargs).fit_sample(x[train_index], y[train_index])

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


def classify(x, y, kfold, smote, classifier='logistic'):
    fitted_classifiers = []
    set_x_train, set_x_test, set_y_train, set_y_test = split_dataset(x, y, kfold, smote)
    for x_train, x_test, y_train, y_test in zip(set_x_train, set_x_test, set_y_train, set_y_test):
        if classifier == 'logistic':
            clf = linear_model.LogisticRegression()
        elif classifier == 'random_forest':
            clf = RandomForestClassifier()
        elif classifier == 'svm':
            clf = svm.SVC()
        elif classifier == 'knn':
            clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
        else:
            raise ValueError('Classifier unknown')

        clf.fit(x_train, y_train)
        cross_validate(clf, x_test, y_test)
        fitted_classifiers.append(clf)

    visualize.plot_roc_curve(fitted_classifiers, set_x_test, set_y_test)


if __name__ == '__main__':
    data, categorical_sets = load_data()
    postprocessed_data = postprocess_data(data)

    #################
    # Visualize task
    #################
    # visualize.fraud_per_feature_category(postprocessed_data)  # was een test
    # visualize.plot_visualizations(pd.DataFrame.from_records(postprocessed_data))

    #################
    # Imbalance task
    #################
    features, labels = create_x_y_sets(postprocessed_data, categorical_sets)
    classify(features, labels, kfold=10, smote=True, classifier='logistic')
    # random_forest(features, labels, kfold=False, smote=False)
    # support_vector_machine(features, labels, kfold=False, smote=False)

    ######################
    # Classification task
    ######################
    # Blackbox
    # classify(features, labels, kfold=True, smote=True, classifier='logistic')

    # Whitebox
    # random_forest(features, labels, kfold=True, smote=False)
