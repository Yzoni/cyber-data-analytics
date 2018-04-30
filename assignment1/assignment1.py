import csv
from collections import defaultdict
from datetime import datetime
from pprint import pprint

import numpy as np
from sklearn import neighbors, linear_model
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

CATEGORICAL = ['issuer_country', 'tx_variant', 'currency', 'shopper_country', 'shopper_interaction', 'verification',
               'account_code']


def cat_to_nr(categorical_set: set, element):
    """
    Modify categorical feature to number in data set
    """
    return list(categorical_set).index(element)


def get_data(max_lines: int = None):
    with open('data_for_student_case.csv') as csv_file:
        reader = csv.DictReader(csv_file)

        categorical_sets = dict.fromkeys(CATEGORICAL, set())

        data = []
        for idx, row in enumerate(reader):

            if max_lines and idx > max_lines - 1:
                break

            if row['simple_journal'] == 'Refused':
                continue

            try:
                data_row = {
                    'id': row['txid'],
                    'booking_date': datetime.strptime(row['bookingdate'], '%Y-%m-%d %H:%M:%S'),
                    'issuer_country': row['issuercountrycode'],
                    'tx_variant': row['txvariantcode'],
                    'issuer_id': float(row['bin']),
                    'amount': float(row['amount']),
                    'currency': row['currencycode'],
                    'shopper_country': row['shoppercountrycode'],
                    'shopper_interaction': row['shopperinteraction'],
                    'fraud': 1 if row['simple_journal'] == 'Chargeback' else 0,
                    'verification': row['cardverificationcodesupplied'],
                    'cvc_response': 3 if int(row['cvcresponsecode']) > 2 else row['cvcresponsecode'],
                    'creation_date': datetime.strptime(row['creationdate'], '%Y-%m-%d %H:%M:%S'),
                    'account_code': row['accountcode'],
                    'mail_id': row['mail_id'], 'ip': row['ip_id'],
                    'card_id': row['card_id']
                }

                for category in categorical_sets:
                    categorical_sets[category].add(data_row[category])

            except ValueError as e:
                print('Error on {}, {}'.format(row['txid'], e))
            else:
                data.append(data_row)

        return data, categorical_sets


def create_x_y_sets(data, categorical_sets):
    selected_features = ['issuer_country', 'issuer_id', 'amount', 'currency', 'shopper_country',
                         'shopper_interaction', 'verification', 'cvc_response', 'account_code']
    features = []
    labels = []

    for row in data:
        features.append([row[x] if x not in CATEGORICAL else cat_to_nr(categorical_sets[x], row[x])
                         for x in selected_features])
        labels.append(row['fraud'])
    return np.array(features).astype(float), np.array(labels).astype(float)


def transpose_to_transactions_per_card(data):
    transactions_per_card = defaultdict(list)
    for row in data:
        transactions_per_card[row['card_id']].append(data)
    return transactions_per_card


def create_1d_heatmaps(data):
    """
    Creates a dictionary noting for every feature the ratio of fraud/non fraud

    :param data:
    :return:
    """
    selected_features = ['issuer_id', 'issuer_country', 'currency', 'verification', 'cvc_response']

    feature_map = defaultdict(dict)

    for row in data:
        for feature in selected_features:
            if row['fraud'] == 1:
                try:
                    feature_map[feature][row[feature]][0] += 1
                except KeyError:
                    feature_map[feature][row[feature]] = [1, 0]
            else:
                try:
                    feature_map[feature][row[feature]][1] += 1
                except KeyError:
                    feature_map[feature][row[feature]] = [0, 1]

    # Add ratio
    for k, v in feature_map.items():
        for k_x, v_x in v.items():
            if v_x[1] == 0:
                v_x.append(0)
            else:
                v_x.append(v_x[0] / v_x[1])

    return feature_map


def split_dataset(features, labels):
    # TODO SMOT: http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html
    return train_test_split(features, labels, test_size=0.2)


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


def logistic_regression(features, labels):
    x_train, x_test, y_train, y_test = split_dataset(features, labels)

    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(x_train, y_train)

    cross_validate(clf, x_test, y_test)


def random_forest(features, labels):
    x_train, x_test, y_train, y_test = split_dataset(features, labels)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(x_train, y_train)

    cross_validate(clf, x_test, y_test)


def support_vector_machine(features, labels):
    x_train, x_test, y_train, y_test = split_dataset(features, labels)

    clf = svm.SVC()
    clf.fit(x_train, y_train)

    cross_validate(clf, x_test, y_test)


def k_nearest_neighbour(features, labels):
    x_train, x_test, y_train, y_test = split_dataset(features, labels)

    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
    clf.fit(x_train, y_train)

    cross_validate(clf, x_test, y_test)


if __name__ == '__main__':
    data, categorical_sets = get_data()
    pprint(create_1d_heatmaps(data))

    features, labels = create_x_y_sets(data, categorical_sets)

    k_nearest_neighbour(features, labels)
    logistic_regression(features, labels)
    random_forest(features, labels)
    support_vector_machine(features, labels)
