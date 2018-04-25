import csv
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
import numpy as np

CATEGORICAL = ['issuer_country', 'tx_variant', 'currency', 'shopper_country', 'shopper_interaction', 'verification',
               'account_code']


def cat_to_nr(categorical_set: set, element):
    """
    Modify categorical feature to number in data set
    """
    return list(categorical_set).index(element)


def get_data():
    with open('data_for_student_case.csv') as csv_file:
        reader = csv.DictReader(csv_file)

        categorical_sets = dict.fromkeys(CATEGORICAL, set())

        data = []
        for row in reader:

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


def create_x_y_sets():
    selected_features = ['issuer_country', 'issuer_id', 'amount', 'currency', 'shopper_country',
                         'shopper_interaction', 'verification', 'cvc_response', 'account_code']
    features = []
    labels = []

    data, categorical_sets = get_data()
    for row in data:
        features.append([row[x] if x not in CATEGORICAL else cat_to_nr(categorical_sets[x], row[x])
                         for x in selected_features])
        labels.append(row['fraud'])
    return np.array(features), np.array(labels)


def k_nearest_neighbour(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.2)  # test_size: proportion of train/test data
    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
    clf.fit(x_train, y_train)
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
    print('TP: ' + str(TP))
    print('FP: ' + str(FP))
    print('FN: ' + str(FN))
    print('TN: ' + str(TN))


if __name__ == '__main__':
    features, labels = create_x_y_sets()
    k_nearest_neighbour(features, labels)
