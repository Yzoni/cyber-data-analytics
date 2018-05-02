import csv
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
import numpy as np

import pandas as pd
import time
import matplotlib.pyplot as plt
#from imblearn.over_sampling import over_sampling
from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import UnderSampler
#from imblearn.over_sampling import UnbalancedDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report


df=pd.read_csv('data_for_student_case.csv')
print ('\nshape data')
print (df.shape)
print ('\ndescribing float data')
print (df.describe())
print ('\nindex types')
print (df.dtypes)




# df_input = (df1[['id','booking_date','issuer_country','tx_variant','issuer_id','amount','currency','shopper_country','shopper_interaction','fraud','verification','cvc_response','creation_date','account_code','mail_id', 'ip','card_id']])
# df_input[['issuer_id','label_int']] = df_input[['issuer_id','label_int']].astype(int)
# print (df_input.dtypes)
# x = df_input[df_input.columns[0:-1]].as_matrix()
# y = df_input[df_input.columns[-1]].as_matrix()
#






df_sort_creation = df.sort_values(by = 'creationdate', ascending = True)#new Frame of data d to leave the original Frame of data the same
df['creationdate'] =pd.to_datetime(df['creationdate'])
df['bookingdate'] =pd.to_datetime(df['bookingdate'])
df['euro'] =map(lambda x,y: currency_dict[y]*x, df['amount'], df['currencycode'])
currency_dict = {'SEK': 0.01*0.11,'MXN': 0.01*0.05, 'AUD': 0.01*0.67, 'NZD': 0.01*0.61,'GBP': 0.01*1.28}
key = lambda k:(k.year,k.month,k.day)
CATEGORICAL = ['issuer_country', 'tx_variant', 'currency', 'shopper_country', 'shopper_interaction', 'verification',
               'account_code']
#
# def missing(x):
#     return sum(x.isnull())
#     df_renamed = df.rename(index=str, columns={'issuer_country', 'issuer_id', 'amount', 'currency', 'shopper_country',
#                          'shopper_interaction', 'verification', 'cvc_response', 'account_code'})
#     df_select = (df_renamed[['issuercountrycode', 'cardtype', 'issuer_id', 'euro', 'currencycode',
#                              'shoppercountrycode', 'shoppingtype', 'label', 'cvcsupply', 'cvcresponse', 'merchant_id',
#                              'mail_id', 'ip_id', 'card_id']])
#     print("Values missing in column:")
#     print(df_select.apply(missing, axis=0))
#     df_clean = df_select.fillna('missing')

    # issuercountrycode_category = pd.Categorical.from_array(df_clean['issuercountrycode'])
    # cardtype_category = pd.Categorical.from_array(df_clean['cardtype'])
    # # issuer_id_category = pd.Categorical.from_array(df_clean['issuer_id'])
    # currencycode_category = pd.Categorical.from_array(df_clean['currencycode'])
    # shoppercountrycode_category = pd.Categorical.from_array(df_clean['shoppercountrycode'])
    # shoppingtype_category = pd.Categorical.from_array(df_clean['shoppingtype'])
    # cvcsupply_category = pd.Categorical.from_array(df_clean['cvcsupply'])
    # merchant_id_category = pd.Categorical.from_array(df_clean['merchant_id'])
    # mail_id_category = pd.Categorical.from_array(df_clean['mail_id'])
    # ip_id_category = pd.Categorical.from_array(df_clean['ip_id'])
    # card_id_category = pd.Categorical.from_array(df_clean['card_id'])
    #
    # issuercountrycode_dict = dict(set(zip(issuercountrycode_category, issuercountrycode_category.codes)))
    # cardtype_dict = dict(set(zip(cardtype_category, cardtype_category.codes)))
    # currencycode_dict = dict(set(zip(currencycode_category, currencycode_category.codes)))
    # shoppercountrycode_dict = dict(set(zip(shoppercountrycode_category, shoppercountrycode_category.codes)))
    # shoppingtype_dict = dict(set(zip(shoppingtype_category, shoppingtype_category.codes)))
    # cvcsupply_dict = dict(set(zip(cvcsupply_category, cvcsupply_category.codes)))
    # merchant_id_dict = dict(set(zip(merchant_id_category, merchant_id_category.codes)))
    # mail_id_dict = dict(set(zip(mail_id_category, mail_id_category.codes)))
    # ip_id_dict = dict(set(zip(ip_id_category, ip_id_category.codes)))
    # card_id_dict = dict(set(zip(card_id_category, card_id_category.codes)))
    #
    # df_clean['issuercountrycode'] = issuercountrycode_category.codes
    # df_clean['cardtype'] = cardtype_category.codes
    # df_clean['currencycode'] = currencycode_category.codes
    # df_clean['shoppercountrycode'] = shoppercountrycode_category.codes
    # df_clean['shoppingtype'] = shoppingtype_category.codes
    # df_clean['cvcsupply'] = cvcsupply_category.codes
    # df_clean['merchant_id'] = merchant_id_category.codes
    # df_clean['mail_id'] = mail_id_category.codes
    # df_clean['ip_id'] = ip_id_category.codes
    # df_clean['card_id'] = card_id_category.codes
    # df_clean['label_int'], df_clean['cvcresponse_int'] = 0, 0
    # df_clean['label_int'] = map(lambda x: 1 if str(x) == 'Chargeback' else 0 if str(x) == 'Settled' else 'unknown',
    #                             df_clean['label'])
    # df_clean['cvcresponse_int'] = map(lambda x: 3 if x > 2 else x + 0, df_clean['cvcresponse'])
    # # 0 = Unknown, 1=Match, 2=No Match, 3=Not checked
    # df1 = df_clean.ix[(df_clean['label_int'] == 1) | (df_clean['label_int'] == 0)]  # 237036 instances
    # df1.head()
    #








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

        df = pd.DataFrame.from_records(data)
        print('\nshape data')
        print(df.shape)
        print('\ndescribing float data')
        print(df.describe())
        print('\nindex types')
        print(df.dtypes)
        # Return de df

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


