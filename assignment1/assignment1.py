import csv
from datetime import datetime
import numpy as np
from collections import defaultdict
import pdb
import pandas as pd
import time
import matplotlib.pyplot as plt
# from imblearn.over_sampling import over_sampling
from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import UnderSampler
# from imblearn.over_sampling import UnbalancedDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Imputer
from sklearn import svm, preprocessing, neighbors, linear_model
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score, precision_score, confusion_matrix


df = pd.read_csv('data_for_student_case.csv')
print('\nshape data')
print(df.shape)
print('\ndescribing float data')
print(df.describe())
print('\nindex types')
print(df.dtypes)
print(df.columns)

#Lukt mij niet om alle columns te transformen.. had iets met datamapper en seriesimputter geprobeerd maar lukte allemaal niet.

#encoding de column txtvariantcode


X = df.iloc[:,:].values
labelencoder_X= LabelEncoder()
X[:,3]= labelencoder_X.fit_transform(X[:,3])
Y=pd.DataFrame(X)
print(Y)


#encoding de column issuercountrycode

# B= df.iloc[:,:].values
# labelencoder_B= LabelEncoder()
# B[:,2]= labelencoder_B.fit_transform(B[:,2])
# G=pd.DataFrame(B)
# print(G)



# Een probeersel met encoding met def cat_to_nr(categorical_set, element) maar lukte ook niet..

# names = []
# grouping_names = range(0, 500)
# counter = 0


CATEGORICAL = ['issuer_country', 'tx_variant', 'currency', 'shopper_country', 'shopper_interaction', 'verification',
               'account_code']



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
                    'fraud': 1 if row['simple_journal'] == 'Chargeback'else 0,
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

        dataframe = pd.DataFrame.from_records(data)

        dataframe_sort_creation = dataframe.sort_values(by='creation_date',ascending=True)  # new Frame of data d to leave the original Frame of data the same
        dataframe['creation_date'] = pd.to_datetime(dataframe['creation_date'])
        dataframe['booking_date'] = pd.to_datetime(dataframe['booking_date'])
        dataframe['euro'] = map(lambda x, y: currency_dict[y] * x, dataframe['amount'], dataframe['currency'])
        currency_dict = {'SEK': 0.01 * 0.11, 'MXN': 0.01 * 0.05, 'AUD': 0.01 * 0.67, 'NZD': 0.01 * 0.61,
                         'GBP': 1.28 * 0.01}
        key = lambda k: (k.year, k.month, k.day)
        # print(dataframe_sort_creation.groupby(dataframe_sort_creation['creation_date'].apply(key)).mean()['amount'])
        # print(dataframe.groupby(dataframe['booking_date'].apply(key)))

        print('\nshape data')
        print(dataframe.shape)
        print('\ndescribing float data')
        print(dataframe.describe())
        print('\nindex types')
        print(dataframe.dtypes)

        Fraud = dataframe[dataframe['fraud'] ==1]
        NonFraud= dataframe[dataframe['fraud'] ==0 ]
        outlier_fraction = len(Fraud) / float(len(NonFraud))
        print(outlier_fraction)
        print('Cases which are Fraud:')
        print(format(len(Fraud)))
        print('Cases which are NonFraud:')
        print(format(len(NonFraud)))


        # correlation matrix: heatmap
        corrmat = dataframe.corr()
        fig = plt.figure(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=0.8, square=True)
        #plt.show()


        # less reliable results but better for computations
        dataframe = dataframe.sample(frac=0.1, random_state=1)
        print(dataframe.shape)

        # plotting histogram of each feature
        dataframe.hist(figsize=(20, 20))
        #plt.show()  # omzetten die type objecten ook naar float waardes.. meer histograms nodig van alle features... en kijk naar outliers


        #all data colums and target data column
        columns = dataframe.columns.tolist()
        target= 'fraud'

        columns= [ c for c in columns if c not in ['fraud']]
        all= dataframe[columns]
        print(all.shape)

        print('our target:')
        target1= dataframe[target]
        print(target1.shape)

        random state
        state = 1
        # outlier detection methods
        classifiers = {
            "Logistic Reggresion": IsolationForest(max_samples=len(all), contamination=outlier_fraction, random_state=state),
            "Support vector machine": LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction),
            "random forest": LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
        }
        #
        # #Fit model
        # n_outliers= len(Fraud)    # fit data and tag outliers
        # for i, (clf_name, clf) in enumerate(classifiers.items()):
        #  if clf_name == "Local Outlier Factor":
        #     predictTarget= clf.fit_predict(all)
        #     scores_pred = clf.negative_outlier_factor_
        # else:
        #     clf.fit(all)
        #     scores_pred = clf.decision_function(all)
        #     predictTarget = clf.predict(all)
        #
        # # Reshape prediction values to 0 for nonFraud and 1 for Fraud
        # predictTarget[predictTarget ==1] = 0
        # predictTarget[predictTarget == -1] = 1
        #
        # numberErrors = (predictTarget != target1).sum()
        #
        # # classification matrix
        # print('{}: {}'.format(clf_name,numberErrors))
        # print(accuracy_score(target1, predictTarget))
        # print(classification_report(target1,predictTarget)) #dit werkt, als we als die objects kunnen omzetten in float..

        return data, dataframe, categorical_sets







def create_x_y_sets(data, categorical_sets):
    selected_features = ['issuer_country', 'issuer_id', 'amount', 'currency', 'shopper_country',
                         'shopper_interaction', 'verification', 'cvc_response', 'account_code']
    features = []
    labels = []
    for row in data:
        features.append([row[x] if x not in CATEGORICAL else cat_to_nr(categorical_sets[x], row[x])
                         for x in selected_features])
        labels.append(row['fraud'])

    # for name in names:
    #     (name = counter, counter+= 1)
    #     for row in data:
    #         features.append #elke verschillende categorische uitkomst een nummer aan verbinden en deze updaten in de dataset met getal.


    return np.array(features).astype(float), np.array(labels).astype(float)




  # def cat_to_nr(categorical_set, element):
  # for element in categorical_set:
  #           categorical_set.append(element if element not in categorical_set






      #return list(categorical_set).index(element)

#
# def toNumber(x):
#     if x == element:
#         return counter
#
#
# #    for rowName in categorical_setsName:
# #        if(rowName not in names):  ( names.append(rowName))




def transpose_to_transactions_per_card(data):
    transactions_per_card = defaultdict(list)
    for row in data:
        transactions_per_card[row['card_id']].append(data)
    return transactions_per_card


def split_dataset(x, y, smote=True, **kwargs):
    """
    Yield 10-fold cross-validation sets and optionally smote the training data
    """
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(features):
        x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
        if smote:
            x_train, y_train = SMOTE(**kwargs).fit_sample(features[train_index], labels[train_index])

        yield x_train, x_test, y_train, y_test


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

    if hasattr(clf, 'decision_function'):
        create_roc_curve(clf, x_test, y_test)
    else:
        print('Classifier does not have decision function')


def create_roc_curve(clf, x_test, y_test):
    y_score = clf.decision_function(x_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = {})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def logistic_regression(x, y):
    for x_train, x_test, y_train, y_test in split_dataset(x, y):
        clf = linear_model.LogistifcRegression(C=1e5)
        clf.fit(x_train, y_train)

        cross_validate(clf, x_test, y_test)



def random_forest(x, y):
    for x_train, x_test, y_train, y_test in split_dataset(x, y):
        clf = RandomForestClassifier(random_state=0)
        clf.fit(x_train, y_train)

        cross_validate(clf, x_test, y_test)


def support_vector_machine(x, y):
    for x_train, x_test, y_train, y_test in split_dataset(x, y):
        clf = svm.SVC()
        clf.fit(x_train, y_train)

        cross_validate(clf, x_test, y_test)


def k_nearest_neighbour(x, y):
    for x_train, x_test, y_train, y_test in split_dataset(x, y):
        clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
        clf.fit(x_train, y_train)

        cross_validate(clf, x_test, y_test)



        metrics= pd.DataFrame(index=['accuracy', 'precision', 'recall'], columns= ['NULL', 'LogisticReg'])


if __name__ == '__main__':
    data, dataframe, categorical_sets = get_data()

    features, labels = create_x_y_sets(data, categorical_sets)

    k_nearest_neighbour(features, labels)
    logistic_regression(features, labels)
    random_forest(features, labels)
    support_vector_machine(features, labels)
    pdb.set_trace()
