import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from imblearn.over_sampling import over_sampling
from imblearn.over_sampling import SMOTE
from scipy._lib.six import xrange
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import UnderSampler
# from imblearn.over_sampling import UnbalancedDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)

CATEGORICAL = ['issuercountrycode', 'cardtype', 'issuer_id', 'euro', 'currencycode',
              'shoppercountrycode', 'shoppingtype', 'label', 'cvcsupply', 'cvcresponse', 'merchant_id',
              'mail_id', 'ip_id', 'card_id']



df = pd.read_csv('data_for_student_case.csv')



print ('\nshape of data')
print (df.shape)
print ('\ntypes of index')
print (df.dtypes)
print ('\ndescribe (only for float data)')
print (df.describe())



# def get_data():
#     with open('data_for_student_case.csv') as csv_file:
#         reader = csv.DictReader(csv_file)
#
#         categorical_sets = dict.fromkeys(CATEGORICAL, set())
#
#         data = []
#         for row in reader:
#
#             if row['simple_journal'] == 'Refused':
#                 continue
#
#             try:
#                 data_row = {
#                     'id': row['txid'],
#                     'booking_date': datetime.strptime(row['bookingdate'], '%Y-%m-%d %H:%M:%S'),
#                     'issuer_country': row['issuercountrycode'],
#                     'tx_variant': row['txvariantcode'],
#                     'issuer_id': float(row['bin']),
#                     'amount': float(row['amount']),
#                     'currency': row['currencycode'],
#                     'shopper_country': row['shoppercountrycode'],
#                     'shopper_interaction': row['shopperinteraction'],
#                     'fraud': 1 if row['simple_journal'] == 'Chargeback' else 0,
#                     'verification': row['cardverificationcodesupplied'],
#                     'cvc_response': 3 if int(row['cvcresponsecode']) > 2 else row['cvcresponsecode'],
#                     'creation_date': datetime.strptime(row['creationdate'], '%Y-%m-%d %H:%M:%S'),
#                     'account_code': row['accountcode'],
#                     'mail_id': row['mail_id'], 'ip': row['ip_id'],
#                     'card_id': row['card_id']
#                 }
#
#                 for category in categorical_sets:
#                     categorical_sets[category].add(data_row[category])
#
#         return  categorical_sets



df['bookingdate'] = pd.to_datetime(df['bookingdate'])
df['creationdate'] = pd.to_datetime(df['creationdate'])
currency_dict = {'MXN': 0.01*0.05, 'SEK': 0.01*0.11, 'AUD': 0.01*0.67, 'GBP': 0.01*1.28, 'NZD': 0.01*0.61}
df['euro'] = map(lambda x,y: currency_dict[y]*x, df['amount'],df['currencycode'])
#key = lambda k:k.day
#print df.groupby(df['bookingdate'].apply(key))

df_sort_creation = df.sort_values(by = 'creationdate', ascending = True)#Returns a new dataframe, leaving the original dataframe unchanged
key = lambda k:(k.year,k.month,k.day)
#print df_sort_creation.groupby(df_sort_creation['creationdate'].apply(key)).mean()['amount']

df_renamed = df.rename(index=str, columns = {'txvariantcode': 'cardtype', 'bin': 'issuer_id', 'shopperinteraction': 'shoppingtype',
                   'simple_journal': 'label', 'cardverificationcodesupplied': 'cvcsupply',
                  'cvcresponsecode': 'cvcresponse', 'accountcode': 'merchant_id'})
df_select = (df_renamed[['issuercountrycode', 'cardtype', 'issuer_id', 'euro', 'currencycode',
              'shoppercountrycode', 'shoppingtype', 'label', 'cvcsupply', 'cvcresponse', 'merchant_id',
              'mail_id', 'ip_id', 'card_id']])
print ("Missing values per column:")
print (df_select.apply(lambda x: sum(x.isnull()), axis=0))
df_clean = df_select.fillna('missing')

issuercountrycode_category = pd.Categorical(df_clean['issuercountrycode'])
cardtype_category = pd.Categorical(df_clean['cardtype'])
issuer_id_category = pd.Categorical.from_array(df_clean['issuer_id'])
currencycode_category = pd.Categorical(df_clean['currencycode'])
shoppercountrycode_category = pd.Categorical(df_clean['shoppercountrycode'])
shoppingtype_category = pd.Categorical(df_clean['shoppingtype'])
cvcsupply_category = pd.Categorical(df_clean['cvcsupply'])
merchant_id_category = pd.Categorical(df_clean['merchant_id'])
mail_id_category = pd.Categorical(df_clean['mail_id'])
ip_id_category = pd.Categorical(df_clean['ip_id'])
card_id_category = pd.Categorical(df_clean['card_id'])

dict.fromkeys(issuercountrycode_category, set())
cardtype_dict = dict(set(zip(cardtype_category, cardtype_category.codes)))
currencycode_dict = dict(set(zip(currencycode_category, currencycode_category.codes)))
shoppercountrycode_dict = dict(set(zip(shoppercountrycode_category, shoppercountrycode_category.codes)))
shoppingtype_dict = dict(set(zip(shoppingtype_category, shoppingtype_category.codes)))
cvcsupply_dict = dict(set(zip(cvcsupply_category, cvcsupply_category.codes)))
merchant_id_dict = dict(set(zip(merchant_id_category, merchant_id_category.codes)))
mail_id_dict = dict(set(zip(mail_id_category, mail_id_category.codes)))
ip_id_dict = dict(set(zip(ip_id_category, ip_id_category.codes)))
card_id_dict = dict(set(zip(card_id_category, card_id_category.codes)))

df_clean['issuercountrycode'] = issuercountrycode_category.codes
df_clean['cardtype'] = cardtype_category.codes
df_clean['currencycode'] = currencycode_category.codes
df_clean['shoppercountrycode'] = shoppercountrycode_category.codes
df_clean['shoppingtype'] = shoppingtype_category.codes
df_clean['cvcsupply'] = cvcsupply_category.codes
df_clean['merchant_id'] = merchant_id_category.codes
df_clean['mail_id'] = mail_id_category.codes
df_clean['ip_id'] = ip_id_category.codes
df_clean['card_id'] = card_id_category.codes
df_clean['label_int'], df_clean['cvcresponse_int']= 0,0
df_clean['label_int'] = map(lambda x:1 if str(x) == 'Chargeback' else 0 if str(x) == 'Settled' else 'unknown', df_clean['label'])
df_clean['cvcresponse_int'] = map(lambda x:3 if x > 2 else x+0, df_clean['cvcresponse'])
#0 = Unknown, 1=Match, 2=No Match, 3=Not checked
df1 = df_clean.ix[(df_clean['label_int']==1) | (df_clean['label_int']==0)]#237036 instances
df1.head()

#hier moet een tabel uitkomen...


print ("Missing values per row:")
print (df_select.apply(num_missing, axis=1).head())
df_clean = df_select.fillna('missing')
df_select.shape
df_clean = df_select.dropna(axis=0)



df_input = (df1[['issuercountrycode', 'cardtype', 'issuer_id', 'currencycode',
              'shoppercountrycode', 'shoppingtype', 'cvcsupply', 'cvcresponse_int', 'merchant_id', 'euro',
              'label_int']])
df_input[['issuer_id','label_int']] = df_input[['issuer_id','label_int']].astype(int)
print(df_input.dtypes)
x = df_input[df_input.columns[0:-1]].as_matrix()
y = df_input[df_input.columns[-1]].as_matrix()



TP, FP, FN, TN = 0, 0, 0, 0
x_array = np.array(x)
x_array = preprocessing.normalize(np.array(x), norm='l2')

enc = preprocessing.OneHotEncoder()
enc = preprocessing.LabelEncoder()
enc.fit(x_array[:,0:-1])
x_encode = enc.transform(x_array[:,0:-1]).toarray()
min_max_scaler = preprocessing.MinMaxScaler()
x_array[:,-1] = min_max_scaler.fit_transform(x_array[:,-1])
x_in = np.c_[x_encode,x_array[:,-1]]
y_in = np.array(y)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_in, y_in, test_size = 0.2)#test_size: proportion of train/test data
print ("Training date set")
#sampler = UnderSampler(ratio=10, verbose = True)
#sampler = OverSampler(verbose = True)
sampler = SMOTE(kind='regular', verbose = True)
usx, usy = sampler.fit_transform(x_train, y_train)

def cutoff_predict(clf, x, cutoff):
    return (clf.predict_proba(x)[:,1]>cutoff).astype(int)
scores = []

def custom_score(cutoff):
    def score_cutoff(clf, x, y):
        ypred = cutoff_predict(clf, x, cutoff)
        return f1_score(y, ypred)
        #return precision_score(y, ypred)
    return score_cutoff
for cutoff in np.arange(0.1, 0.9, 0.1):
    clf = LogisticRegression()
    validated = cross_validation.cross_val_score(clf, usx , usy, cv = 10, scoring = custom_score(cutoff))
    scores.append(validated)
sns.boxplot(np.arange(0.1, 0.9, 0.1), scores)
plt.title('F scores for each cutoff setting')
plt.xlabel('each cutoff value')
plt.ylabel('custom score')
plt.show()

cutoff = 0.6
#clf = LogisticRegression()
#clf = svm.SVC(kernel='poly', degree=3)
#clf = svm.SVC(probability=True)
clf = RandomForestClassifier(n_estimators=50, criterion='gini')
#clf = svm.SVC(kernel='sigmoid')
clf.fit(usx, usy)
#y_predict = clf.predict(x_test)#output label
predict_proba = clf.predict_proba(x_test)
y_predict = (predict_proba[:,1]>cutoff).astype(int)
false_positive_rate, true_positive_rate, thresholds1 = roc_curve(y_test, predict_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.subplot(1, 2, 1)
plt.plot(false_positive_rate, true_positive_rate, 'b', label = 'AUC = %0.2f'% roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
#plt.legend('AUC = %0.2f'% roc_auc)
plt.legend(loc="lower right")
precision, recall, thresholds2 = precision_recall_curve(y_test, predict_proba[:,1])
plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
for i in xrange(len(y_predict)):
    if y_test[i]==1 and y_predict[i]==1:
        TP += 1
    if y_test[i]==0 and y_predict[i]==1:
        FP += 1
    if y_test[i]==1 and y_predict[i]==0:
        FN += 1
    if y_test[i]==0 and y_predict[i]==0:
        TN += 1
print ('TP: '+ str(TP))
print ('FP: '+ str(FP))
print ('FN: '+ str(FN))
print ('TN: '+ str(TN))


def create_x_y_sets(data, categorical_sets):
    selected_features = ['issuercountrycode', 'cardtype', 'issuer_id', 'euro', 'currencycode',
              'shoppercountrycode', 'shoppingtype', 'label', 'cvcsupply', 'cvcresponse', 'merchant_id',
              'mail_id', 'ip_id', 'card_id']
    features = []
    labels = []

    for row in data:
        features.append([row[x] if x not in CATEGORICAL else cat_to_nr(categorical_sets[x], row[x])
                         for x in selected_features])
        labels.append(row['fraud'])
    return np.array(features).astype(float), np.array(labels).astype(float)

def cat_to_nr(categorical_set: set, element):
    """
    Modify categorical feature to number in data set
    """
    return list(categorical_set).index(element)