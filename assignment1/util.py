from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def transpose_to_transactions_per_card(data: list):
    transactions_per_card = defaultdict(list)
    for row in data:
        transactions_per_card[row['card_id']].append(data)
    return transactions_per_card


def group_by_booking_date(dataframe: pd.DataFrame):
    key = lambda k: (k.year, k.month, k.day)
    dataframe_sort_creation = dataframe.sort_values(by='creation_date',
                                                    ascending=True)  # new Frame of data d to leave the original Frame of data the same
    print(dataframe_sort_creation.groupby(dataframe_sort_creation['creation_date'].apply(key)).mean()['amount'])
    print(dataframe.groupby(dataframe['booking_date'].apply(key)))


def nonnr_to_nr(categorical_set, element):
    """
    String data values need to be converted to numerical value
    """
    if element == -1:  # Missing value substituted
        return element
    else:
        return list(categorical_set).index(element)


# TODO: ik begrijp niet wat dit moet doen
def unknown():
    df = pd.read_csv('data_for_student_case.csv')
    print('\nshape data')
    print(df.shape)
    print('\ndescribing float data')
    print(df.describe())
    print('\nindex types')
    print(df.dtypes)
    print(df.columns)

    # Ik kan niet alleen de correcte  ge encoded column uit elke krijgen.

    # encoding de column txtvariantcode
    X = df.iloc[:, :].values
    labelencoder_X = LabelEncoder()

    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    Y = pd.DataFrame(X)
    print(Y)

    # encoding de column issuercountrycode
    B = df.iloc[:, :].values
    labelencoder_B = LabelEncoder()
    B[:, 2] = labelencoder_B.fit_transform(B[:, 3])
    G = pd.DataFrame(B)
    print(G)

    names = []
    grouping_names = range(0, 500)
    counter = 0
