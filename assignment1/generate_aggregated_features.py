import sqlite3
from pprint import pprint
from dateutil.relativedelta import relativedelta

from assignment1 import load_data, postprocess_data
from datetime import datetime
import progressbar


def create(conn, data: list):
    c = conn.cursor()
    c.execute('''CREATE TABLE transactions
                 (id int, booking_date date, issuer_country text, tx_variant text, issuer_id float, amount float, 
                 currency text, shopper_country int, shopper_interaction text, fraud int, verification text, 
                 cvc_response int, creation_date date, account_code text, mail_id text, card_id text)''')

    rows = [(row['id'], row['booking_date'], row['issuer_country'], row['tx_variant'], row['issuer_id'], row['amount'],
             row['currency'], row['shopper_country'], row['shopper_interaction'], row['fraud'], row['verification'],
             row['cvc_response'], row['creation_date'], row['account_code'],
             row['mail_id'], row['card_id']) for row in postprocessed_data]

    c.executemany('''INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', rows)
    conn.commit()


def txn_amount_over_month(conn, data: list):
    print('txn_amount_over_month')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('average_daily_over_month.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(months=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT amount FROM transactions WHERE card_id = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['card_id'], begin, row['creation_date']))
            transactions = c.fetchall()
            f.write('{}\n'.format(sum([amount[0] for amount in transactions]) / 30))

            bar.update(idx)


def average_daily_over_month(conn, data: list):
    print('average_daily_over_month')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('average_daily_over_month.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(months=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT amount FROM transactions WHERE card_id = ? AND account_code = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['card_id'], row['account_code'], begin, row['creation_date']))
            transactions = c.fetchall()
            f.write('{}\n'.format(sum([amount[0] for amount in transactions]) / 30))

            bar.update(idx)


if __name__ == '__main__':
    conn = sqlite3.connect('data.db')

    data, categorical_sets = load_data()
    postprocessed_data = postprocess_data(data)

    # create(conn, postprocessed_data)

    txn_amount_over_month(conn, postprocessed_data)
    average_daily_over_month(conn, postprocessed_data)

    conn.close()