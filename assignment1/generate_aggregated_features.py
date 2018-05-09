import sqlite3
from dateutil.relativedelta import relativedelta

from assignment1 import load_data
import progressbar
from multiprocessing import Pool, Process


def create(data: list):
    conn = sqlite3.connect(':memory:')

    c = conn.cursor()
    c.execute('''CREATE TABLE transactions
                 (id int, booking_date date, issuer_country text, tx_variant text, issuer_id float, amount float, 
                 currency text, shopper_country int, shopper_interaction text, fraud int, verification text, 
                 cvc_response int, creation_date date, account_code text, mail_id text, card_id text)''')

    rows = [(row['id'], row['booking_date'], row['issuer_country'], row['tx_variant'], row['issuer_id'], row['amount'],
             row['currency'], row['shopper_country'], row['shopper_interaction'], row['fraud'], row['verification'],
             row['cvc_response'], row['creation_date'], row['account_code'],
             row['mail_id'], row['card_id']) for row in data]

    c.executemany('''INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', rows)
    conn.commit()

    return conn


def txn_amount_over_month(data: list):
    conn = create(data)

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
    conn.close()


def average_daily_over_month(data: list):
    conn = create(data)

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
    conn.close()


def amount_same_day(data: list):
    conn = create(data)

    print('amount_same_day')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('amount_same_day.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(days=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT amount FROM transactions WHERE card_id = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['card_id'], begin, row['creation_date']))
            transactions = c.fetchall()
            f.write('{}\n'.format(sum([amount[0] for amount in transactions])))

            bar.update(idx)
    conn.close()


def number_same_day(data: list):
    conn = create(data)

    print('number_same_day')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('number_same_day.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(days=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT COUNT(*) FROM transactions WHERE card_id = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['card_id'], begin, row['creation_date']))
            transactions = c.fetchone()
            f.write('{}\n'.format(sum([count[0] for count in transactions])))

            bar.update(idx)
    conn.close()


def amount_same_merchant_month(data: list):
    conn = create(data)

    print('amount_same_merchant_month')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('number_same_day.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(days=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT amount FROM transactions WHERE account_code = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['account_code'], begin, row['creation_date']))
            transactions = c.fetchone()
            f.write('{}\n'.format(sum([amount[0] for amount in transactions]) / 30))

            bar.update(idx)


def number_same_merchant_month(data: list):
    conn = create(data)

    print('number_same_merchant_month')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('number_same_merchant_month.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(months=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT COUNT(*) FROM transactions WHERE account_code = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['account_code'], begin, row['creation_date']))
            transactions = c.fetchone()
            f.write('{}\n'.format(sum([amount[0] for amount in transactions])))

            bar.update(idx)
    conn.close()


def amount_same_currency_month(data: list):
    conn = create(data)

    print('amount_same_currency_month')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('amount_same_currency_month.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(months=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT amount FROM transactions WHERE currency = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['currency'], begin, row['creation_date']))
            transactions = c.fetchone()
            f.write('{}\n'.format(sum([amount[0] for amount in transactions]) / 30))

            bar.update(idx)
    conn.close()


def number_same_currency_month(data: list):
    conn = create(data)

    print('number_same_currency_month')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('number_same_currency_month.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(months=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT COUNT(*) FROM transactions WHERE currency = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['currency'], begin, row['creation_date']))
            transactions = c.fetchone()
            f.write('{}\n'.format(sum([amount[0] for amount in transactions])))

            bar.update(idx)
    conn.close()


def amount_same_shopper_country_month(data: list):
    conn = create(data)

    print('amount_same_shopper_country_month')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('amount_same_shopper_country_month.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(months=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT amount FROM transactions WHERE shopper_country = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['shopper_country'], begin, row['creation_date']))
            transactions = c.fetchone()
            f.write('{}\n'.format(sum([amount[0] for amount in transactions]) / 30))

            bar.update(idx)


def number_same_shopper_country_month(data: list):
    conn = create(data)

    print('number_same_shopper_country_month')
    bar = progressbar.ProgressBar(max_value=len(data))

    with open('number_same_shopper_country_month.csv', mode='w') as f:
        for idx, row in enumerate(data):
            begin = row['creation_date'] + relativedelta(months=-1)
            c = conn.cursor()
            c.execute(
                '''SELECT COUNT(*) FROM transactions WHERE shopper_country = ? AND creation_date BETWEEN date(?) AND date(?)''',
                (row['shopper_country'], begin, row['creation_date']))
            transactions = c.fetchone()
            f.write('{}\n'.format(sum([amount[0] for amount in transactions])))

            bar.update(idx)
    conn.close()


def create_all(data: list):
    p1 = Process(target=txn_amount_over_month, args=(data,))
    p1.start()
    p2 = Process(target=txn_amount_over_month, args=(data,))
    p2.start()
    p3 = Process(target=average_daily_over_month, args=(data,))
    p3.start()
    p4 = Process(target=amount_same_day, args=(data,))
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    p1 = Process(target=number_same_day, args=(data,))
    p1.start()
    p2 = Process(target=amount_same_merchant_month, args=(data,))
    p2.start()
    p3 = Process(target=number_same_merchant_month, args=(data,))
    p3.start()
    p4 = Process(target=amount_same_currency_month, args=(data,))
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    p1 = Process(target=number_same_currency_month, args=(data,))
    p1.start()
    p2 = Process(target=amount_same_shopper_country_month, args=(data,))
    p2.start()
    p3 = Process(target=number_same_shopper_country_month, args=(data,))
    p3.start()

    p1.join()
    p2.join()
    p3.join()


if __name__ == '__main__':
    data, categorical_sets = load_data(use_cached=False)
    create_all(data)
