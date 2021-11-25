'''
@author: Laura, Vandana
'''
import pandas as pd


def compute_adj_close_data(input_col):
    return input_col['Adj Close']


def compute_simple_moving_average(input_col, window):
    return input_col.rolling(window=window, center=True).mean()


def calculate_wma(weight):
    def calculate(x):
        return sum(weight * x) / sum(weight)

    return calculate


def compute_weighted_moving_average(input_col, window):
    weight_list = list(reversed([(window - n) * window for n in range(window)]))
    return input_col.rolling(window=window).apply(calculate_wma(weight_list), raw=True)


def compute_exponential_moving_average(input_col, span):
    return input_col.ewm(span=span, adjust=False).mean()


def compute_macd_signal(input_col, a, b, c):
    ema1 = input_col.ewm(span=a, adjust=False).mean()
    ema2 = input_col.ewm(span=b, adjust=False).mean()
    macd = pd.DataFrame(ema1 - ema2)
    return pd.DataFrame(macd.ewm(span=c, adjust=False).mean())


def compute_moving_averages(input_df, col_name, window):
    input_col = input_df[col_name]
    input_df['SMA'] = compute_simple_moving_average(input_col, window)
    input_df['WMA'] = compute_weighted_moving_average(input_col, window)
    input_df['EMA'] = compute_exponential_moving_average(input_col, window)
    input_df['MACD'] = compute_macd_signal(input_col, 26, 12, 9)
    return input_df
