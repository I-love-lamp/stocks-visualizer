'''
@author: Laura, Vandana
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -------------------plotting Stock Prediction Graph------------------------------------
def plot_linear_regression(x_train, x_test, y_history, y_predict, company_name, df, predict_parameter):
    plt.figure(figsize=(16, 8))
    plt.title("{0} {1} Price Predictions".format(company_name, predict_parameter))
    plt.xlabel("Date")
    plt.ylabel("Price USD ($)")
    plt.plot(x_train, df["Close"], label="Historical Price", color="Green")
    plt.plot(x_train, y_history, label="Mathematical Model", color="tab:blue")
    plt.plot(x_test, y_predict, label="Stock Predictions", color="Red")
    plt.legend(loc="lower right")
    plt.show()


# ------------Calculates RMSE & R2 from linear regression prediction model-----------------
def compute_rmse_and_r2_values(y_train, y_history):
    lr_mse = sqrt(mean_squared_error(y_train, y_history, squared=False))
    r2 = r2_score(y_train, y_history)
    return round(lr_mse, 4), round(r2, 4)


# ------------------ prediction model with linear regression----------------------------------
def linear_reg(df, input_days, company_name):
    df['DateDays'] = (df.Date - pd.to_datetime("1970-01-01")).days
    
    # sample and split data into test and training sets
    
    # target variable is Closing price
    y_train = np.asarray(df["Close"])
    
    # predictors: 'Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Company', 'Action'
    x_train = df
    
    regression_model = LinearRegression()
    regression_model.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    y_history = regression_model.predict(x_train.reshape(-1, 1))
    x_test = np.asarray(pd.RangeIndex(start=x_train[-1], stop=x_train[-1] + input_days))
    y_predict = regression_model.predict(x_test.reshape(-1, 1))
    x_train = pd.to_datetime(df.index, origin="1970-01-01", unit="D")
    x_test = pd.to_datetime(x_test, origin="1970-01-01", unit="D")
    compute_rmse_and_r2_values(y_train, y_history)
    plot_linear_regression(x_train, x_test, y_history, y_predict, company_name, df, "Close")
