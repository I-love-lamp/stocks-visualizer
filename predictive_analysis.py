'''
@author: Laura, Vandana
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


# ------------Calculates RMSE & R2 from linear regression prediction model-----------------
def compute_rmse_and_r2_values(y_train, y_history):
    lr_mse = sqrt(mean_squared_error(y_train, y_history, squared=False))
    r2 = r2_score(y_train, y_history)
    return round(lr_mse, 4), round(r2, 4)


# ------------------ prediction model with linear regression----------------------------------
def linear_reg(df, input_days, company_name):
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateDays'] = (df.Date - pd.to_datetime("1970-01-01")).dt.days
    
    # sample and split data into test and training sets
    
    # target variable is Closing price
    y_train = np.asarray(df["Close"])
    X_train = np.asarray(df['DateDays'])

    # create the regressor and fit to training data
    regression_model = LinearRegression()
    regression_model.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    y_history = regression_model.predict(X_train.reshape(-1, 1))
    
    # project days beyond today
    future_days = np.asarray(range(X_train[-1]+1, X_train[-1] + input_days))
     
    # future prediction
    y_predict = regression_model.predict(future_days.reshape(-1, 1))
    
    # build Timeline series for descriptive and prescriptive
    future_days= pd.to_datetime(future_days, origin="1970-01-01", unit="D")
    mse, r_squared = compute_rmse_and_r2_values(y_train, y_history)
    return X_train, future_days, y_history, y_predict, company_name, df, mse, r_squared

    