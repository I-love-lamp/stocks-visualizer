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
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateDays'] = (df.Date - pd.to_datetime("1970-01-01")).dt.days
    
    # sample and split data into test and training sets
    
    # target variable is Closing price
    y_train = np.asarray(df["Close"])
    x_train = np.asarray(df['DateDays'])
   
    # create the regressor and fit to training data
    regression_model = LinearRegression()
    regression_model.fit(x_train.reshape(1, -1), y_train.reshape(1, -1))
    y_history = regression_model.predict(x_train.reshape(-1, 1))
    
    # project days beyond today
    future_days = np.asarray(range(x_train[-1]+1, x_train[-1] + input_days))
    x_test = np.append(x_train, future_days)
     
    # future prediction
    y_predict = regression_model.predict(x_test.reshape(-1, 1))
    
    # build Timeline series for descriptive and prescriptive
    #x_train = pd.to_datetime(df.Date, origin="1970-01-01", unit="D")
    x_test = pd.to_datetime(x_test, origin="1970-01-01", unit="D")
    compute_rmse_and_r2_values(y_train, y_history)
    plot_linear_regression(x_train, x_test, y_history, y_predict, company_name, df, "Close")
    
    ''' # project days beyond today
     future_days = np.asarray(range(X_test[-1], X_test[-1] + input_days))
     X_project = np.append(X_test, future_days)
     
     y_project = regression_model.predict(X_project.reshape(1, -1))
     #X_test = pd.to_datetime(df.Date, origin="1970-01-01", unit="D")
     #X_project = X_project
     X_project = pd.to_datetime(X_project, origin="1970-01-01", unit="D")
     compute_rmse_and_r2_values(y_test, y_project)
     plot_linear_regression(X_test, X_project, y_test, y_project, company_name, df, "Close")'''

    
def linear_reg_v2(df, input_days, company_name):  
    df = df.reset_index()   
    df['Date'] = pd.to_datetime(df['Date'])
    
    # need numeric date representation - using Days instead of Date
    df['DateDays'] = (df['Date'] - pd.to_datetime("1970-01-01")).dt.days
    df.set_index('Date', inplace=True, drop=True)
    
    # prepare the data (impute Nans)
    imputer = SimpleImputer(strategy="median")
    imputer.fit_transform(df)
    
    
    # Only interested in Closing price   
    df = df.loc[:, ['DateDays', 'Close']]
    
    # sample and split data into test and training sets
    X = df
    
    # target variable is Closing price
    y = np.asarray(df["Close"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
    # train the model
    regression_model = LinearRegression()
    regression_model.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
       
    # historical and projected
    X_project = np.asarray(pd.RangeIndex(start=X_test[-1], stop=X_test[-1] + input_days))
    y_project = regression_model.predict(X_project.reshape(-1, 1))
    
    # historical
    y_test = regression_model.predict(X_test.reshape(-1, 1))
    
    # convert x-axis back to date format
    X_test = pd.to_datetime(X_test, origin="1970-01-01", unit="D")
    X_project = pd.to_datetime(X_project, origin="1970-01-01", unit="D")
    compute_rmse_and_r2_values(y_test, y_project)
    plot_linear_regression(X_test, X_project, y_test, y_project, company_name, df, 'Close' )   
    