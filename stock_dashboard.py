# -*- coding: utf-8 -*-
'''
@author: daire
'''
import os
from datetime import date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from matplotlib.dates import DateFormatter
from moving_averages import compute_moving_averages
from stocks import Stocks
from predictive_analysis import linear_reg
from model import Model
from streamlit_toggle import st_toggleswitch
import requests
from requests.exceptions import ConnectionError

# initialize stocks object to load data from the beginning of the chosen year to current date
stocks_start_year = 2000
stocks = Stocks(stocks_start_year)

# retrieve and save stocks data (if trading data has not been saved)
if not os.path.exists('datasets/stocks'):
    stocks.save_stock_files()

companies = stocks.get_all_tickers_overview()  # retrieves companies and descriptions, autofils NaNs
companies = companies.set_index('Symbol')

# ----------------- Build UI filter menu ------------------------------#
# unique sorted set of company names
stock_list = sorted(set(companies.loc[:, "Name"]))

st.write("""
         # Stock analyzer
         Select a stock of interest to see its historical data.
         """)

# -------------------- Stock selection ------------------------------#
stock_name = st.sidebar.selectbox("Stock name",
                                  options=stock_list)

visualizations = ["Stock price", "Stock volume", "Moving averages", "Weighted moving average", "Moving average converging/diverging"]
selected_viz = st.sidebar.multiselect("Visualization", visualizations)


# --- date selection --- #
# --- Compute today's date and Year's back date to set as default --- #
date_today = date.today()
date_year_back = date_today.replace(year=date_today.year - 1, month=date_today.month, day=date_today.day)
time_start = st.sidebar.date_input("Timeline start date", 
                                   value=date_year_back,
                                   min_value=stocks.START_DATE,
                                   max_value=date_today)

time_end = st.sidebar.date_input("Timeline end date", 
                                   value=date_today,
                                   max_value=date_today)
# dictionary to hold stock data
params = dict()


def create_stock_item(source):
    '''
    Parameters
    ----------
    num : Integer
        Integer for each stock selected.

    Returns
    -------
    A dictionary object representation a the selected stock

    '''

    return {'stock': companies.loc[companies['Name'] == source].index[0],
            'name': stock_name,
            'sector': companies.loc[companies['Name'] == source].Sector[0],
            'industry': companies.loc[companies['Name'] == source].Industry[0],
            'country': companies.loc[companies['Name'] == source].Country[0],
            'IPO_year': companies.loc[companies['Name'] == source]['IPO Year'][0]
            }


params = create_stock_item(stock_name)


# --- Print company details for selected stock --- #
st.write(f"""## *{params['stock']} : {params['name']}*""")


# ------------------ Load dataset from filter parameters -----------#
date_year_back = date_today.replace(year=date_today.year - 1, month=date_today.month, day=date_today.day)
df = stocks.get_trading_history(params["stock"], 
                                time_start, 
                                time_end,
                                save=True)


# ------------------ Plot data using filter parameters -------------#
# --- time series plot function - Seaborn --- #
def plot_time_series_sns(title, y_label, Y, col, df=df):
    # create timeline for each stock
    start_date = df.index[0]
    end_date = df.index[len(df) - 1]

    # set aesthetics for the chart
    sns.set(font_scale=2)
    sns.set_style("whitegrid")
    sns.color_palette("bright")
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    ax.set_xlabel("Date", fontsize = 24)
    ax.set_ylabel(y_label, fontsize = 24)
    ax.set_title(f"{title}", fontsize = 24)
    sns.lineplot(x = mdates.date2num(df.index), y = Y, 
                    data = df, color = 'blue', err_style = 'band',
                    linewidth = 3)
    plt.xticks(rotation = 90)
    
    
    # format date axis
    date_formatter = DateFormatter('%Y')
    ax.xaxis.set_major_formatter(date_formatter)

    # draw a trend line - if at least two points
    if len(df.index) > 1:
        # TODO: remove this fix when Nans sorted
        if "Moving averages" not in selected_viz:
            slope, intercept = np.polyfit(mdates.date2num(df.index), Y, 1)
            reg_line = slope*mdates.date2num(df.index) + intercept
            plt.plot(df.index, reg_line, color='orange', linewidth=6, linestyle="dashed")
            # plot selected timeframe
            col.pyplot(fig)
        else:
            col.pyplot(fig)
    else:
        col.write("Not enough data points available.")
    # TODO: create interactions

    return start_date, end_date


st.subheader("Time period last 5-day Performance")
st.write(df.tail(5))

# ------------------ Create layout (2 X 4) -----------------------------#
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)

# --- stock price visualization and price prediction --- #
if 'Stock price' in selected_viz:
    price_start, price_end = plot_time_series_sns('stock price', 'USD ($)', df["Adj Close"], col1)
    
    # NodeJs server needed for toggleswitch - fallback to standard radio button
    server_url = 'http://localhost:3001'
    try:
        server = requests.get('http://localhost:3001')
        predict_yn_toggle = st_toggleswitch("Predict stock price?", False)
        if predict_yn_toggle:
            pred_slider = True
        else:
            pred_slider = False
    except ConnectionError:
        predict_yn = col1.radio("Predict stock price?", options=['Yes', 'No'], index=1, 
                          help="Select \"Yes\" to get options for stock prediction.")
        if predict_yn == "Yes":
            pred_slider = True
        else:
            pred_slider = False

    if pred_slider:
        # ------------------ Predictive model -----------------------------#
        # dynamically set the model age
        current_year = date_today.year
        model_recency = col1.slider('Model recency (years)', min_value=1, max_value=10, step=1)
        model_start = f'{current_year - model_recency}-01-01'
        model = Model(model_start)
        # function runs once and caches the result
        @st.cache(persist=True, show_spinner=True)
        def build_model(stock, model):
            # build model if no columns
            model_df = model.create_model(stock)
            return model_df, stock
       
        
       # -------------------plotting Stock Prediction Graph----------------#
        def plot_linear_regression(x_train, x_test, y_regression, y_predict, company_name, model_df, predict_parameter):
            plt.figure(figsize=(12, 16))
            plt.title("{0} {1} Price Predictions".format(company_name, predict_parameter))
            plt.xlabel("Date")
            plt.ylabel("Price USD ($)")
            plt.plot(x_train, model_df["Close"], label="Historical Price", color="blue", linewidth=3)
            plt.plot(x_train, y_regression, label="Mathematical Model", color="orange", linewidth=6, linestyle='dashed')
            plt.plot(x_test, y_predict, label="Stock Predictions", color="Red", linewidth=8)
            plt.legend(loc="lower right")
            plt.xticks(rotation = 90)
            return plt
        
        
         # ----- build a model for selected company
        stock_model, company = build_model(params['stock'], model)
        pred_window = col1.slider("Prediction window (days)", min_value=0, max_value=365, step=30)
        st.write(f'Model dimensions for {company}: {stock_model.shape}')
         
         # train the model, make predictions, return metrics
        if pred_window > 1:
             x_train, x_test, y_regression, y_predict, company_name, model_df, mse, r_squared = linear_reg(stock_model, pred_window, params['name'])
             
             # model metrics
             st.write("Model mean squared error: ", mse)
             st.write("Model R-squared value: ", r_squared)
             
             # plot the model for a prediction window
             pred_plt = plot_linear_regression(x_train, x_test, y_regression, y_predict, company_name, model_df, "Close")
             col1.pyplot(pred_plt)
             st.balloons()
                
 
# --- stock volume --- #
if "Stock volume" in selected_viz:
    # area plot example
    volume_start, volume_end = plot_time_series_sns('trading volume', 'shares', df.loc[:, 'Volume'], col2)

if "Moving averages" in selected_viz:
    # add moving averages columns to the trading dataframe
    # --- Slidebar to choose length for computing Moving average
    window = st.sidebar.slider(label='Span to Compute Moving Average', 
                               min_value=2, max_value=200, value=20, step=1)
    compute_moving_averages(df, 'Adj Close', window)
    plot_time_series_sns('Moving Averages', 'Moving Avg.', df.loc[:, 'SMA'], col3)

    

