import os.path
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from datetime import date
from matplotlib.dates import DateFormatter
from stocks import Stocks
from moving_averages import compute_moving_averages

# initialize stocks object to load data from the beginning of the chosen year to current date
stocks = Stocks(1980)

# retrieve and save stocks data (if trading data has not been saved)
if not os.path.exists('datasets/stocks'):
    stocks.save_stock_files()

companies = stocks.get_all_tickers_overview()  # retrieves companies and descriptions, autofils NaNs
companies = companies.set_index('Symbol')

# ----------------- Build UI filter menu ------------------------------#
# create sets to store unique instances to populate select boxes
sectors = set(companies.loc[:, "Sector"])
# TODO: remove all stocks with unusual characters, ^ in symbol, 
stock_list = sorted(set(companies.loc[:, "Name"]))

st.write("""
         # Stock analyzer
         Select a stock of interest to see its historical data. \
             Use the Sector to identify stocks of interest.
         """)
stock_name = st.sidebar.selectbox("Stock name",
                                  options=stock_list)

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

# ------------------- visualization selection ------------------------------#
visualizations = ["Stock price", "Stock volume", "Moving averages"]
selected_viz = st.sidebar.multiselect("Visualization", visualizations)

# TODO: show how old the data is - and provide a refresh
# pull in daily data
st.sidebar.button("Refresh data")

# ------------------ Load dataset from filter parameters -----------#
date_today = date.today()
date_year_back = date_today.replace(year=date_today.year - 1, month=date_today.month, day=date_today.day)
df = stocks.get_trading_history(params["stock"], stocks.START_DATE, date_today)
df.index = pd.to_datetime(df.index).date

# -------------------- Date selection ------------------------------#
# filter dates
time_start = st.sidebar.date_input("Timeline start date",
                                   value=date_year_back,
                                   max_value=date_today)

time_end = st.sidebar.date_input("Timeline end date",
                                 value=date_today,
                                 max_value=date_today)

# --- Slidebar to choose length for computing Moving average
window = st.sidebar.slider(label='Span to Compute Moving Average', min_value=2, max_value=200, value=20, step=1)

# ------------------ Plot data using filter parameters -------------#
# --- time series plot function - Matplotlib --- #
def plot_time_series(title, y_label, Y, col, df=df):
    # set plot font
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 8
            }
    matplotlib.rc('font', **font)

    # timeline for each stock
    # TODO: convert dates from timestamp to %Y-%m-%d
    start_date = df.index[0]
    end_date = df.index[len(df) - 1]
    date_format = mdates.DateFormatter('%Y')
    pd.plotting.plot_params = {'x_compat': True, }
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    plt.plot(df.index, Y)
    ax.set(
        xlabel="date",
        ylabel=y_label,
        title=f"""{params['name']} {title}""",
        xlim=(start_date, end_date)
    )
    ax.xaxis.set_major_formatter(date_format)
    # TODO: decide on matplotlib or streamlit plot
    plt.xticks(rotation=90)
    plt.grid()
    fig.canvas.toolbar_visible = True
    fig.canvas.header_visible = True

    # draw a trend line
    slope, intercept = np.polyfit(mdates.date2num(df.index), Y, 1)
    reg_line = slope * mdates.date2num(df.index) + intercept
    plt.plot(df.index, reg_line)
    col.write(fig)

    # return filter date range
    return start_date, end_date


# --- time series plot function - Seaborn --- #
def plot_time_series_sns(title, y_label, Y, col, df=df):
    # create timeline for each stock
    start_date = df.index[0]
    end_date = df.index[len(df) - 1]

    # set aesthetics for the chart
    sns.set(rc={'figure.figsize': (32, 32)})
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.color_palette("bright")
    fig, ax = plt.subplots()
    fig.set_figheight(32)
    ax.set_xlabel("Date", fontsize=32)
    ax.set_ylabel(y_label, fontsize=32)
    ax.set_title(f"{params['name']} {title}", fontsize=36)
    sns.lineplot(x=mdates.date2num(df.index), y=Y,
                 data=df, color='blue', err_style='band')
    plt.xticks(rotation=90)

    # format date axis
    date_formatter = DateFormatter('%Y')
    ax.xaxis.set_major_formatter(date_formatter)

    # draw a trend line - if at least two points
    if len(df.index) > 1:
        slope, intercept = np.polyfit(mdates.date2num(df.index), Y, 1)
        reg_line = slope * mdates.date2num(df.index) + intercept
        plt.plot(df.index, reg_line, color='orange', linewidth=4, linestyle="dashed")
        # plot selected timeframe
        col.pyplot(fig)
    else:
        col.write("Not enough data points available.")
    # TODO: create interactions

    return start_date, end_date


st.subheader("Last 5-day Performance")
st.write(df.tail(5))

# ------------------ Create layout (2 X 4) -----------------------------#
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)

# --- stock price --- #
# Streamlit area plot
if 'Stock price' in selected_viz:
    price_start, price_end = plot_time_series_sns('stock price', 'USD ($)', df["Adj Close"], col1)

# --- stock volume --- #
if "Stock volume" in selected_viz:
    # area plot example
    volume_start, volume_end = plot_time_series_sns('trading volume', 'shares (millions)', df.loc[:, 'Volume'], col2)

# --- Plot Moving Average --- #
if "Moving averages" in selected_viz:
    compute_moving_averages(df, 'Adj Close', window)
    print(df.head())
    # to do - plot the graphs for received df

# ---------------------- Stock performance KPIs ---------------------#
# analyze = Analyzer(df, '2000-01-01', '2021-11-10')
# day_price = analyze.get_day_price(date_price)
# col6.print(f"Stock price on {date_price} was {day_price}")
