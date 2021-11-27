#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:48:49 2021

@author: daire
"""
import pandas as pd
import os
from pandas_datareader import data
from datetime import date
from datetime import datetime as dt
from datetime import timedelta

class Stocks(object):
    """
        Class that retrieves and manages stock trading data
    """
    def __init__(self, year):
        # path to data files
        self.STOCKS_BASE_PATH = os.path.join("datasets", "stocks")
        self.STOCKS_DIR = os.makedirs(self.STOCKS_BASE_PATH, exist_ok=True)
        
        
        # financial data source
        self.STOCK_SOURCE = 'yahoo'
        self.DIVIDEND_SOURCE = 'yahoo-dividends'
        self.START_DATE=dt.strptime(f"{year}-01-01", '%Y-%m-%d').date()
        
        
        # list of companies
        self.TICKER_OVERVIEW = pd.DataFrame(pd.read_csv("nasdaq_screener.csv")).fillna("Unknown") # handle NaN values
        self.stocks_list = [] # populates UI with
        self.CURRENT_STOCK_DF = pd.DataFrame()
        self.CURRENT_TICKER = ''
    
    def clean_data(self, df):
        # set date as index and format to datetime     
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index, format = '%Y-%m-%d').date
            # df.index = [dt.strptime(date, '%Y-%m-%d').date for date in df.index]
            return df
        
    def get_current_ticker(self):
        """        
        Returns the current ticker symbol name.
        
        Returns
        -------
        string
            Ticker symbol for current stock.

        """
        return self.CURRENT_TICKER
    
    def set_current_stock(self, ticker, startDt):       
        """
        Function that sets the full trading data for the selected stock from a 
        selected date.

        Parameters
        ----------
        ticker : string
            A trading entity ticker.
        startDt : Date
            Earliest trading date.

        Returns
        -------
        None.

        """
        try:
            self.CURRENT_STOCK_DF = data.DataReader(ticker, self.STOCK_SOURCE, startDt)
            print('Retrieved ', len(self.CURRENT_STOCK_DF), ' rows for ', ticker)
        except Exception as e:
            print('Unable to retrieve data for ', ticker, '. ', e)
        
    def get_current_stock(self):
        """
        Function that retrieves financial data for the currently selected trading entity.
        
        Returns
        -------
        DataFrame
            Full trading data for the selected stock.
        """
        return self.CURRENT_STOCK_DF
    
    def save_trading_data(self, ticker_sym):
        """
        Retrieve trading data for a single trading entity and save it to a file.
        Parameters
        ----------
        ticker_sym : String
            Trading entity.

        Returns
        -------
        None.

        """
        try:
            self.set_current_stock(ticker_sym, self.START_DATE)
            self.get_current_stock().to_csv(self.STOCKS_BASE_PATH + f"/{ticker_sym}.csv")
            print("Saved stock history for ", ticker_sym, ".csv")
            self.stocks_list.append(ticker_sym) # add ticker to valid list (for UI)
        except Exception as e :
           print(f"Could not retrieve stock for {ticker_sym}. {e}")
                    
            
    # function to retrieve stock from list of companies
    def save_stock_files(self):
        """
        Retrieve and save a file trading entities from a list of trading companies.
    
        Returns
        -------
        None.
    
        """
        # loop through a file containing a list of trading companies
        for ticker in self.TICKER_OVERVIEW.index:
            self.save_trading_data(str(ticker)) # need to find earliest date for the stock
    
    
    def get_ticker_trading_history(self, ticker):
        """
        Function that retrieves trading data from saved files for a trading entity

        Parameters
        ----------
        ticker : String
            Trading entity.

        Returns
        -------
        trading_history : Data Frame
            Trading data for the selected period.

        """
        # open the source file and return it as a DF
        try:
            trading_history = pd.read_csv(self.STOCKS_BASE_PATH + f"/{ticker}.csv") # remove for performance
            self.set_current_stock(ticker, self.START_DATE)
            trading_history = self.get_current_stock()
            print(f"Retrieved {len(trading_history)} rows from file for {ticker}") # returns today as start date
        except FileNotFoundError:
            trading_history = self.get_current_stock() # returns today as start date
        #self.clean_data(trading_history)
        self.CURRENT_TICKER = ticker # saves stock symbol as a class variable
        return trading_history
     
    def get_all_tickers_overview(self):
        """
        Retrieves company details for all companies in the companies dataset.

        Returns
        -------
        DataFrame
            DataFrame containing overview for all companies.

        """
        return self.TICKER_OVERVIEW
    
    
    def get_trading_companies(self):
       """
       This function is called by the app to retrieve a list of trading entities
       that have been loaded to the application.
    
       Returns
       -------
       List
           Trading entities that have been loaded to the application.
    
       """
       return self.stocks_list
        
   
    def get_trading_history(self, ticker, startDt, endDt, save):
        """
        Function that retrieves trading data from saved files between two dates for a trading entity

        Parameters
        ----------
        ticker : String
            Trading entity
        fromDt : Date
            Start date for historical trading data.
        toDt : Date
            End date for historical trading data.
        save: Boolean
            Flag for saving csv files.
        Returns
        -------
        trading_history : Data Frame
            Trading data for the selected period.

        """
        
        # open the source file and return it as a DF
        if save:
            try:
                trading_history = pd.read_csv(f"datasets/stocks/{ticker}.csv")
                
                # clean step
                self.clean_data(trading_history)
                
                print("Retrieved ", len(trading_history), " rows for ", ticker, "between ", startDt, " and ", endDt)
            except FileNotFoundError:
                self.save_trading_data(ticker) # save the stock to CSV
                # TODO: Error handling
                trading_history = self.get_current_stock()
                
                # clean step
                self.clean_data(trading_history)
        else:
            # get company data, dont save a CSV
            trading_history = data.DataReader(ticker, self.STOCK_SOURCE, startDt)
            
                
        # filter data using startDt and endDt
        # trading_history = trading_history.loc[(trading_history.index >= startDt) & (trading_history.index <= endDt)]
        return trading_history                 
        
        