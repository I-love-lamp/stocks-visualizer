#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:53:22 2021

@author: daire
"""
import os
import pandas as pd
from pandas_datareader import data

class Model(object):
    """
        Class that builds the stock prediction model from a data source.
    """
    def __init__(self, date):
        self.MODEL_BASE_PATH = os.path.join("datasets", "model")
        self.MODEL_DIR = os.makedirs(self.MODEL_BASE_PATH, exist_ok=True)
        self.MODEL_BUILT = False
        self.MODEL_DF = pd.DataFrame()
        
        
        # financial data source
        self.STOCK_SOURCE = 'yahoo'
        self.DIVIDEND_SOURCE = 'yahoo-dividends'
        self.START_DATE = date
        
    # -------------------    Flag for model build state ------------------------------------
    def set_model_state(self,):
        if len(self.MODEL_DF) > 0:
            self.MODEL_BUILT = True
        else:
            self.MODEL_BUILT = False
            
    def get_model_state(self):
        return self.MODEL_BUILT
    
    # -------------------        build the model        ------------------------------------
    # get dividends data
    def get_dividend_history(self, companies, startDt, endDt):
        '''      
        Parameters
        ----------
        companies : Iterable
            List of company tickers.
        startDt : Timestamp
            DESCRIPTION.
        endDt : TYPE
            DESCRIPTION.

        Returns
        -------
        dividend_hist : DataFrame
            List of dividend amounts and dates for all companies provided.

        '''      
        dividend_hist = pd.DataFrame()
        for company in companies:
            special_characters = "!@#$%^&*()-+?_=,<>/"
            if not any(char in special_characters for char in company):
                company_dividend = data.DataReader(company, self.DIVIDEND_SOURCE, startDt, endDt)
                if len(company_dividend) > 0:
                    company_dividend = company_dividend.reset_index()
                    company_dividend['Date'] = company_dividend.iloc[:, 0]
                    company_dividend['Action'] = 'dividend'
                    company_dividend['Company'] = company
                    if len(dividend_hist) == 0:
                        dividend_hist = company_dividend
                    else:
                        dividend_hist = dividend_hist.append(company_dividend, ignore_index=True)      
                    print(f"Retrieved dividend history for {company}")
                else:
                    print(f"No dividend history available for {company}")
            
        # reformat dividend dataframe to align with model dataframe
        dividend_hist.drop('index', 1, inplace=True)
        dividend_hist.drop('action', 1, inplace=True)
        dividend_hist = dividend_hist[['Date', 'Company', 'Action', 'value']]
            
        return dividend_hist

    # construct the model
    def create_model(self, company):
        '''
        Parameters
        ----------
        companies : Iterable
            List of trading companies to build the model.

        Returns
        -------
        None.

        '''
        row_count = 0
        print("Building the model...")
         
        # API fails for tickers containing special characters
        special_characters = "!@#$%^&*()-+?_=,<>/"
        if not any(char in special_characters for char in company):
            try:
                company_prices = data.DataReader(company, self.STOCK_SOURCE, self.START_DATE)
                company_prices = company_prices.reset_index()
                if row_count == 0:
                    self.MODEL_DF = company_prices
                else:
                    self.MODEL_DF = self.MODEL_DF.append(company_prices.loc[:, :'Adj Close'], ignore_index=True)
                    
                if 'Company' not in self.MODEL_DF.columns:
                    self.MODEL_DF['Company'] = ''
                
                self.MODEL_DF.iloc[row_count:1+(row_count+len(company_prices)), 7] = company
                self.MODEL_DF['Day Range'] = self.MODEL_DF['High'] - self.MODEL_DF['Low']
                row_count += len(company_prices)
                print(f"Added trading history for {company} to the model.")
            except Exception as e:
                print('Unable to retrieve data for ', company, '. ', e)                
        else:
            print("Invalid Characters in company ticker string - unable to retrieve trading history.")
        # re-order columns
        cols = ['Date', 'Open', 'Adj Close', 'Low', 'High', 'Day Range', 'Close']
        self.MODEL_DF = self.MODEL_DF[cols]
        self.MODEL_DF = self.MODEL_DF.reset_index(drop=True)
        self.MODEL_DF.set_index('Date', inplace=True)
        
        # set flag to identify the model build status 
        self.set_model_state()
        
        # TODO: incorporate earnings per share into the model
        #dividends = self.get_dividend_history(set(companies), self.START_DATE, date.today().strftime('%Y-%m-%d'))
        if self.get_model_state:
            print("Model built!")
        else:
            print("Unable to build the model at this time. Try again.")
        return self.MODEL_DF
    
    
    # ----------------------------- Save the model to CSV -------------------------------------
    def save_model_file(self):
        """
        Save the model to CSV.
    
        Returns
        -------
        None.
    
        """
        try:
            path = self.MODEL_BASE_PATH + "/stock_data_model.csv"
            self.MODEL_DF.to_csv(path)
            print("Saved stock model data: ", path)
        except Exception as e :
           print(f"Could not save stock model data. {e}")
           