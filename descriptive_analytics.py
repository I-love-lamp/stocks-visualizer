# -*- coding: utf-8 -*-
'''
@author: Mia
'''
import numpy as np

def AveragePrice(prices):
    return np.mean(prices) 

def RangePrice(prices):
    return np.ptp(prices) 

def quartileprice(numbers):
    length = len(numbers) 
    numbers.sort()
    index1 = int((length+1)/4)-1 
    index2= int((length+1)/2)-1 
    index3 = int(3*(length+1)/4)-1
    q1 = numbers[index1] 
    q2 = numbers[index2] 
    q3= numbers[index3]
    return q1, q2, q3

def StdPrices(prices):
    return np.std(prices)

def cov(prices):
    return np.std(prices)/np.mean(prices)
    
def get_day_price(self, dt):
        price = self.STOCK_DF.loc[self.STOCK_DF.index == dt].Close[0]
        return price # add error handling
    
        