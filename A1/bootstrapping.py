#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 13:49:46 2019

@author: zikunchen
"""

import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

# CONSTANTS
PAYMENT_PERIOD = 6
PERIOD = 5 # in years
DAYS_IN_YEAR = 365
FIVE_YEARS = [i for i in range(1,6)]

class bootstrap():   
    
    def __init__(self, date):
        self.date = date
        
    def preprocess(raw_bonds, T0):
        
        bonds = raw_bonds
        
        # Format Maturity
        bonds['Eff. Maturity'] = pd.to_datetime(bonds['Eff. Maturity'])
    
        # Today's Date
        bonds['T0'] = pd.to_datetime(T0)
        
        # Time to Maturity In Month
        bonds['T'] = round((bonds['Eff. Maturity'] - bonds['T0'])/np.timedelta64(1, 'M'))
    
        # Filter desired bonds for rate calculations
        start_maturity = int(bonds['T'][1])
        threshold = start_maturity + 12 * PERIOD
        bonds = bonds[bonds['T'] <= threshold]
        desired_maturities =  [maturity for maturity in \
                               range(start_maturity, threshold + 1, PAYMENT_PERIOD)]
        bonds = bonds[bonds['T'].isin(desired_maturities)]
    
        # Calculate Clean Price
        bonds['Last_Coupon_T'] = np.repeat((bonds.loc[[1]]['Eff. Maturity'] - \
                 pd.DateOffset(months=PAYMENT_PERIOD)).values, bonds.shape[0])
        bonds['Accured_Interest'] = ((bonds['T0'] - bonds['Last_Coupon_T']) \
                /np.timedelta64(1, 'D'))/ DAYS_IN_YEAR * bonds['Coupon']
        bonds['Price'] = bonds['Price'] + bonds['Accured_Interest']
        
        # Delete irrelevant columns
        bonds = bonds.drop(columns="T0")
        bonds = bonds.drop(columns="Eff. Maturity")
        bonds = bonds.drop(columns="Last_Coupon_T")
        bonds = bonds.drop(columns="Accured_Interest")
        
        # Adjust Coupon to reflect semiannual payment
        bonds['Coupon'] = bonds['Coupon']/2
        bonds['T'] = bonds['T']/12
        
        return bonds
    
    
    def ytm(bonds):
        """Return 5 year ytm and maturity from a set of bonds"""        
        ytm = bonds['Yield'].values
        maturity = bonds['T'].values
        return ytm, maturity
    
    
    def spot_rates(bonds):
        """Return 5 year zero rates and maturity from a set of bonds"""
        principle = 100
        
        coupon = bonds['Coupon'].values
        price = bonds['Price'].values
        maturity = bonds['T'].values
        
        n = coupon.shape[0]
        rates = [0]*n
        
        for m in range(n):
            payments = 0
            if m > 0:
                for i in range(m-1):
                    payments += coupon[m] * np.exp(-rates[i] * maturity[i])
            rate = - np.log((price[m] - payments)/(principle + coupon[m])) / maturity[m]
            rates[m] = rate
            
        spl = make_interp_spline(maturity, rates, k=3) #BSpline object
        return spl(FIVE_YEARS), np.array(FIVE_YEARS)    
    
    def one_year_forward_rates(spot_rates):
        """Generate 1 year forward rates from zero rates from 2 to 5 years out."""
        m = len(spot_rates)
        forward_rates = [0] * (m-1)
        for i in range(1, m):
            riti = spot_rates[i]*FIVE_YEARS[i]
            r0t0 = spot_rates[0]*FIVE_YEARS[0]
            forward_rates[i-1] = (riti - r0t0)/(FIVE_YEARS[i]-FIVE_YEARS[0])
        return np.array(forward_rates)
    
    def cov(rates):
        """Generate covariance matrix of time series"""
        df = pd.DataFrame.from_dict(rates)
        n = df.shape[0] # number of assets 
        m = df.shape[1] # number of features
        
        df.columns = np.arange(1, m+1)
        df.index = np.arange(1, n+1)
        
        df = df.shift(periods=1, axis='columns') / df
        df = df.drop(columns = 1)
        
        cov_matrix = np.cov(df)

        return cov_matrix, np.linalg.eig(cov_matrix)
        


        