#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:24:28 2019

@author: zikunchen
"""

from bootstrapping import bootstrap as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline



FILENAME = "raw_bond_data.xls"

WEEK_ONE_START = '2019-01-07'
WEEK_TWO_START = '2019-01-14'

JANFIRST = '2019-01-01'
FEBFIRST = '2019-02-01'

FIVE_YEARS = [i for i in range(1,6)]


if __name__ == "__main__":
    
# =============================================================================
#     Pre-Processing
# =============================================================================
    
    bonds_dict = {}
    
    for i in range(10):
        if i < 5:
            date = str((pd.to_datetime(WEEK_ONE_START) + pd.Timedelta(i, unit='d')).date())
        else:
            date = str((pd.to_datetime(WEEK_TWO_START) + pd.Timedelta(i-5, unit='d')).date())
        
        raw_bonds = pd.read_excel(FILENAME, sheet_name = date)
        
        # Round today's date to the start of next month if necessary for easier calculation
        if int(date[-2:]) < 15:
            bonds = bs.preprocess(raw_bonds, JANFIRST)
        else:
            bonds = bs.preprocess(raw_bonds, FEBFIRST)
        
        bonds_dict[date] = bonds
        
    
# =============================================================================
#     Yield Curve
# =============================================================================
    
    ### all ytm
    yields = {}
    legend = [0]*20
    for i in range(10):
        if i < 5:
            date = str((pd.to_datetime(WEEK_ONE_START) + pd.Timedelta(i, unit='d')).date())
        else:
            date = str((pd.to_datetime(WEEK_TWO_START) + pd.Timedelta(i-5, unit='d')).date())

        bonds = bonds_dict[date]
        ytm, maturity = bs.ytm(bonds)
        legend[2*i] = date
        
        xnew = np.linspace(maturity.min(),maturity.max(),300)
        spl = make_interp_spline(maturity, ytm, k=3) #BSpline object
        ytm_smooth = spl(xnew)
        plt.plot(xnew, ytm_smooth, '-', linewidth=2) 
        
        ytm = spl(FIVE_YEARS)
        yields[date] = ytm
        plt.plot(FIVE_YEARS, ytm, 'ro', markersize=2, label='_nolegend_')

    plt.title('Yield To Maturity')
    plt.ylabel('Yield (%)')
    plt.xlabel('Year')
    plt.legend(legend[::2], prop={'size': 5})
    plt.show()
    
# =============================================================================
#     Spot Curve
# =============================================================================

    spot_rates = {}
    legend = [0]*20
    for i in range(10):
        if i < 5:
            date = str((pd.to_datetime(WEEK_ONE_START) + pd.Timedelta(i, unit='d')).date())
        else:
            date = str((pd.to_datetime(WEEK_TWO_START) + pd.Timedelta(i-5, unit='d')).date())

        bonds = bonds_dict[date]
        spot_rate, maturity = bs.spot_rates(bonds)
        legend[2*i] = date

        xnew = np.linspace(maturity.min(),maturity.max(),300)
        spl_spot = make_interp_spline(maturity, spot_rate, k=3) #BSpline object
        spot_rate_smooth = spl_spot(xnew)
        plt.plot(xnew, spot_rate_smooth, '-', linewidth=0.5)

        spot_rate = spl_spot(FIVE_YEARS)
        plt.plot(FIVE_YEARS, spot_rate, 'ro', markersize=2, label='_nolegend_')
        spot_rates[date] = spot_rate
    
    plt.title('Spot Curve')
    plt.ylabel('Spot Rate')
    plt.xlabel('Year')
    plt.legend(legend[::2], prop={'size': 5}, loc = 'lower right')
    plt.show()

    
# =============================================================================
#     Forward Rates
# =============================================================================
    
    forward_rates = {}
    legend = [0]*10
    for i in range(10):
        if i < 5:
            date = str((pd.to_datetime(WEEK_ONE_START) + pd.Timedelta(i, unit='d')).date())
        else:
            date = str((pd.to_datetime(WEEK_TWO_START) + pd.Timedelta(i-5, unit='d')).date())

        bonds = bonds_dict[date]
        spot_rates, maturity = bs.spot_rates(bonds)
        forward_rate = bs.one_year_forward_rates(spot_rates)
        forward_rates[date] = forward_rate
        legend[i] = date
        plt.plot(FIVE_YEARS[1:], forward_rate*100, 'o-', linewidth=0.5, markersize=2) 
    
    plt.title('Forward Curve')
    plt.ylabel('Forward Rate')
    plt.xlabel('Year')
    plt.legend(legend, prop={'size': 5})
    plt.show()


# =============================================================================
#     Covariance Matrix
# =============================================================================    
    
    ## Yield Rates
    
    cov_yield, (eigenvalue, eigenvectors) = bs.cov(yields)
    
    print('Covariance Matrix: \n')
    print(cov_yield)
    
    df_ev_yield = pd.DataFrame(eigenvectors)
    df_ev_yield.columns  = eigenvalue
    
    fig = plt.figure()
    
    plt.plot(FIVE_YEARS, eigenvalue/sum(eigenvalue)*100, 'o-', linewidth=0.5, markersize=2) 
    
    plt.title('Yield Curve Precentage Variance Explained')
    plt.ylabel('Precentage')
    plt.xlabel('Factor Number')
    plt.show()
        
    print('Eigenvalues: \n')
    print(eigenvalue)
    print('Eigenvectors (By Columns): \n')
    print(eigenvectors)
    
    PC = pd.DataFrame(eigenvectors)
    PC.columns  = FIVE_YEARS
    PC.index = range(1,6)
    PC.iloc[:, :3].plot()
    
    ## Forward Rates
    
    cov_forward, (eigenvalue, eigenvectors) = bs.cov(forward_rates)
    
    print('Covariance Matrix: \n')
    print(cov_forward)
    
    df_ev_forward = pd.DataFrame(eigenvectors)
    df_ev_forward.columns  = eigenvalue
    
    fig = plt.figure()
    
    plt.plot(FIVE_YEARS[1:], eigenvalue/sum(eigenvalue)*100, 'o-', linewidth=0.5, markersize=2) 
    
    plt.title('Forward Curve Percentage Variance Explained')
    plt.ylabel('Percentage')
    plt.xlabel('Factor Number')    
    plt.show()
    
    print('Eigenvalues: \n')
    print(eigenvalue)
    print('Eigenvectors (By Columns): \n')
    print(eigenvectors)
