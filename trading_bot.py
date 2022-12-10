#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

#Supress Warnings
import warnings
warnings.filterwarnings("ignore")

'''This script represents a trading bot that uses the 10 days of trading within the backtesting output. What is currently represented in the function is buying shares of a random stock from the list of output and investing 20% of the bankroll into it. When a profit is realized, it will sell and use that to re-invest throuhghout the year of 2021. You can change the year to which every you would like that is represented in the backtesting dataset.'''


def profit(n,files,cash):
    #Create the array and fill with starting cash value
    cash_flow = np.zeros((len(files)+1))
    cash_flow[0] = cash

    for c,f in enumerate(files):
        d = pd.read_csv(f)

        #Pick stock at random and sell when we see any kind of profit
        idx = np.random.randint(len(d))
        pct = (d[['Close{}'.format(val) for val in np.arange(0,10)]].values[idx] - d['Open0'].values[idx]) / d['Open0'].values[idx]
        try:
            pct = pct[np.where(pct>0)[0][0]]
        except IndexError:
            pct = np.nanmax(pct)
            
        #Invest only 20% of bankroll
        cash_flow[c+1] = cash_flow[c] + ((cash_flow[c]*0.2)*pct)

    return pct


if __name__ == '__main__':
    #Get output from the year 2021
    starting_cash = 500
    year = 2021
    files = sorted(glob('/Users/admin/Desktop/psa/output/psa_{}**buy*'.format(year)))

    #Bootstrap all possbile outcomes when trading SHARES of stock
    n = 1000
    p = Pool(20)
    cash_flow_bs = np.array(list(tqdm(p.imap(partial(profit,files=files,cash=starting_cash),range(n)),total=n)))
    p.close()
    
    #plot a histogram -> buying shares only
    plt.figure(figsize=(6,6))
    ax = plt.subplot(1,1,1)

    #Plot the ending cash for 2021 so far
    plt.hist(cash_flow_bs[:,-1],bins=50,color='blue')
    p05,p95 = np.nanpercentile(cash_flow_bs[:,-1],[5,95])

    plt.axvline(x=p05,color='k')
    plt.axvline(x=p95,color='k')

    plt.show()
    plt.close()

    print('5th percentile: ${:.2f}'.format(p05))
    print('50th percentile: ${:.2f}'.format(np.nanmedian(cash_flow_bs[:,-1])))
    print('95th percentile: ${:.2f}'.format(p95))
