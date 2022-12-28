# This script was created by Tyler Pardun (https://github.com/TylerPardun).
#
#                                15 June 2022
#
# This piece of code will create CSV files from previous trading dates to
# get an idea of how accurate this model performs. Backtesting on this algorithm is
# just like running it for the first time as if it were the date. There is no data leakage.
# This code runs in parallel and is best suited for supercomputing since it will take a
# very long time to run on a local machine, depending upon how many trading days you would
# like to go back. Set the parameters at the bottom.
# ==============================================================================

#Supress Warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

from mpi4py import MPI
import pandas as pd
import pandas_market_calendars as mcal

import sys
from glob import glob
import time
import yfinance as yf
import talib
from datetime import datetime,timedelta
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def SMA_buy_sell(SMA_30,SMA_100,df):
    sig = np.zeros((len(df)))*np.nan
    position = False

    for i in range(len(df)):
        if SMA_30[i] > SMA_100[i]:
            if position == False :
                sig[i] = 1
                position = True
            else:
                sig[i] = 0
        elif SMA_30[i] < SMA_100[i]:
            if position == True:
                sig[i] = -1
                position = False
            else:
                sig[i] = 0
        else:
            sig[i] = 0
    return sig

def predictors(df):
    '''
    This function will return a dataframe with all oscillators and indicators inside of it.
    
    INPUTS:
        df -> pandas dataframe containing the data (returned from get_stock_data function)
    RETURNS:
        df -> pandas dataframe of all possible indicators and oscillators
    '''
    close = df['Close'].values
    df['macd'],df['macd_s'],df['macd_h'] = talib.MACD(df['Close'])
    df['RSI'] = talib.RSI(df['Close'])
    df['CCI'] = talib.CCI(df.High,df.Low,df['Close'])
    
    df['spread']=((df['Close']/df['Open'])-1).abs()
    
    #Overlap Studies Functions
    df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(df['Close'])
    df['DEMA'] = talib.DEMA(df['Close'],timeperiod=5)
    df['EMA'] = talib.EMA(df['Close'],timeperiod=5)
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['Close'])
    df['KAMA'] = talib.KAMA(df['Close'],timeperiod=5)
    df['MA'] = talib.MA(df['Close'],timeperiod=5)
    df['MAMA'], df['FAMA'] = talib.MAMA(df['Close'])
    df['MIDPOINT'] = talib.MIDPOINT(df['Close'],timeperiod=5)
    df['MIDPRICE'] = talib.MIDPRICE(df['High'],df['Low'],timeperiod=5)
    df['SAR'] = talib.SAR(df['High'],df['Low'])
    df['SAREXT'] = talib.SAREXT(df['High'],df['Low'])
    df['SMA'] = talib.SMA(df['Close'],timeperiod=5)
    df['T3'] = talib.T3(df['Close'],timeperiod=5)
    df['TEMA'] = talib.TEMA(df['Close'],timeperiod=5)
    df['TRIMA'] = talib.TRIMA(df['Close'],timeperiod=5)
    df['WMA'] = talib.WMA(df['Close'],timeperiod=5)
    

    #Momentum indicators
    df['ADX'] = talib.ADX(df.High,df.Low,df['Close'])
    df['APO'] = talib.APO(df['Close'])
    df['AROON_down'],df['AROON_up'] = talib.AROON(df.High,df.Low)
    df['AROONOSC'] = talib.AROONOSC(df.High,df.Low)
    df['BOP'] = talib.BOP(df.Open,df.High,df.Low,df['Close'])
    df['CMO'] = talib.CMO(df['Close'])
    df['DX'] = talib.DX(df.High,df.Low,df['Close'])
    df['MFI'] = talib.MFI(df.High,df.Low,df['Close'],df.Volume)
    df['MINUS_DI'] = talib.MINUS_DI(df.High,df.Low,df['Close'])
    df['MINUS_DM'] = talib.MINUS_DM(df.High,df.Low)
    df['MOM'] = talib.MOM(df['Close'])
    df['PLUS_DI'] = talib.PLUS_DI(df.High,df.Low,df['Close'])
    df['PLUS_DM'] = talib.PLUS_DM(df.High,df.Low)
    df['PPO'] = talib.PPO(df['Close'])
    df['SLOWK'],df['SLOWD'] = talib.STOCH(df.High,df.Low,df['Close'])
    df['FASTK'],df['FASTD'] = talib.STOCHF(df.High,df.Low,df['Close'])
    df['RSIK'],df['RSID'] = talib.STOCHRSI(df['Close'])
    df['ULTOSC'] = talib.ULTOSC(df.High,df.Low,df['Close'])
    df['WILLR'] = talib.WILLR(df.High,df.Low,df['Close'])

    #Volume indicators
    df['AD'] = talib.AD(df.High,df.Low,df['Close'],df.Volume)
    df['ADOSC'] = talib.ADOSC(df.High,df.Low,df['Close'],df.Volume)
    df['OBV'] = talib.OBV(df['Close'],df.Volume)

    #Volatility indicators
    df['ATR'] = talib.ATR(df.High,df.Low,df['Close'])
    df['NATR'] = talib.NATR(df.High,df.Low,df['Close'])
    df['TRANGE'] = talib.TRANGE(df.High,df.Low,df['Close'])

    df['BETA'] = talib.BETA(df.High,df.Low)
    df['CORREL'] = talib.CORREL(df.High,df.Low)
    df['LINEARREG'] = talib.LINEARREG(df['Close'])
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df['Close'])
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(df['Close'])
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(df['Close'])
    df['STDDEV'] = talib.STDDEV(df['Close'])
    df['TSF'] = talib.TSF(df['Close'])
    df['VAR'] = talib.VAR(df['Close'])
    
    #Cycle indicators
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['Close'])
    df['HT_DCPHASE'] = talib.HT_DCPHASE(df['Close'])
    df['inphase'],df['quadrature'] = talib.HT_PHASOR(df['Close'])
    df['sine'],df['leadsine'] = talib.HT_SINE(df['Close'])
    
    #Price transform functions
    df['AVGPRICE'] = talib.AVGPRICE(df['Open'], df['High'], df['Low'], df['Close'])
    df['MEDPRICE'] = talib.MEDPRICE(df['High'],df['Low'])
    df['TYPPRICE'] = talib.TYPPRICE(df['High'],df['Low'],df['Close'])
    df['WCLPRICE'] = talib.WCLPRICE(df['High'],df['Low'],df['Close'])
    
    '''
    #Pattern recognition stuff
    df['CDL2CROWS'] = talib.CDL2CROWS(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDL3INSIDE'] = talib.CDL3INSIDE(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLBELTHOLD'] = talib.CDLBELTHOLD(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLDOJI'] = talib.CDLDOJI(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLDOJISTAR'] = talib.CDLDOJISTAR(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLENGULFING'] = talib.CDLENGULFING(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLHAMMER'] = talib.CDLHAMMER(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLHARAMI'] = talib.CDLHARAMI(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLHIKKAKE'] = talib.CDLHIKKAKE(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLINNECK'] = talib.CDLINNECK(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLKICKING'] = talib.CDLKICKING(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLLONGLINE'] = talib.CDLLONGLINE(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLMARUBOZU'] = talib.CDLMARUBOZU(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLMATHOLD'] = talib.CDLMATHOLD(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLONNECK'] = talib.CDLONNECK(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLPIERCING'] = talib.CDLPIERCING(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLSHORTLINE'] = talib.CDLSHORTLINE(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLTAKURI'] = talib.CDLTAKURI(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLTHRUSTING'] = talib.CDLTHRUSTING(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLTRISTAR'] = talib.CDLTRISTAR(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(df['Open'],df['High'],df['Low'],df['Close'])
    df['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(df['Open'],df['High'],df['Low'],df['Close'])
    '''
    
    #MACD
    macd,macd_s,macd_h = talib.MACD(df['Close'])
    macd_sig = np.zeros((len(df)))*np.nan
    for i in range(len(df)):
        if (macd[i] < macd_s[i]) & (np.abs(macd_h[i])<=0.05):
            macd_sig[i] = 1
        elif (macd[i] > macd_s[i]) & (np.abs(macd_h[i])<=0.05):
            macd_sig[i] = -1
        else:
            macd_sig[i] = 0
    df['macd_sig'] = macd_sig

    #RSI
    rsi = talib.RSI(df['Close'])
    rsi_sig = np.zeros((len(df)))*np.nan
    for i in range(len(df)):
        if rsi[i]<=35:
            rsi_sig[i] = 1
        elif rsi[i]>=65:
            rsi_sig[i] = -1
        else:
            rsi_sig[i] = 0
    df['rsi_sig'] = rsi_sig

    #CCI
    cci = talib.CCI(df.High,df.Low,close)
    CCI_signal = np.zeros((len(df)))

    icross = np.where(np.diff(np.signbit(cci)))[0] #Gives indices of values prior to the cross
    for i,val in enumerate(cci[icross]):
        if val < 0:
            CCI_signal[icross[i]] = 1
        else:
            CCI_signal[icross[i]] = -1
    df['CCI_sig'] = CCI_signal

    #DEMA
    dema = talib.DEMA(df['Close'])
    dem = np.zeros((len(df)))*np.nan
    for i in range(len(df)):
        if close[i]>dema[i]:
            dem[i] = 1
        elif close[i]<=dema[i]:
            dem[i] = -1
        else:
            dem[i] = 0
    df['DEMA_sig'] = dem

    #EMA
    df['SMA_sig'] = SMA_buy_sell(talib.SMA(close,30),talib.SMA(close,100),df)
    
    #MAMA FAMA
    mama,fama = talib.MAMA(close)
    diff = mama-close
    icross = np.where(np.diff(np.signbit(diff)))[0]
    sig = np.zeros((len(df)))
    for i,val in enumerate(diff[icross]):
        if val < 0:
            sig[icross[i]] = -1
        else:
            sig[icross[i]] = 1
    df['MAMA_sig'] = sig


    #SAR (Stop and Reverse) -> from negative: SAR approaches close from below close: SAR approaches from above close
    sar = talib.SAR(df['High'],df['Low'])
    diff = sar-close
    icross = np.where(np.diff(np.signbit(diff)))[0]
    sig = np.zeros((len(df)))
    for i,val in enumerate(diff[icross]):
        if val < 0:
            sig[icross[i]] = 1
        else:
            sig[icross[i]] = -1
    df['SAR_sig'] = sig


    #T3
    t3,t8 = talib.T3(df['Close'],timeperiod=3),talib.T3(df['Close'],timeperiod=8)
    diff = t8-t3
    icross = np.where(np.diff(np.signbit(diff)))[0]
    sig = np.zeros((len(df)))
    for i,val in enumerate(diff[icross]):
        if val > 0:
            sig[icross[i]] = 1
        else:
            sig[icross[i]] = -1
    df['T3_sig'] = sig
    
    
    #Make a buy,hold,sell signal using the CCI. Crosses 0 from negative, buy / crosses zero from positive, sell
    CCI_signal = np.zeros((len(df)))

    icross = np.where(np.diff(np.signbit(df['CCI'])))[0] #Gives indices of values prior to the cross
    for i,val in enumerate(df['CCI'].values[icross]):
        if val < 0:
            CCI_signal[icross[i]] = 1
        else:
            CCI_signal[icross[i]] = -1
    df['CCI_signal'] = CCI_signal

    df['signal'] = 0.0
    df.loc[(df['RSI'] <= 43) & (df['CCI_signal'] >= 1) & (df['macd'] > df['macd_s']), 'signal'] = 1
    df.loc[(df['RSI'] >= 70) & (df['CCI_signal'] <= -1) & (df['macd'] < df['macd_s']), 'signal'] = -1
    
    return df

def get_stock_data(day,ticker_vals,lag_val,forecast_days,tick_list=False):
    '''
    This function will take in a date and return the dataset as well as a validation dataset if the validation date
    has already ocurred. If not, the validation dataset will be nan.
    
    INPUTS:
        day -> date of initalization [str. (yyyy-mm-dd)] *This is prior to market open on the day specified*
        ticker_vals -> could be a list or string of ticker symbol(s).
        lag_val -> integer corresponding to number of years of data in the dataset
        forecast_days -> integer corresponding to the number of *TRADING DAYS* to forecast into the future
        tick_list -> bool if the input ticker symbol is a list (True) or a string (False). Fefault is set to False.
    
    RETURNS:
        df -> Pandas dataframe containing the data needed
        d_true -> the validation dataframe (if existing)
        
    '''
    
    #Get the start and end dates of the total weeks wanting to predict
    cal = mcal.get_calendar('NYSE').schedule(start_date=datetime.strptime(day,'%Y-%m-%d')-timedelta(days=(365*(lag_val*2))), end_date=datetime.strptime(day,'%Y-%m-%d')+timedelta(days=forecast_days+20))
    date_list = pd.to_datetime(mcal.date_range(cal, frequency='1D'))
    date_list = np.array([datetime(val.year,val.month,val.day) for val in date_list])

    #Get the pred date for the validation dataset
    pred_date = date_list[np.where(date_list==datetime.strptime(day,'%Y-%m-%d'))[0][0]+forecast_days-1]

    #Index the trading dates to get all the dates we need
    #date_list = date_list[:np.where(date_list==datetime.strptime(day,'%Y-%m-%d'))[0][0]+forecast_days+1]
    start,end = date_list[0],date_list[-1]

    #Download all of the data in bulk here
    if tick_list:
        df = yf.download(ticker_vals.tolist(),start=start,end=end,interval='1d',progress=True)
    else:
        df = yf.download(ticker_vals,start=start,end=end,interval='1d',progress=False)

    #Get the 10-year treasury data
    #df['tres'] = yf.download('^TNX',start=start,end=end,interval='1d',progress=False)['Close']
    df.reset_index(inplace=True)
    
    #Get the actual observation
    try:
        if tick_list:
            d_true = df.loc[np.where(df['Date']>=pred_date)[0]]
            d_true = d_true.swaplevel(0,1,1)
            d_true.reset_index(inplace=True,drop=True)
            d_true = d_true.loc[:4]
            
            df = df.loc[np.where(df['Date']<pred_date)[0]]
            df = df.swaplevel(0,1,1)
        else:
            d_true = df.loc[np.where(df['Date']>=pred_date)[0]]
            d_true.reset_index(inplace=True,drop=True)
            d_true = d_true.loc[:9] #one week of trading to test (inclusive)
            df = df.loc[np.where(df['Date']<pred_date)[0]]

    except IndexError: #If current data
        d_true = np.nan
        
    #Get the full dataset
    d_dates = np.array([datetime(val.year,val.month,val.day) for val in pd.to_datetime(df['Date'].values)])
    try:
        df = df.loc[:np.where(d_dates==date_list[-forecast_days-1])[0][0]-1]
    except IndexError: #Usually ocurrs when running real-time
        pass
    
    #Organize data if needed
    if tick_list:
        df = df.swaplevel(0,1,1)
        
    return df,d_true

def main(t,ticker_vals,day,lag_val,forecast_days,outpath,real_time=False):
    tval = ticker_vals[t]
    time.sleep(0.5)
    
    df,d_true = get_stock_data(day,tval,lag_val,forecast_days,tick_list=False)
    try:
        d = predictors(df)
    except:
        return

    #Remove the nan values from lagged oscillators
    inan = np.where(np.isnan(d))[0][-1]
    d = d.loc[inan+1:]
    d.reset_index(inplace=True)
    d.drop(columns='index',inplace=True)

    #Do some machine learning here
    d.set_index('Date',inplace=True)

    X = d.drop(columns = ['signal'])
    y = d['signal'].values
    yr = (d['Close'] - d['Open']) / d['Open']
    try:
        X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[1:], random_state = 0) #Shift by one for prediction
        X_trainr, X_testr, y_trainr, y_testr = train_test_split(X[:-1], yr[1:], random_state = 0) #Shift by one for prediction
    except ValueError:
        return

    #Scaling the X_train and X_test data
    try:
        scaler = StandardScaler()
        X_scaler = scaler.fit(X_train)
        X_train_scaled = X_scaler.transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        X_train_scaledr = X_scaler.transform(X_trainr)
        X_test_scaledr = X_scaler.transform(X_testr)
    except ValueError:
        return

    # Use RandomOverSampler to resample the dataset using random_state=1
    ros = RandomOverSampler(random_state=1)
    try:
        X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
    except ValueError:
        return

    '''
    day_num = 7
    dayx_train = np.zeros((X_resampled.shape[0]-day_num,X_resampled.shape[1]*day_num))*np.nan
    dayy_train = np.zeros((X_resampled.shape[0]-day_num))*np.nan

    dayx_test = np.zeros((X_test_scaled.shape[0]-day_num,X_test_scaled.shape[1]*day_num))*np.nan
    dayy_test = np.zeros((X_test_scaled.shape[0]-day_num))*np.nan

    for i in range(X_resampled.shape[0]-day_num):
        dayx_train[i,:] = X_resampled[i:i+day_num].ravel()
        dayy_train[i] = y_resampled[i+day_num]

    for i in range(X_test_scaled.shape[0]-day_num):
        dayx_test[i,:] = X_test_scaled[i:i+day_num].ravel()
        dayy_test[i] = y_test[i+day_num]
        
    #Create the forecast models
    svc_model = SVC().fit(dayx_train, dayy_train)
    svc_test_report = accuracy_score(dayy_test, svc_model.predict(dayx_test))

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(dayx_train, dayy_train)
    lr_test_report = accuracy_score(dayy_test, lr.predict(dayx_test))
    
    rc = RandomForestClassifier(random_state=20).fit(dayx_train, dayy_train)
    rc_acc = accuracy_score(dayy_test, rc.predict(dayx_test))

    #Make the actual prediction
    x_pred = d.drop(columns='signal')
    x_pred = X_scaler.transform(x_pred)
    x_pred = x_pred[-day_num:].ravel().reshape(1,-1)
    lrpred,svcpred,rcpred = lr.predict(x_pred)[0],svc_model.predict(x_pred)[0],rc.predict(x_pred)[0]
    '''
    
    #Creating the SVC model instance and fitting to the resampled data
    svc_model = SVC()
    svc_model = svc_model.fit(X_resampled, y_resampled)
    svc_test_report = accuracy_score(y_test, svc_model.predict(X_test_scaled))

    #Creating a LogisticRegression model for comparison
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr = lr.fit(X_resampled, y_resampled)
    lr_test_report = accuracy_score(y_test, lr.predict(X_test_scaled))

    #Make the final prediction
    x_pred = d.drop(columns='signal')
    x_pred = X_scaler.transform(x_pred.values[-1].reshape(1,-1))
    lrpred,svcpred = lr.predict(x_pred)[0],svc_model.predict(x_pred)[0]

    #Print and organize the output
    if (lrpred==1) & (svcpred==1):
    
        #Do some regression
        rfr = RandomForestRegressor(random_state=20).fit(X_trainr, y_trainr)
        rfr_pred = rfr.predict(x_pred)[0]

        #Make a regression onto the prediction
        predr = rfr.predict(X_testr)
        mod = sm.OLS(sm.add_constant(predr),y_testr).fit()
        rfr_pred_reg = mod.predict(rfr_pred)[0][-1]

        #Get a target on the final price
        final_price = d['Close'].values[-1] + (d['Close'].values[-1]*rfr_pred_reg)
    
            
        #Grab all the data to make a CSV
        try:
            high_price = np.nanmax(d_true['Close'].values)
            ihigh = np.where(d_true['Close'].values==high_price)[0][0]
            pct_change = ((high_price-d.iloc[-1]['Close'])/d.iloc[-1]['Close'])*100
            pct_change_1 = ((d_true['Close'].values[0]-d.iloc[-1]['Close'])/d.iloc[-1]['Close'])*100
            obs = True if pct_change>0 else False
        except ValueError:
            pct_change,ihigh,obs = np.nan,np.nan,np.nan
        
        #Get the data ready for the outfile
        data_dict = {'symbol':tval,'price_init':d['Close'].values[-1], 'pred_price':final_price, 'pred_pct':rfr_pred_reg*100, 'obs_price':d_true['Close'].values[0], 'obs_pct':pct_change_1, 'highest_price':d['Close'].values[-1]+(d['Close'].values[-1]*(pct_change/100)), 'highest_pct':pct_change, 'highest_pct_day':ihigh, 'bool':obs}
        
        #Append the price history
        obs_cols0 = ['Open{}'.format(i) for i in range(len(d_true))]
        obs_cols1 = ['High{}'.format(i) for i in range(len(d_true))]
        obs_cols2 = ['Low{}'.format(i) for i in range(len(d_true))]
        obs_cols3 = ['Close{}'.format(i) for i in range(len(d_true))]

        obs_cols = [[obs_cols0[i],obs_cols1[i],obs_cols2[i],obs_cols3[i]] for i in range(len(d_true))]
        obs_cols = [item for sublist in obs_cols for item in sublist]

        d_true_flat = d_true[['Open','High','Low','Close']].values.ravel()
        for i,val in enumerate(obs_cols):
            data_dict[val] = d_true_flat[i]
        
        outloc = outpath+'psa_{}_buy.csv'.format(datetime.strptime(day,'%Y-%m-%d').strftime('%Y%m%d'))
        
        #Append or write the file
        try:
            foo = pd.read_csv(outloc)
            #pd.DataFrame(data_dict,index=[t]).to_csv(outloc,mode='a',header=False)
        except FileNotFoundError:
            #pd.DataFrame(data_dict,index=[t]).to_csv(outloc,mode='w',header=True)
            pass
        return
        
        #Print and organize the output
    if (lrpred==-1) & (svcpred==-1):
    
        #Do some regression
        rfr = RandomForestRegressor(random_state=20).fit(X_trainr, y_trainr)
        rfr_pred = rfr.predict(x_pred)[0]

        #Make a regression onto the prediction
        predr = rfr.predict(X_testr)
        mod = sm.OLS(sm.add_constant(predr),y_testr).fit()
        rfr_pred_reg = mod.predict(rfr_pred)[0][-1]

        #Get a target on the final price
        final_price = d['Close'].values[-1] + (d['Close'].values[-1]*rfr_pred_reg)
    
            
        #Grab all the data to make a CSV
        try:
            high_price = np.nanmin(d_true['Close'].values)
            ihigh = np.where(d_true['Close'].values==high_price)[0][0]
            pct_change = ((high_price-d.iloc[-1]['Close'])/d.iloc[-1]['Close'])*100
            pct_change_1 = ((d_true['Close'].values[0]-d.iloc[-1]['Close'])/d.iloc[-1]['Close'])*100
            obs = True if pct_change<0 else False
        except ValueError:
            pct_change,ihigh,obs = np.nan,np.nan,np.nan
        
        #Get the data ready for the outfile
        data_dict = {'symbol':tval,'price_init':d['Close'].values[-1], 'pred_price':final_price, 'pred_pct':rfr_pred_reg*100, 'obs_price':d_true['Close'].values[0], 'obs_pct':pct_change_1, 'highest_price':d['Close'].values[-1]+(d['Close'].values[-1]*(pct_change/100)), 'highest_pct':pct_change, 'highest_pct_day':ihigh, 'bool':obs}
        
        #Append the price history
        obs_cols0 = ['Open{}'.format(i) for i in range(len(d_true))]
        obs_cols1 = ['High{}'.format(i) for i in range(len(d_true))]
        obs_cols2 = ['Low{}'.format(i) for i in range(len(d_true))]
        obs_cols3 = ['Close{}'.format(i) for i in range(len(d_true))]

        obs_cols = [[obs_cols0[i],obs_cols1[i],obs_cols2[i],obs_cols3[i]] for i in range(len(d_true))]
        obs_cols = [item for sublist in obs_cols for item in sublist]

        d_true_flat = d_true[['Open','High','Low','Close']].values.ravel()
        for i,val in enumerate(obs_cols):
            data_dict[val] = d_true_flat[i]
        
        outloc = outpath+'psa_{}_sell.csv'.format(datetime.strptime(day,'%Y-%m-%d').strftime('%Y%m%d'))
        
        #Append or write the file
        try:
            foo = pd.read_csv(outloc)
            #pd.DataFrame(data_dict,index=[t]).to_csv(outloc,mode='a',header=False)
        except FileNotFoundError:
            #pd.DataFrame(data_dict,index=[t]).to_csv(outloc,mode='w',header=True)
            pass
        return

if __name__ == '__main__':

    #########################################################
    # Set the parameter space
    #########################################################

    forecast_days = 1 #Trading days to predict into the future
    lag_val = 20 #years to go back to train
    ticker_vals = pd.read_csv('Symbols.csv')['symbol'].values
    outpath = 'output/'

    #List of dates to test
    cal = mcal.get_calendar('NYSE').schedule(start_date='2009-03-12',end_date='2022-11-21')
    days = pd.to_datetime(mcal.date_range(cal, frequency='1D'))
    days = np.array([datetime(val.year,val.month,val.day).strftime('%Y-%m-%d') for val in days])
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    perrank = int(len(days)/size)
    a = 1
    for p in range(int(a + rank*perrank), int(a + (rank+1)*perrank)):
        pl = Pool(10)
        data = list(pl.imap(partial(main, ticker_vals=ticker_vals, day=days[p], lag_val=lag_val, forecast_days=forecast_days, outpath=outpath), range(len(ticker_vals))))
        pl.close()
