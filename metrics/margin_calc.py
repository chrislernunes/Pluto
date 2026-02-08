import os, glob, datetime, time, sys, traceback
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import direct_redis
from scipy.optimize import curve_fit
import concurrent.futures
import multiprocessing, asyncio, threading
import operator
from itertools import repeat
from copy import copy
import inspect
import re
import math
from strategies import *
from utils import *

from engine.ems import DataInterface, lot_size_dict

r = direct_redis.DirectRedis(host='localhost', port=6379, db=0)
di = DataInterface()
lists=di.get_all_ticks_by_symbol('NIFTYSPOT')
lists['timestamp']=pd.to_datetime(lists['timestamp'])
ACTIVE_DATES = list(lists['timestamp'].dt.date.unique())
ACTIVE_DATES = [x for x in ACTIVE_DATES if type(x) == datetime.date and pd.isna(x) == False and type(x) != pd.NaT]
ACTIVE_DATES.sort()

def get_unit_margin(symbol):
    """
    This function takes trading symbols or strategy uids as input. 
    Eg - NIFTY, BANKNIFTY-I, FINNIFTY24011622550CE, al_99_x0_3_1_NIFTY_PCT_0.001_10_0.55_0.99_5_False_True
    """
    if '_' in symbol:
        if 'BANKNIFTY' in symbol: return 50000
        elif 'FINNIFTY' in symbol: return 60000
        elif 'MIDCPNIFTY' in symbol: return 60000
        elif '_NIFTY' in symbol: return 70000
        elif 'SENSEX' in symbol: return 50000
        
    else:
        if 'BANKNIFTY' in symbol: return 50000
        elif 'FINNIFTY' in symbol: return 60000
        elif 'MIDCPNIFTY' in symbol: return 60000
        elif 'NIFTY' in symbol: return 70000
        elif 'SENSEX' in symbol: return 50000


#####################################################################################################################

def strike_from_symbol(tsym):

    if tsym[0] == "B":
        return int(tsym[15:-2])
    elif tsym[0] == 'N':
        return int(tsym[11:-2])
    elif tsym[0] == 'F':
        return int(tsym[14:-2])
    elif tsym[0] == 'M':
        op = tsym[-2:]
        strike = tsym.strip('MIDCPNIFTY').strip(op)[6:]
        return int(strike)
    elif tsym[0] == 'S':
        return int(tsym[12:-2]) 
    else:
        return np.nan

def parse_strike_from_symbol(symbol):
    # ...
    if symbol.startswith('BANKNIFTY'):
        underlying = 'BANKNIFTY'
    elif symbol.startswith('NIFTY'):
        underlying = 'NIFTY'
    elif symbol.startswith('FINNIFTY'):
        underlying = 'FINNIFTY'
    elif symbol.startswith('MIDCPNIFTY'):
        underlying = 'MIDCPNIFTY'
    elif symbol.startswith('SENSEX'):
        underlying = 'SENSEX'
    elif symbol.startswith('BANKEX'):
        underlying = 'BANKEX'
    else:
        raise ValueError(f"Unknown symbol: {symbol}")
    # for ticker in stock_tickers:
    #     if symbol.startswith(ticker) : underlying =  ticker
    # if underlying == '':
    #     print(f'WARNING: {symbol} underlying not found')
    #     return np.nan
    # ...
    return int(float(symbol.strip(underlying).strip('CE').strip('PE')[6:]))


def parse_date_from_symbol(symbol):
    if symbol.startswith('BANKNIFTY'):
        underlying = 'BANKNIFTY'
    elif symbol.startswith('NIFTY'):
        underlying = 'NIFTY'
    elif symbol.startswith('FINNIFTY'):
        underlying = 'FINNIFTY'
    elif symbol.startswith('MIDCPNIFTY'):
        underlying = 'MIDCPNIFTY'
    elif symbol.startswith('SENSEX'):
        underlying = 'SENSEX'
    elif symbol.startswith('BANKEX'):
        underlying = 'BANKEX'
    for ticker in stock_tickers:
        if symbol.startswith(ticker) : underlying =  ticker

    return datetime.datetime.strptime(symbol.strip(underlying).strip('CE').strip('PE')[:6], '%y%m%d').date()

def single_strike_payoff_cal(strike, option_type, option_price, var_range, qty):
    option_price_neg_qty = -option_price * qty  # Precompute to avoid redundant calculations
    payoff_array = [
        int((var_strike - strike) * qty if var_strike >= strike else option_price_neg_qty)
        if option_type == "CE"
        else int((strike - var_strike) * qty if var_strike < strike else option_price_neg_qty)
        for var_strike in var_range
    ]
    return payoff_array


# def single_strike_payoff_cal( strike, option_type, option_price, var_range, qty):
#     tic = time.perf_counter()
    
#     payoff_array = []
#     if option_type == "CE":
#         for var_strike in var_range:
#             if var_strike < strike:
#                 payoff_array.append(int((option_price)*(-1)*(qty)))
#             else:
#                 payoff_array.append(int((var_strike-strike) * qty))
#     elif option_type == "PE":
#         for var_strike in var_range:
#             if var_strike < strike:
#                 payoff_array.append(int((strike-var_strike) * qty))
#             else:
#                 payoff_array.append(int((option_price)*(-1)*(qty)))

#     toc = time.perf_counter()
#     # print(f"single_strike_payoff_cal took {toc - tic:0.4f} seconds")

#     return payoff_array

def get_day_margin(date, temp_tb_):

    try:

        var_per = 0.09
        marign_over_time = {}
        var_over_time = {}
        di = DataInterface()
        
        for index in temp_tb_['symbol'].apply(lambda x : x[0]).unique():
            
            temp_tb = temp_tb_[(temp_tb_['symbol'].str[0] == index) & (temp_tb_['timestamp'].apply(lambda x: x.date()) == date)]

            if index == 'B': mysymbol = 'BANKNIFTYSPOT'; strike_diff = 100 ; lot_size = lot_size_dict['BANKNIFTY']
            elif index == 'N': mysymbol = 'NIFTYSPOT'; strike_diff = 50 ; lot_size = lot_size_dict['NIFTY']
            elif index == 'F': mysymbol = 'FINNIFTYSPOT'; strike_diff = 50 ; lot_size = lot_size_dict['FINNIFTY']
            elif index == 'M' : mysymbol = 'MIDCPNIFTYSPOT'; strike_diff = 25 ; lot_size = lot_size_dict['MIDCPNIFTY']
            elif index == 'S' : mysymbol = 'SENSEXSPOT'; strike_diff = 100 ; lot_size = lot_size_dict['SENSEX']

            for timestamp in temp_tb['timestamp'].unique():
                try:
                    print(timestamp, sep='',end="\r",flush=True)
                    spot_price = di.get_tick(timestamp, mysymbol)['c']

                    min_var_price = spot_price*(1-var_per)
                    max_var_price = spot_price*(1+var_per)
                    atm_strike = round(spot_price/strike_diff)*strike_diff
                    min_var_strike = round(min_var_price/strike_diff)*strike_diff
                    max_var_strike = round(max_var_price/strike_diff)*strike_diff
                    var_range = np.arange(min_var_strike, max_var_strike, strike_diff)

                    try:
                        work_df = temp_tb[temp_tb['timestamp'] <= timestamp].copy()
                        temp = work_df.groupby('symbol')[['qty_dir', 'value']].sum()
                        final_df = temp.copy().reset_index()
                        final_df['price'] = abs(final_df['value']/final_df['qty_dir'])

                        final_df['qty_dir'] = final_df['qty_dir'].replace(0, np.nan)
                        final_df = final_df.dropna()
                        final_df['strike'] = final_df['symbol'].map(parse_strike_from_symbol)
                        

                        # unhedged_pos_pe = final_df[final_df['symbol'].str[-2:] == 'PE']["qty_dir"].sum()
                        total_hedged_lot = (abs(final_df[final_df['qty_dir'] < 0]['qty_dir'].sum()/lot_size))

                        final_dict = final_df.to_dict('records')

                        first_loop = True
                        final_output = []
                        elm_margin = 0
                        for symbols in final_dict:
                            temp_array = np.array(single_strike_payoff_cal(symbols['strike'], symbols['symbol'][-2:], symbols['price'], var_range, symbols['qty_dir']))
                            # print(temp_array)
                            # print('SYMBOLS - ', symbols)
                            if symbols['qty_dir'] < 0 :
                                expiry_date = parse_date_from_symbol(symbols['symbol'])
                                dte = (expiry_date-timestamp.date()).days
                                if dte == 0:
                                    elm_margin += abs(symbols['qty_dir']*spot_price*0.02)
            
                            if first_loop:
                                final_output = temp_array
                                first_loop = False
                            else:
                                final_output += temp_array
                        # print('final_output - ', final_output)
                        exposure_margin = total_hedged_lot*spot_price*0.02*lot_size
                        var_over_time[index+'_'+str(timestamp)] = abs(min(final_output))
                        marign_over_time[index+'_'+str(timestamp)] = abs(min(final_output))+(exposure_margin) + elm_margin
                        
                    except Exception as e:
                        # print(e)
                        # traceback.print_exc()
                        pass
                except:
                    pass
        return date, marign_over_time, var_over_time

    except Exception as e:
        print(f'ERROR IN get_day_margin {e} {date}')
        traceback.print_exc()
        return date, {}, {}        
    

class MARGINCALCULATOR(DataInterface):

    def __init__(self, tb) -> None:
        
        self.tb = tb.copy()

    def get_margin_dict_single_core(self, entrie_period=True):

        # self.tb['expiry'] = self.tb['symbol'].apply(lambda x: expiry_from_symbol(x))
        
        margin_date_dict = {}
        self.var_date_dict = {}

        self.tb['date'] = self.tb['timestamp'].apply(lambda x: x.date())
        end_date = self.tb.timestamp.sort_values().iloc[-1].date()

        # MARGIN PERIOD
        if not entrie_period:
            start_date = end_date - datetime.timedelta(days=30)
            self.temp_tb = self.tb[(self.tb['date'] >= start_date) & (self.tb['date'] <= end_date)]
        else:
            self.temp_tb = self.tb

        dates = self.temp_tb['timestamp'].apply(lambda x: x.date()).unique()
        # tb_by_date = [self.temp_tb[self.temp_tb['date'] == date] for date in dates]

        for date in dates:
            date, margin_over_time, var_over_time = get_day_margin(date, self.temp_tb[self.temp_tb['date'] == date].copy())
            margin_date_dict[date] = margin_over_time
            self.var_date_dict[date] = var_over_time

        return margin_date_dict


    def get_margin_dict(self, entrie_period=True):
        manager = multiprocessing.Manager()

        # self.tb['expiry'] = self.tb['symbol'].apply(lambda x: expiry_from_symbol(x))
        
        margin_date_dict = {}
        self.var_date_dict = {}

        self.tb['date'] = self.tb['timestamp'].apply(lambda x: x.date())
        end_date = self.tb.timestamp.sort_values().iloc[-1].date()

        # MARGIN PERIOD
        if not entrie_period:
            start_date = end_date - datetime.timedelta(days=60)
            self.temp_tb = self.tb[(self.tb['date'] >= start_date) & (self.tb['date'] <= end_date)]
        else:
            self.temp_tb = self.tb

        dates = self.temp_tb['timestamp'].apply(lambda x: x.date()).unique()
        tb_by_date = manager.list([self.temp_tb[self.temp_tb['date'] == date] for date in dates])
        
        with multiprocessing.Pool() as pool:
            results = pool.starmap(get_day_margin, zip(dates, tb_by_date))

        for date, margin_over_time, var_over_time in results:
            margin_date_dict[date] = margin_over_time
            self.var_date_dict[date] = var_over_time

        return margin_date_dict

        
    def get_VAR(self):
        margindf = []
        for date in self.var_date_dict.keys():
            day_margin = self.var_date_dict[date]
            #print(day_margin)
            # time.sleep(0.5)
            for key, value in day_margin.items():
                timestamp = pd.to_datetime(key.split('_')[1])
                index = key.split('_')[0]
                # print(timestamp, index, value)
                margindf.append([timestamp, index, value])
                # print(key, value)
                # break
        marginDf = pd.DataFrame(margindf, columns=['timestamp', 'index', 'margin'])
            
        total_margin = pd.DataFrame()
        for i in marginDf['index'].unique():
            temp = marginDf[marginDf['index']==i]
            temp = temp.set_index('timestamp')
            temp_margin = temp['margin']
            temp_margin.name = i
            # print(temp_margin)
            total_margin = pd.concat([total_margin, temp_margin], axis=1)
            # break
        total_margin = total_margin.sort_index()
        total_margin = total_margin.fillna(0)

        total_margin['total'] = total_margin.sum(axis=1)

        return total_margin['total']


    def get_margin_utilization(self, entire_period=False, multiprocess=True):
        if multiprocess: self.margin_dict = self.get_margin_dict(entire_period)
        else: self.margin_dict = self.get_margin_dict_single_core(entire_period)

        margindf = []
        for date in self.margin_dict.keys():
            day_margin = self.margin_dict[date]
            #print(day_margin)
            # time.sleep(0.5)
            for key, value in day_margin.items():
                timestamp = pd.to_datetime(key.split('_')[1])
                index = key.split('_')[0]
                # print(timestamp, index, value)
                margindf.append([timestamp, index, value])
                # print(key, value)
                # break
        marginDf = pd.DataFrame(margindf, columns=['timestamp', 'index', 'margin'])
            
        total_margin = pd.DataFrame()
        for i in marginDf['index'].unique():
            temp = marginDf[marginDf['index']==i]
            temp = temp.set_index('timestamp')
            temp_margin = temp['margin']
            temp_margin.name = i
            total_margin = pd.concat([total_margin, temp_margin], axis=1)
            # break
        total_margin = total_margin.sort_index()
        total_margin = total_margin.fillna(0)

        total_margin['total'] = total_margin.sum(axis=1)
        self.marginDF = margindf
        return total_margin['total']

def get_margin_utilization_check(self, entire_period=False):
        self.margin_dict = self.get_margin_dict(entire_period)
        margindf = []
        for date in self.margin_dict.keys():
            day_margin = self.margin_dict[date]
            #print(day_margin)
            # time.sleep(0.5)
            for key, value in day_margin.items():
                timestamp = pd.to_datetime(key.split('_')[1])
                index = key.split('_')[0]
                # print(timestamp, index, value)
                margindf.append([timestamp, index, value])
                # print(key, value)
                # break
        marginDf = pd.DataFrame(margindf, columns=['timestamp', 'index', 'margin'])
            
def get_weekday(date):
    expiries = r.hgetall('list_of_expiries')
    if date in expiries['BANKNIFTY'] and date in expiries['NIFTY']:
        weekday = 5
    else:
        weekday = date.weekday()
    return weekday

