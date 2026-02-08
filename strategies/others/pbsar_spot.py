import datetime, time
import pandas as pd
import numpy as np

from utils.definitions import *
from utils.sessions import *
import direct_redis

if REDIS:
    from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
else:
    from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

r = direct_redis.DirectRedis()

class ParabolicSAR:
    def __init__(self, initial_af=0.02, max_af=0.02):
        """
        Initializes the Parabolic SAR indicator.
        
        Parameters:
        - initial_af (float): Initial acceleration factor.
        - max_af (float): Maximum acceleration factor.
        """
        self.initial_af = initial_af
        self.max_af = max_af
        self.af = initial_af  # Acceleration factor
        self.ep = None  # Extreme price
        self.sar = None  # Current SAR value
        self.trend = None  # Current trend ("up" or "down")
        self.prev_high = None  # Previous high
        self.prev_low = None  # Previous low

    def initialize_from_df(self, initial_df):
        for i, d in initial_df.iterrows():
            pasr, ternd = self.update(d['h'], d['l'], d['c'])
            # self.psar_df.append({'ts': d['ts'], 'psar': pasr, 'o':d['o'], 'h':d['h'], 'l':d['l'], 'c':d['c'], 'trend': ternd})

    # def get_psar_df(self):
    #     return self.psar_df

    def initialize(self, high, low, trend):
        """
        Initializes the SAR with the first values.
        
        Parameters:
        - high (float): Initial high price.
        - low (float): Initial low price.
        - trend (str): Initial trend ("up" or "down").
        """
        self.trend = trend
        self.sar = low if trend == "up" else high
        self.ep = high if trend == "up" else low
        self.prev_high = high
        self.prev_low = low

    def update(self, high, low, close):
        """
        Updates the Parabolic SAR with a new price tick.
        
        Parameters:
        - high (float): High price of the new tick.
        - low (float): Low price of the new tick.
        - close (float): Close price of the new tick.
        
        Returns:
        - float: Updated SAR value, or None if not initialized.
        """
        if self.sar is None or self.trend is None:
            # Initialize trend and SAR based on the first tick
            initial_trend = "up" if self.prev_high is None or close < high else "down"
            self.initialize(high, low, initial_trend)
            return None, None

        # Update SAR
        self.sar += self.af * (self.ep - self.sar)

        # Check for trend reversal
        if self.trend == "up" and low < self.sar:
            # Downtrend detected
            self.trend = "down"
            self.sar = self.ep
            self.ep = low
            self.af = self.initial_af
        elif self.trend == "down" and high > self.sar:
            # Uptrend detected
            self.trend = "up"
            self.sar = self.ep
            self.ep = high
            self.af = self.initial_af
        else:
            # Continue in the same trend, update EP and AF
            if self.trend == "up" and high > self.ep:
                self.ep = high
                self.af = min(self.af + self.initial_af, self.max_af)
            elif self.trend == "down" and low < self.ep:
                self.ep = low
                self.af = min(self.af + self.initial_af, self.max_af)

        # Update previous prices
        self.prev_high = high
        self.prev_low = low
        return self.sar, self.trend

class PBSARSPOT(EventInterfacePositional):

    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()
        self.signal = 0
        self.last_expiry = None
        self.futprice = None
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.exit_price = None
        self.trade_log = []

    def get_random_uid(self):
        # Select
        self.active_weekday = 99#np.random.choice(weekdays)
        self.session = 'x0'# np.random.choice(sessions)
        self.underlying = np.random.choice(underlyings)
        self.selector = 'PCT' # np.random.choice(selectors)
        if self.selector == 'M':
            self.selector_val = 0  # Placeholder or replace with a valid list if needed
        elif self.selector == 'P':
            self.selector_val = 0  # Placeholder or replace with a valid list if needed
        elif self.selector == 'PCT':
            self.selector_val = np.random.choice([0.0025, 0.0015, 0.00075, 0.0001, 0.0002, 0.00015, 0.0005, 0.0003, 0.00045])
        self.hedge_shift = np.random.choice([5, 7, 10])
        self.sl_pct = round(.05 * round(np.random.choice(np.random.rand(10)*0.5).round(2)/.05), 2)
        self.tgt_pct = round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2) #np.random.choice(tgt_pcts)
        self.max_reset = np.random.choice([0, 1, 2, 3, 4, 5, 6])
        self.trail_on = np.random.choice([True, False])
        self.delay = np.random.choice(range(0, 30, 5))
        # ...
        self.af = np.random.choice([0.005, 0.01, 0.015, 0.02, 0.025])
        self.max_af = np.random.choice([0.005, 0.01, 0.015, 0.02, 0.025])
        timeframe = [3, 5, 10]
        self.timeframe = np.random.choice(timeframe)

        # ...
        return self.get_uid_from_params()

    def set_params_from_uid(self, uid):
        # print(uid)
        s = uid.split('_')
        try:
            assert s[0] == self.strat_id
        except AssertionError:
            raise ValueError(f'Invalid UID {uid} for strat ID {self.strat_id}')
        s = s[1:]
        self.active_weekday = int(s.pop(0))
        self.session = s.pop(0)
        self.delay = int(s.pop(0))#=='True'
        self.underlying = s.pop(0)
        self.selector = s.pop(0)
        if self.selector == 'P' or self.selector == 'M':
            self.selector_val = int(s.pop(0))
        elif self.selector == 'PCT' or self.selector == 'D':
            self.selector_val = float(s.pop(0))
        self.hedge_shift = int(s.pop(0))
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.trail_on = s.pop(0)=='True'
        self.max_reset = int(s.pop(0))
        # ...
        self.af = float(s.pop(0))
        self.max_af = float(s.pop(0))
        self.timeframe = int(s.pop(0))
        self.itmperc = float(s.pop(0))
        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        # print(self.gen_uid)
        # print("debug →", repr(uid), repr(self.gen_uid))  # add this line temporarily
        assert uid == self.gen_uid
        self.uid = uid
        # print(self.uid)

    def get_uid_from_params(self):
        return f"""
        {self.strat_id}_
        {self.active_weekday}_
        {self.session}_
        {self.delay}_
        {self.underlying}_
        {self.selector}_
        {self.selector_val}_
        {self.hedge_shift}_
        {self.sl_pct}_
        {self.tgt_pct}_
        {self.trail_on}_
        {self.max_reset}_
        {self.af}_
        {self.max_af}_
        {self.timeframe}_
        {self.itmperc}_
        """.replace('\n', '').replace(' ', '').strip('_')

    def on_new_day(self):
        # Only NIFTYSPOT is used
        self.underlying = 'GOLD'
        self.mysymbol = 'GOLD'
        self.lot_size = 1  # For spot, lot size is 1 (or as per your trading logic)
        self.session_vals = sessions_dict[self.session]
        self.start_time = self.session_vals['start_time']
        self.stop_time = self.session_vals['stop_time']
        dtn = datetime.datetime.now()
        self.start_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=self.delay)).time()
        if self.futprice is None:
            self.futprice = self.get_all_ticks_by_symbol(self.mysymbol)
            self.futprice = self.futprice[self.futprice['ts'] < self.now]
            self.futprice = self.futprice.tail(30000).reset_index().drop(['index'], axis=1)
            self.psar_obj = ParabolicSAR(self.af, self.max_af)
            self.psar_obj.initialize_from_df(self.futprice.resample(f'{self.timeframe}min', on='ts').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}))
        self.premium_for_day_selection = True
        self.entry_price = None
        self.exit_price = None
        self.trade_log = []

    def on_event(self):
        pass

    def on_bar_complete(self):
        if self.last_expiry is None:
            self.last_expiry = str(self.now.date())
        self.current_tick = self.get_tick(self.now, self.mysymbol)
        self.underlying_close = self.current_tick['c']
        self.futprice.loc[len(self.futprice.index)] = ([self.current_tick['ts'], self.current_tick['o'], self.current_tick['h'], self.current_tick['l'], self.current_tick['c']])

        # INDICATOR CALC
        if ((self.now + datetime.timedelta(minutes=1)).time().minute % self.timeframe == 0 and self.now.time() < self.stop_time) or \
            (self.now.time() == datetime.time(9, 7) and self.underlying == 'NIFTY'):
            resampled_candle = self.futprice.tail(self.timeframe).resample(f'{self.timeframe}min', on='ts').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).to_dict('records')[-1]
            self.signal_val, self.trend = self.psar_obj.update(resampled_candle['h'], resampled_candle['l'], resampled_candle['c'])
            if self.signal_val > self.underlying_close:
                self.signal = -1
            elif self.signal_val < self.underlying_close:
                self.signal = 1
            else:
                self.signal = 0

        # ENTRY/EXIT LOGIC FOR SPOT USING place_trade
        if self.now.time() >= self.start_time and self.now.time() < datetime.time(15, 27):
            # Entry Long
            if self.position == 0 and self.signal == 1:
                self.success, self.entry_price = self.place_trade(self.now, 'BUY', self.lot_size, self.mysymbol, note='SPOT ENTRY')
                if self.success:
                    self.position = 1
                    self.trade_log.append({'action': 'BUY', 'price': self.entry_price, 'ts': self.now})
            # Entry Short
            elif self.position == 0 and self.signal == -1:
                self.success, self.entry_price = self.place_trade(self.now, 'SELL', self.lot_size, self.mysymbol, note='SPOT ENTRY')
                if self.success:
                    self.position = -1
                    self.trade_log.append({'action': 'SELL', 'price': self.entry_price, 'ts': self.now})
            # Signal inversion: Close long and go short
            elif self.position == 1 and self.signal == -1:
                self.success, self.exit_price = self.place_trade(self.now, 'SELL', self.lot_size, self.mysymbol, note='SPOT EXIT')
                if self.success:
                    self.trade_log.append({'action': 'SELL', 'price': self.exit_price, 'ts': self.now, 'note': 'Signal Inversion'})
                    self.position = -1
                    self.success, self.entry_price = self.place_trade(self.now, 'SELL', self.lot_size, self.mysymbol, note='SPOT ENTRY')
                    if self.success:
                        self.trade_log.append({'action': 'SELL', 'price': self.entry_price, 'ts': self.now, 'note': 'Signal Inversion'})
            # Signal inversion: Close short and go long
            elif self.position == -1 and self.signal == 1:
                self.success, self.exit_price = self.place_trade(self.now, 'BUY', self.lot_size, self.mysymbol, note='SPOT EXIT')
                if self.success:
                    self.trade_log.append({'action': 'BUY', 'price': self.exit_price, 'ts': self.now, 'note': 'Signal Inversion'})
                    self.position = 1
                    self.success, self.entry_price = self.place_trade(self.now, 'BUY', self.lot_size, self.mysymbol, note='SPOT ENTRY')
                    if self.success:
                        self.trade_log.append({'action': 'BUY', 'price': self.entry_price, 'ts': self.now, 'note': 'Signal Inversion'})