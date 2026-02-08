import numpy as np
import datetime
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

class PERPETUA(EventInterfacePositional):

    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()

        self.position = 0        # 1 = long, -1 = short, 0 = flat
        self.entry_price = None
        self.sl_price = None
        self.base_data = None

    # ------------------------------------------------------------------
    # PARAM HANDLING
    # ------------------------------------------------------------------

    def get_random_uid(self):
        self.active_weekday = 99
        self.session = 'x0'
        self.underlying = np.random.choice(underlyings)
        self.timeframe = np.random.choice([3, 5, 10])
        self.bb_period = np.random.choice([20, 30, 40])
        self.std_mult = np.random.choice([2.0, 2.5])
        self.delay = np.random.choice(range(0, 30, 5))
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
        self.underlying = s.pop(0)
        self.timeframe = int(s.pop(0))
        self.bb_period = int(s.pop(0))#=='True'
        self.std_mult = int(s.pop(0))
        self.delay = int(s.pop(0))#=='True'
        #Sample UID: perpetua_99_x0_NIFTY_1_30_1_0

        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        print(self.gen_uid)
        assert uid == self.gen_uid
        self.uid = uid
        # print(self.uid)


    def get_uid_from_params(self):
        return f"{self.strat_id}_{self.active_weekday}_{self.session}_" \
               f"{self.underlying}_{self.timeframe}_{self.bb_period}_{self.std_mult}_{self.delay}"

    # ------------------------------------------------------------------
    # DAY SETUP
    # ------------------------------------------------------------------

    def on_new_day(self):

        symbol_map = {
            'BANKNIFTY': 'BANKNIFTYSPOT',
            'NIFTY': 'NIFTYSPOT',
            'FINNIFTY': 'FINNIFTYSPOT',
            'MIDCPNIFTY': 'MIDCPNIFTYSPOT',
            'SENSEX': 'SENSEXSPOT'
        }

        self.mysymbol = symbol_map[self.underlying]
        self.lot_size = self.get_lot_size(self.underlying)

        session_vals = sessions_dict[self.session]
        dtn = datetime.datetime.now()

        self.start_time = (
            datetime.datetime.combine(dtn.date(), session_vals['start_time']) +
            datetime.timedelta(minutes=self.delay)
        ).time()

        self.end_time = datetime.time(15, 10)

        self.position = 0
        self.entry_price = None
        self.sl_price = None

        if self.base_date is None:
            self.base_data = self.get_all_ticks_by_symbol(self.mysymbol)
            self.active_data = self.calculate_bbands()
            

    # ------------------------------------------------------------------
    # INDICATORS
    # ------------------------------------------------------------------

    def calculate_bbands(self):

        if self.base_data is None or len(self.base_data) < self.bb_period:
            return None

        df = self.base_data.set_index('ts').sort_index()
        df['sma'] = df['c'].rolling(f'{self.bb_period}min').mean()  # time-based rolling if timestamps irregular
        df['std'] = df['c'].rolling(f'{self.bb_period}min').std(ddof=0)
        df['plus_1'] = df['sma'] + self.std_mult * df['std'] * 1
        df['minus_1'] = df['sma'] - self.std_mult * df['std'] * 1
        df['plus_2'] = df['sma'] + self.std_mult * df['std'] * 2
        df['minus_2'] = df['sma'] - self.std_mult * df['std'] * 2
        df.reset_index(inplace=True)

        return df

    # ------------------------------------------------------------------
    # SIGNAL LOGIC (MEAN REVERSION + TREND)
    # ------------------------------------------------------------------

    def entry_signal(self, price, prev_price, bb):

        if len(self.futprice) < 2:
            return 0
        
        prev = self.futprice['c'].iloc[-2]

        # Long signals
        if prev_price <= bb['minus_1'] and price > bb['minus_1']:
            return 1          # mean reversion
        if prev_price <= bb['plus_2'] and price > bb['plus_2']:
            return 1          # trend breakout

        # Short signals
        if prev_price >= bb['plus_1'] and price < bb['plus_1']:
            return -1         # mean reversion
        if prev_price >= bb['minus_2'] and price < bb['minus_2']:
            return -1         # trend breakdown

        return 0

    # ------------------------------------------------------------------
    # BAR HANDLER
    # ------------------------------------------------------------------

    def on_bar_complete(self):

        tick = self.get_tick(self.now, self.mysymbol)
        prev_tick = self.get_tick(self.now - datetime.timedelta(minutes=1), self.mysymbol)
        
        if tick is None or prev_tick is None:
            return

        price = tick['c']
        prev_price = prev_tick['c']

        self.futprice = self.all_ticks[self.all_ticks['ts'] <= self.now]

        if len(self.futprice) < self.bb_period + 1:
            return

        bb = self.calculate_bbands()
        if bb is None:
            return

        exited_this_bar = False

        # ---------------- EXIT ----------------

        if self.position == 1:
            self.sl_price = max(self.sl_price or bb['minus_2'], bb['minus_2'])
            if price <= self.sl_price:
                self.place_trade(self.now, 'SELL', self.lot_size, self.mysymbol, note='SL_LONG EXIT')
                self.position = 0
                exited_this_bar = True

        elif self.position == -1:
            self.sl_price = min(self.sl_price or bb['plus_2'], bb['plus_2'])
            if price >= self.sl_price:
                self.place_trade(self.now, 'BUY', self.lot_size, self.mysymbol, note='SL_SHORT EXIT')
                self.position = 0
                exited_this_bar = True

        # ---------------- ENTRY ----------------

        if (not exited_this_bar and
            self.position == 0 and
            self.start_time <= self.now.time() < self.end_time):

            sig = self.entry_signal(price,prev_price, bb)

            if sig == 1:
                self.place_trade(self.now, 'BUY', self.lot_size, self.mysymbol, note='LONG ENTRY')
                self.position = 1
                self.sl_price = bb['minus_2']

            elif sig == -1:
                self.place_trade(self.now, 'SELL', self.lot_size, self.mysymbol, note='SHORT ENTRY')
                self.position = -1
                self.sl_price = bb['plus_2']

        # ---------------- EOD FLAT ----------------

        if self.now.time() >= self.end_time and self.position != 0:
            side = 'SELL' if self.position == 1 else 'BUY'
            self.place_trade(self.now, side, self.lot_size, self.mysymbol,note='EOD EXIT')
            self.position = 0
            self.sl_price = None