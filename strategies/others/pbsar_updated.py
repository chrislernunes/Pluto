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

class PBSARHL(EventInterfacePositional):

    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()
        self.signal = 0
        self.last_expiry = None
        self.futprice = None
        self.expiry_ce = None  # Track expiry of CE position
        self.expiry_pe = None  # Track expiry of PE position

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
    
    def sync_signal_from_csv(self,df):
        
        # Ensure datetime comparison
        df["ts"] = pd.to_datetime(df["ts"])
        now_ts = pd.to_datetime(self.now)

        # Find matching row
        match = df[df["ts"] == now_ts]

        if not match.empty:
            row = match.iloc[0]
            self.signal = int(row["SIGNAL"])
            self.trend = row["TREND"]
        else:
            return

    def on_new_day(self):
        # ...
        if self.underlying == 'BANKNIFTY':
            self.mysymbol = 'BANKNIFTYSPOT'
        elif self.underlying == 'NIFTY':
            self.mysymbol = 'NIFTYSPOT'
        elif self.underlying == 'FINNIFTY':
            self.mysymbol = 'FINNIFTYSPOT'
        elif self.underlying == 'MIDCPNIFTY':
            self.mysymbol = "MIDCPNIFTYSPOT" 
        elif self.underlying == 'SENSEX':
            self.mysymbol = "SENSEXSPOT"
        elif self.underlying == 'SPXW':
            self.mysymbol = "SPXW"
        else:
            raise ValueError(f'Unknown Underlying: {self.underlying}')   
        
        self.lot_size = self.get_lot_size(self.underlying) 
        # ...
        self.session_vals = sessions_dict[self.session]
        self.start_time = self.session_vals['start_time']# #datetime.time(9, 15)#
        # self.stop_time = self.session_vals['stop_time']#datetime.time (15, 14)#
        self.stop_time = datetime.time (15, 14)
        # ADD DELAY TO START TIME
        dtn = datetime.datetime.now()
        self.stop_time = (datetime.datetime.combine(dtn.date(), self.stop_time) - datetime.timedelta(minutes=self.delay)).time()
        # ...
        self.signal_df = pd.read_csv(f'/home/mridul/jupiter/notebooks/Mridul/signals_nifty_csv/{self.af}_signals_pbsar_itmpct.csv')
        self.premium_for_day_selection = True
        
        
    def on_event(self):
        pass

    def on_bar_complete(self):
        #print('---------------------------------------------------')
        # print(self.now)
        # print(self.now, self.symbol_ce, self.symbol_pe)
        if self.last_expiry is None:
            self.last_expiry = str(self.now.date())
        # ...
        if self.premium_for_day_selection:
            if self.symbol_ce is None and self.symbol_pe is None:
                self.dte = self.get_dte_by_underlying(self.now, self.underlying)
                self.premium_for_day_selection = False

            elif self.symbol_ce is not None:
                self.dte = self.get_dte(self.now, self.symbol_ce)
                self.premium_for_day_selection = False
                
            elif self.symbol_pe is not None:
                self.dte = self.get_dte(self.now, self.symbol_pe)
                self.premium_for_day_selection = False

        # ...
        if self.dte != self.active_weekday and self.active_weekday != 99:
            return
        
        # ROLLOVER LOGIC
        if self.dte == 0 and self.now.time() == datetime.time(14, 50):
            self.roll_over = True
            expiry_idx = 1
        elif self.dte == 0 and self.now.time() > datetime.time(14, 50):
            self.roll_over = False
            expiry_idx = 1
        else:
            self.roll_over = False
            expiry_idx = 0
        
        if ((self.now + datetime.timedelta(minutes=1)).time().minute % self.timeframe == 0 and self.now.time() < self.stop_time) or \
            (self.now.time() == datetime.time(9, 7) and self.underlying == 'NIFTY') or \
                (self.now.time() == datetime.time(9, 9) and self.underlying == 'SENSEX') :
            
            self.sync_signal_from_csv(self.signal_df)

        # EXIT
        if (self.now.time() >= self.start_time and self.now.time() < datetime.time(15, 29) ) :

            if self.position_ce == 1:
                self.to_exit = False
                
                # # ROLL-OVER
                if self.roll_over and self.expiry_ce == self.now.date():
                    self.to_exit = True
                    self.reason_ce = 'ROLL-OVER'
                
                if self.signal == 1:
                    self.to_exit = True
                    self.reason_ce = 'SIGNAL INVERSION'


                if self.to_exit:
                    self.success_ce, _ = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_ce,note=self.reason_ce)
                    if self.success_ce:
                        self.position_ce = -1
                        self.symbol_ce = None
                        self.expiry_ce = None

                        
            if self.position_pe == 1:
                self.to_exit = False

                # # ROLL-OVER
                if self.roll_over and self.expiry_pe == self.now.date():
                    self.to_exit = True
                    self.reason_pe = 'ROLL-OVER'
                
                if self.signal == -1:
                    self.to_exit = True
                    self.reason_pe = 'SIGNAL INVERSION'
                
                if self.to_exit:
                    self.success_pe, _ = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_pe,note=self.reason_pe)
                    if self.success_pe:
                        self.position_pe = -1
                        self.symbol_pe = None
                        self.expiry_pe = None
        
        
        # ENTRY
        if (self.now.time() >= self.start_time and self.now.time() < datetime.time(15, 27) ) :   #or (self.now.time() == datetime.time(9, 7))
            if self.position_ce == 0 and self.signal == -1 :
                #print(round(abs(selectorval)))
                # select symbol CE
                if self.selector == 'P':
                    self.symbol_ce = self.find_symbol_by_premium(
                        self.now, self.underlying, expiry_idx, 'CE', self.selector_val
                    )
                    self.note_ce = "ENTRY"
                elif self.selector == 'M':
                    self.symbol_ce = self.find_symbol_by_moneyness(
                        self.now, self.underlying, expiry_idx, 'CE', self.selector_val
                    )
                    self.note_ce = "ENTRY"
                elif self.selector == 'PCT':
                    self.symbol_ce = self.find_symbol_by_premium(
                    self.now, self.underlying, expiry_idx, 'CE', self.selector_val*self.underlying_close
                    )
                    self.note_ce = "ENTRY"
                elif self.selector == 'D':
                    self.symbol_ce, self.delta_ce = self.find_symbol_by_itm_percent(
                        self.now, self.underlying, expiry_idx, 'CE', self.selector_val, default_to_atm =True
                    )
                    self.note_ce = "ENTRY"         
                if self.symbol_ce is not None:            
                    self.success_ce, self.entry_price_ce = self.place_trade(self.now, 'SELL', self.lot_size, self.symbol_ce,note=self.note_ce)
                    # print("####","|",self.now,"|",self.symbol_ce,"|",self.delta_ce)
                    if self.success_ce:
                        self.position_ce = 1
                        self.expiry_ce = self.parse_date_from_symbol(self.symbol_ce)
                        

            if self.position_pe == 0 and self.signal == 1 :
                #print(round(-selectorval))
                # select symbol P1
                if self.selector == 'P':
                    self.symbol_pe = self.find_symbol_by_premium(
                        self.now, self.underlying, expiry_idx, 'PE', self.selector_val
                    )
                    self.note_pe = "ENTRY"
                elif self.selector == 'M':
                    self.symbol_pe = self.find_symbol_by_moneyness(
                        self.now, self.underlying, expiry_idx, 'PE', self.selector_val
                    )
                    self.note_pe = "ENTRY"
                elif self.selector == 'PCT':
                    self.symbol_pe = self.find_symbol_by_premium(
                    self.now, self.underlying, expiry_idx, 'PE', self.selector_val*self.underlying_close
                    )
                    self.note_pe = "ENTRY"
                elif self.selector == 'D':
                    self.symbol_pe, self.delta_pe = self.find_symbol_by_itm_percent(
                        self.now, self.underlying, expiry_idx, 'PE', self.selector_val, default_to_atm =True
                    )
                    self.note_pe = "ENTRY"
                if self.symbol_pe is not None:
                    self.success_pe, self.entry_price_pe= self.place_trade(self.now, 'SELL', self.lot_size, self.symbol_pe,note=self.note_pe)
                    # print("####","|",self.now,"|",self.symbol_pe,"|",self.delta_pe)
                    if self.success_pe:
                        self.position_pe = 1
                        self.expiry_pe = self.parse_date_from_symbol(self.symbol_pe)
                        
                        

        # RE-ENTRY
        if self.position_ce == -1:
            #print(self.now, self.current_price_ce)
            if (self.reason_ce == 'SL' or self.reason_ce == 'TGT') and self.reset_count_ce < self.max_reset:
                self.position_ce = 0
                self.reset_count_ce += 1
            if self.reason_ce in ['ROLL-OVER', 'SIGNAL INVERSION']:
                self.position_ce = 0
                
        if self.position_pe == -1:    
            #print(self.now, self.current_price_pe)
            if (self.reason_pe == 'SL' or self.reason_pe == 'TGT') and self.reset_count_pe < self.max_reset:
                self.position_pe = 0
                self.reset_count_pe += 1
            if self.reason_pe in ['ROLL-OVER', 'SIGNAL INVERSION']:
                self.position_pe = 0

        # CHANGE ACTIVE DTE
        if self.roll_over:
            self.last_expiry = str(self.now.date())