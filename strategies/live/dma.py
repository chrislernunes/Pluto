import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import direct_redis
# r = direct_redis.DirectRedis(host='localhost', port=6379, db=0)

from utils.definitions import *
from utils.utility import *

if REDIS:
    from engine.ems import EventInterfacePositional
else:
    from engine.ems_db import EventInterfacePositional


class DMA(EventInterfacePositional):
    def __init__(self, conn=None):
         super().__init__(conn)
         self.strat_id = self.__class__.__name__.lower()
         self.position_pe=0
         self.note=""
         self.note2=""

    
    def get_random_uid(self):
        # Select
        self.active_weekday = 99#np.random.choice(weekdays)
        self.underlying = 'NIFTY'#np.random.choice(underlyings)
        self.selector = np.random.choice(selectors)
        if self.selector == 'M':
            self.selector_val = np.random.choice(moneynesses)
        self.delay = np.random.choice(delays)
        # ...
        return self.get_uid_from_params()
    
    def set_params_from_uid(self, uid):
        s = uid.split('_')
        try:
            assert s[0] == self.strat_id
        except AssertionError:
            raise ValueError(f'Invalid UID {uid} for strat ID {self.strat_id}')
        s = s[1:]
        self.active_weekday = int(s.pop(0))
        self.dma_val = int(s.pop(0))
        self.underlying = s.pop(0)
        self.delay = int(s.pop(0))
        self.system_tag = s.pop(0)
        self.uid=uid
        # self.now=datetime.datetime.now()
        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        print(self.gen_uid)
        assert uid == self.gen_uid
        self.uid = uid
        # print(self.uid)
    
    def get_uid_from_params(self):
        return f"""
        {self.strat_id}_
        {self.active_weekday}_
        {self.dma_val}_
        {self.underlying}_
        {self.delay}_
        {self.system_tag}
        """.replace('\n', '').replace(' ', '').strip('_')
    
    #DMA CALCULATION
    def on_new_day(self):
        if self.underlying=="NIFTY":
            self.symbol='NIFTYSPOT'
        # self.dma=0
        # self.closes=[]
        self.lot_size = self.get_lot_size(self.underlying)
        # self.sell=False
        # self.buy=False

    def on_bar_complete(self):
        if self.now.time() == datetime.time(15,0):
            self.data=self.get_all_ticks_by_symbol(self.symbol)
            self.data.sort_values("ts")
            self.now_close= int(self.get_tick(self.now, self.symbol)['c'])
            # print("data",self.data[(self.data["ts"].dt.time==datetime.time(15,29)) & ( self.data["ts"].dt.date<self.now.date())]['c'].tail(9))
            #CAlculating DMA
            self.closes=self.data[(self.data["ts"].dt.time==datetime.time(15,29)) & ( self.data["ts"].dt.date<self.now.date())]['c']
            # print("Closes",self.closes.tail(self.dma_val))
            self.dma=int(self.closes.tail(self.dma_val-1).sum())
            self.dma=int((self.dma+self.now_close)/self.dma_val)
            # print(self.dma)
            # print("now_close",self.now_close, self.now)
            # self.exp_date=self.get_monthly_expiry_code(self.now,self.underlying)
            # print("expiry date:", self.exp_date)
            # print("today's date:", self.now.date())

            #ROLL OVER LOGIC
            # self.exp_id=0
            if self.position_pe==1 and self.get_dte(self.now,self.symbol_pe)==0:
                self.note='ROLL-OVER'
                self.note2='ROLL-OVER'
                # self.list_of_expiries = r.hget('list_of_expiries', 'NIFTY')
                # if self.now.month==12:
                #     self.list_of_expiries=[self.x for self.x in self.list_of_expiries if self.x.year == int(self.now.year)+1 and self.x.month == 1]
                # else:
                #     self.list_of_expiries=[self.x for self.x in self.list_of_expiries if self.x.year == int(self.now.year) and self.x.month == int(self.now.month)+1]
                # self.exp_id=len(self.list_of_expiries)               
                # print("  ")
                # print("  ")
                # print("  ")
                # print(self.exp_id)
                # jdsfbsidv



            #Exit Logic
            if (self.now_close>self.dma and self.position_pe==1) or (self.note=='ROLL-OVER'):
                if self.now_close>self.dma and self.position_pe==1 and self.note=='ROLL-OVER':
                    self.note='EXIT'
                    self.note2=""
                if self.note=='ROLL-OVER':
                    self.list_of_expiries = self.get_expiry_dates(self.underlying)
                    if self.now.month==12:
                        self.list_of_expiries=[self.x for self.x in self.list_of_expiries if self.x.year == int(self.now.year)+1 
                                               and self.x.month == 1]
                        self.exp_id2=len(self.list_of_expiries)
                    else:
                        self.list_of_expiries=[self.x for self.x in self.list_of_expiries if self.x.year == int(self.now.year)
                                                and self.x.month == int(self.now.month)+1]
                        self.exp_id2=len(self.list_of_expiries)
                else:
                    self.note="EXIT"

                self.success_pe, self.entry_price_pe= self.place_trade(self.now,'SELL',self.lot_size,self.symbol_pe,note=self.note)
                # if self.note=='ROLL-OVER':
                self.position_pe=0
                    


            #Entry Logic
            if self.now_close<self.dma and self.position_pe==0:
                
                # self.buy=True
                # self.position=1
                # self.exp_id=0
                self.list_of_expiries = self.get_expiry_dates(self.underlying)
                self.list_of_expiries=[self.x for self.x in self.list_of_expiries if self.x.year == int(self.now.year) 
                                       and self.x.month == int(self.now.month)]
                self.list_of_expiries.sort()
                
                
                if self.list_of_expiries[-1].day-self.now.day<7 and self.list_of_expiries[-1].day-self.now.day>0:
                    self.exp_id=0
                elif  (self.list_of_expiries[-1].day-self.now.day>=7) and (self.list_of_expiries[-1].day-self.now.day<14):
                    self.exp_id=1
                elif self.list_of_expiries[-1].day-self.now.day>=14 and self.list_of_expiries[-1].day-self.now.day<21:
                    self.exp_id=2
                elif self.list_of_expiries[-1].day-self.now.day>=21 and self.list_of_expiries[-1].day-self.now.day<28:
                    self.exp_id=3
                elif self.list_of_expiries[-1].day-self.now.day>=28 and self.list_of_expiries[-1].day-self.now.day<35:
                    self.exp_id=4
                elif self.list_of_expiries[-1].day-self.now.day<0:
                    self.list_of_expiries = self.get_expiry_dates(self.underlying)
                    if self.now.month==12:
                        self.list_of_expiries=[self.x for self.x in self.list_of_expiries if self.x.year == int(self.now.year)+1 
                                               and self.x.month == 1]
                    else:
                        self.list_of_expiries=[self.x for self.x in self.list_of_expiries if self.x.year == int(self.now.year)
                                                and self.x.month == int(self.now.month)+1]
                    self.exp_id=len(self.list_of_expiries)-1
                elif self.list_of_expiries[-1].day-self.now.day==0 and self.note2!="ROLL-OVER":
                    self.list_of_expiries = self.get_expiry_dates(self.underlying)
                    if self.now.month==12:
                        self.list_of_expiries=[self.x for self.x in self.list_of_expiries if self.x.year == int(self.now.year)+1 
                                               and self.x.month == 1]
                        self.exp_id=len(self.list_of_expiries)
                    else:
                        self.list_of_expiries=[self.x for self.x in self.list_of_expiries if self.x.year == int(self.now.year)
                                                and self.x.month == int(self.now.month)+1]
                        self.exp_id=len(self.list_of_expiries)
                if self.note2=="ROLL-OVER":
                    self.exp_id=self.exp_id2
                    self.note2=""
                self.note="ENTER"
                self.symbol_pe=self.find_symbol_by_moneyness(self.now, self.underlying,self.exp_id,'PE',0)
                if self.symbol_pe is not None:
                    self.success_pe, self.entry_price_pe= self.place_trade(self.now, 'BUY',self.lot_size,self.symbol_pe,note=self.note)
                    self.note=""
                    self.position_pe=1

            

            #Roll-Over Logic
            # if self.position==1 and self.get_dte(self.now,self.symbol_pe)==0:
            #     self.note="Roll over "
            #     self.success_pe, self.entry_price_pe= self.place_trade(self.now, 'SELL',self.lot_size,self.symbol_pe,note=self.note)
            #     self.exp_id=1
            #     self.buy=True

            # Actions 
            # if self.buy:
                # self.symbol_pe=self.find_symbol_by_moneyness(self.now, self.underlying,self.exp_id,'PE',0)
                # self.success_pe, self.entry_price_pe= self.place_trade(self.now, 'BUY',self.lot_size,self.symbol_pe,note=self.note)
                # self.position=1
# =====================================================================

            # #Entry Logic
            # if self.now_close<self.dma and self.position==0:
            #     self.note="DMA Below"
            #     self.buy=True
            #     self.position=1
            #     self.exp_id=0
            #     self.symbol_pe=self.find_symbol_by_moneyness(self.now, self.underlying,self.exp_id,'PE',0)
                       
            # #Exit Logic
            # if self.now_close>self.dma and self.position==1:
            #     self.note="DMA Above"
            #     self.success_pe, self.entry_price_pe= self.place_trade(self.now, 'SELL',self.lot_size,self.symbol_pe,note=self.note)
            #     self.position=0
            # #Roll-Over Logic
            # if self.position==1 and self.get_dte(self.now,self.symbol_pe)==0:
            #     self.note="Roll over "
            #     self.success_pe, self.entry_price_pe= self.place_trade(self.now, 'SELL',self.lot_size,self.symbol_pe,note=self.note)
            #     self.exp_id=1
            #     self.buy=True
            # #Actions 
            # if self.buy:
            #     self.symbol_pe=self.find_symbol_by_moneyness(self.now, self.underlying,self.exp_id,'PE',0)
            #     self.success_pe, self.entry_price_pe= self.place_trade(self.now, 'BUY',self.lot_size,self.symbol_pe,note=self.note)
            #     self.position=1
