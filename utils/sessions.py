import datetime

market_start_time = datetime.time(9, 15)

sessions_dict = {
        'x0': {
            'start_time': datetime.time(9,15), 
            'stop_time': datetime.time(15,13),
            },
        'x1': {
            'start_time': datetime.time(11,15),
            'stop_time': datetime.time(15,14),
            },
        'x2': {
            'start_time': datetime.time(9,15),
            'stop_time': datetime.time(13,15),
            },
        's0': {
            'start_time': datetime.time(9,15),
            'stop_time': datetime.time(11,15),
            },
        's1': {
            'start_time': datetime.time(11,15),
            'stop_time': datetime.time(13,15),
            },
        's2': {
            'start_time': datetime.time(13,15),
            'stop_time': datetime.time(15,14),
            },

        'y0': {
            'start_time': datetime.time(10,15),
            'stop_time': datetime.time(15,14),
            },
        'z0': {
            'start_time': datetime.time(12,15),
            'stop_time': datetime.time(15,14),
            },
        
        't0': {
            'start_time': datetime.time(9,15),
            'stop_time': datetime.time(10,15),
            },
        't1': {
            'start_time': datetime.time(10,15),
            'stop_time': datetime.time(14,15),
            },
        't2': {
            'start_time': datetime.time(14,15),
            'stop_time': datetime.time(15,14),
            },
        
        # Overnight Session for STBT
        'o1': {
            'start_time': datetime.time(15,20),
            'stop_time': datetime.time(9,20)
        },
        # BTST Session
        'o2': {
            'start_time': datetime.time(9,15),
            'stop_time': datetime.time(15,15),
            'sq_off_time': datetime.time(9,30)
        },
        'o3': {
            'start_time': datetime.time(12,0),
            'stop_time': datetime.time(15,15),
            'sq_off_time': datetime.time(9,30)
        },
        'o4':{
            'start_time': datetime.time(15,20),
            'stop_time': datetime.time(9,30)
        },
        'o5':{
            'start_time': datetime.time(15,20),
            'stop_time': datetime.time(9,45)
        },
        'o6':{
            'start_time': datetime.time(15,20),
            'stop_time': datetime.time(10,0)
        },


        # NIFTY TBS SESSION TIME
        'n0': {
            'start_time': datetime.time(9,20),
            'stop_time': datetime.time(15,15),
            },
        'n1': {
            'start_time': datetime.time(10,30),
            'stop_time': datetime.time(15,15),
            },
        'n2': {
            'start_time': datetime.time(11,15),
            'stop_time': datetime.time(15,15),
            },
        'n3': {
            'start_time': datetime.time(13,15),
            'stop_time': datetime.time(15,15),
            },
        'n4': {
            'start_time': datetime.time(14,15),
            'stop_time': datetime.time(15,15),
            },

        # BANKNIFTY TBS SESSION TIME
        'b0': {
            'start_time': datetime.time(9,20),
            'stop_time': datetime.time(15,15),
            },
        'b1': {
            'start_time': datetime.time(10,10),
            'stop_time': datetime.time(15,15),
            },
        'b2': {
            'start_time': datetime.time(11,10),
            'stop_time': datetime.time(15,15),
            },
        'b3': {
            'start_time': datetime.time(12,10),
            'stop_time': datetime.time(15,15),
            },
        'b4': {
            'start_time': datetime.time(13,10),
            'stop_time': datetime.time(15,15),
            },

        # FINNIFTY TBS SESSION TIME
        'f0': {
            'start_time': datetime.time(9,20),
            'stop_time': datetime.time(15,15),
            },
        'f1': {
            'start_time': datetime.time(10,20),
            'stop_time': datetime.time(15,15),
            },
        'f2': {
            'start_time': datetime.time(11,20),
            'stop_time': datetime.time(15,15),
            },
        'f3': {
            'start_time': datetime.time(12,20),
            'stop_time': datetime.time(15,15),
            },
        'f4': {
            'start_time': datetime.time(13,20),
            'stop_time': datetime.time(15,15),
            },
        
        # atbs 
        'a1': {
            'start_time': datetime.time(9, 29),
            'stop_time': datetime.time(11, 57),
            },
        'a2': {
            'start_time': datetime.time(11, 59),
            'stop_time': datetime.time(15, 14),
            },
            
        
        
        ##### Short options 2 System #####
        'sot01': {
            'start_time': datetime.time(9,30),
            'stop_time': datetime.time(15,10),
            'reentry_limit_time': datetime.time(14,30),
            },
        'sot02': {
            'start_time': datetime.time(10,30),
            'stop_time': datetime.time(15,11),
            'reentry_limit_time': datetime.time(14,30),
            },
        
}

hedge_prices_dict = {
    'BANKNIFTY': 2,
    'NIFTY': 2,
    'FINNIFTY': 2,
    'SENSEX': 10
}


# nse2022holidays = [datetime.date(2022, 1, 26),
#                   datetime.date(2022, 3, 1),
#                   datetime.date(2022, 3, 18),
#                   datetime.date(2022, 4, 14),
#                   datetime.date(2022, 4, 15),
#                   datetime.date(2022, 5, 3),
#                   datetime.date(2022, 8, 9),
#                   datetime.date(2022, 8, 15),
#                   datetime.date(2022, 8, 31),
#                   datetime.date(2022, 10, 5),
#                   datetime.date(2022, 10, 24),
#                   datetime.date(2022, 10, 26),
#                   datetime.date(2022, 11, 8)]

# nse2021holidays = [datetime.date(2021, 1, 26),
#                   datetime.date(2021, 3, 11),
#                   datetime.date(2021, 3, 29),
#                   datetime.date(2021, 4, 2),
#                   datetime.date(2021, 4, 14),
#                   datetime.date(2021, 4, 21),
#                   datetime.date(2021, 5, 13),
#                   datetime.date(2021, 7, 21),
#                   datetime.date(2021, 8, 19),
#                   datetime.date(2021, 9, 10),
#                   datetime.date(2021, 10, 15),
#                   datetime.date(2021, 11, 4),
#                   datetime.date(2021, 11, 5),
#                   datetime.date(2021, 11, 19)]

# nse2023holidays = [datetime.date(2023, 1, 26),
#                     datetime.date(2023, 3, 7),
#                     datetime.date(2023, 3, 30),
#                     datetime.date(2023, 4, 4),
#                     datetime.date(2023, 4, 7),
#                     datetime.date(2023, 4, 14),
#                     datetime.date(2023, 5, 1),
#                     datetime.date(2023, 6, 29),
#                     datetime.date(2023, 8, 15),
#                     datetime.date(2023, 9, 19),
#                     datetime.date(2023, 10, 2),
#                     datetime.date(2023, 10, 24),
#                     datetime.date(2023, 11, 14),
#                     datetime.date(2023, 11, 27),
#                     datetime.date(2023, 12, 25)]

# nse2024holdiays = [datetime.date(2024, 1, 22),
#                     datetime.date(2024, 1, 26),
#                     datetime.date(2024, 3, 8),
#                     datetime.date(2024, 3, 25),
#                     datetime.date(2024, 3, 29),
#                     datetime.date(2024, 4, 11),
#                     datetime.date(2024, 4, 17),
#                     datetime.date(2024, 5, 1),
#                     datetime.date(2024, 6, 17),
#                     datetime.date(2024, 7, 17),
#                     datetime.date(2024, 8, 15),
#                     datetime.date(2024, 10, 2),
#                     datetime.date(2024, 11, 1),
#                     datetime.date(2024, 11, 15),
#                     datetime.date(2024, 12, 25)]

# holidays = nse2021holidays+nse2022holidays+nse2023holidays+nse2024holdiays
#############

# GENERAL PARAMS SPACE
weekdays = range(0, 5)
sessions = list(sessions_dict.keys())
timeframes = range(1, 6)
underlyings = ['BANKNIFTY', 'NIFTY', 'FINNIFTY'] # ,'BANKNIFTY', 'FINNIFTY'
selectors = ['P', 'PCT', 'M'] # , 'M'
moneynesses = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
seek_prices = [15, 25, 50, 75, 100] #[50,75,100,125,150,175,200,250,300,350,400]
hedge_shifts = [10, 15, 20]
sl_pcts = [0.1,0.2,0.3,0.4,0.5,0.75]
tgt_pcts = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
resets = [1, 2, 4, 5, 6, 10, 15]
delays = range(30, 300, 30)
# t = [1, 3, 5, 7, 9, 11, 13, 15]
normalizers = [0.0005,0.00075,0.001,0.00]

