import datetime
import time
from functools import wraps
import utils.definitions as defs
import pandas as pd


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} took {time.time()-start} seconds')
        return result
    return wrapper

market_start_time = datetime.time(9, 15)
market_close_time = datetime.time(15, 30)

nse_calendar = {datetime.date(2023, 1, 26): 'Republic Day',
                datetime.date(2023, 3, 7): 'Holi',
                datetime.date(2023, 3, 30): 'Ram Navami',
                datetime.date(2023, 4, 4): 'Mahavir Jayanti',
                datetime.date(2023, 4, 7):  'Good Friday',
                datetime.date(2023, 4, 14):  'Dr.Baba Saheb Ambedkar Jayanti',
                datetime.date(2023, 5, 1): 'Maharashtra Day',
                datetime.date(2023, 6, 29): 'Bakri Id',
                datetime.date(2023, 8, 15): 'Independence Day',
                datetime.date(2023, 9, 19): 'Ganesh Chaturthi',
                datetime.date(2023, 10, 2): 'Mahatma Gandhi Jayanti',
                datetime.date(2023, 10, 24): 'Dussehra',
                datetime.date(2023, 11, 14): 'Diwali-Balipratipada',
                datetime.date(2023, 11, 27): 'Gurunanak Jayanti',
                datetime.date(2023, 12, 25): 'Christmas' ,

                datetime.date(2024, 1, 26): 'Republic Day',
                datetime.date(2024, 3, 8): 'Mahashivratri',
                datetime.date(2024, 3, 25): 'Holi',
                datetime.date(2024, 3, 29): 'Good Friday',
                datetime.date(2024, 4, 11): 'Id-Ul-Fitr (Ramadan Eid)',
                datetime.date(2024, 4, 17): 'Shri Ram Navmi',
                datetime.date(2024, 5, 1): 'Maharashtra Day',
                datetime.date(2024, 6, 17): 'Bakri Id',
                datetime.date(2024, 7, 17): 'Moharram',
                datetime.date(2024, 8, 15): 'Independence Day/Parsi New Year',
                datetime.date(2024, 10, 2): 'Mahatma Gandhi Jayanti',
                datetime.date(2024, 11, 1): 'Diwali Laxmi Pujan*',
                datetime.date(2024, 11, 15): 'Gurunanak Jayanti',
                datetime.date(2024, 12, 25): 'Christmas',
                }


                
def time_now():return datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')

def market_hours_sython(open_time = datetime.time(9, 0, 0), close_time= datetime.time(15, 29, 59)):
    date_today = datetime.datetime.today().date()
    day_today = time.strftime("%A", time.localtime())
    if date_today in nse_calendar or day_today in ['Saturday', 'Sunday']:
        return 60*60*8
    
    market_time_open1 = datetime.datetime.today().time() > open_time
    market_time_open2 = datetime.datetime.today().time() < close_time
    if market_time_open1 and market_time_open2:
        return -1

    current = datetime.datetime.time(datetime.datetime.now())
    diff = datetime.datetime.combine(datetime.date.today(), open_time) - datetime.datetime.combine(datetime.date.today(), current)
    sleep_secs = diff.seconds + 5
    get_up_time=datetime.datetime.fromtimestamp(sleep_secs+time.time()).strftime('%Y-%m-%d %H:%M:%S %A')
    print('System at sleep :), will wake up again @ ',get_up_time)
    return sleep_secs


def market_hours(open_time = datetime.time(9, 0, 0), close_time= datetime.time(15, 29, 59)):
    date_today = datetime.datetime.today().date()
    day_today = time.strftime("%A", time.localtime())
    if date_today in nse_calendar or day_today in ['Saturday', 'Sunday']:
        return 60*60*8
    
    market_time_open1 = datetime.datetime.today().time() > open_time
    market_time_open2 = datetime.datetime.today().time() < close_time
    if market_time_open1 and market_time_open2:
        return 'OPEN'

    current = datetime.datetime.time(datetime.datetime.now())
    diff = datetime.datetime.combine(datetime.date.today(), open_time) - datetime.datetime.combine(datetime.date.today(), current)
    sleep_secs = diff.seconds + 5
    get_up_time=datetime.datetime.fromtimestamp(sleep_secs+time.time()).strftime('%Y-%m-%d %H:%M:%S %A')
    print('System at sleep :), will wake up again @ ',get_up_time)
    return sleep_secs

def wake_up_time(wakeup_at = datetime.time(9, 15, 0)):
    current = datetime.datetime.time(datetime.datetime.now())
    if datetime.datetime.combine(datetime.date.today(), wakeup_at) > datetime.datetime.combine(datetime.date.today(), current):
        diff = datetime.datetime.combine(datetime.date.today(), wakeup_at) - datetime.datetime.combine(datetime.date.today(), current)
    else:
        diff = datetime.datetime.combine(datetime.date.today(), wakeup_at) - datetime.datetime.combine(datetime.date.today(), current) + datetime.timedelta(days=1)
    sleep_secs = diff.seconds + .51
    get_up_time=datetime.datetime.fromtimestamp(sleep_secs+time.time()).strftime('%Y-%m-%d %H:%M:%S %A')
    print('System at sleep :), will wake up again @ ',get_up_time)
    return sleep_secs


def IsDateExpiry(date, Index):
    ExpiryToday = False
    year = datetime.datetime.today().date().year

    if Index in ['BN', 'BANKNIFTY'] :
        # Accomodating for Banknifty expiry change from Thursday to Wednesday
        if date < datetime.date(2023, 9, 4):
            bn_expiry = defs.THU
        else:
            bn_expiry = defs.BNEXPIRY
        
        FirstExpiryDate = datetime.date(year, 1, 1)
        while True :
            if FirstExpiryDate.weekday() == bn_expiry:
                break
            else:
                FirstExpiryDate += datetime.timedelta(days=1)

        ExpiryList = pd.date_range(start=FirstExpiryDate, end=datetime.date(year, 12, 31), freq='7D').date
        if date in ExpiryList and date not in holidays :
            ExpiryToday = True
        elif ([x for x in ExpiryList if x > date][0] in holidays) :
            DayBeforeExpiry = [x for x in ExpiryList if x > date][0] - datetime.timedelta(days=1)
            while True:
                if DayBeforeExpiry in holidays or DayBeforeExpiry.weekday() in [defs.SAT, defs.SUN] :
                    DayBeforeExpiry -= datetime.timedelta(days=1)
                else:
                    break
            if date == DayBeforeExpiry :
                ExpiryToday = True

    if Index in ['N', 'NIFTY']:
        FirstExpiryDate = datetime.date(year, 1, 1)
        while True :
            if FirstExpiryDate.weekday() == defs.NEXPIRY:
                break
            else:
                FirstExpiryDate += datetime.timedelta(days=1)
        ExpiryList = pd.date_range(start=FirstExpiryDate, end=datetime.date(year, 12, 31), freq='7D').date
        if date in ExpiryList and date not in holidays :
            ExpiryToday = True
        elif ([x for x in ExpiryList if x > date][0] in holidays) :
            DayBeforeExpiry = [x for x in ExpiryList if x > date][0] - datetime.timedelta(days=1)
            while True:
                if DayBeforeExpiry in holidays or DayBeforeExpiry.weekday() in [defs.SAT, defs.SUN] :
                    DayBeforeExpiry = DayBeforeExpiry - datetime.timedelta(days=1)
                else:
                    break
            if date == DayBeforeExpiry :
                ExpiryToday = True
    
    if Index in ['FN', 'FINNIFTY']:
        FirstExpiryDate = datetime.date(year, 1, 1)
        while True :
            if FirstExpiryDate.weekday() == defs.FNEXPIRY:
                break
            else:
                FirstExpiryDate += datetime.timedelta(days=1)
        ExpiryList =  pd.date_range(start=FirstExpiryDate, end=datetime.date(year, 12, 31), freq='7D').date
        if date in ExpiryList and date not in holidays :
            ExpiryToday = True
        elif ([x for x in ExpiryList if x > date][0] in holidays) :
            DayBeforeExpiry = [x for x in ExpiryList if x > date][0] - datetime.timedelta(days=1)
            while True:
                if DayBeforeExpiry in holidays or DayBeforeExpiry.weekday() in [defs.SAT, defs.SUN] :
                    DayBeforeExpiry = DayBeforeExpiry - datetime.timedelta(days=1)
                else:
                    break
            if date == DayBeforeExpiry :
                ExpiryToday = True

    return ExpiryToday

def IsDateAfterExpiry(date, Index):
    PrevDate = date - datetime.timedelta(days=1)
    while True:
        if PrevDate in holidays or PrevDate.weekday() in [defs.SAT, defs.SUN]:
            PrevDate -= datetime.timedelta(days=1)
        else:
            break
    if IsDateExpiry(PrevDate, Index):
        return True
    else:
        return False
    
def FindExpiry(date, Index, expiry_num):
    for date in pd.date_range(start=date, end=datetime.date(date.year, 12, 31), freq='1D').date:
        if IsDateExpiry(date, Index):
            expiry_num -= 1
        if expiry_num == 0:
            break
    return date

import datetime

nse2019holidays = [
    datetime.date(2019, 3, 4),
    datetime.date(2019, 3, 21),
    datetime.date(2019, 4, 17),
    datetime.date(2019, 4, 19),
    datetime.date(2019, 5, 1),
    datetime.date(2019, 6, 5),
    datetime.date(2019, 8, 12),
    datetime.date(2019, 8, 15),
    datetime.date(2019, 9, 2),
    datetime.date(2019, 9, 10),
    datetime.date(2019, 10, 2),
    datetime.date(2019, 10, 8),
    datetime.date(2019, 10, 28),
    datetime.date(2019, 11, 12),
    datetime.date(2019, 12, 25),
]

nse2022holidays = [datetime.date(2022, 1, 26),
                  datetime.date(2022, 3, 1),
                  datetime.date(2022, 3, 18),
                  datetime.date(2022, 4, 14),
                  datetime.date(2022, 4, 15),
                  datetime.date(2022, 5, 3),
                  datetime.date(2022, 8, 9),
                  datetime.date(2022, 8, 15),
                  datetime.date(2022, 8, 31),
                  datetime.date(2022, 10, 5),
                  datetime.date(2022, 10, 24),
                  datetime.date(2022, 10, 26),
                  datetime.date(2022, 11, 8)]

nse2021holidays = [datetime.date(2021, 1, 26),
                  datetime.date(2021, 3, 11),
                  datetime.date(2021, 3, 29),
                  datetime.date(2021, 4, 2),
                  datetime.date(2021, 4, 14),
                  datetime.date(2021, 4, 21),
                  datetime.date(2021, 5, 13),
                  datetime.date(2021, 7, 21),
                  datetime.date(2021, 8, 19),
                  datetime.date(2021, 9, 10),
                  datetime.date(2021, 10, 15),
                  datetime.date(2021, 11, 4),
                  datetime.date(2021, 11, 5),
                  datetime.date(2021, 11, 19)]

nse2023holidays = [datetime.date(2023, 1, 26),
                    datetime.date(2023, 3, 7),
                    datetime.date(2023, 3, 30),
                    datetime.date(2023, 4, 4),
                    datetime.date(2023, 4, 7),
                    datetime.date(2023, 4, 14),
                    datetime.date(2023, 5, 1),
                    datetime.date(2023, 6, 29),
                    datetime.date(2023, 8, 15),
                    datetime.date(2023, 9, 19),
                    datetime.date(2023, 10, 2),
                    datetime.date(2023, 10, 24),
                    datetime.date(2023, 11, 14),
                    datetime.date(2023, 11, 27),
                    datetime.date(2023, 12, 25)]

nse2024holidays = [datetime.date(2024, 1, 22),
                    datetime.date(2024, 1, 26),
                    datetime.date(2024, 3, 8),
                    datetime.date(2024, 3, 25),
                    datetime.date(2024, 3, 29),
                    datetime.date(2024, 4, 11),
                    datetime.date(2024, 4, 17),
                    datetime.date(2024, 5, 1),
                    datetime.date(2024, 5, 20),
                    datetime.date(2024, 6, 17),
                    datetime.date(2024, 7, 17),
                    datetime.date(2024, 8, 15),
                    datetime.date(2024, 10, 2),
                    datetime.date(2024, 11, 1),
                    datetime.date(2024, 11, 15),
                    datetime.date(2024, 12, 25)]

nse2025holidays = [
    datetime.date(2025, 2, 26)  ,
                datetime.date(2025, 3, 14)	,
                datetime.date(2025, 3, 31)	,
                datetime.date(2025, 4, 10)	,
                datetime.date(2025, 4, 14)	,
                datetime.date(2025, 4, 18)	,
                datetime.date(1015, 5, 1)	,
                datetime.date(2025, 8, 15)	,
                datetime.date(2025, 8, 27)	,
                datetime.date(2025, 10, 2)	,
                datetime.date(2025, 10, 21)	,
                datetime.date(2025, 10, 22)	,
                datetime.date(2025, 11, 5)	,
                datetime.date(2025, 12, 25)	,
]

nse2020holidays = [
    datetime.date(2020, 2, 21),  # Mahashivratri
    datetime.date(2020, 3, 10),  # Holi
    datetime.date(2020, 3, 29),  # Holi (second session)
    datetime.date(2020, 4, 2),   # Ram Navami
    datetime.date(2020, 4, 6),   # Mahavir Jayanti
    datetime.date(2020, 4, 10),  # Good Friday
    datetime.date(2020, 4, 14),  # Dr. B. R. Ambedkar Jayanti
    datetime.date(2020, 5, 1),   # Maharashtra Day
    datetime.date(2020, 5, 25),  # Id-Ul-Fitr (Ramzan Id)
    datetime.date(2020, 10, 2),  # Mahatma Gandhi Jayanti
    datetime.date(2020, 11, 16), # Diwali – Balipratipada
    datetime.date(2020, 11, 30), # Gurunanak Jayanti
    datetime.date(2020, 12, 25), # Christmas
]



holidays = nse2019holidays+nse2020holidays+nse2021holidays+nse2022holidays+nse2023holidays+nse2024holidays+nse2025holidays


fno_tickers = ['AARTIIND', 'ABB', 'ABBOTINDIA', 'ABCAPITAL', 'ABFRL', 'ACC',
                'ADANIENT', 'ADANIPORTS', 'ALKEM', 'AMBUJACEM', 'APOLLOHOSP',
                'APOLLOTYRE', 'ASHOKLEY', 'ASIANPAINT', 'ASTRAL', 'ATUL', 'AUBANK',
                'AUROPHARMA', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJFINANCE',
                'BALKRISIND', 'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA',
                'BANKNIFTY', 'BATAINDIA', 'BEL', 'BERGEPAINT', 'BHARATFORG',
                'BHARTIARTL', 'BHEL', 'BIOCON', 'BOSCHLTD', 'BPCL', 'BRITANNIA',
                'BSOFT', 'CANBK', 'CANFINHOME', 'CHAMBLFERT', 'CHOLAFIN', 'CIPLA',
                'COALINDIA', 'COFORGE', 'COLPAL', 'CONCOR', 'COROMANDEL',
                'CROMPTON', 'CUB', 'CUMMINSIND', 'DABUR', 'DALBHARAT', 'DEEPAKNTR',
                'DELTACORP', 'DIVISLAB', 'DIXON', 'DLF', 'DRREDDY', 'EICHERMOT',
                'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'FINNIFTY', 'FSL', 'GAIL',
                'GLENMARK', 'GMRINFRA', 'GNFC', 'GODREJCP', 'GODREJPROP',
                'GRANULES', 'GRASIM', 'GUJGASLTD', 'HAL', 'HAVELLS', 'HCLTECH',
                'HDFC', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO',
                'HINDALCO', 'HINDCOPPER', 'HINDPETRO', 'HINDUNILVR', 'HONAUT',
                'IBULHSGFIN', 'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDEA', 'IDFC',
                'IDFCFIRSTB', 'IEX', 'IGL', 'INDHOTEL', 'INDIACEM', 'INDIAMART',
                'INDIGO', 'INDUSINDBK', 'INDUSTOWER', 'INFY', 'INTELLECT', 'IOC',
                'IPCALAB', 'IRCTC', 'ITC', 'JINDALSTEL', 'JKCEMENT', 'JSWSTEEL',
                'JUBLFOOD', 'KOTAKBANK', 'L&TFH', 'LALPATHLAB', 'LAURUSLABS',
                'LICHSGFIN', 'LT', 'LTIM', 'LTTS', 'LUPIN', 'M&M', 'M&MFIN',
                'MANAPPURAM', 'MARICO', 'MARUTI', 'MCDOWELL-N', 'MCX',
                'METROPOLIS', 'MFSL', 'MGL', 'MIDCPNIFTY', 'MOTHERSON', 'MPHASIS',
                'MRF', 'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI', 'NAVINFLUOR',
                'NESTLEIND', 'NIFTY', 'NMDC', 'NTPC', 'OBEROIRLTY', 'OFSS', 'ONGC',
                'PAGEIND', 'PEL', 'PERSISTENT', 'PETRONET', 'PFC', 'PIDILITIND',
                'PIIND', 'PNB', 'POLYCAB', 'POWERGRID', 'PVR', 'RAIN', 'RAMCOCEM',
                'RBLBANK', 'RECLTD', 'RELIANCE', 'SAIL', 'SBICARD', 'SBILIFE',
                'SBIN', 'SHREECEM', 'SHRIRAMFIN', 'SIEMENS', 'SRF', 'SUNPHARMA',
                'SUNTV', 'SYNGENE', 'TATACHEM', 'TATACOMM', 'TATACONSUM',
                'TATAMOTORS', 'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN',
                'TORNTPHARM', 'TORNTPOWER', 'TRENT', 'TVSMOTOR', 'UBL',
                'ULTRACEMCO', 'UPL', 'VEDL', 'VOLTAS', 'WHIRLPOOL', 'WIPRO',
                'ZEEL', 'ZYDUSLIFE']


nifty_tickers = ['ADANIENT',  'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
                'BAJAJFINSV',  'BEL', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GRASIM',
                'HCLTECH',  'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC', 'INDUSINDBK',
                'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE',
                'SBILIFE',  'SHRIRAMFIN', 'SBIN', 'SUNPHARMA',  'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN',
                'TRENT', 'ULTRACEMCO', 'WIPRO']

banknifty_tickers = ['AUBANK',
                    'AXISBANK',
                    'BANKBARODA',
                    'CANBK',
                    'FEDERALBNK',
                    'HDFCBANK',
                    'ICICIBANK',
                    'IDFCFIRSTB',
                    'INDUSINDBK',
                    'KOTAKBANK',
                    'PNB',
                    'SBIN']


finnifty_tickers = ['AXISBANK',
                    'BAJFINANCE',
                    'BAJAJFINSV',
                    'CHOLAFIN',
                    'HDFCAMC',
                    'HDFCBANK',
                    'HDFCLIFE',
                    'ICICIBANK',
                    'ICICIGI',
                    'ICICIPRULI',
                    'KOTAKBANK',
                    'LICHSGFIN',
                    'MCX',
                    'MUTHOOTFIN',
                    'PFC',
                    'RECLTD',
                    'SBICARD',
                    'SBILIFE',
                    'SHRIRAMFIN',
                    'SBIN']

midcap_tickers = ['AUBANK',
                'ASHOKLEY',
                'AUROPHARMA',
                'BHARATFORG',
                'COFORGE',
                'COLPAL',
                'CONCOR',
                'CUMMINSIND',
                'DIXON',
                'FEDERALBNK',
                'GODREJPROP',
                'HDFCAMC',
                'HINDPETRO',
                'IDFCFIRSTB',
                'INDHOTEL',
                'INDUSTOWER',
                'LUPIN',
                'MRF',
                'MPHASIS',
                'PIIND',
                'PERSISTENT',
                'POLYCAB',
                'SRF',
                'IDEA',
                'VOLTAS']

index_tickers = nifty_tickers + banknifty_tickers + finnifty_tickers + midcap_tickers
stock_tickers = list(pd.Series(index_tickers).unique())


lot_size_dict = {
    # INDEX
    'BANKNIFTY' :15,
    'NIFTY': 75,
    'FINNIFTY': 65,
    'MIDCPNIFTY': 50,
    'SENSEX': 20,

    # STOCKS
    'ADANIENT': 300,
    'ADANIPORTS': 400,
    'APOLLOHOSP': 125,
    'ASIANPAINT': 200,
    'AUBANK': 1000,
    'AXISBANK': 625,
    'BAJAJFINSV': 500,
    'BAJFINANCE': 125,
    'BANDHANBNK': 2800,
    'BANKBARODA': 2925,
    'BHARTIARTL': 475,
    'BRITANNIA': 200,
    'CHOLAFIN': 625,
    'CIPLA': 650,
    'COALINDIA': 2100,
    'DIVISLAB': 200,
    'DRREDDY': 125,
    'EICHERMOT': 175,
    'FEDERALBNK': 5000,
    'GRASIM': 250,
    'HCLTECH': 350,
    'HDFCAMC': 150,
    'HDFCBANK': 550,
    'HDFCLIFE': 1100,
    'HEROMOTOCO': 150,
    'HINDALCO': 1400,
    'HINDUNILVR': 300,
    'ICICIBANK': 700,
    'ICICIGI': 500,
    'ICICIPRULI': 1500,
    'IDFC': 5000,
    'IDFCFIRSTB': 7500,
    'INDUSINDBK': 500,
    'INFY': 400,
    'ITC': 1600,
    'JSWSTEEL': 675,
    'KOTAKBANK': 400,
    'LICHSGFIN': 1000,
    'LT': 150,
    'MARUTI': 50,
    'MUTHOOTFIN': 550,
    'NESTLEIND': 200,
    'ONGC': 1925,
    'PFC': 1300,
    'PNB': 8000,
    'POWERGRID': 3600,
    'RECLTD': 2000,
    'RELIANCE': 250,
    'SBICARD': 800,
    'SBILIFE': 375,
    'SBIN': 750,
    'SHREECEM': 25,
    'SHRIRAMFIN': 300,
    'SUNPHARMA': 350,
    'TATACONSUM': 456,
    'TATAMOTORS': 550,
    'TATASTEEL': 5500,
    'TCS': 175,
    'TECHM': 600,
    'TITAN': 175,
    'ULTRACEMCO': 100,
    'UPL': 1300,
    'VEDL': 2300,
    'WIPRO': 1500
}

freeze_qty_dict = {
  'BANKNIFTY': 900,
  'NIFTY': 1800,
  'FINNIFTY': 1800,
  'MIDCPNIFTY': 4200,
  'SENSEX': 1000
}

strike_diff_dict = {
    # INDEX
    'BANKNIFTY': 100 ,
    'NIFTY': 50,
    'FINNIFTY': 50,
    'MIDCPNIFTY': 25,
    'SENSEX': 100,
    'BANKEX': 100,

    # STOCKS
    'RELIANCE': 20,
    'TCS': 50,
    'INFY': 20,
    'HDFCBANK': 10,
    'ICICIBANK': 10,
    'HINDUNILVR': 20,
    'KOTAKBANK': 20,
    'SBIN': 10,
    'BHARTIARTL': 20,
    'ITC': 5,
    'ASIANPAINT': 20,
    'BAJFINANCE': 100,
    'MARUTI': 100,
    'AXISBANK': 10,
    'LT': 50,
    'HCLTECH': 20,
    'SUNPHARMA': 20,
    'WIPRO': 5,
    'ULTRACEMCO': 100,
    'TITAN': 10,
    'TECHM': 20,
    'NESTLEIND': 20,
    'JSWSTEEL': 10,
    'TATASTEEL': 2,
    'POWERGRID': 5,
    'ONGC': 5,
    'COALINDIA': 5,
    'INDUSINDBK': 20,
    'BAJAJFINSV': 20,
    'GRASIM': 20,
    'CIPLA': 20,
    'ADANIPORTS': 20,
    'TATAMOTORS': 10,
    'DRREDDY': 50,
    'BRITANNIA': 50,
    'HEROMOTOCO': 100,
    'DIVISLAB': 50,
    'EICHERMOT': 50,
    'SHREECEM': 250,
    'APOLLOHOSP': 50,
    'UPL': 5,
    'TATACONSUM': 10,
    'HINDALCO': 10,
    'SBILIFE': 20,
    'VEDL': 10,
    'BANKBARODA': 2,
    'FEDERALBNK': 2,
    'PNB': 0,
    'IDFCFIRSTB': 1,
    'AUBANK': 10,
    'BANDHANBNK': 2,
    'RECLTD': 10,
    'ICICIPRULI': 10,
    'ICICIGI': 20,
    'HDFCLIFE': 5,
    'LICHSGFIN': 10,
    'CHOLAFIN': 20,
    'PFC': 10,
    'SHRIRAMFIN': 50,
    'HDFCAMC': 50,
    'IDFC': 1,
    'SBICARD': 5,
    'MUTHOOTFIN': 20
}


# GENERAL PARAMS SPACE
weekdays = range(0, 5)
timeframes = range(1, 6)
underlyings = ['BANKNIFTY', 'NIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX'] # ,'BANKNIFTY', 'FINNIFTY'
selectors = ['P', 'PCT', 'M'] # , 'M'
moneynesses = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
seek_prices = [25, 50, 75, 100, 125, 150] #[50,75,100,125,150,175,200,250,300,350,400]
hedge_shifts = [10, 15, 20]
sl_pcts = [0.1,0.2,0.3,0.4,0.5,0.75]
tgt_pcts = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
resets = [1, 2, 4, 5, 6, 10, 15]
delays = range(30, 300, 30)
# t = [1, 3, 5, 7, 9, 11, 13, 15]
normalizers = [0.0005,0.00075,0.001,0.00]
