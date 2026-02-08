# Pluto – Backtesting Engine 🚀

Pluto_test is a **Python-based quantitative backtesting engine** built for systematic traders and researchers.  
It focuses on **clean architecture, reproducibility, and fast experimentation** for intraday and swing trading strategies.

This repository is an active research sandbox for building, testing, and optimizing rule-based trading systems.

---

## 📂 Repository Structure

```
Pluto_test/
│
├── backtest/               # Backtest orchestration & control flow
├── engine/                 # Core execution engine (entries, exits, positions)
├── strategies/             # Strategy logic (ORB, mean reversion, etc.)
├── metrics/                # Performance & risk metrics
├── optimization/           # Parameter sweeps / grid search
├── utils/                  # Helper utilities
├── benchmark_results/      # Stored backtest & optimization outputs
├── cython_modules/         # Optional performance-optimized modules
├── tests/                  # Test cases
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🧠 Design Philosophy

- **Tradebook-first**: All metrics are derived from executed trades
- **Deterministic runs**: Same inputs → same outputs
- **Strategy isolation**: Each configuration is evaluated independently
- **Research-focused**: Built for testing ideas, not live trading

---

## ✨ Key Features

- Event-driven backtesting loop  
- Intraday & multi-day strategy support  
- Strict session / time-window controls  
- Parameter optimization via grid search  
- Detailed trade logs & benchmark results  
- Clean separation of data, logic, execution, and analytics  
- Optional Cython acceleration  

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/chrislernunes/Pluto.git
cd Pluto
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(Optional) Build Cython modules:

```bash
python setup.py build_ext --inplace
```

---

## 🚀 Running a Backtest

Example (conceptual):

```python
import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from utils.definitions import *
from utils.sessions import *
import direct_redis, math

if REDIS:
    from engine.ems import EventInterfacePositional
else:
    from engine.ems_db import EventInterfacePositional

r = direct_redis.DirectRedis()


class BTSTDIRV2(EventInterfacePositional):
    
    def __init__(self):
        super().__init__()
        self.strat_id = self.__class__.__name__.lower()

        self.position_ce = 0
        self.position_pe = 0

        self.symbol_ce = None
        self.prices_ce = []

        self.symbol_pe = None
        self.prices_pe = []

        self.symbol_ce_hedge = None
        self.symbol_pe_hedge = None
        self.last_active_date = None

        self.sl_updated_ce = False
        self.sl_updated_pe = False

    def get_random_uid(self):
        # Select
        self.active_weekday = 99#np.random.choice(weekdays)
        self.session = np.rand.random.choice(timeframes)
        self.underlying = np.random.choice(['MIDCPNIFTY']) # 'NIFTY', 'FINNIFTY', 'BANKNIFTY', 'SENSEX', 
        self.selector = 'P' # nom.choice(['x0'])
        self.timeframe = 1 #npp.random.choice(selectors)
        if self.selector == 'M':
            self.selector_val = np.random.choice(moneynesses)
        elif self.selector == 'P':
            self.selector_val = np.random.choice(range(5, 20, 5)) # np.random.choice([15, 25, 50, 75, 100])
        # self.hedge_shift = np.random.choice(hedge_shifts)

        self.sl_pct = round(np.random.choice(np.arange(0.3, 0.5, 0.05)), 2) #round(.05 * round(np.random.choice(np.random.rand(10)*0.5).round(2)/.05), 2)
        self.tgt_pct = round(np.random.choice(np.arange(0.6, 0.9, 0.05)), 2) #round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2) #np.random.choice(tgt_pcts)
        self.max_reset = np.random.choice([0,1])
        self.trail_on = np.random.choice([True])
        self.delay = np.random.choice(range(0, 120, 30))
        # ...
        if self.session in ['x0', 'x1', 'x2', 'y0', 't1']:
            orb_sizes = [15, 30, 45, 60, 75, 90]
        else:
            orb_sizes = [5, 10, 15, 20, 25, 30]
        self.orb_size = np.random.choice(orb_sizes)
        self.breakout_factor = round(np.random.choice(np.arange(1.0, 1.5, 0.05)), 2)
        
        self.ohlc = np.random.choice(['o', 'c'])
        self.delay_exit = np.random.choice(range(0, 10, 1))
        self.strat_type = np.random.choice(['r', 'n']) # r - Roll over at EOD , n - directly enter next expiry
        self.trail_pct = np.random.choice([0.05, 0.025, 0.01])
        # ...
        return self.get_uid_from_params()

    def set_params_from_uid(self, uid):
        s = uid.split('_')
        try:
            print(s[0], self.strat_id)
            assert s[0] == self.strat_id
        except AssertionError:
            raise ValueError(f'Invalid UID {uid} for strat ID {self.strat_id}')
        s = s[1:]
        self.active_weekday = int(s.pop(0))
        self.session = s.pop(0)
        self.delay = int(s.pop(0))#=='True'
        self.timeframe = int(s.pop(0))
        self.underlying = s.pop(0)
        self.selector = s.pop(0)
        self.selector_val = int(s.pop(0))
        # self.hedge_shift = int(s.pop(0))
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.max_reset = int(s.pop(0))
        self.trail_on = s.pop(0)=='True'
        # ...
        self.orb_size = int(s.pop(0))
        self.breakout_factor = float(s.pop(0))
        self.ohlc = s.pop(0)
        self.delay_exit = int(s.pop(0))
        self.strat_type = s.pop(0)
        self.trail_pct = float(s.pop(0))
        self.roll_or_no=s.pop(0)=='True'
        # self.system_tag = s.pop(0)

        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        assert uid == self.gen_uid
        self.uid = uid
        print(self.uid)
    
    def get_uid_from_params(self):
        return f"""
        {self.strat_id}_
        {self.active_weekday}_
        {self.session}_
        {self.delay}_
        {self.timeframe}_
        {self.underlying}_
        {self.selector}_
        {self.selector_val}_
        {self.sl_pct}_
        {self.tgt_pct}_
        {self.max_reset}_
        {self.trail_on}_
        {self.orb_size}_
        {self.breakout_factor}_
        {self.ohlc}_
        {self.delay_exit}_
        {self.strat_type}_
        {self.trail_pct}_
        {self.roll_or_no}
        """.replace('\n', '').replace(' ', '').strip('_')
```

> Exact APIs may vary as the engine evolves.

---

## 📊 Metrics Included

- Total PnL  
- Win rate  
- Expectancy  
- Max & average drawdown  
- Sharpe & Sortino ratios  
- Best / worst trades  
- Trade frequency  

All metrics are calculated **from the tradebook**, not candle-level assumptions.

---

## 🔍 Optimization

Pluto_test supports parameter sweeps for strategy research:

- Grid search over strategy parameters
- Stored benchmark results
- Easy comparison across variants

Ideal for robustness testing and drawdown control research.

---

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Tests cover execution logic, metrics accuracy, and strategy behavior.

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.  
It is **not** financial advice and is **not intended for live trading**.

Past performance does not guarantee future results.

---

## 🌌 Final Note

Pluto_test is built to answer one question:

**“Does this idea actually work — under real constraints?”**

If you can define your edge in rules, Pluto can test it.

Happy researching 📈
