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
from backtest.engine import BacktestEngine
from strategies.orb import ORBStrategy

engine = BacktestEngine(
    symbol="XAUUSD",
    timeframe="5m",
    initial_capital=100_000
)

results = engine.run(
    strategy=ORBStrategy,
    params={
        "orb_minutes": 5,
        "stop_loss_pct": 0.3,
        "take_profit_pct": 0.6
    },
    session=("07:00", "23:00")
)

print(results.summary)
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
