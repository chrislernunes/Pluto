import pandas as pd
import numpy as np

SLIPPAGE_RATE = 0.01  # 1% slippage applied to each exit trade
def apply_slippage(tradebook: pd.DataFrame) -> pd.DataFrame:
    """
    Applies slippage to exit trades by adjusting the 'value' column.
    Slippage = SLIPPAGE_RATE * abs(price * qty)
    """
    df = tradebook.copy()
    slippage_cost = SLIPPAGE_RATE * (df['turnover'].abs())
    df['value'] -= slippage_cost
    return df

def apply_brokerage(tradebook: pd.DataFrame, brokerage:float) -> pd.DataFrame:
    """
    Applies slippage to exit trades by adjusting the 'value' column.
    Slippage = SLIPPAGE_RATE * abs(price * qty)
    """
    df = tradebook.copy()
    df['value'] -= brokerage
    return df


# ───────────────────────────────────────────────────────────────
# DAY-LEVEL METRICS
# ───────────────────────────────────────────────────────────────


def compute_daily_pnl(tradebook: pd.DataFrame) -> pd.Series:
    daily_pnl = tradebook.groupby(tradebook['date'])['value'].sum()    
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    return daily_pnl

def pnl_by_weekday(daily_pnl: pd.Series) -> pd.Series:
    """
    Computes total PNL aggregated by day of the week (Monday–Sunday).
    """
    weekday_pnl = daily_pnl.groupby(daily_pnl.index.day_name()).sum()
    # Reorder to standard weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_pnl = weekday_pnl.reindex(weekday_order).dropna()
    return weekday_pnl

def total_pnl(daily_pnl: pd.Series) -> float:
    return daily_pnl.sum()

def sharpe_ratio(daily_pnl: pd.Series, risk_free_rate=0.0) -> float:
    excess_returns = daily_pnl - risk_free_rate / 252
    return excess_returns.mean() / (excess_returns.std() + 1e-9) * np.sqrt(252)

def max_drawdown(daily_pnl: pd.Series) -> float:
    equity = daily_pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return drawdown.min()


def avg_drawdown(daily_pnl: pd.Series) -> float:
    equity = daily_pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return drawdown.mean()

def max_drawdown_duration(daily_pnl: pd.Series) -> int:
    equity = daily_pnl.cumsum()
    max_duration = 0
    peak_time = equity.index[0]
    peak_value = equity.iloc[0]
    for time, value in zip(equity.index, equity):
        if value > peak_value:
            peak_value = value
            peak_time = time
        else:
            duration = (time - peak_time).days
            max_duration = max(max_duration, duration)
    return max_duration

def max_drawdown_pct(daily_pnl: pd.Series, initial_capital: float = 1.0) -> float:    
    
    # Cum-equity curve
    equity = initial_capital + daily_pnl.cumsum()
    # Running peaks then drawdowns
    rolling_peak = equity.cummax()
    drawdown     = equity / rolling_peak - 1.0          # already in -pct form

    return drawdown.min()                         # % and negative

def calmar_ratio(daily_pnl: pd.Series) -> float:
    total_days = (daily_pnl.index[-1] - daily_pnl.index[0]).days
    annual_return = daily_pnl.sum() / (total_days / 365.0 + 1e-9)
    max_dd = abs(max_drawdown(daily_pnl))
    return annual_return / (max_dd + 1e-9)

def cagr(daily_pnl: pd.Series, margin: float) -> float:
    """
    Computes CAGR based on cumulative PNL and initial capital (margin).
    
    Args:
        daily_pnl (pd.Series): Series of daily PNL values indexed by date (must be datetime).
        margin (float): Starting capital or margin used to generate the returns.
    
    Returns:
        float: Compounded Annual Growth Rate (CAGR)
    """
    if margin <= 0 or daily_pnl.empty:
        return 0.0
    equity = daily_pnl.cumsum()
    total_days = (daily_pnl.index[-1] - daily_pnl.index[0]).days
    years = total_days / 365.0
    total_return = equity.iloc[-1] / margin 
    return (1 + total_return) ** (1 / (years + 1e-9)) - 1

def win_days_pct(daily_pnl: pd.Series) -> float:
    return (daily_pnl > 0).sum() / (len(daily_pnl) + 1e-9)

def lose_days_pct(daily_pnl: pd.Series) -> float:
    return (daily_pnl < 0).sum() / (len(daily_pnl) + 1e-9)

def average_profit_per_day(daily_pnl: pd.Series) -> float:
    return daily_pnl.mean()

def payoff_ratio(daily_pnl: pd.Series) -> float:
    avg_gain = daily_pnl[daily_pnl > 0].mean()
    avg_loss = abs(daily_pnl[daily_pnl < 0].mean())
    return avg_gain / (avg_loss + 1e-9)

def average_negative_day_pnl_pct(daily_pnl: pd.Series, margin: float) -> float:
    """Average PnL % on days with losses (negative daily PnL only)."""
    eq = margin + daily_pnl.cumsum()                     # end-of-day equity
    pct = daily_pnl.div(eq.shift(fill_value=margin))
    return pct[daily_pnl < 0].mean()                      # NaN if no loss days


def average_negative_day_pnl(daily_pnl: pd.Series) -> float:
    """Average PnL on days with losses (negative daily PnL only)."""
    losses = daily_pnl[daily_pnl < 0]
    return losses.mean() if not losses.empty else 0.0

def average_positive_day_pnl_pct(daily_pnl: pd.Series, margin: float) -> float:
    """Average PnL % on days with profits (positive daily PnL only)."""
    eq = margin + daily_pnl.cumsum()                     # end-of-day equity
    pct = daily_pnl.div(eq.shift(fill_value=margin))
    return pct[daily_pnl > 0].mean()                      # NaN if no loss days

def average_positive_day_pnl(daily_pnl: pd.Series) -> float:
    """Average PnL on days with gains (positive daily PnL only)."""
    gains = daily_pnl[daily_pnl > 0]
    return gains.mean() if not gains.empty else 0.0

def worst_single_day_pnl(daily_pnl: pd.Series) -> float:
    """Worst (most negative) single-day PnL."""
    return daily_pnl.min() if not daily_pnl.empty else 0.0

def worst_single_day_pnl_pct(daily_pnl: pd.Series, margin: float) -> float:
    """Average PnL % on days with profits (positive daily PnL only)."""
    eq = margin + daily_pnl.cumsum()                     # end-of-day equity
    pct = daily_pnl.div(eq.shift(fill_value=margin))
    return pct.min()                      # NaN if no loss days

def best_single_day_pnl_pct(daily_pnl: pd.Series, margin: float) -> float:
    """Average PnL % on days with profits (positive daily PnL only)."""
    eq = margin + daily_pnl.cumsum()                     # end-of-day equity
    pct = daily_pnl.div(eq.shift(fill_value=margin))
    return pct.max()                      # NaN if no loss days

def yearly_pnl(daily_pnl: pd.Series) -> pd.Series:
    return daily_pnl.groupby(daily_pnl.index.to_period('Y')).sum()

def monthly_pnl(daily_pnl: pd.Series) -> pd.Series:
    return daily_pnl.groupby(daily_pnl.index.to_period('M')).sum()

def weekly_pnl(daily_pnl: pd.Series) -> pd.Series:
    return daily_pnl.groupby(daily_pnl.index.to_period('W')).sum()

def monthly_weekday_pnl(daily_pnl: pd.Series) -> dict:
    """
    Computes a nested dictionary of monthly PNL broken down by day of the week.
    Output format:
    {
        '2025-01': {'Monday': 1000.0, 'Tuesday': -500.0, ...},
        '2025-02': {'Monday': 300.0, 'Wednesday': 700.0, ...},
        ...
    }
    """
    df = daily_pnl.reset_index()
    df.columns = ['date', 'pnl']
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['weekday'] = df['date'].dt.day_name()
    grouped = df.groupby(['month', 'weekday'])['pnl'].sum().unstack(fill_value=0)
    # Convert to dict of dicts for JSON compatibility and readability
    return grouped.to_dict(orient='index')

def yearly_weekday_pnl(daily_pnl: pd.Series) -> dict:
    """
    Computes total PNL by weekday for each year.

    Returns:
        dict: {
            '2025': {'Monday': 1000.0, 'Tuesday': -500.0, ...},
            ...
        }
    """
    df = daily_pnl.reset_index()
    df.columns = ['date', 'pnl']
    df['year'] = df['date'].dt.to_period('Y').astype(str)
    df['weekday'] = df['date'].dt.day_name()
    result = df.groupby(['year', 'weekday'])['pnl'].sum().unstack(fill_value=0)
    return result.to_dict(orient='index')

import pandas as pd
import numpy as np

# ───────────────────────────────────────────────────────────────
# TRADE-LEVEL METRICS
# ───────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

def add_trade_id(tradebook: pd.DataFrame,
                 symbol_col: str = "symbol",
                 qty_col: str = "qty",
                 side_col: str = "action",
                 time_col: str = "ts") -> pd.DataFrame:
    """
    Annotate each fill with a `trade_id` that stays constant from
    the first opening fill until the position is closed (flat).

    Assumptions
    -----------
    • A “trade” = position goes from flat → non-zero → flat again (FIFO).
    • `side` column holds 'BUY' for +qty, 'SELL' for –qty.  
      If your conventions differ, adjust the np.where() line.

    Returns
    -------
    The *same* DataFrame instance with two new columns:
        signed_qty, trade_id
    """

    # Filtering out Psuedo Trades except the last one

    # 0)  make sure rows are in fill-time order
    tb  = tradebook.sort_values(time_col)

    # Find the trade_id of the last *real* trade (turnover ≠ 0).
    last_trade_ts = tb.loc[tb['turnover'] != 0, 'ts'].iloc[-1]

    # Build a mask: keep every real trade and rows after the last actual trade.
    mask = (tb['turnover'] != 0) | (tb['ts'] >= last_trade_ts)

    tradebook = tb.loc[mask].reset_index(drop=True)

    # 2)  Running position **after** each fill
    tradebook["pos_after"] = (
        tradebook.groupby(symbol_col)["qty_dir"]
                 .cumsum()
    )

    # 3)  Trade counter: increment *after* every time we return to flat (0)
    counter = (
        tradebook.groupby(symbol_col)["pos_after"]
                 .apply(lambda s: s.eq(0).cumsum().shift(fill_value=0))
                 .reset_index(level=0, drop=True)
    )

    # 4)  Concatenate for uniqueness across symbols
    tradebook["trade_id"] = (
        tradebook[symbol_col].astype(str) + "_" + counter.astype(int).astype(str)
    )

    return tradebook

def total_trades(tradebook: pd.DataFrame) -> int:
    """Count completed trades (rows) in the tradebook."""
    return len(tradebook[tradebook["turnover"] != 0])

def avg_trades_per_day(tradebook: pd.DataFrame) -> float:
    """Average number of trades executed per calendar day."""
    if tradebook.empty:
        return 0.0
    daily_pnl = tradebook.groupby(tradebook['date'])['value'].sum()    
    trade_days =  len(daily_pnl)
    return total_trades(tradebook) / (trade_days + 1e-9)

def count_ce_trades(tradebook: pd.DataFrame) -> int:
    """Number of Call-option trades."""
    return (tradebook['option_type'] == 'CE').sum()

def count_pe_trades(tradebook: pd.DataFrame) -> int:
    """Number of Put-option trades."""
    return (tradebook['option_type'] == 'PE').sum()

def profitable_trades_ce(tradebook: pd.DataFrame) -> int:
    """Count of Call trades with positive PnL."""
    mask = (tradebook['option_type'] == 'CE') & (tradebook['pnl'] > 0)
    return mask.sum()

def profitable_trades_pe(tradebook: pd.DataFrame) -> int:
    """Count of Put trades with positive PnL."""
    mask = (tradebook['option_type'] == 'PE') & (tradebook['pnl'] > 0)
    return mask.sum()

def average_pnl_percent_ce(tradebook: pd.DataFrame) -> float:
    """Mean % return for Call trades."""
    m = (tradebook["option_type"] == "CE") & (tradebook["turnover"] != 0)
    return (tradebook.loc[m, "pnl"] / tradebook.loc[m, "turnover"]).mean()
    
def average_pnl_percent_pe(tradebook: pd.DataFrame) -> float:
    """Mean % return for Call trades."""
    m = (tradebook["option_type"] == "PE") & (tradebook["turnover"] != 0)
    return (tradebook.loc[m, "pnl"] / tradebook.loc[m, "turnover"]).mean()
    
def trading_edge(tradebook: pd.DataFrame) -> float:

    if tradebook.empty or len(tradebook) == 0:
        return 0.0
    win_pct  = (tradebook['pnl'] > 0).sum()/len(tradebook)
    loss_pct  = (tradebook['pnl'] <= 0).sum()/len(tradebook)
    tradebook = tradebook[tradebook['turnover'] != 0].copy()
    tradebook["pnl_pct"] = tradebook["pnl"] / tradebook["turnover"]
    avg_win_pnl_pct = tradebook.loc[tradebook['pnl'] > 0, 'pnl_pct'].mean()
    avg_loss_pnl_pct = tradebook.loc[tradebook['pnl'] <= 0, 'pnl_pct'].mean()
    td_edge = (win_pct * avg_win_pnl_pct) + (loss_pct *avg_loss_pnl_pct)
    return td_edge

# ───────────────────────────────────────────────────────────────
# CALCULATING AND DISPLAYING METRICS
# ───────────────────────────────────────────────────────────────

def compute_all_metrics_daily(daily_pnl: pd.Series, margin: float) -> dict:
    return {
        "total_pnl": total_pnl(daily_pnl),
        "sharpe_ratio": sharpe_ratio(daily_pnl),
        "calmar_ratio": calmar_ratio(daily_pnl),
        "max_drawdown": max_drawdown(daily_pnl),
        "max_drawdown_duration": max_drawdown_duration(daily_pnl),
        "max_drawdown %": max_drawdown_pct(daily_pnl,margin),
        "cagr": cagr(daily_pnl, margin),
        "return_on_margin": total_pnl(daily_pnl) / (margin + 1e-9),
        "Win Days %": win_days_pct(daily_pnl),
        "Lose Days %": lose_days_pct(daily_pnl),
        "avg_profit_per_day": average_profit_per_day(daily_pnl),
        "payoff_ratio": payoff_ratio(daily_pnl),
        "Average Negative Day PnL %": average_negative_day_pnl_pct(daily_pnl,margin),
        "Average Positive Day PnL %": average_positive_day_pnl_pct(daily_pnl,margin),
        "Worst Single Day PnL %": worst_single_day_pnl_pct(daily_pnl,margin),
        "Best Single Day PnL %": best_single_day_pnl_pct(daily_pnl,margin),
        "Avg Negative Day PNL": average_negative_day_pnl(daily_pnl),
        "Avg Positive Day PNL": average_positive_day_pnl(daily_pnl),
        "Worst Single Day PNL": worst_single_day_pnl(daily_pnl),
        "Avg Drawdown": avg_drawdown(daily_pnl),
        "yearly_pnl": yearly_pnl(daily_pnl),
        "monthly_pnl": monthly_pnl(daily_pnl),
        "weekly_pnl": weekly_pnl(daily_pnl),
        "weekday_pnl": pnl_by_weekday(daily_pnl),
        "monthly_weekday_pnl": monthly_weekday_pnl(daily_pnl),
        "yearly_weekday_pnl": yearly_weekday_pnl(daily_pnl)
    }

def compute_all_metrics(tradebook, margin):
    """
    Compute all trading performance metrics after slippage is applied.
    """
    daily_pnl = compute_daily_pnl(tradebook)
    return compute_all_metrics_daily(daily_pnl, margin)

def compute_summary_metrics_daily(daily_pnl: pd.Series, margin: float) -> dict:
    return {
        "total_pnl": total_pnl(daily_pnl),
        "sharpe_ratio": sharpe_ratio(daily_pnl),
        "calmar_ratio": calmar_ratio(daily_pnl),
        "max_drawdown": max_drawdown(daily_pnl),
        "max_drawdown_duration": max_drawdown_duration(daily_pnl),
        "max_drawdown %": max_drawdown_pct(daily_pnl,margin),
        "cagr": cagr(daily_pnl, margin),
        "return_on_margin": total_pnl(daily_pnl) / (margin + 1e-9),
        "Win Days %": win_days_pct(daily_pnl),
        "Lose Days %": lose_days_pct(daily_pnl),
        "avg_profit_per_day": average_profit_per_day(daily_pnl),
        "payoff_ratio": payoff_ratio(daily_pnl),
        "Avg Negative Day PNL": average_negative_day_pnl(daily_pnl),
        "Avg Positive Day PNL": average_positive_day_pnl(daily_pnl),
        "Worst Single Day PNL": worst_single_day_pnl(daily_pnl),
        "Average Negative Day PnL %": average_negative_day_pnl_pct(daily_pnl,margin),
        "Average Positive Day PnL %": average_positive_day_pnl_pct(daily_pnl,margin),
        "Worst Single Day PnL %": worst_single_day_pnl_pct(daily_pnl,margin),
        "Best Single Day PnL %": best_single_day_pnl_pct(daily_pnl,margin),
        "Avg Drawdown": avg_drawdown(daily_pnl),
    }

def compute_trade_level_metrics(tradebook: pd.Series, margin: float) -> dict:
    combined_tradebook = (
        add_trade_id(tradebook)          # add a unique id per round-trip
        .sort_values('ts')      # make sure rows are in chronological order
        .groupby('trade_id')
        .agg(
            timestamp_entry = ('ts', 'first'),   # first time-stamp in the trade
            timestamp_exit  = ('ts', 'last'),    # last time-stamp in the trade
            value           = ('value',     'sum'),     # P&L or qty over the trade
            turnover        = ('turnover',  'first'),   # turnover recorded on the opening leg
            symbol_count    = ('symbol',    'count'),   # how many legs / fills
            underlying_spot_entry    = ('underlying',    'first'),   # how many legs / fills
            underlying_spot_exit    = ('underlying',    'last'),   # how many legs / fills
            entry_price     = ("value", "first"),  # ← Value at Entry
            exit_price     = ("value", "last"),  # ← Value at Exit
        ).reset_index()
    )
    uid = tradebook['uid'].iloc[0]
    combined_tradebook["option_type"] = combined_tradebook['trade_id'].str[-4:-2] 
    combined_tradebook["pnl"] = combined_tradebook['value']
    # combined_tradebook.to_csv(f"combined_tb_{uid}.csv", index=False)
    return {
        "uid": tradebook['uid'].iloc[0] if not tradebook.empty else None,
        "total_trades": total_trades(tradebook),
        "Avg Trades Per Day": avg_trades_per_day(tradebook),
        "Count of CE Trades": count_ce_trades(combined_tradebook),
        "Count of PE Trades": count_pe_trades(combined_tradebook),
        "Profitable Trades with CE": profitable_trades_ce(combined_tradebook),
        "Profitable Trades with PE": profitable_trades_pe(combined_tradebook),
        "Average CE Pnl %": average_pnl_percent_ce(combined_tradebook),
        "Average PE Pnl %": average_pnl_percent_pe(combined_tradebook),
        "Trading Edge": trading_edge(combined_tradebook)
        }

def compute_summary_metrics(tradebook, margin):
    """
    Compute all trading performance metrics after slippage is applied.
    """
    daily_pnl = compute_daily_pnl(tradebook)
    return compute_summary_metrics_daily(daily_pnl, margin)


def compute_perf_metrics(tradebook,margin):

    print("Calculating Trade Level Metrics")
    tl_metrics = pd.DataFrame([compute_trade_level_metrics(tradebook,margin)])
    daily_pnl = compute_daily_pnl(tradebook)
    daily_metrics = pd.DataFrame([compute_summary_metrics_daily(daily_pnl, margin)])
    all_metrics = pd.concat([tl_metrics, daily_metrics], axis=1)
    return all_metrics
