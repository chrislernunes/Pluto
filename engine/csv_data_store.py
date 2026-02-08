import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.utility import stock_tickers

INDEXES = (
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTY",
    "SENSEX",
    "BANKEX",
    "SPXW",
    "NDXP",
    "GOLDM",
)


@dataclass
class SymbolData:
    df: pd.DataFrame
    ts_values: np.ndarray


class CSVDataStore:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = self._load_csv(csv_path)
        self.data_by_symbol = self._build_symbol_lookup(self.data)
        self.option_metadata = self._build_option_metadata(self.data_by_symbol)
        self.expiries_by_underlying = self._build_expiry_lookup(self.option_metadata)

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        timestamp_col = "ts" if "ts" in df.columns else "timestamp"
        if timestamp_col not in df.columns:
            raise ValueError("CSV must include a 'ts' or 'timestamp' column.")
        if "symbol" not in df.columns:
            raise ValueError("CSV must include a 'symbol' column.")
        df = df.rename(columns={timestamp_col: "ts"})
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
        return df

    def _build_symbol_lookup(self, df: pd.DataFrame) -> Dict[str, SymbolData]:
        data_by_symbol: Dict[str, SymbolData] = {}
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values("ts").reset_index(drop=True)
            data_by_symbol[symbol] = SymbolData(group, group["ts"].values)
        return data_by_symbol

    def _build_option_metadata(self, data_by_symbol: Dict[str, SymbolData]) -> pd.DataFrame:
        records = []
        for symbol in data_by_symbol.keys():
            if not (symbol.endswith("CE") or symbol.endswith("PE")):
                continue
            try:
                underlying = self._get_underlying_from_symbol(symbol)
                expiry_date = self._parse_date_from_symbol(symbol)
                strike = self._parse_strike_from_symbol(symbol)
            except ValueError:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "underlying": underlying,
                    "expiry_date": expiry_date,
                    "strike": strike,
                    "opt_type": "CE" if symbol.endswith("CE") else "PE",
                }
            )
        return pd.DataFrame(records)

    def _build_expiry_lookup(self, option_metadata: pd.DataFrame) -> Dict[str, List[datetime.date]]:
        expiries: Dict[str, List[datetime.date]] = {}
        if option_metadata.empty:
            return expiries
        for underlying, group in option_metadata.groupby("underlying"):
            expiries[underlying] = sorted(group["expiry_date"].unique().tolist())
        return expiries

    def _get_underlying_from_symbol(self, symbol: str) -> str:
        for underlying in INDEXES + tuple(stock_tickers):
            if symbol.startswith(underlying):
                return underlying
        raise ValueError(f"Unrecognized underlying for symbol: {symbol}")

    def _parse_date_from_symbol(self, symbol: str) -> datetime.date:
        underlying = self._get_underlying_from_symbol(symbol)
        symbol_wo_underlying = symbol[len(underlying):]
        date_part = symbol_wo_underlying[:6]
        return datetime.datetime.strptime(date_part, "%y%m%d").date()

    def _parse_strike_from_symbol(self, symbol: str) -> int:
        underlying = self._get_underlying_from_symbol(symbol)
        symbol_wo_underlying = symbol[len(underlying):]
        remainder = symbol_wo_underlying[6:].rstrip("CE").rstrip("PE")
        return int(remainder)

    def _resolve_symbol(self, symbol: str) -> Optional[str]:
        if symbol in self.data_by_symbol:
            return symbol
        if symbol.endswith("SPOT"):
            base_symbol = symbol.replace("SPOT", "")
            if base_symbol in self.data_by_symbol:
                return base_symbol
        return None

    def get_all_ticks_by_symbol(self, symbol: str) -> pd.DataFrame:
        resolved_symbol = self._resolve_symbol(symbol)
        if resolved_symbol is None:
            return pd.DataFrame()
        return self.data_by_symbol[resolved_symbol].df.copy()

    def get_last_available_tick(self, symbol: str) -> Optional[pd.Series]:
        resolved_symbol = self._resolve_symbol(symbol)
        if resolved_symbol is None:
            return None
        symbol_data = self.data_by_symbol[resolved_symbol]
        if symbol_data.df.empty:
            return None
        return symbol_data.df.iloc[-1]

    def get_tick(self, timestamp: datetime.datetime, symbol: str) -> Optional[pd.Series]:
        resolved_symbol = self._resolve_symbol(symbol)
        if resolved_symbol is None:
            return None
        symbol_data = self.data_by_symbol[resolved_symbol]
        if symbol_data.df.empty:
            return None
        ts_values = symbol_data.ts_values
        idx = np.searchsorted(ts_values, np.datetime64(timestamp), side="left")
        if idx < len(ts_values):
            return symbol_data.df.iloc[idx]
        return symbol_data.df.iloc[-1]

    def get_ticks_between(self, symbol: str, from_timestamp=None, to_timestamp=None) -> pd.DataFrame:
        df = self.get_all_ticks_by_symbol(symbol)
        if df.empty:
            return df
        if from_timestamp is not None:
            df = df[df["ts"] >= pd.to_datetime(from_timestamp)]
        if to_timestamp is not None:
            df = df[df["ts"] <= pd.to_datetime(to_timestamp)]
        return df

    def get_expiry_dates(self, underlying: str) -> List[datetime.date]:
        return self.expiries_by_underlying.get(underlying, [])

    def get_option_chain_snapshot(
        self,
        timestamp: datetime.datetime,
        underlying: str,
        expiry_date: datetime.date,
        opt_type: str,
    ) -> pd.DataFrame:
        if self.option_metadata.empty:
            return pd.DataFrame()
        subset = self.option_metadata[
            (self.option_metadata["underlying"] == underlying)
            & (self.option_metadata["expiry_date"] == expiry_date)
            & (self.option_metadata["opt_type"] == opt_type)
        ]
        if subset.empty:
            return pd.DataFrame()
        rows = []
        for symbol in subset["symbol"].tolist():
            tick = self.get_tick(timestamp, symbol)
            if tick is None:
                continue
            tick_data = tick.to_dict()
            tick_data["symbol"] = symbol
            tick_data["strike"] = subset.loc[subset["symbol"] == symbol, "strike"].iloc[0]
            rows.append(tick_data)
        return pd.DataFrame(rows)
