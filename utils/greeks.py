import math
from typing import Literal, Optional

import numpy as np
import pandas as pd


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * x**2)


def bs_greeks(
    spot: np.ndarray,
    strike: np.ndarray,
    time_to_expiry: np.ndarray,
    iv: np.ndarray,
    rate: float = 0.0,
    option_type: Literal["CE", "PE"] = "CE",
) -> pd.DataFrame:
    """
    Calculate Black-Scholes greeks for European options.

    Parameters
    ----------
    spot : array-like
        Spot price.
    strike : array-like
        Strike price.
    time_to_expiry : array-like
        Time to expiry in years.
    iv : array-like
        Implied volatility (decimal).
    rate : float
        Risk-free rate (decimal).
    option_type : {"CE", "PE"}
        Call ("CE") or Put ("PE").
    """
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    time_to_expiry = np.asarray(time_to_expiry, dtype=float)
    iv = np.asarray(iv, dtype=float)

    eps = 1e-12
    time_to_expiry = np.maximum(time_to_expiry, eps)
    iv = np.maximum(iv, eps)

    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * iv**2) * time_to_expiry) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    pdf_d1 = _norm_pdf(d1)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)

    if option_type == "CE":
        delta = cdf_d1
        theta = (
            -spot * pdf_d1 * iv / (2.0 * sqrt_t)
            - rate * strike * np.exp(-rate * time_to_expiry) * cdf_d2
        )
        rho = strike * time_to_expiry * np.exp(-rate * time_to_expiry) * cdf_d2
    else:
        delta = cdf_d1 - 1.0
        theta = (
            -spot * pdf_d1 * iv / (2.0 * sqrt_t)
            + rate * strike * np.exp(-rate * time_to_expiry) * _norm_cdf(-d2)
        )
        rho = -strike * time_to_expiry * np.exp(-rate * time_to_expiry) * _norm_cdf(-d2)

    gamma = pdf_d1 / (spot * iv * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t

    return pd.DataFrame(
        {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
            "d1": d1,
            "d2": d2,
        }
    )


def add_greeks(
    df: pd.DataFrame,
    *,
    spot_col: str = "spot",
    strike_col: str = "strike",
    iv_col: str = "iv",
    time_to_expiry_col: str = "time_to_expiry",
    rate: float = 0.0,
    option_type_col: Optional[str] = "option_type",
    default_option_type: Literal["CE", "PE"] = "CE",
) -> pd.DataFrame:
    """
    Add Black-Scholes greeks to a DataFrame.

    Expects time_to_expiry in years and iv as a decimal (e.g., 0.2 for 20%).
    """
    if option_type_col and option_type_col in df.columns:
        option_types = df[option_type_col].fillna(default_option_type).astype(str).str.upper()
    else:
        option_types = pd.Series([default_option_type] * len(df))

    results = []
    for opt_type in option_types.unique():
        mask = option_types == opt_type
        greeks = bs_greeks(
            df.loc[mask, spot_col].values,
            df.loc[mask, strike_col].values,
            df.loc[mask, time_to_expiry_col].values,
            df.loc[mask, iv_col].values,
            rate=rate,
            option_type=opt_type,
        )
        results.append((mask, greeks))

    output = df.copy()
    for mask, greeks in results:
        output.loc[mask, greeks.columns] = greeks.values

    return output
