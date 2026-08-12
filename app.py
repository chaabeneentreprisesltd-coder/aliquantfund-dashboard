# -*- coding: utf-8 -*-
"""
⚡ AliQuantFund Institutional Architecture v4.4
================================================
CORE QUANT ENGINE

DATA
 ├─ Binance Spot / Bybit fallback
 ├─ Binance Futures OI + Funding / Bybit fallback
 └─ Binance Trade Tape

ANALYSIS
 ├─ ATR
 ├─ Ichimoku
 ├─ Session / Weekly / Monthly VWAP
 ├─ Smart Anchored VWAP
 ├─ Real / Approx CVD
 ├─ OI + Funding
 └─ Multi-Timeframe 1D → 4H → 1H → 15M → 5M

DECISION
 ├─ Market State
 ├─ HTF Context
 ├─ Setup Detection
 ├─ Trigger Detection
 ├─ Layered Quant Score
 ├─ Signal Grade
 └─ Trade Management

RISK
 ├─ Risk-Based Position Sizing
 ├─ Fees
 ├─ Slippage
 ├─ Partial TP
 ├─ Break-Even
 ├─ Trailing Stop
 └─ Time Stop

BACKTESTING
 ├─ Historical Simulation
 ├─ Realistic Execution
 ├─ Equity Curve
 ├─ Win Rate
 ├─ Profit Factor
 ├─ Sharpe Ratio
 ├─ Max Drawdown
 └─ Detailed Trade Log

IMPORTANT
-----------
This system generates analysis and signals.
It does NOT execute live trades.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 0. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AliQuantFund - Core Quant Engine v4.4",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("AliQuantFund")


# ============================================================
# 1. ENUMS
# ============================================================

class MarketState(Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGE_COMPRESSION = "RANGE_COMPRESSION"
    VOLATILE_EXPANSION = "VOLATILE_EXPANSION"


class SetupType(Enum):
    RECLAIM = "VWAP / Level Reclaim"
    REJECTION = "VWAP / Level Rejection"
    BREAKOUT = "Volatile Breakout"
    MEAN_REVERSION = "Overextended Reversion"
    NO_SETUP = "No Valid Setup"


class TriggerType(Enum):
    CONFIRMED_BUY = "BUY (Full Alignment)"
    CONFIRMED_SELL = "SELL (Full Alignment)"
    WAIT = "WAIT (Awaiting Trigger)"
    INVALID = "INVALID (Setup Failed)"


class SignalGrade(Enum):
    INSTITUTIONAL_STRONG = "A+ (Institutional Strong)"
    CONFIRMED = "A (Confirmed Setup)"
    MODERATE = "B (Moderate Alignment)"
    NEUTRAL = "C (Neutral / High Risk)"
    NO_TRADE = "NO TRADE"


class DataStatus(Enum):
    LIVE = "LIVE (Direct)"
    FALLBACK = "FALLBACK (Secondary API)"
    APPROXIMATED = "APPROXIMATED"
    UNAVAILABLE = "UNAVAILABLE"


# ============================================================
# 2. DATA STRUCTURES
# ============================================================

@dataclass
class QuantitativeMetrics:

    vwap_session: float = 0.0
    vwap_weekly: float = 0.0
    vwap_monthly: float = 0.0
    vwap_anchored: Optional[float] = None

    atr_14: float = 0.0

    cvd_slope: float = 0.0
    cvd_divergence: str = "NONE"

    oi_change_pct: float = 0.0
    funding_rate: Optional[float] = None

    tenkan: float = 0.0
    kijun: float = 0.0
    span_a: float = 0.0
    span_b: float = 0.0


@dataclass
class ScoringBreakdown:

    direction_score: float = 0.0
    flow_score: float = 0.0
    positioning_score: float = 0.0
    location_score: float = 0.0

    total_score: float = 0.0

    data_quality_pct: float = 100.0


@dataclass
class SetupResult:

    setup: SetupType
    reason: str
    direction: str
    quality: float


@dataclass
class TradePlan:

    direction: str = "NONE"

    entry_low: Optional[float] = None
    entry_high: Optional[float] = None

    stop_loss: Optional[float] = None

    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None

    rr_tp1: Optional[float] = None
    rr_tp2: Optional[float] = None
    rr_tp3: Optional[float] = None

    position_size: Optional[float] = None
    risk_amount: Optional[float] = None

    invalidation: str = "No active trade plan"


@dataclass
class BacktestResult:

    total_trades: int = 0

    winning_trades: int = 0
    losing_trades: int = 0

    win_rate: float = 0.0

    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0

    max_drawdown_pct: float = 0.0

    profit_factor: float = 0.0

    sharpe_ratio: float = 0.0

    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    trades_log: List[Dict[str, Any]] = field(
        default_factory=list
    )

    equity_curve: List[Dict[str, Any]] = field(
        default_factory=list
    )


# ============================================================
# 3. DATA LOADER
# ============================================================

class MarketDataLoader:

    BYBIT_TF_MAP = {
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240",
        "1d": "D"
    }

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_klines(
        symbol: str,
        interval: str,
        limit: int = 500
    ) -> Tuple[Optional[pd.DataFrame], str]:

        symbol = symbol.replace("/", "").upper()

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        endpoints = [

            (
                "https://data-api.binance.vision/api/v3/klines"
                f"?symbol={symbol}"
                f"&interval={interval}"
                f"&limit={limit}"
            ),

            (
                "https://api1.binance.com/api/v3/klines"
                f"?symbol={symbol}"
                f"&interval={interval}"
                f"&limit={limit}"
            ),

            (
                "https://api2.binance.com/api/v3/klines"
                f"?symbol={symbol}"
                f"&interval={interval}"
                f"&limit={limit}"
            ),

            (
                "https://api3.binance.com/api/v3/klines"
                f"?symbol={symbol}"
                f"&interval={interval}"
                f"&limit={limit}"
            )
        ]

        for url in endpoints:

            try:

                r = requests.get(
                    url,
                    headers=headers,
                    timeout=4
                )

                if r.status_code != 200:
                    continue

                data = r.json()

                if not isinstance(data, list) or not data:
                    continue

                df = pd.DataFrame(
                    data,
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_av",
                        "trades",
                        "tb_base_av",
                        "tb_quote_av",
                        "ignore"
                    ]
                )

                numeric_cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]

                for c in numeric_cols:

                    df[c] = pd.to_numeric(
                        df[c],
                        errors="coerce"
                    )

                df["timestamp"] = pd.to_datetime(
                    df["timestamp"],
                    unit="ms",
                    utc=True
                ).dt.tz_localize(None)

                return (
                    df[
                        [
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume"
                        ]
                    ].dropna().reset_index(drop=True),

                    DataStatus.LIVE.value
                )

            except Exception:
                continue

        # ----------------------------------------------------
        # BYBIT FALLBACK
        # ----------------------------------------------------

        try:

            bybit_tf = (
                MarketDataLoader.BYBIT_TF_MAP.get(
                    interval,
                    "5"
                )
            )

            url = (
                "https://api.bybit.com/v5/market/kline"
                f"?category=spot"
                f"&symbol={symbol}"
                f"&interval={bybit_tf}"
                f"&limit={limit}"
            )

            r = requests.get(
                url,
                headers=headers,
                timeout=5
            )

            if r.status_code == 200:

                data = (
                    r.json()
                    .get("result", {})
                    .get("list", [])
                )

                if data:

                    df = pd.DataFrame(
                        data,
                        columns=[
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "turnover"
                        ]
                    )

                    for c in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume"
                    ]:

                        df[c] = pd.to_numeric(
                            df[c],
                            errors="coerce"
                        )

                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"].astype(float),
                        unit="ms",
                        utc=True
                    ).dt.tz_localize(None)

                    df = (
                        df.iloc[::-1]
                        .reset_index(drop=True)
                    )

                    return (
                        df[
                            [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume"
                            ]
                        ].dropna(),

                        DataStatus.FALLBACK.value
                    )

        except Exception as e:

            logger.warning(
                f"Bybit spot error: {e}"
            )

        return (
            None,
            DataStatus.UNAVAILABLE.value
        )

    # ========================================================
    # FUTURES
    # ========================================================

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_futures_metrics(
        symbol: str,
        interval: str,
        limit: int = 50
    ):

        symbol = symbol.replace("/", "").upper()

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        funding = {
            "available": False,
            "current": None,
            "history": []
        }

        try:

            oi_interval = (
                interval
                if interval in [
                    "5m",
                    "15m",
                    "1h",
                    "4h",
                    "1d"
                ]
                else "5m"
            )

            url_oi = (
                "https://fapi.binance.com/futures/data/"
                f"openInterestHist?"
                f"symbol={symbol}"
                f"&period={oi_interval}"
                f"&limit={limit}"
            )

            url_fr = (
                "https://fapi.binance.com/fapi/v1/"
                f"fundingRate?"
                f"symbol={symbol}"
                f"&limit=30"
            )

            r_oi = requests.get(
                url_oi,
                headers=headers,
                timeout=5
            )

            r_fr = requests.get(
                url_fr,
                headers=headers,
                timeout=5
            )

            if r_oi.status_code == 200:

                data = r_oi.json()

                if isinstance(data, list) and len(data) >= 2:

                    df = pd.DataFrame(data)

                    df["openInterest"] = pd.to_numeric(
                        df["sumOpenInterest"],
                        errors="coerce"
                    )

                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"],
                        unit="ms",
                        utc=True
                    ).dt.tz_localize(None)

                    df = (
                        df.dropna(
                            subset=["openInterest"]
                        )
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )

                    if r_fr.status_code == 200:

                        fr = r_fr.json()

                        if isinstance(fr, list) and fr:

                            hist = [
                                float(
                                    x.get(
                                        "fundingRate",
                                        0
                                    )
                                )
                                for x in reversed(fr)
                            ]

                            funding["available"] = True

                            if hist:

                                funding["current"] = hist[0]
                                funding["history"] = hist

                    return (
                        df,
                        DataStatus.LIVE.value,
                        funding
                    )

        except Exception as e:

            logger.warning(
                f"Binance futures error: {e}"
            )

        # ----------------------------------------------------
        # BYBIT FALLBACK
        # ----------------------------------------------------

        try:

            tf = (
                MarketDataLoader.BYBIT_TF_MAP.get(
                    interval,
                    "5"
                )
            )

            url = (
                "https://api.bybit.com/v5/market/open-interest"
                f"?category=linear"
                f"&symbol={symbol}"
                f"&intervalTime={tf}"
                f"&limit={limit}"
            )

            r = requests.get(
                url,
                headers=headers,
                timeout=5
            )

            if r.status_code == 200:

                data = (
                    r.json()
                    .get("result", {})
                    .get("list", [])
                )

                if data:

                    df = pd.DataFrame(data)

                    df["openInterest"] = pd.to_numeric(
                        df["openInterest"],
                        errors="coerce"
                    )

                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"].astype(float),
                        unit="ms",
                        utc=True
                    ).dt.tz_localize(None)

                    df = (
                        df.dropna(
                            subset=["openInterest"]
                        )
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )

                    return (
                        df,
                        DataStatus.FALLBACK.value,
                        funding
                    )

        except Exception as e:

            logger.warning(
                f"Bybit OI error: {e}"
            )

        return (
            None,
            DataStatus.UNAVAILABLE.value,
            funding
        )

    # ========================================================
    # TRADE TAPE
    # ========================================================

    @staticmethod
    @st.cache_data(ttl=10)
    def fetch_recent_trades(
        symbol: str,
        limit: int = 1000
    ):

        symbol = symbol.replace("/", "").upper()

        endpoints = [

            (
                "https://data-api.binance.vision/api/v3/trades"
                f"?symbol={symbol}&limit={limit}"
            ),

            (
                "https://api1.binance.com/api/v3/trades"
                f"?symbol={symbol}&limit={limit}"
            ),

            (
                "https://api3.binance.com/api/v3/trades"
                f"?symbol={symbol}&limit={limit}"
            )
        ]

        for url in endpoints:

            try:

                r = requests.get(
                    url,
                    timeout=4
                )

                if r.status_code != 200:
                    continue

                data = r.json()

                if not data:
                    continue

                df = pd.DataFrame(data)

                df["price"] = pd.to_numeric(
                    df["price"],
                    errors="coerce"
                )

                df["qty"] = pd.to_numeric(
                    df["qty"],
                    errors="coerce"
                )

                df["time"] = pd.to_datetime(
                    df["time"],
                    unit="ms",
                    utc=True
                ).dt.tz_localize(None)

                df["is_buy"] = ~df["isBuyerMaker"]

                return (
                    df.dropna(
                        subset=[
                            "price",
                            "qty"
                        ]
                    ).reset_index(drop=True),

                    DataStatus.LIVE.value
                )

            except Exception:
                continue

        return (
            None,
            DataStatus.UNAVAILABLE.value
        )


# ============================================================
# 4. QUANTITATIVE ENGINE
# ============================================================

class QuantitativeEngine:

    # ========================================================
    # ATR
    # ========================================================

    @staticmethod
    def atr(df, period=14):

        prev_close = df["close"].shift(1)

        tr = pd.concat(
            [
                df["high"] - df["low"],

                (
                    df["high"] - prev_close
                ).abs(),

                (
                    df["low"] - prev_close
                ).abs()
            ],
            axis=1
        ).max(axis=1)

        return (
            tr.rolling(
                period,
                min_periods=period
            )
            .mean()
            .bfill()
        )

    # ========================================================
    # ICHIMOKU
    # ========================================================

    @staticmethod
    def ichimoku(df):

        d = df.copy()

        d["tenkan"] = (
            d["high"].rolling(9).max()
            +
            d["low"].rolling(9).min()
        ) / 2

        d["kijun"] = (
            d["high"].rolling(26).max()
            +
            d["low"].rolling(26).min()
        ) / 2

        d["span_a"] = (
            (
                d["tenkan"]
                +
                d["kijun"]
            ) / 2
        ).shift(26)

        d["span_b"] = (
            (
                d["high"].rolling(52).max()
                +
                d["low"].rolling(52).min()
            ) / 2
        ).shift(26)

        return d

    # ========================================================
    # VWAP
    # ========================================================

    @staticmethod
    def vwap(df, mode):

        typical = (
            df["high"]
            +
            df["low"]
            +
            df["close"]
        ) / 3

        pv = typical * df["volume"]

        if mode == "SESSION":

            group = df["timestamp"].dt.date

        elif mode == "WEEKLY":

            group = (
                df["timestamp"]
                .dt.to_period("W")
            )

        elif mode == "MONTHLY":

            group = (
                df["timestamp"]
                .dt.to_period("M")
            )

        else:

            group = pd.Series(
                0,
                index=df.index
            )

        cumulative_pv = (
            pv.groupby(group)
            .cumsum()
        )

        cumulative_volume = (
            df["volume"]
            .groupby(group)
            .cumsum()
        )

        return (
            cumulative_pv
            /
            cumulative_volume.replace(
                0,
                np.nan
            )
        )

    # ========================================================
    # SMART ANCHOR
    # ========================================================

    @staticmethod
    def smart_anchor(df):

        if len(df) < 30:

            return (
                max(0, len(df) - 1),
                "Initial"
            )

        d = df.tail(100).copy()

        volume_std = (
            d["volume"].std()
        )

        if volume_std == 0 or pd.isna(volume_std):

            return (
                d.index[0],
                "No Volume Spike"
            )

        z = (
            d["volume"]
            -
            d["volume"].mean()
        ) / (
            volume_std
        )

        idx = z.idxmax()

        return (
            int(idx),
            f"Volume Spike Z={z.loc[idx]:.2f}"
        )

    # ========================================================
    # ANCHORED VWAP
    # ========================================================

    @staticmethod
    def anchored_vwap(df, anchor):

        if anchor < 0 or anchor >= len(df):

            return pd.Series(
                np.nan,
                index=df.index
            )

        typical = (
            df["high"]
            +
            df["low"]
            +
            df["close"]
        ) / 3

        pv = typical * df["volume"]

        pv = pv.copy()
        vol = df["volume"].copy()

        pv.iloc[:anchor] = 0
        vol.iloc[:anchor] = 0

        cumulative_pv = pv.cumsum()
        cumulative_vol = vol.cumsum()

        avwap = (
            cumulative_pv
            /
            cumulative_vol.replace(
                0,
                np.nan
            )
        )

        avwap.iloc[:anchor] = np.nan

        return avwap

    # ========================================================
    # CVD
    # ========================================================

    @staticmethod
    def cvd(
        df,
        trades,
        timeframe
    ):

        # ----------------------------------------------------
        # REAL TRADE CVD
        # ----------------------------------------------------

        if trades is not None and not trades.empty:

            t = trades.copy()

            t["signed"] = np.where(
                t["is_buy"],
                t["qty"],
                -t["qty"]
            )

            rule = {
                "5m": "5min",
                "15m": "15min",
                "1h": "1h",
                "4h": "4h",
                "1d": "1D"
            }.get(
                timeframe,
                "5min"
            )

            delta = (
                t.set_index("time")["signed"]
                .resample(rule)
                .sum()
            )

            cvd = delta.cumsum()

            candle_index = pd.DatetimeIndex(
                df["timestamp"]
            )

            cvd = (
                cvd.reindex(
                    candle_index,
                    method="ffill"
                )
                .fillna(0)
            )

            cvd = cvd.reset_index(drop=True)

            return (
                cvd,
                DataStatus.LIVE.value,
                QuantitativeEngine.cvd_stats(
                    df,
                    cvd
                )
            )

        # ----------------------------------------------------
        # APPROX CVD
        # ----------------------------------------------------

        candle_range = (
            df["high"]
            -
            df["low"]
        ).replace(
            0,
            1e-9
        )

        delta = (
            df["volume"]
            *
            (
                (
                    df["close"]
                    -
                    df["open"]
                )
                /
                candle_range
            )
        )

        cvd = delta.cumsum()

        return (
            cvd,
            DataStatus.APPROXIMATED.value,
            QuantitativeEngine.cvd_stats(
                df,
                cvd
            )
        )

    # ========================================================
    # CVD STATS
    # ========================================================

    @staticmethod
    def cvd_stats(df, cvd):

        if len(df) < 20:

            return {
                "slope": 0.0,
                "normalized_slope": 0.0,
                "divergence": "NONE"
            }

        lookback = min(
            10,
            len(df) - 1
        )

        price_change = (
            df["close"].iloc[-1]
            -
            df["close"].iloc[-lookback]
        )

        cvd_change = (
            cvd.iloc[-1]
            -
            cvd.iloc[-lookback]
        )

        volume_base = (
            df["volume"]
            .tail(20)
            .mean()
        )

        if volume_base <= 0:
            volume_base = 1

        normalized_slope = (
            cvd_change
            /
            (
                volume_base
                *
                lookback
            )
        )

        divergence = "NONE"

        if (
            price_change < 0
            and cvd_change > 0
        ):

            divergence = "BULLISH_ABSORPTION"

        elif (
            price_change > 0
            and cvd_change < 0
        ):

            divergence = "BEARISH_ABSORPTION"

        return {
            "slope": float(
                normalized_slope
            ),

            "normalized_slope": float(
                normalized_slope
            ),

            "divergence": divergence
        }


# ============================================================
# 5. MARKET STATE ENGINE
# ============================================================

class MarketStateEngine:

    @staticmethod
    def classify(df, atr):

        close = df["close"].iloc[-1]

        ema20 = (
            df["close"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        ema50 = (
            df["close"]
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        recent_range = (
            df["high"].tail(10).max()
            -
            df["low"].tail(10).min()
        )

        if recent_range > 3.5 * atr:

            return (
                MarketState.VOLATILE_EXPANSION
            )

        if recent_range < 1.5 * atr:

            return (
                MarketState.RANGE_COMPRESSION
            )

        if close > ema20 > ema50:

            return (
                MarketState.TRENDING_BULL
            )

        if close < ema20 < ema50:

            return (
                MarketState.TRENDING_BEAR
            )

        return (
            MarketState.RANGE_COMPRESSION
        )


# ============================================================
# 6. MULTI-TIMEFRAME ENGINE
# ============================================================

class MultiTimeframeEngine:

    @staticmethod
    def evaluate(symbol):

        tfs = [
            "1d",
            "4h",
            "1h",
            "15m",
            "5m"
        ]

        result = {}

        for tf in tfs:

            df, status = (
                MarketDataLoader.fetch_klines(
                    symbol,
                    tf,
                    150
                )
            )

            if (
                df is None
                or len(df) < 60
            ):

                result[tf] = {
                    "score": 50,
                    "bias": "NEUTRAL",
                    "status": status
                }

                continue

            d = (
                QuantitativeEngine
                .ichimoku(df)
            )

            close = d["close"].iloc[-1]

            tenkan = d["tenkan"].iloc[-1]
            kijun = d["kijun"].iloc[-1]

            span_a = d["span_a"].iloc[-1]
            span_b = d["span_b"].iloc[-1]

            if pd.isna(span_a):
                span_a = close

            if pd.isna(span_b):
                span_b = close

            cloud_high = max(
                span_a,
                span_b
            )

            cloud_low = min(
                span_a,
                span_b
            )

            if (
                close > cloud_high
                and tenkan > kijun
            ):

                bias = "BULLISH"
                score = 80

            elif (
                close < cloud_low
                and tenkan < kijun
            ):

                bias = "BEARISH"
                score = 20

            else:

                if close > kijun:

                    bias = "BULLISH"
                    score = 60

                elif close < kijun:

                    bias = "BEARISH"
                    score = 40

                else:

                    bias = "NEUTRAL"
                    score = 50

            result[tf] = {
                "score": score,
                "bias": bias,
                "status": status
            }

        context = (
            result["1d"]["score"] * 0.60
            +
            result["4h"]["score"] * 0.40
        )

        if context >= 65:

            context_bias = "BULLISH"

        elif context <= 35:

            context_bias = "BEARISH"

        else:

            context_bias = "NEUTRAL"

        execution = (
            result["15m"]["score"] * 0.50
            +
            result["5m"]["score"] * 0.50
        )

        return {
            "frames": result,

            "context_score": context,

            "context_bias": context_bias,

            "direction_bias":
                result["1h"]["bias"],

            "execution_score":
                execution
        }


# ============================================================
# 7. SETUP ENGINE
# ============================================================

class SetupEngine:

    @staticmethod
    def detect(
        df,
        metrics,
        market_state,
        htf
    ):

        close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        vwap = metrics.vwap_session

        atr = max(
            metrics.atr_14,
            close * 0.001
        )

        # ----------------------------------------------------
        # VWAP RECLAIM
        # ----------------------------------------------------

        bullish_reclaim = (
            prev_close < vwap
            and
            close > vwap
        )

        bearish_reclaim = (
            prev_close > vwap
            and
            close < vwap
        )

        if bullish_reclaim:

            return SetupResult(
                SetupType.RECLAIM,
                "Price reclaimed Session VWAP",
                "LONG",
                80
            )

        if bearish_reclaim:

            return SetupResult(
                SetupType.RECLAIM,
                "Price lost Session VWAP",
                "SHORT",
                80
            )

        # ----------------------------------------------------
        # BREAKOUT
        # ----------------------------------------------------

        if len(df) >= 25:

            resistance = (
                df["high"]
                .iloc[-21:-1]
                .max()
            )

            support = (
                df["low"]
                .iloc[-21:-1]
                .min()
            )

            volume_avg = (
                df["volume"]
                .iloc[-21:-1]
                .mean()
            )

            volume_expansion = (
                df["volume"].iloc[-1]
                >
                volume_avg * 1.5
            )

            if (
                close > resistance
                and volume_expansion
            ):

                return SetupResult(
                    SetupType.BREAKOUT,
                    "Resistance breakout with volume expansion",
                    "LONG",
                    85
                )

            if (
                close < support
                and volume_expansion
            ):

                return SetupResult(
                    SetupType.BREAKOUT,
                    "Support breakdown with volume expansion",
                    "SHORT",
                    85
                )

        # ----------------------------------------------------
        # REJECTION
        # ----------------------------------------------------

        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]
        open_ = df["open"].iloc[-1]

        upper_wick = (
            high
            -
            max(
                close,
                open_
            )
        )

        lower_wick = (
            min(
                close,
                open_
            )
            -
            low
        )

        if (
            upper_wick > atr * 0.7
            and
            close < vwap
        ):

            return SetupResult(
                SetupType.REJECTION,
                "Upper rejection below VWAP",
                "SHORT",
                70
            )

        if (
            lower_wick > atr * 0.7
            and
            close > vwap
        ):

            return SetupResult(
                SetupType.REJECTION,
                "Lower rejection above VWAP",
                "LONG",
                70
            )

        # ----------------------------------------------------
        # MEAN REVERSION
        # ----------------------------------------------------

        distance = (
            close - vwap
        ) / atr

        if distance > 2.5:

            return SetupResult(
                SetupType.MEAN_REVERSION,
                "Price excessively above VWAP",
                "SHORT",
                60
            )

        if distance < -2.5:

            return SetupResult(
                SetupType.MEAN_REVERSION,
                "Price excessively below VWAP",
                "LONG",
                60
            )

        return SetupResult(
            SetupType.NO_SETUP,
            "No high-quality structural setup",
            "NONE",
            0
        )


# ============================================================
# 8. TRIGGER ENGINE
# ============================================================

class TriggerEngine:

    @staticmethod
    def detect(
        df,
        metrics,
        setup,
        htf
    ):

        if (
            setup.setup
            == SetupType.NO_SETUP
        ):

            return TriggerType.WAIT

        close = df["close"].iloc[-1]

        ema20 = (
            df["close"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
            .iloc[-1]
        )

        bullish_flow = (
            metrics.cvd_slope > 0
            or
            metrics.cvd_divergence
            ==
            "BULLISH_ABSORPTION"
        )

        bearish_flow = (
            metrics.cvd_slope < 0
            or
            metrics.cvd_divergence
            ==
            "BEARISH_ABSORPTION"
        )

        if setup.direction == "LONG":

            alignment = 0

            if close > ema20:
                alignment += 1

            if bullish_flow:
                alignment += 1

            if (
                htf["context_bias"]
                == "BULLISH"
            ):
                alignment += 1

            if (
                htf["direction_bias"]
                == "BULLISH"
            ):
                alignment += 1

            if alignment >= 3:

                return (
                    TriggerType.CONFIRMED_BUY
                )

            if (
                metrics.cvd_divergence
                ==
                "BEARISH_ABSORPTION"
            ):

                return TriggerType.INVALID

            return TriggerType.WAIT

        if setup.direction == "SHORT":

            alignment = 0

            if close < ema20:
                alignment += 1

            if bearish_flow:
                alignment += 1

            if (
                htf["context_bias"]
                == "BEARISH"
            ):
                alignment += 1

            if (
                htf["direction_bias"]
                == "BEARISH"
            ):
                alignment += 1

            if alignment >= 3:

                return (
                    TriggerType.CONFIRMED_SELL
                )

            if (
                metrics.cvd_divergence
                ==
                "BULLISH_ABSORPTION"
            ):

                return TriggerType.INVALID

            return TriggerType.WAIT

        return TriggerType.WAIT


# ============================================================
# 9. FACTOR SCORING ENGINE
# ============================================================

class FactorScoringEngine:

    @staticmethod
    def score(
        market_state,
        df,
        metrics,
        htf,
        futures_status,
        cvd_status
    ):

        close = df["close"].iloc[-1]

        atr = max(
            metrics.atr_14,
            close * 0.001
        )

        # ----------------------------------------------------
        # DIRECTION
        # ----------------------------------------------------

        direction = 0

        if (
            htf["context_bias"]
            == "BULLISH"
        ):

            direction += 15

        elif (
            htf["context_bias"]
            == "BEARISH"
        ):

            direction -= 15

        if (
            htf["direction_bias"]
            == "BULLISH"
        ):

            direction += 10

        elif (
            htf["direction_bias"]
            == "BEARISH"
        ):

            direction -= 10

        # ----------------------------------------------------
        # FLOW
        # ----------------------------------------------------

        if (
            metrics.cvd_divergence
            ==
            "BULLISH_ABSORPTION"
        ):

            flow = 20

        elif (
            metrics.cvd_divergence
            ==
            "BEARISH_ABSORPTION"
        ):

            flow = -20

        else:

            flow = float(
                np.clip(
                    metrics.cvd_slope * 100,
                    -20,
                    20
                )
            )

        # ----------------------------------------------------
        # POSITIONING
        # ----------------------------------------------------

        positioning = 0

        if (
            futures_status
            !=
            DataStatus.UNAVAILABLE.value
        ):

            # OI expansion
            if metrics.oi_change_pct > 2:

                if direction > 0:

                    positioning += 15

                elif direction < 0:

                    positioning -= 15

            # OI contraction
            elif metrics.oi_change_pct < -2:

                if direction > 0:

                    positioning -= 8

                elif direction < 0:

                    positioning += 8

            # Funding
            if (
                metrics.funding_rate
                is not None
            ):

                funding = (
                    metrics.funding_rate
                )

                if funding < -0.0001:

                    positioning += 8

                elif funding > 0.0003:

                    positioning -= 8

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        distance = (
            close
            -
            metrics.vwap_session
        ) / atr

        if abs(distance) <= 0.5:

            location = (
                20
                if direction >= 0
                else -20
            )

        elif distance > 2:

            location = -15

        elif distance < -2:

            location = 15

        else:

            location = (
                8
                if distance > 0
                else -8
            )

        # ----------------------------------------------------
        # MARKET REGIME WEIGHTS
        # ----------------------------------------------------

        if (
            market_state
            ==
            MarketState.RANGE_COMPRESSION
        ):

            dw, fw, pw, lw = (
                0.15,
                0.30,
                0.15,
                0.40
            )

        elif market_state in [

            MarketState.TRENDING_BULL,

            MarketState.TRENDING_BEAR

        ]:

            dw, fw, pw, lw = (
                0.35,
                0.25,
                0.20,
                0.20
            )

        else:

            dw = fw = pw = lw = 0.25

        total = (
            direction * dw
            +
            flow * fw
            +
            positioning * pw
            +
            location * lw
        )

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        quality = 100

        if (
            futures_status
            ==
            DataStatus.UNAVAILABLE.value
        ):

            quality -= 25

        elif (
            futures_status
            ==
            DataStatus.FALLBACK.value
        ):

            quality -= 10

        if (
            cvd_status
            ==
            DataStatus.APPROXIMATED.value
        ):

            quality -= 15

        elif (
            cvd_status
            ==
            DataStatus.UNAVAILABLE.value
        ):

            quality -= 25

        return ScoringBreakdown(

            direction_score=round(
                direction,
                1
            ),

            flow_score=round(
                flow,
                1
            ),

            positioning_score=round(
                positioning,
                1
            ),

            location_score=round(
                location,
                1
            ),

            total_score=round(
                total,
                1
            ),

            data_quality_pct=max(
                0,
                quality
            )
        )


# ============================================================
# 10. SIGNAL ENGINE
# ============================================================

class SignalEngine:

    @staticmethod
    def grade(
        score,
        setup,
        trigger,
        quality,
        htf
    ):

        if (
            trigger
            ==
            TriggerType.INVALID
        ):

            return SignalGrade.NO_TRADE

        if (
            setup.setup
            ==
            SetupType.NO_SETUP
        ):

            return SignalGrade.NEUTRAL

        if quality < 60:

            return SignalGrade.NO_TRADE

        abs_score = abs(score)

        if (
            abs_score >= 30
            and
            quality >= 85
            and
            trigger in [
                TriggerType.CONFIRMED_BUY,
                TriggerType.CONFIRMED_SELL
            ]
        ):

            return (
                SignalGrade.INSTITUTIONAL_STRONG
            )

        if (
            abs_score >= 22
            and
            trigger in [
                TriggerType.CONFIRMED_BUY,
                TriggerType.CONFIRMED_SELL
            ]
        ):

            return SignalGrade.CONFIRMED

        if abs_score >= 15:

            return SignalGrade.MODERATE

        return SignalGrade.NEUTRAL


# ============================================================
# 11. TRADE MANAGEMENT
# ============================================================

class TradeManagement:

    @staticmethod
    def build(
        df,
        metrics,
        direction,
        capital,
        risk_pct
    ):

        if direction not in [
            "LONG",
            "SHORT"
        ]:

            return TradePlan()

        price = df["close"].iloc[-1]

        atr = max(
            metrics.atr_14,
            price * 0.001
        )

        risk_amount = (
            capital
            *
            risk_pct
            /
            100
        )

        entry_low = (
            price
            -
            atr * 0.15
        )

        entry_high = (
            price
            +
            atr * 0.15
        )

        entry = (
            entry_low
            +
            entry_high
        ) / 2

        if direction == "LONG":

            stop = (
                entry
                -
                atr * 1.5
            )

            risk_per_unit = (
                entry
                -
                stop
            )

            tp1 = (
                entry
                +
                atr * 1.5
            )

            tp2 = (
                entry
                +
                atr * 2.5
            )

            tp3 = (
                entry
                +
                atr * 4.0
            )

            invalidation = (
                f"LONG invalid if price closes "
                f"below ${stop:,.4f}"
            )

        else:

            stop = (
                entry
                +
                atr * 1.5
            )

            risk_per_unit = (
                stop
                -
                entry
            )

            tp1 = (
                entry
                -
                atr * 1.5
            )

            tp2 = (
                entry
                -
                atr * 2.5
            )

            tp3 = (
                entry
                -
                atr * 4.0
            )

            invalidation = (
                f"SHORT invalid if price closes "
                f"above ${stop:,.4f}"
            )

        position_size = (
            risk_amount
            /
            max(
                risk_per_unit,
                1e-9
            )
        )

        def rr(target):

            if direction == "LONG":

                return (
                    target
                    -
                    entry
                ) / risk_per_unit

            return (
                entry
                -
                target
            ) / risk_per_unit

        return TradePlan(

            direction=direction,

            entry_low=entry_low,
            entry_high=entry_high,

            stop_loss=stop,

            tp1=tp1,
            tp2=tp2,
            tp3=tp3,

            rr_tp1=rr(tp1),
            rr_tp2=rr(tp2),
            rr_tp3=rr(tp3),

            position_size=position_size,

            risk_amount=risk_amount,

            invalidation=invalidation
        )


# ============================================================
# 12. BACKTEST ENGINE v4.4
# ============================================================

class BacktestEngine:

    @staticmethod
    def run_backtest(
        df,
        initial_capital=100.0,
        risk_pct=2.0,
        fee_pct=0.10,
        slippage_pct=0.05,
        tp1_fraction=0.40,
        tp2_fraction=0.30,
        tp3_fraction=0.30,
        breakeven_after_tp1=True,
        trailing_atr=1.0,
        max_bars_in_trade=48
    ) -> BacktestResult:

        if (
            df is None
            or len(df) < 80
        ):

            return BacktestResult()

        d = (
            df.copy()
            .reset_index(drop=True)
        )

        d = (
            QuantitativeEngine
            .ichimoku(d)
        )

        d["atr"] = (
            QuantitativeEngine.atr(d)
        )

        d["vwap"] = (
            QuantitativeEngine.vwap(
                d,
                "SESSION"
            )
        )

        capital = float(
            initial_capital
        )

        peak_capital = capital

        max_drawdown = 0.0

        trades = []

        equity_curve = []

        in_position = False

        position = None

        # ====================================================
        # LOOP
        # ====================================================

        for i in range(60, len(d)):

            row = d.iloc[i]

            close = float(
                row["close"]
            )

            high = float(
                row["high"]
            )

            low = float(
                row["low"]
            )

            atr = float(
                row["atr"]
            )

            vwap = float(
                row["vwap"]
            )

            # ------------------------------------------------
            # Equity Snapshot
            # ------------------------------------------------

            equity_curve.append({
                "time": row["timestamp"],
                "equity": capital
            })

            # =================================================
            # MANAGE ACTIVE TRADE
            # =================================================

            if in_position:

                position["bars"] += 1

                direction = (
                    position["direction"]
                )

                entry = (
                    position["entry"]
                )

                stop = (
                    position["stop"]
                )

                tp1 = (
                    position["tp1"]
                )

                tp2 = (
                    position["tp2"]
                )

                tp3 = (
                    position["tp3"]
                )

                remaining_qty = (
                    position["remaining_qty"]
                )

                realized_pnl = 0.0

                events = []

                # --------------------------------------------
                # LONG
                # --------------------------------------------

                if direction == "LONG":

                    # Stop has priority if both
                    # stop and target are touched
                    if low <= stop:

                        exit_price = (
                            stop
                            *
                            (
                                1
                                -
                                slippage_pct
                                /
                                100
                            )
                        )

                        pnl = (
                            exit_price
                            -
                            entry
                        ) * remaining_qty

                        fee = (
                            (
                                exit_price
                                *
                                remaining_qty
                            )
                            *
                            fee_pct
                            /
                            100
                        )

                        pnl -= fee

                        realized_pnl += pnl

                        events.append(
                            "SL"
                        )

                        remaining_qty = 0

                    else:

                        # TP1
                        if (
                            not position["tp1_hit"]
                            and
                            high >= tp1
                        ):

                            qty = (
                                position["initial_qty"]
                                *
                                tp1_fraction
                            )

                            exit_price = (
                                tp1
                                *
                                (
                                    1
                                    -
                                    slippage_pct
                                    /
                                    100
                                )
                            )

                            pnl = (
                                exit_price
                                -
                                entry
                            ) * qty

                            fee = (
                                exit_price
                                *
                                qty
                                *
                                fee_pct
                                /
                                100
                            )

                            realized_pnl += (
                                pnl - fee
                            )

                            remaining_qty -= qty

                            position[
                                "tp1_hit"
                            ] = True

                            events.append(
                                "TP1"
                            )

                            if breakeven_after_tp1:

                                position[
                                    "stop"
                                ] = entry

                        # TP2
                        if (
                            position["tp1_hit"]
                            and
                            not position["tp2_hit"]
                            and
                            high >= tp2
                        ):

                            qty = (
                                position["initial_qty"]
                                *
                                tp2_fraction
                            )

                            exit_price = (
                                tp2
                                *
                                (
                                    1
                                    -
                                    slippage_pct
                                    /
                                    100
                                )
                            )

                            pnl = (
                                exit_price
                                -
                                entry
                            ) * qty

                            fee = (
                                exit_price
                                *
                                qty
                                *
                                fee_pct
                                /
                                100
                            )

                            realized_pnl += (
                                pnl - fee
                            )

                            remaining_qty -= qty

                            position[
                                "tp2_hit"
                            ] = True

                            events.append(
                                "TP2"
                            )

                        # TP3
                        if (
                            position["tp2_hit"]
                            and
                            not position["tp3_hit"]
                            and
                            high >= tp3
                        ):

                            qty = remaining_qty

                            exit_price = (
                                tp3
                                *
                                (
                                    1
                                    -
                                    slippage_pct
                                    /
                                    100
                                )
                            )

                            pnl = (
                                exit_price
                                -
                                entry
                            ) * qty

                            fee = (
                                exit_price
                                *
                                qty
                                *
                                fee_pct
                                /
                                100
                            )

                            realized_pnl += (
                                pnl - fee
                            )

                            remaining_qty = 0

                            position[
                                "tp3_hit"
                            ] = True

                            events.append(
                                "TP3"
                            )

                        # Trailing Stop
                        if (
                            remaining_qty > 0
                            and
                            position["tp1_hit"]
                        ):

                            new_stop = (
                                close
                                -
                                atr * trailing_atr
                            )

                            if (
                                new_stop
                                >
                                position["stop"]
                            ):

                                position[
                                    "stop"
                                ] = new_stop

                # --------------------------------------------
                # SHORT
                # --------------------------------------------

                else:

                    if high >= stop:

                        exit_price = (
                            stop
                            *
                            (
                                1
                                +
                                slippage_pct
                                /
                                100
                            )
                        )

                        pnl = (
                            entry
                            -
                            exit_price
                        ) * remaining_qty

                        fee = (
                            exit_price
                            *
                            remaining_qty
                            *
                            fee_pct
                            /
                            100
                        )

                        pnl -= fee

                        realized_pnl += pnl

                        events.append(
                            "SL"
                        )

                        remaining_qty = 0

                    else:

                        # TP1
                        if (
                            not position["tp1_hit"]
                            and
                            low <= tp1
                        ):

                            qty = (
                                position["initial_qty"]
                                *
                                tp1_fraction
                            )

                            exit_price = (
                                tp1
                                *
                                (
                                    1
                                    +
                                    slippage_pct
                                    /
                                    100
                                )
                            )

                            pnl = (
                                entry
                                -
                                exit_price
                            ) * qty

                            fee = (
                                exit_price
                                *
                                qty
                                *
                                fee_pct
                                /
                                100
                            )

                            realized_pnl += (
                                pnl - fee
                            )

                            remaining_qty -= qty

                            position[
                                "tp1_hit"
                            ] = True

                            events.append(
                                "TP1"
                            )

                            if breakeven_after_tp1:

                                position[
                                    "stop"
                                ] = entry

                        # TP2
                        if (
                            position["tp1_hit"]
                            and
                            not position["tp2_hit"]
                            and
                            low <= tp2
                        ):

                            qty = (
                                position["initial_qty"]
                                *
                                tp2_fraction
                            )

                            exit_price = (
                                tp2
                                *
                                (
                                    1
                                    +
                                    slippage_pct
                                    /
                                    100
                                )
                            )

                            pnl = (
                                entry
                                -
                                exit_price
                            ) * qty

                            fee = (
                                exit_price
                                *
                                qty
                                *
                                fee_pct
                                /
                                100
                            )

                            realized_pnl += (
                                pnl - fee
                            )

                            remaining_qty -= qty

                            position[
                                "tp2_hit"
                            ] = True

                            events.append(
                                "TP2"
                            )

                        # TP3
                        if (
                            position["tp2_hit"]
                            and
                            not position["tp3_hit"]
                            and
                            low <= tp3
                        ):

                            qty = remaining_qty

                            exit_price = (
                                tp3
                                *
                                (
                                    1
                                    +
                                    slippage_pct
                                    /
                                    100
                                )
                            )

                            pnl = (
                                entry
                                -
                                exit_price
                            ) * qty

                            fee = (
                                exit_price
                                *
                                qty
                                *
                                fee_pct
                                /
                                100
                            )

                            realized_pnl += (
                                pnl - fee
                            )

                            remaining_qty = 0

                            position[
                                "tp3_hit"
                            ] = True

                            events.append(
                                "TP3"
                            )

                        # Trailing
                        if (
                            remaining_qty > 0
                            and
                            position["tp1_hit"]
                        ):

                            new_stop = (
                                close
                                +
                                atr * trailing_atr
                            )

                            if (
                                new_stop
                                <
                                position["stop"]
                            ):

                                position[
                                    "stop"
                                ] = new_stop

                # ------------------------------------------------
                # TIME STOP
                # ------------------------------------------------

                if (
                    remaining_qty > 0
                    and
                    position["bars"]
                    >=
                    max_bars_in_trade
                ):

                    if direction == "LONG":

                        exit_price = (
                            close
                            *
                            (
                                1
                                -
                                slippage_pct
                                /
                                100
                            )
                        )

                        pnl = (
                            exit_price
                            -
                            entry
                        ) * remaining_qty

                    else:

                        exit_price = (
                            close
                            *
                            (
                                1
                                +
                                slippage_pct
                                /
                                100
                            )
                        )

                        pnl = (
                            entry
                            -
                            exit_price
                        ) * remaining_qty

                    fee = (
                        exit_price
                        *
                        remaining_qty
                        *
                        fee_pct
                        /
                        100
                    )

                    realized_pnl += (
                        pnl - fee
                    )

                    remaining_qty = 0

                    events.append(
                        "TIME_STOP"
                    )

                # ------------------------------------------------
                # Update Position
                # ------------------------------------------------

                capital += realized_pnl

                position[
                    "remaining_qty"
                ] = remaining_qty

                position[
                    "realized_pnl"
                ] += realized_pnl

                # ------------------------------------------------
                # Close Trade
                # ------------------------------------------------

                if remaining_qty <= 1e-12:

                    total_pnl = (
                        position["realized_pnl"]
                    )

                    notional = (
                        position["entry"]
                        *
                        position["initial_qty"]
                    )

                    pnl_pct = (
                        total_pnl
                        /
                        max(
                            notional,
                            1e-9
                        )
                        *
                        100
                    )

                    trades.append({

                        "entry_time":
                            position["entry_time"],

                        "exit_time":
                            row["timestamp"],

                        "type":
                            direction,

                        "entry":
                            round(
                                position["entry"],
                                6
                            ),

                        "initial_qty":
                            round(
                                position["initial_qty"],
                                8
                            ),

                        "pnl":
                            round(
                                total_pnl,
                                4
                            ),

                        "pnl_pct":
                            round(
                                pnl_pct,
                                3
                            ),

                        "events":
                            ",".join(events),

                        "bars":
                            position["bars"],

                        "result":
                            (
                                "WIN"
                                if total_pnl > 0
                                else "LOSS"
                            )
                    })

                    in_position = False
                    position = None

                # Continue to next candle
                continue

            # =================================================
            # NEW SIGNAL
            # =================================================

            prev = d.iloc[i - 1]

            prev_close = float(
                prev["close"]
            )

            # -------------------------------------------------
            # Reclaim
            # -------------------------------------------------

            bullish_reclaim = (
                prev_close
                <
                float(prev["vwap"])
                and
                close
                >
                vwap
            )

            bearish_reclaim = (
                prev_close
                >
                float(prev["vwap"])
                and
                close
                <
                vwap
            )

            # -------------------------------------------------
            # Volume
            # -------------------------------------------------

            volume_avg = (
                d["volume"]
                .iloc[
                    max(0, i - 20):i
                ]
                .mean()
            )

            volume_expansion = (
                row["volume"]
                >
                volume_avg * 1.5
            )

            # -------------------------------------------------
            # Breakout
            # -------------------------------------------------

            resistance = (
                d["high"]
                .iloc[
                    max(0, i - 21):i
                ]
                .max()
            )

            support = (
                d["low"]
                .iloc[
                    max(0, i - 21):i
                ]
                .min()
            )

            bullish_breakout = (
                close > resistance
                and
                volume_expansion
            )

            bearish_breakout = (
                close < support
                and
                volume_expansion
            )

            # -------------------------------------------------
            # Direction decision
            # -------------------------------------------------

            direction = None

            if bullish_reclaim:

                if (
                    close
                    >
                    float(row["kijun"])
                ):

                    direction = "LONG"

            elif bearish_reclaim:

                if (
                    close
                    <
                    float(row["kijun"])
                ):

                    direction = "SHORT"

            elif bullish_breakout:

                direction = "LONG"

            elif bearish_breakout:

                direction = "SHORT"

            # -------------------------------------------------
            # Enter
            # -------------------------------------------------

            if direction is None:
                continue

            # Avoid weak data
            if (
                pd.isna(atr)
                or
                atr <= 0
            ):
                continue

            # -------------------------------------------------
            # Execution Slippage
            # -------------------------------------------------

            if direction == "LONG":

                entry = (
                    close
                    *
                    (
                        1
                        +
                        slippage_pct
                        /
                        100
                    )
                )

                stop = (
                    entry
                    -
                    atr * 1.5
                )

                tp1 = (
                    entry
                    +
                    atr * 1.5
                )

                tp2 = (
                    entry
                    +
                    atr * 2.5
                )

                tp3 = (
                    entry
                    +
                    atr * 4.0
                )

            else:

                entry = (
                    close
                    *
                    (
                        1
                        -
                        slippage_pct
                        /
                        100
                    )
                )

                stop = (
                    entry
                    +
                    atr * 1.5
                )

                tp1 = (
                    entry
                    -
                    atr * 1.5
                )

                tp2 = (
                    entry
                    -
                    atr * 2.5
                )

                tp3 = (
                    entry
                    -
                    atr * 4.0
                )

            # -------------------------------------------------
            # Risk
            # -------------------------------------------------

            risk_amount = (
                capital
                *
                risk_pct
                /
                100
            )

            risk_per_unit = abs(
                entry - stop
            )

            if risk_per_unit <= 0:
                continue

            qty = (
                risk_amount
                /
                risk_per_unit
            )

            # -------------------------------------------------
            # Entry fee
            # -------------------------------------------------

            entry_fee = (
                entry
                *
                qty
                *
                fee_pct
                /
                100
            )

            capital -= entry_fee

            # -------------------------------------------------
            # Position
            # -------------------------------------------------

            position = {

                "direction":
                    direction,

                "entry_time":
                    row["timestamp"],

                "entry":
                    entry,

                "initial_qty":
                    qty,

                "remaining_qty":
                    qty,

                "stop":
                    stop,

                "tp1":
                    tp1,

                "tp2":
                    tp2,

                "tp3":
                    tp3,

                "tp1_hit":
                    False,

                "tp2_hit":
                    False,

                "tp3_hit":
                    False,

                "bars":
                    0,

                "realized_pnl":
                    -entry_fee
            }

            in_position = True

        # ====================================================
        # FINAL OPEN POSITION
        # ====================================================

        if in_position and position is not None:

            row = d.iloc[-1]

            close = float(
                row["close"]
            )

            direction = (
                position["direction"]
            )

            qty = (
                position["remaining_qty"]
            )

            if direction == "LONG":

                exit_price = (
                    close
                    *
                    (
                        1
                        -
                        slippage_pct
                        /
                        100
                    )
                )

                pnl = (
                    exit_price
                    -
                    position["entry"]
                ) * qty

            else:

                exit_price = (
                    close
                    *
                    (
                        1
                        +
                        slippage_pct
                        /
                        100
                    )
                )

                pnl = (
                    position["entry"]
                    -
                    exit_price
                ) * qty

            fee = (
                exit_price
                *
                qty
                *
                fee_pct
                /
                100
            )

            pnl -= fee

            capital += pnl

            total_pnl = (
                position["realized_pnl"]
                +
                pnl
            )

            notional = (
                position["entry"]
                *
                position["initial_qty"]
            )

            pnl_pct = (
                total_pnl
                /
                max(
                    notional,
                    1e-9
                )
                *
                100
            )

            trades.append({

                "entry_time":
                    position["entry_time"],

                "exit_time":
                    row["timestamp"],

                "type":
                    direction,

                "entry":
                    round(
                        position["entry"],
                        6
                    ),

                "initial_qty":
                    round(
                        position["initial_qty"],
                        8
                    ),

                "pnl":
                    round(
                        total_pnl,
                        4
                    ),

                "pnl_pct":
                    round(
                        pnl_pct,
                        3
                    ),

                "events":
                    "END_OF_DATA",

                "bars":
                    position["bars"],

                "result":
                    (
                        "WIN"
                        if total_pnl > 0
                        else "LOSS"
                    )
            })

        # ====================================================
        # STATISTICS
        # ====================================================

        if not trades:

            return BacktestResult()

        wins = [
            t
            for t in trades
            if t["result"] == "WIN"
        ]

        losses = [
            t
            for t in trades
            if t["result"] == "LOSS"
        ]

        gross_win = sum(
            max(
                t["pnl"],
                0
            )
            for t in trades
        )

        gross_loss = abs(
            sum(
                min(
                    t["pnl"],
                    0
                )
                for t in trades
            )
        )

        if gross_loss > 0:

            profit_factor = (
                gross_win
                /
                gross_loss
            )

        else:

            profit_factor = (
                gross_win
                if gross_win > 0
                else 0
            )

        win_rate = (
            len(wins)
            /
            len(trades)
            *
            100
        )

        total_pnl = (
            capital
            -
            initial_capital
        )

        total_pnl_pct = (
            total_pnl
            /
            initial_capital
            *
            100
        )

        # ----------------------------------------------------
        # Equity Curve
        # ----------------------------------------------------

        equity_df = pd.DataFrame(
            equity_curve
        )

        if not equity_df.empty:

            equity_df["equity"] = (
                equity_df["equity"]
                .astype(float)
            )

            equity_df["peak"] = (
                equity_df["equity"]
                .cummax()
            )

            equity_df["drawdown"] = (
                (
                    equity_df["equity"]
                    -
                    equity_df["peak"]
                )
                /
                equity_df["peak"]
                *
                100
            )

            max_drawdown = abs(
                equity_df["drawdown"].min()
            )

        # ----------------------------------------------------
        # Sharpe
        # ----------------------------------------------------

        trade_returns = np.array([
            t["pnl_pct"]
            for t in trades
        ])

        if (
            len(trade_returns) > 1
            and
            np.std(trade_returns) > 0
        ):

            sharpe = (
                np.mean(trade_returns)
                /
                np.std(
                    trade_returns,
                    ddof=1
                )
                *
                np.sqrt(
                    len(trade_returns)
                )
            )

        else:

            sharpe = 0.0

        avg_win = (
            np.mean(
                [
                    t["pnl_pct"]
                    for t in wins
                ]
            )
            if wins
            else 0.0
        )

        avg_loss = (
            np.mean(
                [
                    t["pnl_pct"]
                    for t in losses
                ]
            )
            if losses
            else 0.0
        )

        return BacktestResult(

            total_trades=len(trades),

            winning_trades=len(wins),

            losing_trades=len(losses),

            win_rate=win_rate,

            total_pnl=total_pnl,

            total_pnl_pct=total_pnl_pct,

            max_drawdown_pct=max_drawdown,

            profit_factor=profit_factor,

            sharpe_ratio=sharpe,

            avg_win_pct=avg_win,

            avg_loss_pct=avg_loss,

            trades_log=trades,

            equity_curve=equity_curve
        )


# ============================================================
# 13. CSS
# ============================================================

def render_css():

    st.markdown(
        """
        <style>

        .stSidebar,
        div[data-testid="stSidebar"],
        div[data-testid="stSidebar"] * {
            word-break: normal !important;
            word-wrap: normal !important;
            white-space: normal !important;
        }

        .status {
            padding: 4px 8px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
        }

        .decision {
            background: #1E222D;
            padding: 18px;
            border-radius: 8px;
            border-left: 5px solid;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    render_css()

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.title("⚡ AliQuantFund")

        st.caption(
            "Institutional Core Quant Engine v4.4"
        )

        st.markdown("---")

        symbol = st.selectbox(
            "Symbol",
            [
                "BTC/USDT",
                "ETH/USDT",
                "ZEC/USDT",
                "SOL/USDT",
                "XRP/USDT",
                "BNB/USDT",
                "SUI/USDT",
                "LTC/USDT"
            ]
        )

        timeframe = st.selectbox(
            "Execution Timeframe",
            [
                "5m",
                "15m",
                "1h",
                "4h"
            ],
            index=1
        )

        st.markdown("---")

        capital = st.number_input(
            "Capital ($)",
            min_value=10.0,
            value=100.0,
            step=10.0
        )

        risk_pct = st.number_input(
            "Risk per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.5
        )

        st.markdown("---")

        st.subheader(
            "Execution Model"
        )

        fee_pct = st.number_input(
            "Trading Fee (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01
        )

        slippage_pct = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01
        )

        max_bars = st.number_input(
            "Maximum Bars in Trade",
            min_value=5,
            max_value=500,
            value=48,
            step=5
        )

        st.markdown("---")

        auto_refresh = st.checkbox(
            "Auto Refresh",
            value=True
        )

        st.info(
            "النظام لا ينفذ الصفقات. "
            "هو محرك تحليل وإشارات وإدارة صفقة "
            "واختبار خلفي."
        )

    # ========================================================
    # DATA
    # ========================================================

    with st.spinner(
        f"جاري تحليل {symbol}..."
    ):

        df, spot_status = (
            MarketDataLoader.fetch_klines(
                symbol,
                timeframe,
                500
            )
        )

        futures, futures_status, funding = (
            MarketDataLoader
            .fetch_futures_metrics(
                symbol,
                timeframe,
                50
            )
        )

        trades, cvd_status = (
            MarketDataLoader
            .fetch_recent_trades(
                symbol
            )
        )

    if (
        df is None
        or df.empty
    ):

        st.error(
            "❌ تعذر الحصول على بيانات السوق."
        )

        return

    # ========================================================
    # INDICATORS
    # ========================================================

    df = (
        QuantitativeEngine
        .ichimoku(df)
    )

    atr_series = (
        QuantitativeEngine
        .atr(df)
    )

    atr = float(
        atr_series.iloc[-1]
    )

    vwap_session = (
        QuantitativeEngine.vwap(
            df,
            "SESSION"
        )
    )

    vwap_weekly = (
        QuantitativeEngine.vwap(
            df,
            "WEEKLY"
        )
    )

    vwap_monthly = (
        QuantitativeEngine.vwap(
            df,
            "MONTHLY"
        )
    )

    anchor_idx, anchor_reason = (
        QuantitativeEngine.smart_anchor(
            df
        )
    )

    avwap = (
        QuantitativeEngine
        .anchored_vwap(
            df,
            anchor_idx
        )
    )

    cvd_series, cvd_type, cvd_stats = (
        QuantitativeEngine.cvd(
            df,
            trades,
            timeframe
        )
    )

    # ========================================================
    # OI
    # ========================================================

    oi_change = 0.0

    if (
        futures is not None
        and
        len(futures) >= 2
    ):

        start = float(
            futures[
                "openInterest"
            ].iloc[0]
        )

        end = float(
            futures[
                "openInterest"
            ].iloc[-1]
        )

        if start != 0:

            oi_change = (
                (
                    end - start
                )
                /
                start
                *
                100
            )

    # ========================================================
    # SAFE LAST
    # ========================================================

    def safe_last(
        series,
        fallback
    ):

        value = series.iloc[-1]

        if pd.isna(value):

            return fallback

        return float(value)

    close = float(
        df["close"].iloc[-1]
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = QuantitativeMetrics(

        vwap_session=safe_last(
            vwap_session,
            close
        ),

        vwap_weekly=safe_last(
            vwap_weekly,
            close
        ),

        vwap_monthly=safe_last(
            vwap_monthly,
            close
        ),

        vwap_anchored=(
            safe_last(
                avwap,
                close
            )
            if not avwap.isna().all()
            else None
        ),

        atr_14=atr,

        cvd_slope=(
            cvd_stats["slope"]
        ),

        cvd_divergence=(
            cvd_stats["divergence"]
        ),

        oi_change_pct=oi_change,

        funding_rate=funding.get(
            "current"
        ),

        tenkan=safe_last(
            df["tenkan"],
            close
        ),

        kijun=safe_last(
            df["kijun"],
            close
        ),

        span_a=safe_last(
            df["span_a"],
            close
        ),

        span_b=safe_last(
            df["span_b"],
            close
        )
    )

    # ========================================================
    # MARKET STATE
    # ========================================================

    market_state = (
        MarketStateEngine.classify(
            df,
            atr
        )
    )

    # ========================================================
    # MTF
    # ========================================================

    htf = (
        MultiTimeframeEngine.evaluate(
            symbol
        )
    )

    # ========================================================
    # SETUP
    # ========================================================

    setup = SetupEngine.detect(
        df,
        metrics,
        market_state,
        htf
    )

    # ========================================================
    # TRIGGER
    # ========================================================

    trigger = TriggerEngine.detect(
        df,
        metrics,
        setup,
        htf
    )

    # ========================================================
    # SCORE
    # ========================================================

    scoring = (
        FactorScoringEngine.score(
            market_state,
            df,
            metrics,
            htf,
            futures_status,
            cvd_type
        )
    )

    # ========================================================
    # GRADE
    # ========================================================

    grade = SignalEngine.grade(
        scoring.total_score,
        setup,
        trigger,
        scoring.data_quality_pct,
        htf
    )

    # ========================================================
    # FINAL DIRECTION
    # ========================================================

    if (
        trigger
        ==
        TriggerType.CONFIRMED_BUY
        and
        scoring.total_score >= 15
        and
        scoring.data_quality_pct >= 60
    ):

        final_decision = (
            "CONFIRMED LONG"
        )

        direction = "LONG"

    elif (
        trigger
        ==
        TriggerType.CONFIRMED_SELL
        and
        scoring.total_score <= -15
        and
        scoring.data_quality_pct >= 60
    ):

        final_decision = (
            "CONFIRMED SHORT"
        )

        direction = "SHORT"

    else:

        final_decision = (
            "NO TRADE / WAIT"
        )

        direction = "NONE"

    # ========================================================
    # TRADE PLAN
    # ========================================================

    trade_plan = (
        TradeManagement.build(
            df,
            metrics,
            direction,
            capital,
            risk_pct
        )
    )

    # ========================================================
    # HEADER
    # ========================================================

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.markdown(
            f"### {symbol}"
        )

        st.caption(
            f"Price: ${close:,.6f}"
        )

    with c2:

        st.metric(
            "Market State",
            market_state.value
        )

    with c3:

        st.metric(
            "Quant Score",
            f"{scoring.total_score:+.1f}"
        )

    with c4:

        st.metric(
            "Data Quality",
            f"{scoring.data_quality_pct:.0f}%"
        )

    st.markdown("---")

    # ========================================================
    # DECISION
    # ========================================================

    if "LONG" in final_decision:

        color = "#00E676"

    elif "SHORT" in final_decision:

        color = "#FF5252"

    else:

        color = "#FFD600"

    st.markdown(
        f"""
        <div class="decision"
             style="border-color:{color}">

            <h2 style="color:{color}">
                {final_decision}
            </h2>

            <b>Setup:</b>
            {setup.setup.value}

            &nbsp;&nbsp;|&nbsp;&nbsp;

            <b>Trigger:</b>
            {trigger.value}

            <br><br>

            <b>Signal Grade:</b>
            {grade.value}

            &nbsp;&nbsp;|&nbsp;&nbsp;

            <b>HTF:</b>
            {htf["context_bias"]}

            <br><br>

            <b>Setup Reason:</b>
            {setup.reason}

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # MTF
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🧭 Multi-Timeframe Hierarchy"
    )

    cols = st.columns(5)

    for i, tf in enumerate(
        [
            "1d",
            "4h",
            "1h",
            "15m",
            "5m"
        ]
    ):

        frame = (
            htf["frames"][tf]
        )

        cols[i].metric(
            tf.upper(),
            frame["bias"],
            f'{frame["score"]:.0f}'
        )

    # ========================================================
    # SCORE
    # ========================================================

    with st.expander(
        "🧩 Layered Quantitative Scoring",
        expanded=True
    ):

        f1, f2, f3, f4 = (
            st.columns(4)
        )

        f1.metric(
            "Direction",
            f"{scoring.direction_score:+.1f}"
        )

        f2.metric(
            "Flow / CVD",
            f"{scoring.flow_score:+.1f}"
        )

        f3.metric(
            "Positioning",
            f"{scoring.positioning_score:+.1f}"
        )

        f4.metric(
            "Location / VWAP",
            f"{scoring.location_score:+.1f}"
        )

    # ========================================================
    # TRADE MANAGEMENT
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🎯 Trade Management"
    )

    if direction == "NONE":

        st.warning(
            "لا توجد صفقة مؤكدة حالياً."
        )

    else:

        t1, t2, t3, t4 = (
            st.columns(4)
        )

        t1.metric(
            "Entry",
            f"{trade_plan.entry_low:,.6f}"
            f" – "
            f"{trade_plan.entry_high:,.6f}"
        )

        t2.metric(
            "Stop Loss",
            f"{trade_plan.stop_loss:,.6f}"
        )

        t3.metric(
            "TP1",
            f"{trade_plan.tp1:,.6f}"
        )

        t4.metric(
            "R:R TP1",
            f"1:{trade_plan.rr_tp1:.2f}"
        )

        t5, t6, t7, t8 = (
            st.columns(4)
        )

        t5.metric(
            "TP2",
            f"{trade_plan.tp2:,.6f}"
        )

        t6.metric(
            "R:R TP2",
            f"1:{trade_plan.rr_tp2:.2f}"
        )

        t7.metric(
            "TP3",
            f"{trade_plan.tp3:,.6f}"
        )

        t8.metric(
            "Position Size",
            f"{trade_plan.position_size:.6f}"
        )

        st.info(
            f"💰 Risk Amount: "
            f"${trade_plan.risk_amount:.2f}"
            f" | "
            f"{trade_plan.invalidation}"
        )

    # ========================================================
    # BACKTEST
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🧪 Backtesting Engine v4.4"
    )

    bt_result = (
        BacktestEngine.run_backtest(
            df,
            capital,
            risk_pct,
            fee_pct,
            slippage_pct,
            max_bars_in_trade=max_bars
        )
    )

    b1, b2, b3, b4, b5 = (
        st.columns(5)
    )

    b1.metric(
        "Trades",
        bt_result.total_trades
    )

    b2.metric(
        "Win Rate",
        f"{bt_result.win_rate:.1f}%"
    )

    b3.metric(
        "Return",
        f"{bt_result.total_pnl_pct:+.2f}%"
    )

    b4.metric(
        "Max DD",
        f"{bt_result.max_drawdown_pct:.2f}%"
    )

    b5.metric(
        "Profit Factor",
        f"{bt_result.profit_factor:.2f}"
    )

    b6, b7, b8 = (
        st.columns(3)
    )

    b6.metric(
        "Sharpe",
        f"{bt_result.sharpe_ratio:.2f}"
    )

    b7.metric(
        "Average Win",
        f"{bt_result.avg_win_pct:+.2f}%"
    )

    b8.metric(
        "Average Loss",
        f"{bt_result.avg_loss_pct:+.2f}%"
    )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    if bt_result.equity_curve:

        st.subheader(
            "📈 Backtest Equity Curve"
        )

        equity_df = pd.DataFrame(
            bt_result.equity_curve
        )

        fig_eq = go.Figure()

        fig_eq.add_trace(
            go.Scatter(
                x=equity_df["time"],
                y=equity_df["equity"],
                mode="lines",
                name="Equity"
            )
        )

        fig_eq.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Time",
            yaxis_title="Capital",
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            )
        )

        st.plotly_chart(
            fig_eq,
            use_container_width=True
        )

    # ========================================================
    # TRADE LOG
    # ========================================================

    if bt_result.trades_log:

        with st.expander(
            "📋 Detailed Backtest Trade Log",
            expanded=False
        ):

            st.dataframe(
                pd.DataFrame(
                    bt_result.trades_log
                ),
                use_container_width=True
            )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    with st.expander(
        "🔧 Engine Diagnostics",
        expanded=False
    ):

        diagnostics = {

            "Market State":
                market_state.value,

            "Setup":
                setup.setup.value,

            "Setup Reason":
                setup.reason,

            "Trigger":
                trigger.value,

            "Signal Grade":
                grade.value,

            "HTF Context":
                htf["context_bias"],

            "Execution Score":
                round(
                    htf["execution_score"],
                    1
                ),

            "CVD Type":
                cvd_type,

            "CVD Slope":
                round(
                    metrics.cvd_slope,
                    5
                ),

            "CVD Divergence":
                metrics.cvd_divergence,

            "OI Change":
                round(
                    metrics.oi_change_pct,
                    2
                ),

            "Funding":
                metrics.funding_rate,

            "Anchor":
                anchor_reason,

            "Spot Data":
                spot_status,

            "Futures Data":
                futures_status,

            "Data Quality":
                scoring.data_quality_pct,

            "Trading Fee":
                fee_pct,

            "Slippage":
                slippage_pct,

            "Max Bars":
                max_bars
        }

        st.json(
            diagnostics
        )

    # ========================================================
    # VWAP PANEL
    # ========================================================

    with st.expander(
        "📊 VWAP / Market Location",
        expanded=False
    ):

        v1, v2, v3, v4 = (
            st.columns(4)
        )

        v1.metric(
            "Session VWAP",
            f"{metrics.vwap_session:,.6f}"
        )

        v2.metric(
            "Weekly VWAP",
            f"{metrics.vwap_weekly:,.6f}"
        )

        v3.metric(
            "Monthly VWAP",
            f"{metrics.vwap_monthly:,.6f}"
        )

        if metrics.vwap_anchored:

            v4.metric(
                "Anchored VWAP",
                f"{metrics.vwap_anchored:,.6f}"
            )

    # ========================================================
    # CHART
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"📈 {symbol} — Institutional Chart"
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[
            0.70,
            0.30
        ]
    )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price"
        ),
        row=1,
        col=1
    )

    # --------------------------------------------------------
    # Ichimoku
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["tenkan"],
            name="Tenkan",
            mode="lines"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["kijun"],
            name="Kijun",
            mode="lines"
        ),
        row=1,
        col=1
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=vwap_session,
            name="Session VWAP",
            mode="lines"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=vwap_weekly,
            name="Weekly VWAP",
            mode="lines",
            line=dict(
                dash="dash"
            )
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=vwap_monthly,
            name="Monthly VWAP",
            mode="lines",
            line=dict(
                dash="dot"
            )
        ),
        row=1,
        col=1
    )

    # --------------------------------------------------------
    # Anchored VWAP
    # --------------------------------------------------------

    if not avwap.isna().all():

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=avwap,
                name="Smart Anchored VWAP",
                mode="lines"
            ),
            row=1,
            col=1
        )

    # --------------------------------------------------------
    # CVD
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=cvd_series,
            name=f"CVD ({cvd_type})",
            mode="lines"
        ),
        row=2,
        col=1
    )

    # --------------------------------------------------------
    # Trade Levels
    # --------------------------------------------------------

    if direction != "NONE":

        fig.add_hline(
            y=trade_plan.stop_loss,
            line_dash="dash",
            annotation_text="SL",
            row=1,
            col=1
        )

        fig.add_hline(
            y=trade_plan.tp1,
            line_dash="dot",
            annotation_text="TP1",
            row=1,
            col=1
        )

        fig.add_hline(
            y=trade_plan.tp2,
            line_dash="dot",
            annotation_text="TP2",
            row=1,
            col=1
        )

        fig.add_hline(
            y=trade_plan.tp3,
            line_dash="dot",
            annotation_text="TP3",
            row=1,
            col=1
        )

    fig.update_layout(
        template="plotly_dark",
        height=750,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 15. RUN
# ============================================================

if __name__ == "__main__":

    main()
