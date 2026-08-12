# -*- coding: utf-8 -*-
"""
⚡ AliQuantFund Institutional Architecture v4.4
================================================

MASTER QUANT ENGINE

DATA
 ├─ Binance Spot
 ├─ Bybit Spot fallback
 ├─ Binance Futures OI + Funding
 └─ Binance Trade Tape

ANALYSIS
 ├─ ATR
 ├─ Ichimoku
 ├─ Session / Weekly / Monthly VWAP
 ├─ Smart Anchored VWAP
 ├─ CVD
 ├─ OI
 ├─ Funding
 └─ Multi-Timeframe

DECISION
 ├─ Market State
 ├─ Setup Detection
 ├─ Trigger
 ├─ Quant Score
 ├─ Signal Grade
 └─ Trade Management

BACKTEST
 ├─ Same core logic as live engine
 ├─ Fees
 ├─ Slippage
 ├─ Risk based position sizing
 ├─ SL / TP
 ├─ Equity curve
 ├─ Drawdown
 ├─ Profit Factor
 └─ Detailed Trade Log

IMPORTANT
-----------
Backtest is intentionally conservative:
If SL and TP are both touched inside the same candle,
SL is assumed to execute first.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 0. CONFIG
# ============================================================

st.set_page_config(
    page_title="AliQuantFund v4.4",
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
    CONFIRMED_BUY = "BUY"
    CONFIRMED_SELL = "SELL"
    WAIT = "WAIT"
    INVALID = "INVALID"


class SignalGrade(Enum):
    INSTITUTIONAL_STRONG = "A+"
    CONFIRMED = "A"
    MODERATE = "B"
    NEUTRAL = "C"
    NO_TRADE = "NO TRADE"


class DataStatus(Enum):
    LIVE = "LIVE"
    FALLBACK = "FALLBACK"
    APPROXIMATED = "APPROXIMATED"
    UNAVAILABLE = "UNAVAILABLE"


# ============================================================
# 2. DATA CLASSES
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

    invalidation: str = ""


@dataclass
class BacktestResult:

    initial_capital: float = 0.0
    final_capital: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    win_rate: float = 0.0

    total_pnl_pct: float = 0.0

    max_drawdown_pct: float = 0.0

    profit_factor: float = 0.0

    average_trade_pct: float = 0.0

    expectancy_pct: float = 0.0

    trades_log: List[Dict[str, Any]] = field(
        default_factory=list
    )

    equity_curve: List[Dict[str, Any]] = field(
        default_factory=list
    )


# ============================================================
# 3. MARKET DATA
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
    ):

        symbol = symbol.replace("/", "").upper()

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        urls = [

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
                "https://api3.binance.com/api/v3/klines"
                f"?symbol={symbol}"
                f"&interval={interval}"
                f"&limit={limit}"
            )
        ]

        for url in urls:

            try:

                r = requests.get(
                    url,
                    headers=headers,
                    timeout=5
                )

                if r.status_code != 200:
                    continue

                data = r.json()

                if not data:
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
                    ].dropna(),
                    DataStatus.LIVE.value
                )

            except Exception:
                continue

        # ----------------------------------------------------
        # BYBIT FALLBACK
        # ----------------------------------------------------

        try:

            tf = MarketDataLoader.BYBIT_TF_MAP.get(
                interval,
                "5"
            )

            url = (
                "https://api.bybit.com/v5/market/kline"
                f"?category=spot"
                f"&symbol={symbol}"
                f"&interval={tf}"
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
                f"Bybit error: {e}"
            )

        return (
            None,
            DataStatus.UNAVAILABLE.value
        )

    # --------------------------------------------------------
    # FUTURES
    # --------------------------------------------------------

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_futures_metrics(
        symbol: str,
        interval: str,
        limit: int = 50
    ):

        symbol = symbol.replace("/", "").upper()

        funding = {
            "available": False,
            "current": None,
            "history": []
        }

        headers = {
            "User-Agent": "Mozilla/5.0"
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
                "openInterestHist"
                f"?symbol={symbol}"
                f"&period={oi_interval}"
                f"&limit={limit}"
            )

            url_fr = (
                "https://fapi.binance.com/fapi/v1/"
                f"fundingRate"
                f"?symbol={symbol}"
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

                    if (
                        r_fr.status_code == 200
                    ):

                        fr = r_fr.json()

                        if isinstance(fr, list) and fr:

                            history = [
                                float(
                                    x.get(
                                        "fundingRate",
                                        0
                                    )
                                )
                                for x in fr
                            ]

                            history.reverse()

                            if history:

                                funding["available"] = True
                                funding["current"] = history[0]
                                funding["history"] = history

                    return (
                        df,
                        DataStatus.LIVE.value,
                        funding
                    )

        except Exception:
            pass

        # ----------------------------------------------------
        # BYBIT
        # ----------------------------------------------------

        try:

            tf = MarketDataLoader.BYBIT_TF_MAP.get(
                interval,
                "5"
            )

            url = (
                "https://api.bybit.com/v5/market/"
                "open-interest"
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

        except Exception:
            pass

        return (
            None,
            DataStatus.UNAVAILABLE.value,
            funding
        )

    # --------------------------------------------------------
    # TRADE TAPE
    # --------------------------------------------------------

    @staticmethod
    @st.cache_data(ttl=10)
    def fetch_recent_trades(
        symbol: str,
        limit: int = 1000
    ):

        symbol = symbol.replace("/", "").upper()

        urls = [

            (
                "https://data-api.binance.vision/api/v3/trades"
                f"?symbol={symbol}&limit={limit}"
            ),

            (
                "https://api1.binance.com/api/v3/trades"
                f"?symbol={symbol}&limit={limit}"
            )
        ]

        for url in urls:

            try:

                r = requests.get(
                    url,
                    timeout=5
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

    @staticmethod
    def atr(
        df: pd.DataFrame,
        period: int = 14
    ):

        prev_close = df["close"].shift(1)

        tr = pd.concat(
            [

                df["high"] - df["low"],

                (
                    df["high"]
                    - prev_close
                ).abs(),

                (
                    df["low"]
                    - prev_close
                ).abs()

            ],
            axis=1
        ).max(axis=1)

        return (
            tr.ewm(
                alpha=1 / period,
                adjust=False
            )
            .mean()
            .bfill()
        )

    # --------------------------------------------------------
    # ICHIMOKU
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    @staticmethod
    def vwap(
        df,
        mode
    ):

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

    # --------------------------------------------------------
    # ANCHORED VWAP
    # --------------------------------------------------------

    @staticmethod
    def smart_anchor(df):

        if len(df) < 30:
            return 0, "Initial"

        d = df.tail(
            min(150, len(df))
        )

        mean = d["volume"].mean()

        std = d["volume"].std()

        z = (
            d["volume"] - mean
        ) / (
            std + 1e-9
        )

        idx = z.idxmax()

        return (
            int(idx),
            f"Volume Spike Z={z.loc[idx]:.2f}"
        )

    @staticmethod
    def anchored_vwap(
        df,
        anchor
    ):

        typical = (
            df["high"]
            +
            df["low"]
            +
            df["close"]
        ) / 3

        pv = typical * df["volume"]

        volume = df["volume"].copy()

        pv = pv.copy()

        if anchor > 0:

            pv.iloc[:anchor] = 0
            volume.iloc[:anchor] = 0

        result = (
            pv.cumsum()
            /
            volume.cumsum().replace(
                0,
                np.nan
            )
        )

        if anchor > 0:
            result.iloc[:anchor] = np.nan

        return result

    # --------------------------------------------------------
    # CVD
    # --------------------------------------------------------

    @staticmethod
    def cvd(
        df,
        trades=None,
        timeframe="5m"
    ):

        if (
            trades is None
            or trades.empty
        ):

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
                DataStatus.APPROXIMATED.value
            )

        t = trades.copy()

        t["signed"] = np.where(
            t["is_buy"],
            t["qty"],
            -t["qty"]
        )

        rules = {
            "5m": "5min",
            "15m": "15min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D"
        }

        rule = rules.get(
            timeframe,
            "5min"
        )

        delta = (
            t.set_index("time")[
                "signed"
            ]
            .resample(rule)
            .sum()
        )

        cvd = delta.cumsum()

        target_index = pd.DatetimeIndex(
            df["timestamp"]
        )

        cvd = (
            cvd.reindex(
                target_index,
                method="ffill"
            )
            .fillna(0)
        )

        return (
            cvd.reset_index(drop=True),
            DataStatus.LIVE.value
        )

    # --------------------------------------------------------
    # CVD STATS
    # --------------------------------------------------------

    @staticmethod
    def cvd_stats(
        df,
        cvd
    ):

        if len(df) < 10:
            return {
                "slope": 0.0,
                "divergence": "NONE"
            }

        price_change = (
            df["close"].iloc[-1]
            -
            df["close"].iloc[-10]
        )

        cvd_change = (
            cvd.iloc[-1]
            -
            cvd.iloc[-10]
        )

        recent = max(
            abs(cvd.iloc[-5]),
            1e-9
        )

        slope = (
            cvd.iloc[-1]
            -
            cvd.iloc[-5]
        ) / recent

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
                np.clip(
                    slope,
                    -5,
                    5
                )
            ),
            "divergence": divergence
        }


# ============================================================
# 5. MARKET STATE
# ============================================================

class MarketStateEngine:

    @staticmethod
    def classify(
        df,
        atr
    ):

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
            df["high"]
            .tail(10)
            .max()
            -
            df["low"]
            .tail(10)
            .min()
        )

        if recent_range > 4.0 * atr:

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

        return MarketState.RANGE_COMPRESSION


# ============================================================
# 6. MULTI TIMEFRAME
# ============================================================

class MultiTimeframeEngine:

    @staticmethod
    def evaluate(
        symbol
    ):

        result = {}

        for tf in [
            "1d",
            "4h",
            "1h",
            "15m",
            "5m"
        ]:

            df, status = (
                MarketDataLoader.fetch_klines(
                    symbol,
                    tf,
                    150
                )
            )

            if (
                df is None
                or len(df) < 80
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

            if pd.isna(
                span_a
            ):

                span_a = close

            if pd.isna(
                span_b
            ):

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
                score = 85

            elif (
                close < cloud_low
                and tenkan < kijun
            ):

                bias = "BEARISH"
                score = 15

            elif close > kijun:

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

        context_score = (
            result["1d"]["score"] * 0.55
            +
            result["4h"]["score"] * 0.45
        )

        if context_score >= 65:

            context_bias = "BULLISH"

        elif context_score <= 35:

            context_bias = "BEARISH"

        else:

            context_bias = "NEUTRAL"

        execution_score = (
            result["15m"]["score"] * 0.50
            +
            result["5m"]["score"] * 0.50
        )

        return {
            "frames": result,
            "context_score": context_score,
            "context_bias": context_bias,
            "direction_bias": result["1h"]["bias"],
            "execution_score": execution_score
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
        # RECLAIM
        # ----------------------------------------------------

        bullish_reclaim = (
            prev_close < vwap
            and close > vwap
        )

        bearish_reclaim = (
            prev_close > vwap
            and close < vwap
        )

        if bullish_reclaim:

            return SetupResult(
                SetupType.RECLAIM,
                "Bullish VWAP reclaim",
                "LONG",
                80
            )

        if bearish_reclaim:

            return SetupResult(
                SetupType.RECLAIM,
                "Bearish VWAP loss",
                "SHORT",
                80
            )

        # ----------------------------------------------------
        # BREAKOUT
        # ----------------------------------------------------

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

        volume_ratio = (
            df["volume"].iloc[-1]
            /
            max(volume_avg, 1e-9)
        )

        if (
            close > resistance
            and volume_ratio >= 1.5
        ):

            return SetupResult(
                SetupType.BREAKOUT,
                "Resistance breakout with volume expansion",
                "LONG",
                85
            )

        if (
            close < support
            and volume_ratio >= 1.5
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
            and close < vwap
        ):

            return SetupResult(
                SetupType.REJECTION,
                "Upper wick rejection below VWAP",
                "SHORT",
                70
            )

        if (
            lower_wick > atr * 0.7
            and close > vwap
        ):

            return SetupResult(
                SetupType.REJECTION,
                "Lower wick rejection above VWAP",
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
            "No structural setup",
            "NONE",
            0
        )


# ============================================================
# 8. TRIGGER
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
            == "BULLISH_ABSORPTION"
        )

        bearish_flow = (
            metrics.cvd_slope < 0
            or
            metrics.cvd_divergence
            == "BEARISH_ABSORPTION"
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
                == "BEARISH_ABSORPTION"
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
                == "BULLISH_ABSORPTION"
            ):

                return TriggerType.INVALID

            return TriggerType.WAIT

        return TriggerType.WAIT


# ============================================================
# 9. SCORING
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
            == "BULLISH_ABSORPTION"
        ):

            flow = 20

        elif (
            metrics.cvd_divergence
            == "BEARISH_ABSORPTION"
        ):

            flow = -20

        else:

            flow = float(
                np.clip(
                    metrics.cvd_slope * 50,
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
            != DataStatus.UNAVAILABLE.value
        ):

            if metrics.oi_change_pct > 2:

                if direction > 0:
                    positioning += 15

                elif direction < 0:
                    positioning -= 15

            elif metrics.oi_change_pct < -2:

                if direction > 0:
                    positioning -= 8

                elif direction < 0:
                    positioning += 8

            if metrics.funding_rate is not None:

                if metrics.funding_rate < -0.0001:

                    positioning += 8

                elif metrics.funding_rate > 0.0003:

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
        # WEIGHTS
        # ----------------------------------------------------

        if (
            market_state
            == MarketState.RANGE_COMPRESSION
        ):

            dw = 0.15
            fw = 0.30
            pw = 0.15
            lw = 0.40

        elif market_state in [

            MarketState.TRENDING_BULL,
            MarketState.TRENDING_BEAR

        ]:

            dw = 0.35
            fw = 0.25
            pw = 0.20
            lw = 0.20

        else:

            dw = 0.25
            fw = 0.25
            pw = 0.25
            lw = 0.25

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
            == DataStatus.UNAVAILABLE.value
        ):

            quality -= 25

        elif (
            futures_status
            == DataStatus.FALLBACK.value
        ):

            quality -= 10

        if (
            cvd_status
            == DataStatus.APPROXIMATED.value
        ):

            quality -= 15

        elif (
            cvd_status
            == DataStatus.UNAVAILABLE.value
        ):

            quality -= 25

        return ScoringBreakdown(

            direction_score=round(
                direction,
                2
            ),

            flow_score=round(
                flow,
                2
            ),

            positioning_score=round(
                positioning,
                2
            ),

            location_score=round(
                location,
                2
            ),

            total_score=round(
                total,
                2
            ),

            data_quality_pct=max(
                0,
                quality
            )
        )


# ============================================================
# 10. SIGNAL
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
            == TriggerType.INVALID
        ):

            return SignalGrade.NO_TRADE

        if (
            setup.setup
            == SetupType.NO_SETUP
        ):

            return SignalGrade.NEUTRAL

        if quality < 60:

            return SignalGrade.NO_TRADE

        absolute = abs(score)

        if (
            absolute >= 30
            and quality >= 85
            and trigger in [

                TriggerType.CONFIRMED_BUY,
                TriggerType.CONFIRMED_SELL

            ]
        ):

            return (
                SignalGrade.INSTITUTIONAL_STRONG
            )

        if (
            absolute >= 22
            and trigger in [

                TriggerType.CONFIRMED_BUY,
                TriggerType.CONFIRMED_SELL

            ]
        ):

            return SignalGrade.CONFIRMED

        if absolute >= 15:

            return SignalGrade.MODERATE

        return SignalGrade.NEUTRAL


# ============================================================
# 11. TRADE MANAGEMENT
# ============================================================

class TradeManagement:

    @staticmethod
    def build(
        price,
        atr,
        direction,
        capital,
        risk_pct
    ):

        if direction not in [
            "LONG",
            "SHORT"
        ]:

            return TradePlan()

        atr = max(
            atr,
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
            atr * 0.10
        )

        entry_high = (
            price
            +
            atr * 0.10
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

        risk_per_unit = abs(
            entry - stop
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

            return abs(
                target - entry
            ) / max(
                risk_per_unit,
                1e-9
            )

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

            invalidation=(
                f"{direction} invalidation "
                f"at {stop:.6f}"
            )
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

        slippage_pct=0.03,

        min_score=15,

        use_trend_filter=True

    ) -> BacktestResult:

        if (
            df is None
            or len(df) < 100
        ):

            return BacktestResult(
                initial_capital=initial_capital,
                final_capital=initial_capital
            )

        d = df.copy().reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # PRE-CALCULATE INDICATORS
        # ----------------------------------------------------

        d = (
            QuantitativeEngine
            .ichimoku(d)
        )

        d["atr"] = (
            QuantitativeEngine
            .atr(d)
        )

        d["vwap_session"] = (
            QuantitativeEngine
            .vwap(
                d,
                "SESSION"
            )
        )

        d["vwap_weekly"] = (
            QuantitativeEngine
            .vwap(
                d,
                "WEEKLY"
            )
        )

        d["vwap_monthly"] = (
            QuantitativeEngine
            .vwap(
                d,
                "MONTHLY"
            )
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        capital = float(
            initial_capital
        )

        peak = capital

        max_drawdown = 0.0

        trades = []

        equity_curve = []

        in_position = False

        direction = None

        entry_price = 0.0

        stop_loss = 0.0

        tp1 = 0.0

        position_size = 0.0

        entry_time = None

        entry_index = None

        # ----------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------

        start_index = 80

        for i in range(
            start_index,
            len(d)
        ):

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
                row["vwap_session"]
            )

            if not np.isfinite(
                atr
            ) or atr <= 0:

                continue

            if not np.isfinite(
                vwap
            ):

                continue

            # =================================================
            # MANAGE ACTIVE POSITION
            # =================================================

            if in_position:

                exit_price = None

                exit_reason = None

                # ---------------------------------------------
                # LONG
                # ---------------------------------------------

                if direction == "LONG":

                    hit_sl = (
                        low <= stop_loss
                    )

                    hit_tp = (
                        high >= tp1
                    )

                    # Conservative assumption:
                    # if both occur in same candle -> SL first

                    if hit_sl:

                        exit_price = stop_loss

                        exit_reason = "SL"

                    elif hit_tp:

                        exit_price = tp1

                        exit_reason = "TP1"

                # ---------------------------------------------
                # SHORT
                # ---------------------------------------------

                elif direction == "SHORT":

                    hit_sl = (
                        high >= stop_loss
                    )

                    hit_tp = (
                        low <= tp1
                    )

                    if hit_sl:

                        exit_price = stop_loss

                        exit_reason = "SL"

                    elif hit_tp:

                        exit_price = tp1

                        exit_reason = "TP1"

                # ---------------------------------------------
                # EXIT
                # ---------------------------------------------

                if exit_price is not None:

                    if direction == "LONG":

                        raw_pnl = (
                            exit_price
                            -
                            entry_price
                        ) * position_size

                    else:

                        raw_pnl = (
                            entry_price
                            -
                            exit_price
                        ) * position_size

                    # Notional based fees
                    entry_notional = (
                        entry_price
                        *
                        position_size
                    )

                    exit_notional = (
                        exit_price
                        *
                        position_size
                    )

                    fees = (
                        entry_notional
                        +
                        exit_notional
                    ) * (
                        fee_pct / 100
                    )

                    net_pnl = (
                        raw_pnl
                        -
                        fees
                    )

                    capital += net_pnl

                    pnl_pct = (
                        net_pnl
                        /
                        max(
                            entry_notional,
                            1e-9
                        )
                    ) * 100

                    trades.append({

                        "entry_time":
                            entry_time,

                        "exit_time":
                            row["timestamp"],

                        "type":
                            direction,

                        "entry":
                            round(
                                entry_price,
                                6
                            ),

                        "exit":
                            round(
                                exit_price,
                                6
                            ),

                        "stop":
                            round(
                                stop_loss,
                                6
                            ),

                        "tp":
                            round(
                                tp1,
                                6
                            ),

                        "size":
                            round(
                                position_size,
                                6
                            ),

                        "gross_pnl":
                            round(
                                raw_pnl,
                                4
                            ),

                        "fees":
                            round(
                                fees,
                                4
                            ),

                        "net_pnl":
                            round(
                                net_pnl,
                                4
                            ),

                        "pnl_pct":
                            round(
                                pnl_pct,
                                3
                            ),

                        "result":
                            (
                                "WIN"
                                if net_pnl > 0
                                else "LOSS"
                            ),

                        "reason":
                            exit_reason,

                        "bars_held":
                            i - entry_index

                    })

                    in_position = False

                    direction = None

                    entry_price = 0.0

                    stop_loss = 0.0

                    tp1 = 0.0

                    position_size = 0.0

                    entry_time = None

                    entry_index = None

            # =================================================
            # EQUITY TRACKING
            # =================================================

            peak = max(
                peak,
                capital
            )

            drawdown = (
                peak - capital
            ) / max(
                peak,
                1e-9
            ) * 100

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

            equity_curve.append({

                "time":
                    row["timestamp"],

                "equity":
                    capital,

                "drawdown":
                    drawdown

            })

            # =================================================
            # NO NEW TRADE IF ACTIVE
            # =================================================

            if in_position:
                continue

            # =================================================
            # LOCAL FEATURES
            # =================================================

            prev_close = float(
                d["close"].iloc[i - 1]
            )

            prev_vwap = float(
                d["vwap_session"].iloc[i - 1]
            )

            ema20 = (
                d["close"]
                .iloc[
                    max(0, i - 100):
                    i + 1
                ]
                .ewm(
                    span=20,
                    adjust=False
                )
                .mean()
                .iloc[-1]
            )

            ema50 = (
                d["close"]
                .iloc[
                    max(0, i - 150):
                    i + 1
                ]
                .ewm(
                    span=50,
                    adjust=False
                )
                .mean()
                .iloc[-1]
            )

            # =================================================
            # SETUP
            # =================================================

            bullish_reclaim = (
                prev_close < prev_vwap
                and close > vwap
            )

            bearish_reclaim = (
                prev_close > prev_vwap
                and close < vwap
            )

            resistance = (
                d["high"]
                .iloc[
                    i - 20:i
                ]
                .max()
            )

            support = (
                d["low"]
                .iloc[
                    i - 20:i
                ]
                .min()
            )

            volume_avg = (
                d["volume"]
                .iloc[
                    i - 20:i
                ]
                .mean()
            )

            volume_ratio = (
                d["volume"].iloc[i]
                /
                max(
                    volume_avg,
                    1e-9
                )
            )

            bullish_breakout = (
                close > resistance
                and volume_ratio >= 1.5
            )

            bearish_breakdown = (
                close < support
                and volume_ratio >= 1.5
            )

            # =================================================
            # TREND FILTER
            # =================================================

            bullish_trend = (
                close > ema20 > ema50
            )

            bearish_trend = (
                close < ema20 < ema50
            )

            # =================================================
            # ICHIMOKU
            # =================================================

            tenkan = d[
                "tenkan"
            ].iloc[i]

            kijun = d[
                "kijun"
            ].iloc[i]

            span_a = d[
                "span_a"
            ].iloc[i]

            span_b = d[
                "span_b"
            ].iloc[i]

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

            ichimoku_bull = (
                close > cloud_high
                and tenkan > kijun
            )

            ichimoku_bear = (
                close < cloud_low
                and tenkan < kijun
            )

            # =================================================
            # QUANT SCORE
            # =================================================

            score = 0.0

            # Trend
            if bullish_trend:
                score += 10

            elif bearish_trend:
                score -= 10

            # Ichimoku
            if ichimoku_bull:
                score += 10

            elif ichimoku_bear:
                score -= 10

            # VWAP location
            distance_vwap = (
                close - vwap
            ) / max(
                atr,
                1e-9
            )

            if (
                distance_vwap > 0
                and distance_vwap < 2
            ):

                score += 5

            elif (
                distance_vwap < 0
                and distance_vwap > -2
            ):

                score -= 5

            # Volume
            if volume_ratio >= 1.5:

                if close > d[
                    "open"
                ].iloc[i]:

                    score += 5

                else:

                    score -= 5

            # =================================================
            # ENTRY CONDITIONS
            # =================================================

            long_signal = False

            short_signal = False

            setup_name = ""

            # -------------------------------------------------
            # LONG RECLAIM
            # -------------------------------------------------

            if bullish_reclaim:

                setup_name = "VWAP_RECLAIM"

                if (
                    close > ema20
                    and close > kijun
                    and (
                        not use_trend_filter
                        or bullish_trend
                    )
                    and score >= min_score
                ):

                    long_signal = True

            # -------------------------------------------------
            # SHORT RECLAIM
            # -------------------------------------------------

            elif bearish_reclaim:

                setup_name = "VWAP_REJECTION"

                if (
                    close < ema20
                    and close < kijun
                    and (
                        not use_trend_filter
                        or bearish_trend
                    )
                    and score <= -min_score
                ):

                    short_signal = True

            # -------------------------------------------------
            # BREAKOUT LONG
            # -------------------------------------------------

            elif bullish_breakout:

                setup_name = "BREAKOUT_LONG"

                if (
                    close > ema20
                    and close > kijun
                    and score >= min_score
                ):

                    long_signal = True

            # -------------------------------------------------
            # BREAKDOWN SHORT
            # -------------------------------------------------

            elif bearish_breakdown:

                setup_name = "BREAKDOWN_SHORT"

                if (
                    close < ema20
                    and close < kijun
                    and score <= -min_score
                ):

                    short_signal = True

            # =================================================
            # ENTRY
            # =================================================

            if not (
                long_signal
                or
                short_signal
            ):

                continue

            # -------------------------------------------------
            # DIRECTION
            # -------------------------------------------------

            if long_signal:

                direction = "LONG"

            else:

                direction = "SHORT"

            # -------------------------------------------------
            # SL / TP
            # -------------------------------------------------

            risk_distance = (
                atr * 1.5
            )

            reward_distance = (
                atr * 1.5
            )

            # -------------------------------------------------
            # SLIPPAGE
            # -------------------------------------------------

            if direction == "LONG":

                entry_price = (
                    close
                    *
                    (
                        1
                        +
                        slippage_pct / 100
                    )
                )

                stop_loss = (
                    entry_price
                    -
                    risk_distance
                )

                tp1 = (
                    entry_price
                    +
                    reward_distance
                )

            else:

                entry_price = (
                    close
                    *
                    (
                        1
                        -
                        slippage_pct / 100
                    )
                )

                stop_loss = (
                    entry_price
                    +
                    risk_distance
                )

                tp1 = (
                    entry_price
                    -
                    reward_distance
                )

            # -------------------------------------------------
            # POSITION SIZE
            # -------------------------------------------------

            risk_amount = (
                capital
                *
                risk_pct
                /
                100
            )

            position_size = (
                risk_amount
                /
                max(
                    risk_distance,
                    1e-9
                )
            )

            # -------------------------------------------------
            # AVOID IMPOSSIBLE POSITION
            # -------------------------------------------------

            if (
                position_size <= 0
                or
                not np.isfinite(
                    position_size
                )
            ):

                continue

            # -------------------------------------------------
            # ACTIVATE
            # -------------------------------------------------

            in_position = True

            entry_time = (
                row["timestamp"]
            )

            entry_index = i

        # =====================================================
        # FINAL RESULT
        # =====================================================

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

        gross_profit = sum(
            max(
                t["net_pnl"],
                0
            )
            for t in trades
        )

        gross_loss = abs(
            sum(
                min(
                    t["net_pnl"],
                    0
                )
                for t in trades
            )
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                /
                gross_loss
            )

        else:

            profit_factor = (
                gross_profit
                if gross_profit > 0
                else 0
            )

        win_rate = (
            len(wins)
            /
            len(trades)
            *
            100
            if trades
            else 0
        )

        total_return = (
            capital
            -
            initial_capital
        )

        total_return_pct = (
            total_return
            /
            initial_capital
            *
            100
        )

        avg_trade = (
            np.mean(
                [
                    t["pnl_pct"]
                    for t in trades
                ]
            )
            if trades
            else 0
        )

        expectancy = avg_trade

        return BacktestResult(

            initial_capital=initial_capital,

            final_capital=capital,

            total_trades=len(
                trades
            ),

            winning_trades=len(
                wins
            ),

            losing_trades=len(
                losses
            ),

            win_rate=win_rate,

            total_pnl_pct=(
                total_return_pct
            ),

            max_drawdown_pct=(
                max_drawdown
            ),

            profit_factor=(
                profit_factor
            ),

            average_trade_pct=(
                avg_trade
            ),

            expectancy_pct=(
                expectancy
            ),

            trades_log=trades,

            equity_curve=(
                equity_curve
            )
        )


# ============================================================
# 13. CSS
# ============================================================

def render_css():

    st.markdown(
        """
        <style>

        .decision {

            background:
                #1E222D;

            padding:
                18px;

            border-radius:
                8px;

            border-left:
                5px solid;

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

        st.title(
            "⚡ AliQuantFund"
        )

        st.caption(
            "Institutional Engine v4.4"
        )

        st.markdown("---")

        symbol = st.selectbox(
            "Symbol",
            [
                "BTC/USDT",
                "ETH/USDT",
                "ZEC/USDT",
                "SOL/USDT",
                "XRP/USDT"
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
            value=1.0,
            step=0.5
        )

        fee_pct = st.number_input(
            "Fee (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01
        )

        slippage_pct = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            step=0.01
        )

        min_score = st.slider(
            "Minimum Backtest Score",
            5,
            30,
            15
        )

        trend_filter = st.checkbox(
            "Use Trend Filter",
            value=True
        )

        st.markdown("---")

        st.info(
            "Backtest v4.4 يستخدم منطق "
            "Setup + Trend + Ichimoku + VWAP "
            "مع رسوم وانزلاق سعري."
        )

    # ========================================================
    # DATA
    # ========================================================

    with st.spinner(
        f"Analyzing {symbol}..."
    ):

        df, spot_status = (
            MarketDataLoader.fetch_klines(
                symbol,
                timeframe,
                500
            )
        )

        futures, futures_status, funding = (
            MarketDataLoader.fetch_futures_metrics(
                symbol,
                timeframe,
                50
            )
        )

        trades, cvd_status = (
            MarketDataLoader.fetch_recent_trades(
                symbol
            )
        )

    if (
        df is None
        or df.empty
    ):

        st.error(
            "❌ Market data unavailable."
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
        QuantitativeEngine
        .vwap(
            df,
            "SESSION"
        )
    )

    vwap_weekly = (
        QuantitativeEngine
        .vwap(
            df,
            "WEEKLY"
        )
    )

    vwap_monthly = (
        QuantitativeEngine
        .vwap(
            df,
            "MONTHLY"
        )
    )

    anchor_idx, anchor_reason = (
        QuantitativeEngine
        .smart_anchor(df)
    )

    avwap = (
        QuantitativeEngine
        .anchored_vwap(
            df,
            anchor_idx
        )
    )

    cvd_series, cvd_type = (
        QuantitativeEngine
        .cvd(
            df,
            trades,
            timeframe
        )
    )

    cvd_stats = (
        QuantitativeEngine
        .cvd_stats(
            df,
            cvd_series
        )
    )

    # ========================================================
    # OI
    # ========================================================

    oi_change = 0.0

    if (
        futures is not None
        and len(futures) >= 2
    ):

        start_oi = float(
            futures[
                "openInterest"
            ].iloc[0]
        )

        end_oi = float(
            futures[
                "openInterest"
            ].iloc[-1]
        )

        if start_oi != 0:

            oi_change = (
                (
                    end_oi
                    -
                    start_oi
                )
                /
                start_oi
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

        oi_change_pct=(
            oi_change
        ),

        funding_rate=(
            funding.get(
                "current"
            )
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
    # LIVE ENGINE
    # ========================================================

    market_state = (
        MarketStateEngine
        .classify(
            df,
            atr
        )
    )

    htf = (
        MultiTimeframeEngine
        .evaluate(
            symbol
        )
    )

    setup = (
        SetupEngine
        .detect(
            df,
            metrics,
            market_state,
            htf
        )
    )

    trigger = (
        TriggerEngine
        .detect(
            df,
            metrics,
            setup,
            htf
        )
    )

    scoring = (
        FactorScoringEngine
        .score(
            market_state,
            df,
            metrics,
            htf,
            futures_status,
            cvd_type
        )
    )

    grade = (
        SignalEngine
        .grade(
            scoring.total_score,
            setup,
            trigger,
            scoring.data_quality_pct,
            htf
        )
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    if (
        trigger
        == TriggerType.CONFIRMED_BUY
        and scoring.total_score >= 15
        and scoring.data_quality_pct >= 60
    ):

        final_decision = (
            "CONFIRMED LONG"
        )

        direction = "LONG"

    elif (
        trigger
        == TriggerType.CONFIRMED_SELL
        and scoring.total_score <= -15
        and scoring.data_quality_pct >= 60
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
            close,
            atr,
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

    c1.metric(
        "Symbol",
        symbol
    )

    c2.metric(
        "Price",
        f"${close:,.6f}"
    )

    c3.metric(
        "Market State",
        market_state.value
    )

    c4.metric(
        "Quant Score",
        f"{scoring.total_score:+.1f}"
    )

    st.markdown("---")

    # ========================================================
    # DECISION
    # ========================================================

    if "LONG" in final_decision:

        decision_color = (
            "#00E676"
        )

    elif "SHORT" in final_decision:

        decision_color = (
            "#FF5252"
        )

    else:

        decision_color = (
            "#FFD600"
        )

    st.markdown(
        f"""
        <div class="decision"
             style="border-color:{decision_color}">

        <h2 style="color:{decision_color}">
        {final_decision}
        </h2>

        <b>Setup:</b>
        {setup.setup.value}

        &nbsp; | &nbsp;

        <b>Trigger:</b>
        {trigger.value}

        &nbsp; | &nbsp;

        <b>Grade:</b>
        {grade.value}

        <br><br>

        <b>HTF Context:</b>
        {htf["context_bias"]}

        &nbsp; | &nbsp;

        <b>Data Quality:</b>
        {scoring.data_quality_pct:.0f}%

        <br><br>

        <b>Reason:</b>
        {setup.reason}

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # MULTI TF
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🧭 Multi-Timeframe"
    )

    cols = st.columns(5)

    for idx, tf in enumerate(
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

        cols[idx].metric(
            tf.upper(),
            frame["bias"],
            f'{frame["score"]:.0f}'
        )

    # ========================================================
    # SCORING
    # ========================================================

    with st.expander(
        "🧩 Quantitative Scoring",
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
            "No confirmed trade."
        )

    else:

        a, b, c, d_ = (
            st.columns(4)
        )

        a.metric(
            "Entry",
            (
                f"{trade_plan.entry_low:.6f}"
                f" - "
                f"{trade_plan.entry_high:.6f}"
            )
        )

        b.metric(
            "SL",
            f"{trade_plan.stop_loss:.6f}"
        )

        c.metric(
            "TP1",
            f"{trade_plan.tp1:.6f}"
        )

        d_.metric(
            "R:R",
            f"1:{trade_plan.rr_tp1:.2f}"
        )

        e, f, g, h = (
            st.columns(4)
        )

        e.metric(
            "TP2",
            f"{trade_plan.tp2:.6f}"
        )

        f.metric(
            "TP3",
            f"{trade_plan.tp3:.6f}"
        )

        g.metric(
            "Position Size",
            f"{trade_plan.position_size:.6f}"
        )

        h.metric(
            "Risk",
            f"${trade_plan.risk_amount:.2f}"
        )

    # ========================================================
    # BACKTEST
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🧪 Backtesting Engine v4.4"
    )

    st.caption(
        "Historical simulation using the same core "
        "structural logic: VWAP + trend + Ichimoku + volume."
    )

    bt = (
        BacktestEngine.run_backtest(
            df,
            initial_capital=capital,
            risk_pct=risk_pct,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
            min_score=min_score,
            use_trend_filter=trend_filter
        )
    )

    b1, b2, b3, b4, b5 = (
        st.columns(5)
    )

    b1.metric(
        "Trades",
        bt.total_trades
    )

    b2.metric(
        "Win Rate",
        f"{bt.win_rate:.1f}%"
    )

    b3.metric(
        "Return",
        f"{bt.total_pnl_pct:+.2f}%"
    )

    b4.metric(
        "Max DD",
        f"{bt.max_drawdown_pct:.2f}%"
    )

    b5.metric(
        "Profit Factor",
        f"{bt.profit_factor:.2f}"
    )

    x1, x2, x3 = (
        st.columns(3)
    )

    x1.metric(
        "Final Capital",
        f"${bt.final_capital:.2f}"
    )

    x2.metric(
        "Average Trade",
        f"{bt.average_trade_pct:+.3f}%"
    )

    x3.metric(
        "Expectancy",
        f"{bt.expectancy_pct:+.3f}%"
    )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    if bt.equity_curve:

        eq = pd.DataFrame(
            bt.equity_curve
        )

        st.subheader(
            "📈 Equity Curve"
        )

        fig_eq = go.Figure()

        fig_eq.add_trace(
            go.Scatter(
                x=eq["time"],
                y=eq["equity"],
                mode="lines",
                name="Equity"
            )
        )

        fig_eq.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Time",
            yaxis_title="Capital"
        )

        st.plotly_chart(
            fig_eq,
            use_container_width=True
        )

    # ========================================================
    # TRADE LOG
    # ========================================================

    if bt.trades_log:

        with st.expander(
            "📋 Backtest Trade Log",
            expanded=False
        ):

            st.dataframe(
                pd.DataFrame(
                    bt.trades_log
                ),
                use_container_width=True
            )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    with st.expander(
        "🔧 Diagnostics",
        expanded=False
    ):

        st.json({

            "Market State":
                market_state.value,

            "Setup":
                setup.setup.value,

            "Trigger":
                trigger.value,

            "Grade":
                grade.value,

            "HTF":
                htf["context_bias"],

            "Execution Score":
                round(
                    htf["execution_score"],
                    2
                ),

            "CVD":
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
                    3
                ),

            "Funding":
                metrics.funding_rate,

            "Spot":
                spot_status,

            "Futures":
                futures_status,

            "Backtest Fee":
                fee_pct,

            "Backtest Slippage":
                slippage_pct,

            "Backtest Minimum Score":
                min_score

        })

    # ========================================================
    # VWAP
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
            f"{metrics.vwap_session:.6f}"
        )

        v2.metric(
            "Weekly VWAP",
            f"{metrics.vwap_weekly:.6f}"
        )

        v3.metric(
            "Monthly VWAP",
            f"{metrics.vwap_monthly:.6f}"
        )

        if (
            metrics.vwap_anchored
            is not None
        ):

            v4.metric(
                "Anchored VWAP",
                f"{metrics.vwap_anchored:.6f}"
            )

    # ========================================================
    # CHART
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"📈 {symbol} Institutional Chart"
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
    # PRICE
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
    # ICHIMOKU
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
    # VWAPS
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

    if not avwap.isna().all():

        fig.add_trace(

            go.Scatter(

                x=df["timestamp"],

                y=avwap,

                name="Anchored VWAP",

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
    # TRADE LEVELS
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
