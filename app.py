# -*- coding: utf-8 -*-
"""
⚡ AliQuantFund Institutional Architecture v4.4 (Adaptive Backtester)
========================================================================
MASTER INTEGRATION & ADAPTIVE MULTI-TIMEFRAME BACKTESTER

DATA
 ├─ Binance Spot / Bybit fallback
 ├─ Binance Futures OI + Funding / Bybit fallback
 └─ Binance Trade Tape

ANALYSIS
 ├─ ATR & Swing High/Low Dynamic Levels
 ├─ Ichimoku (HTF Trend Filtering)
 ├─ Session / Weekly / Monthly VWAP
 ├─ Smart Anchored VWAP
 ├─ Real / Approx CVD
 ├─ OI + Funding
 └─ Multi-Timeframe 1D → 4H → 1H → 15M → 5M

DECISION & MANAGEMENT
 ├─ Market State Classification
 ├─ Setup & Trigger Engines
 ├─ Dynamic R:R & Swing-Based Invalidation
 └─ Multi-Timeframe Adaptive Backtesting
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
    page_title="AliQuantFund - Master Engine v4.4",
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
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    trades_log: List[Dict[str, Any]] = field(default_factory=list)


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
            f"https://data-api.binance.vision/api/v3/klines"
            f"?symbol={symbol}&interval={interval}&limit={limit}",

            f"https://api1.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval={interval}&limit={limit}",

            f"https://api2.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval={interval}&limit={limit}",

            f"https://api3.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval={interval}&limit={limit}",
        ]

        for url in endpoints:
            try:
                r = requests.get(
                    url,
                    headers=headers,
                    timeout=4
                )

                if r.status_code == 200:
                    data = r.json()

                    if isinstance(data, list) and data:

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
                            ],
                            DataStatus.LIVE.value
                        )

            except Exception:
                continue

        # -----------------------------
        # BYBIT FALLBACK
        # -----------------------------

        try:

            bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(
                interval,
                "5"
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
                        ],
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

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        funding = {
            "available": False,
            "current": None,
            "history": []
        }

        # Binance Futures
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
                f"fundingRate?symbol={symbol}&limit=30"
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

                            if hist:

                                funding["available"] = True
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

        # Bybit fallback
        try:

            tf = MarketDataLoader.BYBIT_TF_MAP.get(
                interval,
                "5"
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

        endpoints = [

            f"https://data-api.binance.vision/api/v3/trades"
            f"?symbol={symbol}&limit={limit}",

            f"https://api1.binance.com/api/v3/trades"
            f"?symbol={symbol}&limit={limit}",

            f"https://api3.binance.com/api/v3/trades"
            f"?symbol={symbol}&limit={limit}"
        ]

        for url in endpoints:

            try:

                r = requests.get(
                    url,
                    timeout=4
                )

                if r.status_code == 200:

                    data = r.json()

                    if data:

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
                                subset=["price", "qty"]
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
    def atr(df, period=14):
        prev_close = df["close"].shift()
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs()
            ],
            axis=1
        ).max(axis=1)

        return (
            tr.rolling(period)
            .mean()
            .bfill()
        )

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
            (d["tenkan"] + d["kijun"]) / 2
        ).shift(26)

        d["span_b"] = (
            (
                d["high"].rolling(52).max()
                +
                d["low"].rolling(52).min()
            ) / 2
        ).shift(26)

        return d

    @staticmethod
    def vwap(df, mode):
        typical = (
            df["high"]
            + df["low"]
            + df["close"]
        ) / 3

        pv = typical * df["volume"]

        if mode == "SESSION":
            group = df["timestamp"].dt.date
        elif mode == "WEEKLY":
            group = df["timestamp"].dt.to_period("W")
        elif mode == "MONTHLY":
            group = df["timestamp"].dt.to_period("M")
        else:
            group = pd.Series(0, index=df.index)

        cumulative_pv = pv.groupby(group).cumsum()
        cumulative_volume = df["volume"].groupby(group).cumsum()

        return (
            cumulative_pv /
            cumulative_volume.replace(0, np.nan)
        )

    @staticmethod
    def smart_anchor(df):
        if len(df) < 30:
            return 0, "Initial"

        d = df.tail(100).copy()
        z = (d["volume"] - d["volume"].mean()) / (d["volume"].std() + 1e-9)
        idx = z.idxmax()

        return int(idx), f"Volume Spike (Z={z.loc[idx]:.1f})"

    @staticmethod
    def anchored_vwap(df, anchor):
        typical = (df["high"] + df["low"] + df["close"]) / 3
        pv = typical * df["volume"]

        pv = pv.copy()
        vol = df["volume"].copy()

        pv.iloc[:anchor] = 0
        vol.iloc[:anchor] = 0

        avwap = pv.cumsum() / vol.cumsum().replace(0, np.nan)
        avwap.iloc[:anchor] = np.nan

        return avwap

    @staticmethod
    def cvd(df, trades, timeframe):
        if trades is None or trades.empty:
            candle_range = (df["high"] - df["low"]).replace(0, 1e-9)
            delta = df["volume"] * ((df["close"] - df["open"]) / candle_range)
            cvd = delta.cumsum()

            return (
                cvd,
                DataStatus.APPROXIMATED.value,
                QuantitativeEngine.cvd_stats(df, cvd)
            )

        t = trades.copy()
        t["signed"] = np.where(t["is_buy"], t["qty"], -t["qty"])

        rule = {
            "5m": "5min",
            "15m": "15min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D"
        }.get(timeframe, "5min")

        delta = t.set_index("time")["signed"].resample(rule).sum()
        cvd = delta.cumsum()

        cvd = cvd.reindex(pd.DatetimeIndex(df["timestamp"]), method="ffill").fillna(0)

        return (
            cvd.reset_index(drop=True),
            DataStatus.LIVE.value,
            QuantitativeEngine.cvd_stats(df, cvd.reset_index(drop=True))
        )

    @staticmethod
    def cvd_stats(df, cvd):
        if len(df) < 10:
            return {"slope": 0, "divergence": "NONE"}

        price_change = df["close"].iloc[-1] - df["close"].iloc[-10]
        cvd_change = cvd.iloc[-1] - cvd.iloc[-10]
        slope = (cvd.iloc[-1] - cvd.iloc[-5]) / (abs(cvd.iloc[-5]) + 1e-9)

        divergence = "NONE"
        if price_change < 0 and cvd_change > 0:
            divergence = "BULLISH_ABSORPTION"
        elif price_change > 0 and cvd_change < 0:
            divergence = "BEARISH_ABSORPTION"

        return {"slope": float(slope), "divergence": divergence}


# ============================================================
# 5. MARKET STATE ENGINE
# ============================================================

class MarketStateEngine:

    @staticmethod
    def classify(df, atr):
        close = df["close"].iloc[-1]
        ema20 = df["close"].ewm(span=20).mean().iloc[-1]
        ema50 = df["close"].ewm(span=50).mean().iloc[-1]

        recent_range = df["high"].tail(10).max() - df["low"].tail(10).min()

        if recent_range > 3.5 * atr:
            return MarketState.VOLATILE_EXPANSION
        if recent_range < 1.5 * atr:
            return MarketState.RANGE_COMPRESSION
        if close > ema20 > ema50:
            return MarketState.TRENDING_BULL
        if close < ema20 < ema50:
            return MarketState.TRENDING_BEAR

        return MarketState.RANGE_COMPRESSION


# ============================================================
# 6. MULTI-TIMEFRAME ENGINE
# ============================================================

class MultiTimeframeEngine:

    @staticmethod
    def evaluate(symbol):
        tfs = ["1d", "4h", "1h", "15m", "5m"]
        result = {}

        for tf in tfs:
            df, status = MarketDataLoader.fetch_klines(symbol, tf, 120)

            if df is None or len(df) < 60:
                result[tf] = {"score": 50, "bias": "NEUTRAL", "status": status}
                continue

            d = QuantitativeEngine.ichimoku(df)
            close = d["close"].iloc[-1]
            tenkan = d["tenkan"].iloc[-1]
            kijun = d["kijun"].iloc[-1]
            span_a = d["span_a"].iloc[-1] if not pd.isna(d["span_a"].iloc[-1]) else close
            span_b = d["span_b"].iloc[-1] if not pd.isna(d["span_b"].iloc[-1]) else close

            cloud_high = max(span_a, span_b)
            cloud_low = min(span_a, span_b)

            if close > cloud_high and tenkan > kijun:
                bias = "BULLISH"
                score = 80
            elif close < cloud_low and tenkan < kijun:
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

            result[tf] = {"score": score, "bias": bias, "status": status}

        context = result["1d"]["score"] * 0.60 + result["4h"]["score"] * 0.40
        context_bias = "BULLISH" if context >= 65 else ("BEARISH" if context <= 35 else "NEUTRAL")
        execution = result["15m"]["score"] * 0.50 + result["5m"]["score"] * 0.50

        return {
            "frames": result,
            "context_score": context,
            "context_bias": context_bias,
            "direction_bias": result["1h"]["bias"],
            "execution_score": execution
        }


# ============================================================
# 7. SETUP DETECTOR
# ============================================================

class SetupEngine:

    @staticmethod
    def detect(df, metrics, market_state, htf):
        close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        vwap = metrics.vwap_session
        atr = max(metrics.atr_14, close * 0.001)

        bullish_reclaim = (prev_close < vwap and close > vwap)
        bearish_reclaim = (prev_close > vwap and close < vwap)

        if bullish_reclaim:
            return SetupResult(SetupType.RECLAIM, "Price reclaimed Session VWAP", "LONG", 80)
        if bearish_reclaim:
            return SetupResult(SetupType.RECLAIM, "Price lost Session VWAP", "SHORT", 80)

        resistance = df["high"].iloc[-21:-1].max()
        support = df["low"].iloc[-21:-1].min()
        volume_avg = df["volume"].iloc[-21:-1].mean()
        volume_expansion = df["volume"].iloc[-1] > volume_avg * 1.5

        if close > resistance and volume_expansion:
            return SetupResult(SetupType.BREAKOUT, "Resistance breakout with volume expansion", "LONG", 85)
        if close < support and volume_expansion:
            return SetupResult(SetupType.BREAKOUT, "Support breakdown with volume expansion", "SHORT", 85)

        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]
        open_ = df["open"].iloc[-1]

        upper_wick = high - max(close, open_)
        lower_wick = min(close, open_) - low

        if upper_wick > atr * 0.7 and close < vwap:
            return SetupResult(SetupType.REJECTION, "Upper rejection below VWAP", "SHORT", 70)
        if lower_wick > atr * 0.7 and close > vwap:
            return SetupResult(SetupType.REJECTION, "Lower rejection above VWAP", "LONG", 70)

        distance = (close - vwap) / atr
        if distance > 2.5:
            return SetupResult(SetupType.MEAN_REVERSION, "Price excessively above VWAP", "SHORT", 60)
        if distance < -2.5:
            return SetupResult(SetupType.MEAN_REVERSION, "Price excessively below VWAP", "LONG", 60)

        return SetupResult(SetupType.NO_SETUP, "No high-quality structural setup", "NONE", 0)


# ============================================================
# 8. TRIGGER ENGINE
# ============================================================

class TriggerEngine:

    @staticmethod
    def detect(df, metrics, setup, htf):
        if setup.setup == SetupType.NO_SETUP:
            return TriggerType.WAIT

        close = df["close"].iloc[-1]
        ema20 = df["close"].ewm(span=20).mean().iloc[-1]

        bullish_flow = (metrics.cvd_slope > 0 or metrics.cvd_divergence == "BULLISH_ABSORPTION")
        bearish_flow = (metrics.cvd_slope < 0 or metrics.cvd_divergence == "BEARISH_ABSORPTION")

        if setup.direction == "LONG":
            alignment = 0
            if close > ema20: alignment += 1
            if bullish_flow: alignment += 1
            if htf["context_bias"] == "BULLISH": alignment += 1
            if htf["direction_bias"] == "BULLISH": alignment += 1

            if alignment >= 3:
                return TriggerType.CONFIRMED_BUY
            if metrics.cvd_divergence == "BEARISH_ABSORPTION":
                return TriggerType.INVALID
            return TriggerType.WAIT

        if setup.direction == "SHORT":
            alignment = 0
            if close < ema20: alignment += 1
            if bearish_flow: alignment += 1
            if htf["context_bias"] == "BEARISH": alignment += 1
            if htf["direction_bias"] == "BEARISH": alignment += 1

            if alignment >= 3:
                return TriggerType.CONFIRMED_SELL
            if metrics.cvd_divergence == "BULLISH_ABSORPTION":
                return TriggerType.INVALID
            return TriggerType.WAIT

        return TriggerType.WAIT


# ============================================================
# 9. FACTOR SCORING
# ============================================================

class FactorScoringEngine:

    @staticmethod
    def score(market_state, df, metrics, htf, futures_status, cvd_status):
        close = df["close"].iloc[-1]
        atr = max(metrics.atr_14, close * 0.001)

        direction = 0
        if htf["context_bias"] == "BULLISH": direction += 15
        elif htf["context_bias"] == "BEARISH": direction -= 15

        if htf["direction_bias"] == "BULLISH": direction += 10
        elif htf["direction_bias"] == "BEARISH": direction -= 10

        if metrics.cvd_divergence == "BULLISH_ABSORPTION": flow = 20
        elif metrics.cvd_divergence == "BEARISH_ABSORPTION": flow = -20
        else: flow = float(np.clip(metrics.cvd_slope * 50, -20, 20))

        positioning = 0
        if futures_status != DataStatus.UNAVAILABLE.value:
            if metrics.oi_change_pct > 2:
                positioning += 15 if direction > 0 else -15
            elif metrics.oi_change_pct < -2:
                positioning += -8 if direction > 0 else 8

            if metrics.funding_rate is not None:
                if metrics.funding_rate < -0.0001: positioning += 8
                elif metrics.funding_rate > 0.0003: positioning -= 8

        distance = (close - metrics.vwap_session) / atr
        if abs(distance) <= 0.5:
            location = 20 if direction >= 0 else -20
        elif distance > 2: location = -15
        elif distance < -2: location = 15
        else: location = 8 if distance > 0 else -8

        if market_state == MarketState.RANGE_COMPRESSION:
            dw, fw, pw, lw = 0.15, 0.30, 0.15, 0.40
        elif market_state in [MarketState.TRENDING_BULL, MarketState.TRENDING_BEAR]:
            dw, fw, pw, lw = 0.35, 0.25, 0.20, 0.20
        else:
            dw = fw = pw = lw = 0.25

        total = direction * dw + flow * fw + positioning * pw + location * lw

        quality = 100
        if futures_status == DataStatus.UNAVAILABLE.value: quality -= 25
        elif futures_status == DataStatus.FALLBACK.value: quality -= 10

        if cvd_status == DataStatus.APPROXIMATED.value: quality -= 15
        elif cvd_status == DataStatus.UNAVAILABLE.value: quality -= 25

        return ScoringBreakdown(
            direction_score=round(direction, 1),
            flow_score=round(flow, 1),
            positioning_score=round(positioning, 1),
            location_score=round(location, 1),
            total_score=round(total, 1),
            data_quality_pct=max(0, quality)
        )


# ============================================================
# 10. SIGNAL GRADING
# ============================================================

class SignalEngine:

    @staticmethod
    def grade(score, setup, trigger, quality, htf):
        if trigger == TriggerType.INVALID: return SignalGrade.NO_TRADE
        if setup.setup == SetupType.NO_SETUP: return SignalGrade.NEUTRAL
        if quality < 60: return SignalGrade.NO_TRADE

        abs_score = abs(score)
        if abs_score >= 30 and quality >= 85 and trigger in [TriggerType.CONFIRMED_BUY, TriggerType.CONFIRMED_SELL]:
            return SignalGrade.INSTITUTIONAL_STRONG
        if abs_score >= 22 and trigger in [TriggerType.CONFIRMED_BUY, TriggerType.CONFIRMED_SELL]:
            return SignalGrade.CONFIRMED
        if abs_score >= 15: return SignalGrade.MODERATE

        return SignalGrade.NEUTRAL


# ============================================================
# 11. TRADE MANAGEMENT
# ============================================================

class TradeManagement:

    @staticmethod
    def build(df, metrics, direction, capital, risk_pct):
        if direction not in ["LONG", "SHORT"]:
            return TradePlan()

        price = df["close"].iloc[-1]
        atr = max(metrics.atr_14, price * 0.001)
        risk_amount = capital * risk_pct / 100

        entry_low = price - atr * 0.15
        entry_high = price + atr * 0.15

        if direction == "LONG":
            entry = (entry_low + entry_high) / 2
            stop = entry - atr * 1.5
            risk_per_unit = entry - stop
            tp1 = entry + atr * 1.5
            tp2 = entry + atr * 2.5
            tp3 = entry + atr * 4.0
            invalidation = f"LONG invalid if price closes below ${stop:,.4f}"
        else:
            entry = (entry_low + entry_high) / 2
            stop = entry + atr * 1.5
            risk_per_unit = stop - entry
            tp1 = entry - atr * 1.5
            tp2 = entry - atr * 2.5
            tp3 = entry - atr * 4.0
            invalidation = f"SHORT invalid if price closes above ${stop:,.4f}"

        position_size = risk_amount / max(risk_per_unit, 1e-9)

        def rr(target):
            if direction == "LONG": return (target - entry) / risk_per_unit
            return (entry - target) / risk_per_unit

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
# 11.5 ADAPTIVE BACKTESTING ENGINE (Multi-TF & Dynamic SL/TP)
# ============================================================

class BacktestEngine:

    @staticmethod
    def run_backtest(df, initial_capital=100.0, risk_pct=2.0) -> BacktestResult:
        if df is None or len(df) < 100:
            return BacktestResult()

        d = df.copy().reset_index(drop=True)
        d = QuantitativeEngine.ichimoku(d)
        atr_series = QuantitativeEngine.atr(d)
        vwap_session = QuantitativeEngine.vwap(d, "SESSION")

        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0.0

        trades = []
        in_position = False
        pos_type = None
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        pos_size = 0.0

        # محاكاة خطة التداول متكيفة بناءً على الاتجاه والتأكيد المؤسسي
        for i in range(50, len(d)):
            current_row = d.iloc[i]
            prev_row = d.iloc[i-1]
            close = current_row["close"]
            high = current_row["high"]
            low = current_row["low"]
            atr = atr_series.iloc[i]
            vwap = vwap_session.iloc[i]

            # 1. Management of Active Position
            if in_position:
                pnl = 0.0
                closed = False
                reason = ""

                if pos_type == "LONG":
                    if low <= stop_loss:
                        pnl = (stop_loss - entry_price) * pos_size
                        closed = True
                        reason = "SL"
                    elif high >= take_profit:
                        pnl = (take_profit - entry_price) * pos_size
                        closed = True
                        reason = "TP"
                elif pos_type == "SHORT":
                    if high >= stop_loss:
                        pnl = (entry_price - stop_loss) * pos_size
                        closed = True
                        reason = "SL"
                    elif low <= take_profit:
                        pnl = (entry_price - take_profit) * pos_size
                        closed = True
                        reason = "TP"

                if closed:
                    capital += pnl
                    trades.append({
                        "time": current_row["timestamp"],
                        "type": pos_type,
                        "entry": round(entry_price, 4),
                        "exit": round(stop_loss if reason == "SL" else take_profit, 4),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round((pnl / (entry_price * pos_size)) * 100, 2),
                        "result": "WIN" if pnl > 0 else "LOSS",
                        "reason": reason
                    })
                    in_position = False

                    if capital > peak_capital:
                        peak_capital = capital
                    dd = (peak_capital - capital) / peak_capital * 100
                    if dd > max_drawdown:
                        max_drawdown = dd

            # 2. Adaptive Setup & Dynamic Entry
            if not in_position:
                # فلتر الاتجاه باستخدام Ichimoku + VWAP Alignment
                htf_bullish = (close > d.iloc[i]["span_a"]) and (close > d.iloc[i]["kijun"])
                htf_bearish = (close < d.iloc[i]["span_a"]) and (close < d.iloc[i]["kijun"])

                bullish_reclaim = (prev_row["close"] < vwap) and (close > vwap)
                bearish_reclaim = (prev_row["close"] > vwap) and (close < vwap)

                # حجم تداول أعلى من المتوسط لتجنب الكسور الكاذبة
                vol_filter = d.iloc[i]["volume"] > d["volume"].iloc[i-10:i].mean()

                # Dynamic SL (Swing Low / High of last 10 candles)
                swing_low = d["low"].iloc[i-10:i].min()
                swing_high = d["high"].iloc[i-10:i].max()

                risk_amt = capital * (risk_pct / 100.0)

                if bullish_reclaim and htf_bullish and vol_filter:
                    in_position = True
                    pos_type = "LONG"
                    entry_price = close
                    
                    # Stop loss under Swing Low or minimum 1.2 ATR
                    stop_loss = min(swing_low, entry_price - (atr * 1.2))
                    risk_per_unit = entry_price - stop_loss
                    
                    # Target with 1:2 Risk to Reward
                    take_profit = entry_price + (risk_per_unit * 2.0)
                    pos_size = risk_amt / max(risk_per_unit, 1e-9)

                elif bearish_reclaim and htf_bearish and vol_filter:
                    in_position = True
                    pos_type = "SHORT"
                    entry_price = close
                    
                    # Stop loss above Swing High or minimum 1.2 ATR
                    stop_loss = max(swing_high, entry_price + (atr * 1.2))
                    risk_per_unit = stop_loss - entry_price
                    
                    # Target with 1:2 Risk to Reward
                    take_profit = entry_price - (risk_per_unit * 2.0)
                    pos_size = risk_amt / max(risk_per_unit, 1e-9)

        wins = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]

        total_gross_win = sum(t["pnl"] for t in wins)
        total_gross_loss = abs(sum(t["pnl"] for t in losses))

        profit_factor = (total_gross_win / total_gross_loss) if total_gross_loss > 0 else (total_gross_win if total_gross_win > 0 else 0.0)
        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
        total_pnl_pct = ((capital - initial_capital) / initial_capital) * 100

        return BacktestResult(
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl_pct=total_pnl_pct,
            max_drawdown_pct=max_drawdown,
            profit_factor=profit_factor,
            trades_log=trades
        )


# ============================================================
# 12. CSS
# ============================================================

def render_css():
    st.markdown(
        """
        <style>
        .stSidebar, div[data-testid="stSidebar"], div[data-testid="stSidebar"] * {
            word-break: normal !important;
            word-wrap: normal !important;
            white-space: normal !important;
        }
        .status { padding: 4px 8px; border-radius: 5px; font-weight: bold; font-size: 12px; }
        .green { background: #133E2B; color: #00E676; }
        .red { background: #4A191B; color: #FF5252; }
        .yellow { background: #3D3214; color: #FFD600; }
        .decision { background: #1E222D; padding: 18px; border-radius: 8px; border-left: 5px solid; }
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    render_css()

    with st.sidebar:
        st.title("⚡ AliQuantFund")
        st.caption("Institutional Master Engine v4.4")
        st.markdown("---")

        symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "ZEC/USDT", "SOL/USDT", "XRP/USDT"])
        timeframe = st.selectbox("Execution Timeframe", ["5m", "15m", "1h", "4h"], index=1)
        st.markdown("---")

        capital = st.number_input("Capital ($)", min_value=10.0, value=100.0, step=10.0)
        risk_pct = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.5)
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        st.markdown("---")
        st.info("النظام لا ينفذ الصفقات. هو محرك تحليل وإشارات وإدارة صفقة واختبار خلفي.")

    with st.spinner(f"جاري تحليل {symbol}..."):
        df, spot_status = MarketDataLoader.fetch_klines(symbol, timeframe, 500)
        futures, futures_status, funding = MarketDataLoader.fetch_futures_metrics(symbol, timeframe, 50)
        trades, cvd_status = MarketDataLoader.fetch_recent_trades(symbol)

    if df is None or df.empty:
        st.error("❌ تعذر الحصول على بيانات السوق.")
        return

    df = QuantitativeEngine.ichimoku(df)
    atr_series = QuantitativeEngine.atr(df)
    atr = atr_series.iloc[-1]

    vwap_session = QuantitativeEngine.vwap(df, "SESSION")
    vwap_weekly = QuantitativeEngine.vwap(df, "WEEKLY")
    vwap_monthly = QuantitativeEngine.vwap(df, "MONTHLY")

    anchor_idx, anchor_reason = QuantitativeEngine.smart_anchor(df)
    avwap = QuantitativeEngine.anchored_vwap(df, anchor_idx)
    cvd_series, cvd_type, cvd_stats = QuantitativeEngine.cvd(df, trades, timeframe)

    oi_change = 0.0
    if futures is not None and len(futures) >= 2:
        start = futures["openInterest"].iloc[0]
        end = futures["openInterest"].iloc[-1]
        if start != 0:
            oi_change = ((end - start) / start) * 100

    def safe_last(series, fallback):
        value = series.iloc[-1]
        return fallback if pd.isna(value) else float(value)

    close = float(df["close"].iloc[-1])

    metrics = QuantitativeMetrics(
        vwap_session=safe_last(vwap_session, close),
        vwap_weekly=safe_last(vwap_weekly, close),
        vwap_monthly=safe_last(vwap_monthly, close),
        vwap_anchored=(safe_last(avwap, close) if not avwap.isna().all() else None),
        atr_14=float(atr),
        cvd_slope=cvd_stats["slope"],
        cvd_divergence=cvd_stats["divergence"],
        oi_change_pct=oi_change,
        funding_rate=funding.get("current"),
        tenkan=safe_last(df["tenkan"], close),
        kijun=safe_last(df["kijun"], close),
        span_a=safe_last(df["span_a"], close),
        span_b=safe_last(df["span_b"], close)
    )

    market_state = MarketStateEngine.classify(df, atr)
    htf = MultiTimeframeEngine.evaluate(symbol)
    setup = SetupEngine.detect(df, metrics, market_state, htf)
    trigger = TriggerEngine.detect(df, metrics, setup, htf)
    scoring = FactorScoringEngine.score(market_state, df, metrics, htf, futures_status, cvd_type)
    grade = SignalEngine.grade(scoring.total_score, setup, trigger, scoring.data_quality_pct, htf)

    if trigger == TriggerType.CONFIRMED_BUY and scoring.total_score >= 15 and scoring.data_quality_pct >= 60:
        final_decision = "CONFIRMED LONG"
        direction = "LONG"
    elif trigger == TriggerType.CONFIRMED_SELL and scoring.total_score <= -15 and scoring.data_quality_pct >= 60:
        final_decision = "CONFIRMED SHORT"
        direction = "SHORT"
    else:
        final_decision = "NO TRADE / WAIT"
        direction = "NONE"

    trade_plan = TradeManagement.build(df, metrics, direction, capital, risk_pct)

    # Header Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"### {symbol}")
    c1.caption(f"Price: ${close:,.4f}")
    c2.metric("Market State", market_state.value)
    c3.metric("Quant Score", f"{scoring.total_score:+.1f}")
    c4.metric("Data Quality", f"{scoring.data_quality_pct:.0f}%")

    st.markdown("---")

    color = "#00E676" if "LONG" in final_decision else ("#FF5252" if "SHORT" in final_decision else "#FFD600")

    st.markdown(
        f"""
        <div class="decision" style="border-color:{color}">
            <h2 style="color:{color}">{final_decision}</h2>
            <b>Setup:</b> {setup.setup.value} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Trigger:</b> {trigger.value}<br><br>
            <b>Signal Grade:</b> {grade.value} &nbsp;&nbsp;|&nbsp;&nbsp; <b>HTF:</b> {htf["context_bias"]}<br><br>
            <b>Setup Reason:</b> {setup.reason}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Multi TF Display
    st.subheader("🧭 Multi-Timeframe Hierarchy")
    cols = st.columns(5)
    for i, tf in enumerate(["1d", "4h", "1h", "15m", "5m"]):
        frame = htf["frames"][tf]
        cols[i].metric(tf.upper(), frame["bias"], f'{frame["score"]:.0f}')

    # Factor Breakdown
    with st.expander("🧩 Layered Quantitative Scoring", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Direction", f"{scoring.direction_score:+.1f}")
        f2.metric("Flow / CVD", f"{scoring.flow_score:+.1f}")
        f3.metric("Positioning", f"{scoring.positioning_score:+.1f}")
        f4.metric("Location / VWAP", f"{scoring.location_score:+.1f}")

    # Trade Management
    st.markdown("---")
    st.subheader("🎯 Trade Management")

    if direction == "NONE":
        st.warning("لا توجد صفقة مؤكدة حالياً. إدارة الصفقة لن تُنشأ قبل اكتمال Trigger.")
    else:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Entry", f"{trade_plan.entry_low:,.4f} – {trade_plan.entry_high:,.4f}")
        t2.metric("Stop Loss", f"{trade_plan.stop_loss:,.4f}")
        t3.metric("TP1", f"{trade_plan.tp1:,.4f}")
        t4.metric("R:R TP1", f"1:{trade_plan.rr_tp1:.2f}")

        t5, t6, t7, t8 = st.columns(4)
        t5.metric("TP2", f"{trade_plan.tp2:,.4f}")
        t6.metric("R:R TP2", f"1:{trade_plan.rr_tp2:.2f}")
        t7.metric("TP3", f"{trade_plan.tp3:,.4f}")
        t8.metric("Position Size", f"{trade_plan.position_size:.6f}")

        st.info(f"💰 Risk Amount: ${trade_plan.risk_amount:.2f} | {trade_plan.invalidation}")

    # BACKTESTING ENGINE SECTION (Adaptive)
    st.markdown("---")
    st.subheader("🧪 Adaptive Backtesting Engine (Dynamic SL/TP & HTF Filter)")

    bt_result = BacktestEngine.run_backtest(df, capital, risk_pct)

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Total Trades", bt_result.total_trades)
    b2.metric("Win Rate", f"{bt_result.win_rate:.1f}%")
    b3.metric("Total Return", f"{bt_result.total_pnl_pct:+.2f}%")
    b4.metric("Max Drawdown", f"{bt_result.max_drawdown_pct:.2f}%")
    b5.metric("Profit Factor", f"{bt_result.profit_factor:.2f}")

    if bt_result.trades_log:
        with st.expander("📋 View Detailed Backtest Trades Log", expanded=False):
            st.dataframe(pd.DataFrame(bt_result.trades_log), use_container_width=True)

    # Diagnostics
    with st.expander("🔧 Engine Diagnostics", expanded=False):
        diagnostics = {
            "Market State": market_state.value,
            "Setup": setup.setup.value,
            "Setup Reason": setup.reason,
            "Trigger": trigger.value,
            "Signal Grade": grade.value,
            "HTF Context": htf["context_bias"],
            "Execution Score": round(htf["execution_score"], 1),
            "CVD Type": cvd_type,
            "CVD Slope": round(metrics.cvd_slope, 4),
            "CVD Divergence": metrics.cvd_divergence,
            "OI Change": round(metrics.oi_change_pct, 2),
            "Funding": metrics.funding_rate,
            "Anchor": anchor_reason,
            "Spot Data": spot_status,
            "Futures Data": futures_status,
            "Data Quality": scoring.data_quality_pct
        }
        st.json(diagnostics)

    # VWAP Panel
    with st.expander("📊 VWAP / Market Location", expanded=False):
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Session VWAP", f"{metrics.vwap_session:,.4f}")
        v2.metric("Weekly VWAP", f"{metrics.vwap_weekly:,.4f}")
        v3.metric("Monthly VWAP", f"{metrics.vwap_monthly:,.4f}")
        if metrics.vwap_anchored:
            v4.metric("Anchored VWAP", f"{metrics.vwap_anchored:,.4f}")

    # Plotly Chart
    st.markdown("---")
    st.subheader(f"📈 {symbol} — Institutional Chart")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.70, 0.30])
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["tenkan"], name="Tenkan", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["kijun"], name="Kijun", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=vwap_session, name="Session VWAP", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=vwap_weekly, name="Weekly VWAP", mode="lines", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=vwap_monthly, name="Monthly VWAP", mode="lines", line=dict(dash="dot")), row=1, col=1)

    if not avwap.isna().all():
        fig.add_trace(go.Scatter(x=df["timestamp"], y=avwap, name="Smart Anchored VWAP", mode="lines"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["timestamp"], y=cvd_series, name=f"CVD ({cvd_type})", mode="lines"), row=2, col=1)

    if direction != "NONE":
        fig.add_hline(y=trade_plan.stop_loss, line_dash="dash", annotation_text="SL", row=1, col=1)
        fig.add_hline(y=trade_plan.tp1, line_dash="dot", annotation_text="TP1", row=1, col=1)
        fig.add_hline(y=trade_plan.tp2, line_dash="dot", annotation_text="TP2", row=1, col=1)
        fig.add_hline(y=trade_plan.tp3, line_dash="dot", annotation_text="TP3", row=1, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 14. RUN
# ============================================================

if __name__ == "__main__":
    main()
