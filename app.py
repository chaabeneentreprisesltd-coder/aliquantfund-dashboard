# -*- coding: utf-8 -*-
"""
⚡ AliQuantFund Institutional Architecture v4.3
=================================================
MASTER INTEGRATION

Integrated:
- Multi-TF Hierarchy
- Market State Engine
- Setup Detection
- Trigger Engine
- Signal Grade
- Ichimoku
- Session / Weekly / Monthly VWAP
- Smart Anchored VWAP
- CVD Real / Approx
- Futures OI
- Funding Rate
- Anti-Double Counting
- Entry Zone
- Stop Loss
- TP1 / TP2 / TP3
- Position Sizing
- Risk / Reward
- Trade Management
- Trailing Stop Reference
- Data Quality
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Tuple, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# 0. CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AliQuantFund - Institutional Engine v4.3",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("AliQuantFund")


# ============================================================================
# 1. ENUMS
# ============================================================================

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


# ============================================================================
# 2. DATA STRUCTURES
# ============================================================================

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
    oi_interpretation: str = "NEUTRAL"

    funding_rate: Optional[float] = None
    funding_bias: str = "NEUTRAL"

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
class TradePlan:
    direction: str = "NONE"

    entry_low: float = 0.0
    entry_high: float = 0.0
    entry_reference: float = 0.0

    stop_loss: float = 0.0

    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0

    risk_per_unit: float = 0.0
    rr_tp1: float = 0.0
    rr_tp2: float = 0.0
    rr_tp3: float = 0.0

    capital: float = 0.0
    risk_pct: float = 0.0
    dollar_risk: float = 0.0

    position_size: float = 0.0
    position_notional: float = 0.0

    trailing_activation: float = 0.0
    trailing_distance: float = 0.0


# ============================================================================
# 3. MARKET DATA LOADER
# ============================================================================

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

        formatted_symbol = symbol.replace("/", "").upper()

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        binance_endpoints = [
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={formatted_symbol}&interval={interval}&limit={limit}",

            f"https://api1.binance.com/api/v3/klines"
            f"?symbol={formatted_symbol}&interval={interval}&limit={limit}",

            f"https://api3.binance.com/api/v3/klines"
            f"?symbol={formatted_symbol}&interval={interval}&limit={limit}",

            f"https://data-api.binance.vision/api/v3/klines"
            f"?symbol={formatted_symbol}&interval={interval}&limit={limit}"
        ]

        for url in binance_endpoints:

            try:

                response = requests.get(
                    url,
                    headers=headers,
                    timeout=4
                )

                if response.status_code != 200:
                    continue

                data = response.json()

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

                for col in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]:
                    df[col] = pd.to_numeric(
                        df[col],
                        errors="coerce"
                    )

                df["timestamp"] = pd.to_datetime(
                    df["timestamp"],
                    unit="ms",
                    utc=True
                ).dt.tz_localize(None)

                df = df[
                    [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume"
                    ]
                ]

                df = df.dropna().reset_index(drop=True)

                return df, DataStatus.LIVE.value

            except Exception as e:
                logger.warning(
                    f"Binance kline endpoint failed: {e}"
                )

        # --------------------------------------------------------------------
        # Bybit fallback
        # --------------------------------------------------------------------

        try:

            bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(
                interval,
                "5"
            )

            url = (
                "https://api.bybit.com/v5/market/kline"
                f"?category=spot"
                f"&symbol={formatted_symbol}"
                f"&interval={bybit_tf}"
                f"&limit={limit}"
            )

            response = requests.get(
                url,
                headers=headers,
                timeout=4
            )

            if response.status_code == 200:

                data = response.json().get(
                    "result",
                    {}
                ).get(
                    "list",
                    []
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

                    for col in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume"
                    ]:
                        df[col] = pd.to_numeric(
                            df[col],
                            errors="coerce"
                        )

                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"].astype(float),
                        unit="ms",
                        utc=True
                    ).dt.tz_localize(None)

                    df = (
                        df
                        .iloc[::-1]
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
                f"Bybit kline fetch failed: {e}"
            )

        return None, DataStatus.UNAVAILABLE.value

    # =========================================================================
    # FUTURES DATA
    # =========================================================================

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_futures_metrics(
        symbol: str,
        interval: str,
        limit: int = 50
    ) -> Tuple[
        Optional[pd.DataFrame],
        str,
        Dict[str, Any]
    ]:

        formatted_symbol = symbol.replace("/", "").upper()

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        funding_meta = {
            "available": False,
            "current": None,
            "history": []
        }

        # --------------------------------------------------------------------
        # Binance Futures
        # --------------------------------------------------------------------

        try:

            oi_interval = (
                interval
                if interval in (
                    "5m",
                    "15m",
                    "1h",
                    "4h",
                    "1d"
                )
                else "5m"
            )

            url_oi = (
                "https://fapi.binance.com/futures/data/"
                f"openInterestHist?symbol={formatted_symbol}"
                f"&period={oi_interval}&limit={limit}"
            )

            url_fr = (
                "https://fapi.binance.com/fapi/v1/"
                f"fundingRate?symbol={formatted_symbol}&limit=30"
            )

            res_oi = requests.get(
                url_oi,
                headers=headers,
                timeout=4
            )

            res_fr = requests.get(
                url_fr,
                headers=headers,
                timeout=4
            )

            if res_oi.status_code == 200:

                oi_data = res_oi.json()

                if isinstance(oi_data, list) and len(oi_data) >= 3:

                    df_oi = pd.DataFrame(
                        oi_data
                    )

                    df_oi["openInterest"] = pd.to_numeric(
                        df_oi["sumOpenInterest"],
                        errors="coerce"
                    )

                    df_oi["timestamp"] = pd.to_datetime(
                        df_oi["timestamp"],
                        unit="ms",
                        utc=True
                    ).dt.tz_localize(None)

                    df_oi = (
                        df_oi
                        .dropna(
                            subset=["openInterest"]
                        )
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )

                    if res_fr.status_code == 200:

                        fr_data = res_fr.json()

                        if (
                            isinstance(fr_data, list)
                            and fr_data
                        ):

                            hist = [
                                float(
                                    row.get(
                                        "fundingRate",
                                        0.0
                                    )
                                )
                                for row in reversed(fr_data)
                                if "fundingRate" in row
                            ]

                            if hist:

                                funding_meta[
                                    "available"
                                ] = True

                                funding_meta[
                                    "current"
                                ] = hist[0]

                                funding_meta[
                                    "history"
                                ] = hist

                    return (
                        df_oi,
                        DataStatus.LIVE.value,
                        funding_meta
                    )

        except Exception as e:

            logger.warning(
                f"Binance futures failed: {e}"
            )

        # --------------------------------------------------------------------
        # Bybit Futures
        # --------------------------------------------------------------------

        try:

            bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(
                interval,
                "5"
            )

            url_oi = (
                "https://api.bybit.com/v5/market/open-interest"
                f"?category=linear"
                f"&symbol={formatted_symbol}"
                f"&intervalTime={bybit_tf}"
                f"&limit={limit}"
            )

            response = requests.get(
                url_oi,
                headers=headers,
                timeout=4
            )

            if response.status_code == 200:

                data = (
                    response
                    .json()
                    .get("result", {})
                    .get("list", [])
                )

                if data:

                    df_oi = pd.DataFrame(
                        data
                    )

                    df_oi["openInterest"] = pd.to_numeric(
                        df_oi["openInterest"],
                        errors="coerce"
                    )

                    df_oi["timestamp"] = pd.to_datetime(
                        df_oi["timestamp"].astype(float),
                        unit="ms",
                        utc=True
                    ).dt.tz_localize(None)

                    df_oi = (
                        df_oi
                        .dropna(
                            subset=["openInterest"]
                        )
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )

                    return (
                        df_oi,
                        DataStatus.FALLBACK.value,
                        funding_meta
                    )

        except Exception as e:

            logger.warning(
                f"Bybit OI failed: {e}"
            )

        return (
            None,
            DataStatus.UNAVAILABLE.value,
            funding_meta
        )

    # =========================================================================
    # RECENT TRADES
    # =========================================================================

    @staticmethod
    @st.cache_data(ttl=10)
    def fetch_recent_trades(
        symbol: str,
        limit: int = 1000
    ) -> Tuple[
        Optional[pd.DataFrame],
        str
    ]:

        formatted_symbol = symbol.replace("/", "").upper()

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        endpoints = [
            f"https://api.binance.com/api/v3/trades"
            f"?symbol={formatted_symbol}&limit={limit}",

            f"https://api1.binance.com/api/v3/trades"
            f"?symbol={formatted_symbol}&limit={limit}",

            f"https://api3.binance.com/api/v3/trades"
            f"?symbol={formatted_symbol}&limit={limit}",

            f"https://data-api.binance.vision/api/v3/trades"
            f"?symbol={formatted_symbol}&limit={limit}"
        ]

        for url in endpoints:

            try:

                response = requests.get(
                    url,
                    headers=headers,
                    timeout=4
                )

                if response.status_code != 200:
                    continue

                trades = response.json()

                if (
                    isinstance(trades, list)
                    and trades
                ):

                    df = pd.DataFrame(
                        trades
                    )

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

                    df["is_buy"] = ~df[
                        "isBuyerMaker"
                    ]

                    df = (
                        df
                        .dropna(
                            subset=[
                                "price",
                                "qty"
                            ]
                        )
                        .sort_values("time")
                        .reset_index(drop=True)
                    )

                    if not df.empty:

                        return (
                            df,
                            DataStatus.LIVE.value
                        )

            except Exception:
                continue

        return (
            None,
            DataStatus.UNAVAILABLE.value
        )


# ============================================================================
# 4. QUANTITATIVE ENGINE
# ============================================================================

class QuantitativeEngine:

    # ------------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_atr(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:

        high_low = (
            df["high"] -
            df["low"]
        )

        high_close = (
            df["high"] -
            df["close"].shift()
        ).abs()

        low_close = (
            df["low"] -
            df["close"].shift()
        ).abs()

        tr = pd.concat(
            [
                high_low,
                high_close,
                low_close
            ],
            axis=1
        ).max(axis=1)

        atr = tr.rolling(
            period,
            min_periods=period
        ).mean()

        return atr.bfill()

    # ------------------------------------------------------------------------
    # Ichimoku
    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_ichimoku(
        df: pd.DataFrame
    ) -> pd.DataFrame:

        df = df.copy()

        df["tenkan"] = (
            df["high"].rolling(9).max() +
            df["low"].rolling(9).min()
        ) / 2

        df["kijun"] = (
            df["high"].rolling(26).max() +
            df["low"].rolling(26).min()
        ) / 2

        df["span_a"] = (
            (
                df["tenkan"] +
                df["kijun"]
            ) / 2
        ).shift(26)

        df["span_b"] = (
            (
                df["high"].rolling(52).max() +
                df["low"].rolling(52).min()
            ) / 2
        ).shift(26)

        return df

    # ------------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_vwap(
        df: pd.DataFrame,
        anchor_type: str
    ) -> pd.Series:

        typical_price = (
            df["high"] +
            df["low"] +
            df["close"]
        ) / 3

        pv = (
            typical_price *
            df["volume"]
        )

        if anchor_type == "SESSION":

            group = df["timestamp"].dt.date

        elif anchor_type == "WEEKLY":

            group = df["timestamp"].dt.to_period("W")

        elif anchor_type == "MONTHLY":

            group = df["timestamp"].dt.to_period("M")

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
            cumulative_pv /
            cumulative_volume.replace(
                0,
                np.nan
            )
        )

    # ------------------------------------------------------------------------
    # Smart Anchor
    # ------------------------------------------------------------------------

    @staticmethod
    def detect_smart_anchor(
        df: pd.DataFrame
    ) -> Tuple[int, str]:

        if len(df) < 50:

            return 0, "Initial"

        sub = df.tail(100).copy()

        volume_std = (
            sub["volume"].std() +
            1e-9
        )

        sub["vol_z"] = (
            sub["volume"] -
            sub["volume"].mean()
        ) / volume_std

        idx = sub["vol_z"].idxmax()

        return (
            int(idx),
            f"Volume Spike "
            f"(Z={sub.loc[idx, 'vol_z']:.1f})"
        )

    # ------------------------------------------------------------------------
    # Anchored VWAP
    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_anchored_vwap(
        df: pd.DataFrame,
        anchor_idx: int
    ) -> pd.Series:

        typical_price = (
            df["high"] +
            df["low"] +
            df["close"]
        ) / 3

        pv = (
            typical_price *
            df["volume"]
        )

        pv_anchored = pv.copy()
        vol_anchored = df["volume"].copy()

        pv_anchored.iloc[:anchor_idx] = 0
        vol_anchored.iloc[:anchor_idx] = 0

        cumulative_pv = pv_anchored.cumsum()
        cumulative_volume = vol_anchored.cumsum()

        avwap = (
            cumulative_pv /
            cumulative_volume.replace(
                0,
                np.nan
            )
        )

        avwap.iloc[:anchor_idx] = np.nan

        return avwap

    # ------------------------------------------------------------------------
    # CVD
    # ------------------------------------------------------------------------

    @staticmethod
    def compute_cvd_metrics(
        df_klines: pd.DataFrame,
        df_trades: Optional[pd.DataFrame],
        timeframe: str
    ) -> Tuple[
        pd.Series,
        str,
        float,
        str
    ]:

        df = df_klines.copy()

        # --------------------------------------------------------------------
        # Real trade tape
        # --------------------------------------------------------------------

        if (
            df_trades is not None
            and not df_trades.empty
        ):

            trades = df_trades.copy()

            trades["signed_vol"] = np.where(
                trades["is_buy"],
                trades["qty"],
                -trades["qty"]
            )

            # Determine candle frequency
            freq_map = {
                "5m": "5min",
                "15m": "15min",
                "1h": "1h",
                "4h": "4h",
                "1d": "1D"
            }

            freq = freq_map.get(
                timeframe,
                "5min"
            )

            # Create candle buckets
            trade_delta = (
                trades
                .set_index("time")["signed_vol"]
                .resample(freq)
                .sum()
            )

            # Align to candle timestamps
            delta = (
                trade_delta
                .reindex(
                    df["timestamp"],
                    fill_value=0
                )
            )

            cvd = (
                delta
                .cumsum()
                .reset_index(drop=True)
            )

            cvd_type = DataStatus.LIVE.value

        # --------------------------------------------------------------------
        # Approximate CVD
        # --------------------------------------------------------------------

        else:

            candle_range = (
                df["high"] -
                df["low"]
            ).replace(
                0,
                1e-9
            )

            delta_approx = (
                df["volume"] *
                (
                    (
                        df["close"] -
                        df["open"]
                    ) /
                    candle_range
                )
            )

            cvd = (
                delta_approx
                .cumsum()
            )

            cvd_type = DataStatus.APPROXIMATED.value

        # --------------------------------------------------------------------
        # CVD slope
        # --------------------------------------------------------------------

        if len(cvd) >= 10:

            cvd_change = (
                cvd.iloc[-1] -
                cvd.iloc[-10]
            )

            price_change = (
                df["close"].iloc[-1] -
                df["close"].iloc[-10]
            )

        else:

            cvd_change = 0
            price_change = 0

        if len(cvd) >= 5:

            denominator = max(
                abs(cvd.iloc[-5]),
                1e-9
            )

            cvd_slope = (
                cvd.iloc[-1] -
                cvd.iloc[-5]
            ) / denominator

        else:

            cvd_slope = 0.0

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

        return (
            cvd,
            cvd_type,
            float(cvd_slope),
            divergence
        )


# ============================================================================
# 5. MARKET STATE ENGINE
# ============================================================================

class MarketStateEngine:

    @staticmethod
    def classify_market_state(
        df: pd.DataFrame,
        atr: float
    ) -> MarketState:

        close = df["close"].iloc[-1]

        ema20 = (
            df["close"]
            .ewm(span=20)
            .mean()
            .iloc[-1]
        )

        ema50 = (
            df["close"]
            .ewm(span=50)
            .mean()
            .iloc[-1]
        )

        recent_range = (
            df["high"].tail(10).max() -
            df["low"].tail(10).min()
        )

        if recent_range > 3.5 * atr:

            return MarketState.VOLATILE_EXPANSION

        if recent_range < 1.5 * atr:

            return MarketState.RANGE_COMPRESSION

        if close > ema20 > ema50:

            return MarketState.TRENDING_BULL

        if close < ema20 < ema50:

            return MarketState.TRENDING_BEAR

        return MarketState.RANGE_COMPRESSION


# ============================================================================
# 6. MULTI TIMEFRAME ENGINE
# ============================================================================

class MultiTimeframeHierarchy:

    @staticmethod
    def evaluate_all(
        symbol: str
    ) -> Dict[str, Any]:

        timeframes = [
            "1d",
            "4h",
            "1h",
            "15m",
            "5m"
        ]

        results = {}

        for tf in timeframes:

            df, status = (
                MarketDataLoader
                .fetch_klines(
                    symbol,
                    tf,
                    150
                )
            )

            if (
                df is None
                or df.empty
            ):

                results[tf] = {
                    "score": 50,
                    "bias": "NEUTRAL",
                    "status": status
                }

                continue

            calc = (
                QuantitativeEngine
                .calculate_ichimoku(df)
            )

            close = calc["close"].iloc[-1]

            tenkan = calc["tenkan"].iloc[-1]
            kijun = calc["kijun"].iloc[-1]

            span_a = calc["span_a"].iloc[-1]
            span_b = calc["span_b"].iloc[-1]

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
                score = 80.0

            elif (
                close < cloud_low
                and tenkan < kijun
            ):

                bias = "BEARISH"
                score = 20.0

            else:

                bias = "NEUTRAL"
                score = 50.0

            results[tf] = {
                "score": score,
                "bias": bias,
                "close": close,
                "status": status
            }

        context_score = (
            results["1d"]["score"] * 0.60 +
            results["4h"]["score"] * 0.40
        )

        if context_score >= 65:

            context_bias = "BULLISH"

        elif context_score <= 35:

            context_bias = "BEARISH"

        else:

            context_bias = "NEUTRAL"

        direction_bias = (
            results["1h"]["bias"]
        )

        execution_score = (
            results["15m"]["score"] * 0.50 +
            results["5m"]["score"] * 0.50
        )

        return {
            "scores": {
                tf: results[tf]["score"]
                for tf in timeframes
            },

            "biases": {
                tf: results[tf]["bias"]
                for tf in timeframes
            },

            "context_bias": context_bias,

            "direction_bias": direction_bias,

            "exec_score": execution_score,

            "raw": results
        }


# ============================================================================
# 7. FACTOR SCORING ENGINE
# ============================================================================

class FactorScoringEngine:

    @staticmethod
    def compute_layered_score(
        market_state: MarketState,
        df: pd.DataFrame,
        metrics: QuantitativeMetrics,
        mft_res: Dict[str, Any],
        data_status_futures: str,
        data_status_cvd: str
    ) -> ScoringBreakdown:

        close = df["close"].iloc[-1]

        atr = (
            metrics.atr_14
            if metrics.atr_14 > 0
            else 1.0
        )

        # ====================================================================
        # DIRECTION
        # ====================================================================

        direction_score = 0.0

        if (
            "BULLISH"
            in mft_res["context_bias"]
        ):

            direction_score += 15

        elif (
            "BEARISH"
            in mft_res["context_bias"]
        ):

            direction_score -= 15

        if (
            close > metrics.span_a
            and close > metrics.span_b
        ):

            direction_score += 10

        elif (
            close < metrics.span_a
            and close < metrics.span_b
        ):

            direction_score -= 10

        # ====================================================================
        # FLOW
        # ====================================================================

        if (
            metrics.cvd_divergence
            == "BULLISH_ABSORPTION"
        ):

            flow_score = 25

        elif (
            metrics.cvd_divergence
            == "BEARISH_ABSORPTION"
        ):

            flow_score = -25

        else:

            flow_score = np.clip(
                metrics.cvd_slope * 50,
                -20,
                20
            )

        # ====================================================================
        # POSITIONING
        # ====================================================================

        positioning_score = 0.0

        if (
            data_status_futures
            != DataStatus.UNAVAILABLE.value
        ):

            if (
                metrics.oi_change_pct > 2
                and direction_score > 0
            ):

                positioning_score += 15

            elif (
                metrics.oi_change_pct > 2
                and direction_score < 0
            ):

                positioning_score -= 15

            if metrics.funding_rate is not None:

                if (
                    metrics.funding_rate
                    < -0.0001
                ):

                    positioning_score += 10

                elif (
                    metrics.funding_rate
                    > 0.0003
                ):

                    positioning_score -= 10

        # ====================================================================
        # LOCATION
        # ====================================================================

        dist_vwap_atr = (
            close -
            metrics.vwap_session
        ) / atr

        if abs(dist_vwap_atr) <= 0.5:

            location_score = (
                25
                if direction_score >= 0
                else -25
            )

        elif dist_vwap_atr > 2:

            location_score = -15

        elif dist_vwap_atr < -2:

            location_score = 15

        else:

            location_score = (
                10
                if dist_vwap_atr > 0
                else -10
            )

        # ====================================================================
        # DYNAMIC WEIGHTS
        # ====================================================================

        if (
            market_state
            == MarketState.RANGE_COMPRESSION
        ):

            dir_weight = 0.15
            flow_weight = 0.30
            positioning_weight = 0.15
            location_weight = 0.40

        elif market_state in (
            MarketState.TRENDING_BULL,
            MarketState.TRENDING_BEAR
        ):

            dir_weight = 0.35
            flow_weight = 0.25
            positioning_weight = 0.20
            location_weight = 0.20

        else:

            dir_weight = 0.25
            flow_weight = 0.25
            positioning_weight = 0.25
            location_weight = 0.25

        total = (
            direction_score * dir_weight +
            flow_score * flow_weight +
            positioning_score * positioning_weight +
            location_score * location_weight
        )

        # ====================================================================
        # DATA QUALITY
        # ====================================================================

        quality = 100.0

        if (
            data_status_futures
            == DataStatus.UNAVAILABLE.value
        ):

            quality -= 30

        elif (
            data_status_futures
            == DataStatus.FALLBACK.value
        ):

            quality -= 10

        if (
            data_status_cvd
            == DataStatus.APPROXIMATED.value
        ):

            quality -= 18

        elif (
            data_status_cvd
            == DataStatus.UNAVAILABLE.value
        ):

            quality -= 30

        return ScoringBreakdown(
            direction_score=round(
                direction_score,
                1
            ),

            flow_score=round(
                flow_score,
                1
            ),

            positioning_score=round(
                positioning_score,
                1
            ),

            location_score=round(
                location_score,
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


# ============================================================================
# 8. SETUP DETECTION ENGINE
# ============================================================================

class SetupEngine:

    @staticmethod
    def detect_setup(
        df: pd.DataFrame,
        metrics: QuantitativeMetrics,
        market_state: MarketState
    ) -> Tuple[
        SetupType,
        str
    ]:

        if len(df) < 20:

            return (
                SetupType.NO_SETUP,
                "Insufficient data"
            )

        close = df["close"].iloc[-1]

        previous_close = (
            df["close"].iloc[-2]
        )

        high_20 = (
            df["high"]
            .tail(20)
            .max()
        )

        low_20 = (
            df["low"]
            .tail(20)
            .min()
        )

        atr = max(
            metrics.atr_14,
            1e-9
        )

        # ====================================================================
        # BREAKOUT
        # ====================================================================

        if (
            close > high_20
            and (
                close -
                previous_close
            ) > 0.5 * atr
        ):

            return (
                SetupType.BREAKOUT,
                "Price expansion above recent structure"
            )

        if (
            close < low_20
            and (
                previous_close -
                close
            ) > 0.5 * atr
        ):

            return (
                SetupType.BREAKOUT,
                "Price expansion below recent structure"
            )

        # ====================================================================
        # VWAP RECLAIM
        # ====================================================================

        if (
            previous_close
            < metrics.vwap_session
            and close
            > metrics.vwap_session
        ):

            return (
                SetupType.RECLAIM,
                "Price reclaimed Session VWAP"
            )

        if (
            previous_close
            > metrics.vwap_session
            and close
            < metrics.vwap_session
        ):

            return (
                SetupType.RECLAIM,
                "Price lost Session VWAP"
            )

        # ====================================================================
        # REJECTION
        # ====================================================================

        candle = df.iloc[-1]

        upper_wick = (
            candle["high"] -
            max(
                candle["open"],
                candle["close"]
            )
        )

        lower_wick = (
            min(
                candle["open"],
                candle["close"]
            ) -
            candle["low"]
        )

        if (
            close < metrics.vwap_session
            and upper_wick > 0.5 * atr
        ):

            return (
                SetupType.REJECTION,
                "Upper rejection near value"
            )

        if (
            close > metrics.vwap_session
            and lower_wick > 0.5 * atr
        ):

            return (
                SetupType.REJECTION,
                "Lower rejection near value"
            )

        # ====================================================================
        # MEAN REVERSION
        # ====================================================================

        distance = (
            close -
            metrics.vwap_session
        ) / atr

        if abs(distance) >= 2.5:

            return (
                SetupType.MEAN_REVERSION,
                "Price is materially extended from VWAP"
            )

        return (
            SetupType.NO_SETUP,
            "No high-quality structural setup"
        )


# ============================================================================
# 9. TRIGGER ENGINE
# ============================================================================

class TriggerEngine:

    @staticmethod
    def evaluate_trigger(
        setup: SetupType,
        score: ScoringBreakdown,
        mft: Dict[str, Any],
        metrics: QuantitativeMetrics
    ) -> TriggerType:

        if setup == SetupType.NO_SETUP:

            return TriggerType.WAIT

        score_value = score.total_score

        context = mft["context_bias"]

        execution = mft["exec_score"]

        # ====================================================================
        # LONG
        # ====================================================================

        long_alignment = (
            score_value >= 12
            and context == "BULLISH"
            and execution >= 55
        )

        # ====================================================================
        # SHORT
        # ====================================================================

        short_alignment = (
            score_value <= -12
            and context == "BEARISH"
            and execution <= 45
        )

        if (
            long_alignment
            and setup in (
                SetupType.RECLAIM,
                SetupType.BREAKOUT
            )
        ):

            return TriggerType.CONFIRMED_BUY

        if (
            short_alignment
            and setup in (
                SetupType.RECLAIM,
                SetupType.BREAKOUT
            )
        ):

            return TriggerType.CONFIRMED_SELL

        # Absorption can act as reversal confirmation

        if (
            metrics.cvd_divergence
            == "BULLISH_ABSORPTION"
            and score_value >= 10
            and context != "BEARISH"
        ):

            return TriggerType.CONFIRMED_BUY

        if (
            metrics.cvd_divergence
            == "BEARISH_ABSORPTION"
            and score_value <= -10
            and context != "BULLISH"
        ):

            return TriggerType.CONFIRMED_SELL

        return TriggerType.WAIT


# ============================================================================
# 10. SIGNAL GRADING
# ============================================================================

class SignalGradingEngine:

    @staticmethod
    def grade_signal(
        score: ScoringBreakdown,
        trigger: TriggerType,
        setup: SetupType
    ) -> SignalGrade:

        strength = abs(
            score.total_score
        )

        quality = (
            score.data_quality_pct
        )

        if (
            trigger
            in (
                TriggerType.CONFIRMED_BUY,
                TriggerType.CONFIRMED_SELL
            )
            and strength >= 18
            and quality >= 85
        ):

            return SignalGrade.INSTITUTIONAL_STRONG

        if (
            trigger
            in (
                TriggerType.CONFIRMED_BUY,
                TriggerType.CONFIRMED_SELL
            )
            and strength >= 14
            and quality >= 70
        ):

            return SignalGrade.CONFIRMED

        if (
            strength >= 10
            and setup != SetupType.NO_SETUP
        ):

            return SignalGrade.MODERATE

        if strength >= 5:

            return SignalGrade.NEUTRAL

        return SignalGrade.NO_TRADE


# ============================================================================
# 11. TRADE MANAGEMENT ENGINE
# ============================================================================

class TradeManagementEngine:

    @staticmethod
    def build_trade_plan(
        direction: str,
        df: pd.DataFrame,
        metrics: QuantitativeMetrics,
        capital: float,
        risk_pct: float
    ) -> TradePlan:

        close = float(
            df["close"].iloc[-1]
        )

        atr = max(
            float(metrics.atr_14),
            close * 0.005
        )

        # --------------------------------------------------------------------
        # Entry
        # --------------------------------------------------------------------

        vwap_values = [
            metrics.vwap_session,
            metrics.vwap_weekly
        ]

        if metrics.vwap_anchored is not None:

            vwap_values.append(
                metrics.vwap_anchored
            )

        vwap_values = [
            x for x in vwap_values
            if x is not None
            and np.isfinite(x)
            and x > 0
        ]

        if vwap_values:

            entry_reference = float(
                np.mean(vwap_values)
            )

        else:

            entry_reference = close

        # Do not allow reference to be too far away
        max_entry_distance = 0.75 * atr

        entry_reference = np.clip(
            entry_reference,
            close - max_entry_distance,
            close + max_entry_distance
        )

        entry_low = (
            min(
                close,
                entry_reference
            ) -
            0.10 * atr
        )

        entry_high = (
            max(
                close,
                entry_reference
            ) +
            0.10 * atr
        )

        # --------------------------------------------------------------------
        # Stop / targets
        # --------------------------------------------------------------------

        if direction == "LONG":

            stop_loss = (
                entry_reference -
                1.25 * atr
            )

            risk_per_unit = (
                entry_reference -
                stop_loss
            )

            tp1 = (
                entry_reference +
                1.0 * risk_per_unit
            )

            tp2 = (
                entry_reference +
                2.0 * risk_per_unit
            )

            tp3 = (
                entry_reference +
                3.0 * risk_per_unit
            )

        elif direction == "SHORT":

            stop_loss = (
                entry_reference +
                1.25 * atr
            )

            risk_per_unit = (
                stop_loss -
                entry_reference
            )

            tp1 = (
                entry_reference -
                1.0 * risk_per_unit
            )

            tp2 = (
                entry_reference -
                2.0 * risk_per_unit
            )

            tp3 = (
                entry_reference -
                3.0 * risk_per_unit
            )

        else:

            return TradePlan(
                direction="NONE",
                capital=capital,
                risk_pct=risk_pct
            )

        # --------------------------------------------------------------------
        # Risk
        # --------------------------------------------------------------------

        dollar_risk = (
            capital *
            risk_pct /
            100
        )

        position_size = (
            dollar_risk /
            max(risk_per_unit, 1e-9)
        )

        position_notional = (
            position_size *
            entry_reference
        )

        # --------------------------------------------------------------------
        # R:R
        # --------------------------------------------------------------------

        rr_tp1 = (
            abs(tp1 - entry_reference) /
            max(risk_per_unit, 1e-9)
        )

        rr_tp2 = (
            abs(tp2 - entry_reference) /
            max(risk_per_unit, 1e-9)
        )

        rr_tp3 = (
            abs(tp3 - entry_reference) /
            max(risk_per_unit, 1e-9)
        )

        # --------------------------------------------------------------------
        # Trailing
        # --------------------------------------------------------------------

        trailing_activation = (
            tp1
        )

        trailing_distance = (
            0.75 * atr
        )

        return TradePlan(
            direction=direction,

            entry_low=float(
                entry_low
            ),

            entry_high=float(
                entry_high
            ),

            entry_reference=float(
                entry_reference
            ),

            stop_loss=float(
                stop_loss
            ),

            tp1=float(tp1),
            tp2=float(tp2),
            tp3=float(tp3),

            risk_per_unit=float(
                risk_per_unit
            ),

            rr_tp1=float(rr_tp1),
            rr_tp2=float(rr_tp2),
            rr_tp3=float(rr_tp3),

            capital=float(capital),
            risk_pct=float(risk_pct),
            dollar_risk=float(
                dollar_risk
            ),

            position_size=float(
                position_size
            ),

            position_notional=float(
                position_notional
            ),

            trailing_activation=float(
                trailing_activation
            ),

            trailing_distance=float(
                trailing_distance
            )
        )


# ============================================================================
# 12. TRADE MANAGEMENT STATUS
# ============================================================================

class TradeStateEngine:

    @staticmethod
    def evaluate_position(
        current_price: float,
        plan: TradePlan
    ) -> str:

        if plan.direction == "NONE":

            return "NO ACTIVE PLAN"

        if plan.direction == "LONG":

            if current_price <= plan.stop_loss:

                return "STOP LOSS ZONE"

            if current_price >= plan.tp3:

                return "TP3 REACHED"

            if current_price >= plan.tp2:

                return "TP2 REACHED / TRAIL"

            if current_price >= plan.tp1:

                return "TP1 REACHED / MOVE SL"

            return "POSITION ACTIVE"

        if plan.direction == "SHORT":

            if current_price >= plan.stop_loss:

                return "STOP LOSS ZONE"

            if current_price <= plan.tp3:

                return "TP3 REACHED"

            if current_price <= plan.tp2:

                return "TP2 REACHED / TRAIL"

            if current_price <= plan.tp1:

                return "TP1 REACHED / MOVE SL"

            return "POSITION ACTIVE"

        return "NO ACTIVE PLAN"


# ============================================================================
# 13. UI CSS
# ============================================================================

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

        .status-badge-green {
            background-color: #133E2B;
            color: #00E676;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        }

        .status-badge-red {
            background-color: #4A191B;
            color: #FF5252;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        }

        .status-badge-yellow {
            background-color: #3D3214;
            color: #FFD600;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        }

        .trade-card {
            background-color: #1E222D;
            border: 1px solid #2A2E39;
            border-radius: 8px;
            padding: 15px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# 14. MAIN
# ============================================================================

def main():

    render_css()

    # ========================================================================
    # SIDEBAR
    # ========================================================================

    with st.sidebar:

        st.title("⚡ AliQuantFund")

        st.caption(
            "Institutional Engine v4.3"
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
            ],
            index=0
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
            "رأس المال ($)",
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

        st.markdown("---")

        auto_refresh = st.checkbox(
            "Auto Refresh",
            value=True
        )

        if auto_refresh:

            st.caption(
                "البيانات يتم تحديثها تلقائياً حسب Cache TTL."
            )

    # ========================================================================
    # DATA
    # ========================================================================

    with st.spinner(
        f"جاري تحليل {symbol}..."
    ):

        df, spot_status = (
            MarketDataLoader
            .fetch_klines(
                symbol,
                timeframe,
                500
            )
        )

        df_futures, futures_status, funding_meta = (
            MarketDataLoader
            .fetch_futures_metrics(
                symbol,
                timeframe
            )
        )

        df_trades, trades_status = (
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
            "❌ تعذر جلب بيانات السوق."
        )

        return

    # ========================================================================
    # INDICATORS
    # ========================================================================

    df_calc = (
        QuantitativeEngine
        .calculate_ichimoku(df)
    )

    atr = (
        QuantitativeEngine
        .calculate_atr(df_calc)
        .iloc[-1]
    )

    vwap_session_series = (
        QuantitativeEngine
        .calculate_vwap(
            df_calc,
            "SESSION"
        )
    )

    vwap_weekly_series = (
        QuantitativeEngine
        .calculate_vwap(
            df_calc,
            "WEEKLY"
        )
    )

    vwap_monthly_series = (
        QuantitativeEngine
        .calculate_vwap(
            df_calc,
            "MONTHLY"
        )
    )

    anchor_idx, anchor_reason = (
        QuantitativeEngine
        .detect_smart_anchor(
            df_calc
        )
    )

    avwap_series = (
        QuantitativeEngine
        .calculate_anchored_vwap(
            df_calc,
            anchor_idx
        )
    )

    cvd_series, cvd_type, cvd_slope, cvd_div = (
        QuantitativeEngine
        .compute_cvd_metrics(
            df_calc,
            df_trades,
            timeframe
        )
    )

    # ========================================================================
    # OI
    # ========================================================================

    oi_change_pct = 0.0

    if (
        df_futures is not None
        and not df_futures.empty
        and len(df_futures) >= 2
    ):

        oi_start = (
            df_futures[
                "openInterest"
            ].iloc[0]
        )

        oi_end = (
            df_futures[
                "openInterest"
            ].iloc[-1]
        )

        oi_change_pct = (
            (
                oi_end -
                oi_start
            ) /
            max(
                abs(oi_start),
                1e-9
            )
        ) * 100

    # ========================================================================
    # FUNDING INTERPRETATION
    # ========================================================================

    funding_rate = (
        funding_meta.get(
            "current"
        )
    )

    if funding_rate is None:

        funding_bias = "NEUTRAL"

    elif funding_rate > 0.0003:

        funding_bias = "LONG_OVERHEATED"

    elif funding_rate < -0.0001:

        funding_bias = "SHORT_HEAVY"

    else:

        funding_bias = "NEUTRAL"

    # ========================================================================
    # OI INTERPRETATION
    # ========================================================================

    if oi_change_pct > 2:

        oi_interpretation = "RISING_OI"

    elif oi_change_pct < -2:

        oi_interpretation = "FALLING_OI"

    else:

        oi_interpretation = "STABLE_OI"

    # ========================================================================
    # METRICS
    # ========================================================================

    metrics = QuantitativeMetrics(

        vwap_session=float(
            vwap_session_series.iloc[-1]
        ),

        vwap_weekly=float(
            vwap_weekly_series.iloc[-1]
        ),

        vwap_monthly=float(
            vwap_monthly_series.iloc[-1]
        ),

        vwap_anchored=(
            float(
                avwap_series.iloc[-1]
            )
            if pd.notna(
                avwap_series.iloc[-1]
            )
            else None
        ),

        atr_14=float(atr),

        cvd_slope=float(
            cvd_slope
        ),

        cvd_divergence=cvd_div,

        oi_change_pct=float(
            oi_change_pct
        ),

        oi_interpretation=(
            oi_interpretation
        ),

        funding_rate=funding_rate,

        funding_bias=funding_bias,

        tenkan=float(
            df_calc[
                "tenkan"
            ].iloc[-1]
        ),

        kijun=float(
            df_calc[
                "kijun"
            ].iloc[-1]
        ),

        span_a=float(
            df_calc[
                "span_a"
            ].iloc[-1]
            if pd.notna(
                df_calc[
                    "span_a"
                ].iloc[-1]
            )
            else df_calc[
                "close"
            ].iloc[-1]
        ),

        span_b=float(
            df_calc[
                "span_b"
            ].iloc[-1]
            if pd.notna(
                df_calc[
                    "span_b"
                ].iloc[-1]
            )
            else df_calc[
                "close"
            ].iloc[-1]
        )
    )

    # ========================================================================
    # MARKET STATE
    # ========================================================================

    market_state = (
        MarketStateEngine
        .classify_market_state(
            df_calc,
            atr
        )
    )

    # ========================================================================
    # MTF
    # ========================================================================

    mft_res = (
        MultiTimeframeHierarchy
        .evaluate_all(
            symbol
        )
    )

    # ========================================================================
    # SCORING
    # ========================================================================

    scoring = (
        FactorScoringEngine
        .compute_layered_score(
            market_state,
            df_calc,
            metrics,
            mft_res,
            futures_status,
            cvd_type
        )
    )

    # ========================================================================
    # SETUP
    # ========================================================================

    setup_type, setup_reason = (
        SetupEngine
        .detect_setup(
            df_calc,
            metrics,
            market_state
        )
    )

    # ========================================================================
    # TRIGGER
    # ========================================================================

    trigger = (
        TriggerEngine
        .evaluate_trigger(
            setup_type,
            scoring,
            mft_res,
            metrics
        )
    )

    # ========================================================================
    # GRADE
    # ========================================================================

    grade = (
        SignalGradingEngine
        .grade_signal(
            scoring,
            trigger,
            setup_type
        )
    )

    # ========================================================================
    # DIRECTION
    # ========================================================================

    if trigger == TriggerType.CONFIRMED_BUY:

        direction = "LONG"

    elif trigger == TriggerType.CONFIRMED_SELL:

        direction = "SHORT"

    else:

        direction = "NONE"

    # ========================================================================
    # TRADE PLAN
    # ========================================================================

    trade_plan = (
        TradeManagementEngine
        .build_trade_plan(
            direction,
            df_calc,
            metrics,
            capital,
            risk_pct
        )
    )

    current_price = float(
        df_calc[
            "close"
        ].iloc[-1]
    )

    position_status = (
        TradeStateEngine
        .evaluate_position(
            current_price,
            trade_plan
        )
    )

    # ========================================================================
    # HEADER
    # ========================================================================

    h1, h2, h3, h4 = st.columns(4)

    with h1:

        st.markdown(
            f"### {symbol}"
        )

        st.caption(
            f"Price: ${current_price:,.4f}"
        )

    with h2:

        badge = (
            "status-badge-green"
            if spot_status
            == DataStatus.LIVE.value
            else
            "status-badge-yellow"
        )

        st.markdown(
            f"**Spot:** "
            f"<span class='{badge}'>"
            f"{spot_status}"
            f"</span>",
            unsafe_allow_html=True
        )

    with h3:

        badge = (
            "status-badge-green"
            if futures_status
            == DataStatus.LIVE.value
            else
            "status-badge-yellow"
        )

        st.markdown(
            f"**Futures:** "
            f"<span class='{badge}'>"
            f"{futures_status}"
            f"</span>",
            unsafe_allow_html=True
        )

    with h4:

        badge = (
            "status-badge-green"
            if cvd_type
            == DataStatus.LIVE.value
            else
            "status-badge-yellow"
        )

        st.markdown(
            f"**CVD:** "
            f"<span class='{badge}'>"
            f"{cvd_type}"
            f"</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ========================================================================
    # DECISION
    # ========================================================================

    decision_col, setup_col = st.columns(
        [1, 1]
    )

    with decision_col:

        st.subheader(
            "🎯 القرار التنفيذي"
        )

        score = scoring.total_score

        if (
            trigger
            == TriggerType.CONFIRMED_BUY
            and scoring.data_quality_pct >= 60
        ):

            decision = "CONFIRMED LONG"

        elif (
            trigger
            == TriggerType.CONFIRMED_SELL
            and scoring.data_quality_pct >= 60
        ):

            decision = "CONFIRMED SHORT"

        else:

            decision = "NO TRADE / WAIT"

        st.markdown(
            f"""
            <div class="trade-card">

            <h2>{decision}</h2>

            <p>
            Score:
            <b>{score:.1f}</b>
            </p>

            <p>
            Grade:
            <b>{grade.value}</b>
            </p>

            <p>
            Data Quality:
            <b>{scoring.data_quality_pct:.0f}%</b>
            </p>

            <p>
            Market State:
            <b>{market_state.value}</b>
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with setup_col:

        st.subheader(
            "🧩 Setup / Trigger"
        )

        st.markdown(
            f"""
            **Setup:**  
            `{setup_type.value}`

            **Reason:**  
            {setup_reason}

            **Trigger:**  
            `{trigger.value}`

            **Grade:**  
            `{grade.value}`
            """
        )

    st.markdown("---")

    # ========================================================================
    # MTF
    # ========================================================================

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

        with cols[i]:

            st.metric(
                tf.upper(),
                f"{mft_res['scores'][tf]:.0f}",
                mft_res["biases"][tf]
            )

    st.info(
        f"HTF Context: "
        f"**{mft_res['context_bias']}** "
        f"| Direction: "
        f"**{mft_res['direction_bias']}** "
        f"| Execution Score: "
        f"**{mft_res['exec_score']:.1f}**"
    )

    # ========================================================================
    # TRADE MANAGEMENT
    # ========================================================================

    st.markdown("---")

    st.subheader(
        "🛡️ إدارة الصفقة"
    )

    if direction != "NONE":

        t1, t2, t3, t4 = st.columns(4)

        with t1:

            st.metric(
                "Entry Reference",
                f"{trade_plan.entry_reference:,.6f}"
            )

        with t2:

            st.metric(
                "Stop Loss",
                f"{trade_plan.stop_loss:,.6f}"
            )

        with t3:

            st.metric(
                "TP1",
                f"{trade_plan.tp1:,.6f}"
            )

        with t4:

            st.metric(
                "TP2",
                f"{trade_plan.tp2:,.6f}"
            )

        t5, t6, t7, t8 = st.columns(4)

        with t5:

            st.metric(
                "TP3",
                f"{trade_plan.tp3:,.6f}"
            )

        with t6:

            st.metric(
                "Risk",
                f"${trade_plan.dollar_risk:.2f}"
            )

        with t7:

            st.metric(
                "Position Size",
                f"{trade_plan.position_size:.6f}"
            )

        with t8:

            st.metric(
                "Notional",
                f"${trade_plan.position_notional:.2f}"
            )

        st.markdown(
            f"""
            **Entry Zone:**  
            `{trade_plan.entry_low:,.6f}`
            → `{trade_plan.entry_high:,.6f}`

            **Risk / Reward:**  
            TP1 = `{trade_plan.rr_tp1:.1f}R`  
            TP2 = `{trade_plan.rr_tp2:.1f}R`  
            TP3 = `{trade_plan.rr_tp3:.1f}R`

            **Trailing Activation:**  
            `{trade_plan.trailing_activation:,.6f}`

            **Trailing Distance:**  
            `{trade_plan.trailing_distance:,.6f}`

            **Current Management Status:**  
            `{position_status}`
            """
        )

        st.warning(
            "إدارة الصفقة هنا هي خطة حسابية وليست تنفيذ أوامر حقيقية."
        )

    else:

        st.info(
            "لا توجد صفقة مؤكدة حالياً، لذلك لم يتم إنشاء Entry/SL/TP."
        )

    # ========================================================================
    # FACTOR BREAKDOWN
    # ========================================================================

    st.markdown("---")

    st.subheader(
        "📊 Layered Quantitative Scoring"
    )

    f1, f2, f3, f4, f5 = st.columns(5)

    f1.metric(
        "Direction",
        scoring.direction_score
    )

    f2.metric(
        "Flow / CVD",
        scoring.flow_score
    )

    f3.metric(
        "Positioning",
        scoring.positioning_score
    )

    f4.metric(
        "Location / VWAP",
        scoring.location_score
    )

    f5.metric(
        "TOTAL",
        scoring.total_score
    )

    # ========================================================================
    # MARKET DATA
    # ========================================================================

    st.markdown("---")

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "OI Change",
            f"{oi_change_pct:+.2f}%"
        )

    with d2:

        funding_text = (
            "N/A"
            if funding_rate is None
            else f"{funding_rate:.5%}"
        )

        st.metric(
            "Funding",
            funding_text
        )

    with d3:

        st.metric(
            "CVD",
            cvd_div
        )

    with d4:

        st.metric(
            "ATR",
            f"{atr:.6f}"
        )

    # ========================================================================
    # CHART
    # ========================================================================

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

    # ------------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------------

    fig.add_trace(
        go.Candlestick(
            x=df_calc["timestamp"],
            open=df_calc["open"],
            high=df_calc["high"],
            low=df_calc["low"],
            close=df_calc["close"],
            name="Price"
        ),
        row=1,
        col=1
    )

    # ------------------------------------------------------------------------
    # Ichimoku
    # ------------------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df_calc["timestamp"],
            y=df_calc["tenkan"],
            mode="lines",
            name="Tenkan"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_calc["timestamp"],
            y=df_calc["kijun"],
            mode="lines",
            name="Kijun"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_calc["timestamp"],
            y=df_calc["span_a"],
            mode="lines",
            name="Span A",
            line=dict(
                width=1
            )
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_calc["timestamp"],
            y=df_calc["span_b"],
            mode="lines",
            name="Span B",
            fill="tonexty",
            line=dict(
                width=1
            )
        ),
        row=1,
        col=1
    )

    # ------------------------------------------------------------------------
    # VWAP Suite
    # ------------------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df_calc["timestamp"],
            y=vwap_session_series,
            mode="lines",
            name="Session VWAP"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_calc["timestamp"],
            y=vwap_weekly_series,
            mode="lines",
            name="Weekly VWAP",
            line=dict(
                dash="dash"
            )
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_calc["timestamp"],
            y=vwap_monthly_series,
            mode="lines",
            name="Monthly VWAP",
            line=dict(
                dash="dot"
            )
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_calc["timestamp"],
            y=avwap_series,
            mode="lines",
            name="Smart Anchored VWAP",
            line=dict(
                width=2
            )
        ),
        row=1,
        col=1
    )

    # ------------------------------------------------------------------------
    # Trade Management Lines
    # ------------------------------------------------------------------------

    if direction != "NONE":

        fig.add_hline(
            y=trade_plan.entry_reference,
            line_dash="dot",
            annotation_text="ENTRY",
            row=1,
            col=1
        )

        fig.add_hline(
            y=trade_plan.stop_loss,
            line_dash="dash",
            annotation_text="STOP",
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

    # ------------------------------------------------------------------------
    # CVD
    # ------------------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df_calc["timestamp"],
            y=cvd_series,
            mode="lines",
            name=f"CVD ({cvd_type})"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================

    with st.expander(
        "🔧 Engine Diagnostics"
    ):

        st.write(
            {
                "Market State":
                    market_state.value,

                "Setup":
                    setup_type.value,

                "Setup Reason":
                    setup_reason,

                "Trigger":
                    trigger.value,

                "Signal Grade":
                    grade.value,

                "HTF Context":
                    mft_res["context_bias"],

                "Execution Score":
                    mft_res["exec_score"],

                "CVD Type":
                    cvd_type,

                "CVD Divergence":
                    cvd_div,

                "OI Change":
                    oi_change_pct,

                "Funding":
                    funding_rate,

                "Anchor":
                    anchor_reason,

                "Data Quality":
                    scoring.data_quality_pct
            }
        )


# ============================================================================
# 15. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
