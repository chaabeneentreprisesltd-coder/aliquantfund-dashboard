import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Any, Tuple, List
from datetime import datetime, timezone
import logging
import time

# =========================================================
# AliQuantFund v4.0
# Market State + Setup Detection Engine
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AliQuantFundEngine")

st.set_page_config(
    page_title="AliQuantFund | Quantitative Market Analysis Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(""" <style> @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap'); html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; direction: rtl; text-align: right; } .stMetric { background-color: #1a1e29; padding: 12px; border-radius: 10px; border: 1px solid #2b3245; } p, span, label, div { word-break: break-word !important; white-space: normal !important; } .status-live { background:#0e382c;color:#00e676;padding:4px 10px; border-radius:6px;font-weight:bold;border:1px solid #00e676; } .status-warn { background:#3d310d;color:#ffb300;padding:4px 10px; border-radius:6px;font-weight:bold;border:1px solid #ffb300; } .status-bad { background:#3a1115;color:#ff5252;padding:4px 10px; border-radius:6px;font-weight:bold;border:1px solid #ff5252; } .decision-box { padding:15px;border-radius:12px;border:1px solid #2b3245; background:#151923;margin-bottom:12px; } </style> """, unsafe_allow_html=True)


# =========================================================
# 1. CONSTANTS / HELPERS
# =========================================================

class DataStatus:
    LIVE = "LIVE"
    FALLBACK = "FALLBACK"
    UNAVAILABLE = "UNAVAILABLE"


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def pct_change_safe(a, b):
    if b is None or not np.isfinite(b) or abs(b) < 1e-12:
        return 0.0
    return float((a - b) / b * 100.0)


# =========================================================
# 2. DATA LAYER
# =========================================================

class MarketDataLoader:

    BYBIT_TF_MAP = {
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240",
        "1d": "D"
    }

@staticmethod
@st.cache_data(ttl=15, show_spinner=False)
    def fetch_spot_ohlcv( symbol: str, interval: str, limit: int = 250 ) -> Tuple[Optional[pd.DataFrame], str, str]:

        formatted = symbol.replace("/", "").upper()
        headers = {"User-Agent": "AliQuantFund/4.0"}

        urls = [
            f"https://api1.binance.com/api/v3/klines?symbol={formatted}&interval={interval}&limit={limit}",
            f"https://api3.binance.com/api/v3/klines?symbol={formatted}&interval={interval}&limit={limit}",
            f"https://data-api.binance.vision/api/v3/klines?symbol={formatted}&interval={interval}&limit={limit}"
        ]

        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=4)
                if r.status_code != 200:
                    continue

                data = r.json()
                if not isinstance(data, list) or not data:
                    continue

                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_av", "trades",
                    "tb_base_av", "tb_quote_av", "ignore"
                ])

                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], unit="ms", utc=True
                )

                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                df = df[
                    ["timestamp", "open", "high", "low", "close", "volume"]
                ].dropna().drop_duplicates("timestamp")

                # Remove currently open candle to reduce partial-candle leakage.
                if len(df) > 2:
                    df = df.iloc[:-1].copy()

                return df.reset_index(drop=True), DataStatus.LIVE, "Binance Spot"

            except Exception as e:
                logger.warning("Binance spot failed: %s", e)

        # Bybit fallback
        try:
            tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, "5")
            url = (
                "https://api.bybit.com/v5/market/kline"
                f"?category=spot&symbol={formatted}&interval={tf}&limit={limit}"
            )

            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                result = r.json().get("result", {}).get("list", [])
                if result:
                    df = pd.DataFrame(
                        result,
                        columns=[
                            "timestamp", "open", "high", "low",
                            "close", "volume", "turnover"
                        ]
                    )
                    df = df.iloc[::-1].reset_index(drop=True)
                    df["timestamp"] = pd.to_datetime(
                        pd.to_numeric(df["timestamp"]),
                        unit="ms",
                        utc=True
                    )

                    for c in ["open", "high", "low", "close", "volume"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                    df = df[
                        ["timestamp", "open", "high", "low", "close", "volume"]
                    ].dropna().drop_duplicates("timestamp")

                    if len(df) > 2:
                        df = df.iloc[:-1].copy()

                    return df.reset_index(drop=True), DataStatus.FALLBACK, "Bybit Spot"

        except Exception as e:
            logger.warning("Bybit spot failed: %s", e)

        return None, DataStatus.UNAVAILABLE, "None"

@staticmethod
@st.cache_data(ttl=20, show_spinner=False)
    def fetch_futures_metrics( symbol: str, interval: str, limit: int = 100 ) -> Tuple[Optional[pd.DataFrame], str]:

        formatted = symbol.replace("/", "").upper()
        headers = {"User-Agent": "AliQuantFund/4.0"}
        tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, "5")

        try:
            if interval == "1d":
                oi_interval = "1d"
            else:
                oi_interval = f"{tf}m"

            url_oi = (
                "https://api.bybit.com/v5/market/open-interest"
                f"?category=linear&symbol={formatted}"
                f"&intervalTime={oi_interval}&limit={limit}"
            )

            url_fr = (
                "https://api.bybit.com/v5/market/funding/history"
                f"?category=linear&symbol={formatted}&limit=20"
            )

            r_oi = requests.get(url_oi, headers=headers, timeout=5)
            r_fr = requests.get(url_fr, headers=headers, timeout=5)

            if r_oi.status_code != 200:
                return None, DataStatus.UNAVAILABLE

            oi_list = r_oi.json().get("result", {}).get("list", [])
            if not oi_list:
                return None, DataStatus.UNAVAILABLE

            df = pd.DataFrame(oi_list)
            df["openInterest"] = pd.to_numeric(
                df["openInterest"], errors="coerce"
            )
            df["timestamp"] = pd.to_datetime(
                pd.to_numeric(df["timestamp"]),
                unit="ms",
                utc=True
            )

            df = df[
                ["timestamp", "openInterest"]
            ].dropna().sort_values("timestamp").reset_index(drop=True)

            funding_rate = np.nan

            if r_fr.status_code == 200:
                fr_list = r_fr.json().get("result", {}).get("list", [])
                if fr_list:
                    funding_rate = safe_float(
                        fr_list[0].get("fundingRate")
                    )

            df["funding_rate"] = funding_rate

            # Derived positioning variables.
            df["oi_change_pct"] = df["openInterest"].pct_change() * 100
            df["oi_momentum"] = df["oi_change_pct"].rolling(5).mean()

            return df, DataStatus.LIVE

        except Exception as e:
            logger.warning("Futures metrics failed: %s", e)
            return None, DataStatus.UNAVAILABLE

@staticmethod
@st.cache_data(ttl=5, show_spinner=False)
    def fetch_trade_level_orderflow( symbol: str, limit: int = 1000 ) -> Tuple[Optional[pd.DataFrame], bool, str]:

        formatted = symbol.replace("/", "").upper()
        url = (
            "https://api1.binance.com/api/v3/trades"
            f"?symbol={formatted}&limit={limit}"
        )

        try:
            r = requests.get(
                url,
                headers={"User-Agent": "AliQuantFund/4.0"},
                timeout=4
            )

            if r.status_code != 200:
                return None, False, DataStatus.UNAVAILABLE

            trades = r.json()

            if not isinstance(trades, list) or not trades:
                return None, False, DataStatus.UNAVAILABLE

            df = pd.DataFrame(trades)

            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
            df["time"] = pd.to_datetime(
                pd.to_numeric(df["time"]),
                unit="ms",
                utc=True
            )

            # Binance isBuyerMaker=True means the buyer was maker,
            # therefore aggressive seller hit the bid.
            df["is_buy"] = ~df["isBuyerMaker"].astype(bool)

            df = df[
                ["time", "price", "qty", "is_buy"]
            ].dropna().sort_values("time")

            return df, True, DataStatus.LIVE

        except Exception as e:
            logger.warning("Trade-level data failed: %s", e)
            return None, False, DataStatus.UNAVAILABLE


# =========================================================
# 3. QUANTITATIVE ENGINE
# =========================================================

class QuantitativeEngine:

@staticmethod
    def calculate_atr(df, period=14):
        prev_close = df["close"].shift(1)

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1).max(axis=1)

        # Wilder-style approximation.
        atr = tr.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()

        return atr.bfill().fillna(df["close"] * 0.01)

@staticmethod
    def calculate_adx_trend(df, period=14):
        high = df["high"]
        low = df["low"]
        close = df["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where(
                (up_move > down_move) & (up_move > 0),
                up_move,
                0.0
            ),
            index=df.index
        )

        minus_dm = pd.Series(
            np.where(
                (down_move > up_move) & (down_move > 0),
                down_move,
                0.0
            ),
            index=df.index
        )

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean().replace(0, np.nan)

        plus_di = (
            100 *
            plus_dm.ewm(alpha=1 / period, adjust=False).mean() /
            atr
        )

        minus_di = (
            100 *
            minus_dm.ewm(alpha=1 / period, adjust=False).mean() /
            atr
        )

        dx = (
            (plus_di - minus_di).abs() /
            (plus_di + minus_di).replace(0, np.nan)
        ) * 100

        adx = dx.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()

        return (
            adx.bfill().fillna(20),
            plus_di.bfill().fillna(20),
            minus_di.bfill().fillna(20)
        )

@staticmethod
    def detect_smart_anchor(df, mode="Automatic"):
        if df is None or len(df) < 5:
            return {
                "index": 0,
                "price": safe_float(df["close"].iloc[0]) if df is not None else 0,
                "timestamp": df["timestamp"].iloc[0] if df is not None else datetime.now(timezone.utc),
                "type": "Default"
            }

        # No future leakage: only select anchors from completed history,
        # but Automatic uses the strongest completed volume spike in the
        # supplied historical window. It does not use future data relative
        # to the current final candle, but it does use the full historical
        # analysis window.
        if mode == "Swing High":
            idx = df["high"].idxmax()
            return {
                "index": idx,
                "price": df.loc[idx, "high"],
                "timestamp": df.loc[idx, "timestamp"],
                "type": "Swing High"
            }

        if mode == "Swing Low":
            idx = df["low"].idxmin()
            return {
                "index": idx,
                "price": df.loc[idx, "low"],
                "timestamp": df.loc[idx, "timestamp"],
                "type": "Swing Low"
            }

        vol_ma = df["volume"].rolling(20, min_periods=10).mean()
        spikes = df["volume"] > vol_ma * 2.2

        if spikes.any():
            idx = df.loc[spikes, "volume"].idxmax()
            typ = "Major Volume Spike"
        else:
            # Recent structural pivot rather than absolute historical low.
            window = min(60, len(df))
            recent = df.iloc[-window:]
            idx = recent["low"].idxmin()
            typ = "Structural Pivot Low"

        return {
            "index": idx,
            "price": df.loc[idx, "close"],
            "timestamp": df.loc[idx, "timestamp"],
            "type": typ
        }

@staticmethod
    def calculate_indicators(df, anchor_info):
        if df is None or df.empty:
            return df

        df = df.copy()

        df["atr"] = QuantitativeEngine.calculate_atr(df, 14)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        df["tp"] = (
            df["high"] + df["low"] + df["close"]
        ) / 3

        df["pv"] = df["tp"] * df["volume"]

        # Session VWAP.
        df["date"] = df["timestamp"].dt.date
        df["session_pv"] = df.groupby("date")["pv"].cumsum()
        df["session_vol"] = df.groupby("date")["volume"].cumsum()
        df["vwap_session"] = (
            df["session_pv"] /
            df["session_vol"].replace(0, np.nan)
        )

        # Weekly VWAP using ISO year/week.
        iso = df["timestamp"].dt.isocalendar()
        df["week_key"] = (
            iso["year"].astype(str) + "-" +
            iso["week"].astype(str)
        )

        df["weekly_pv"] = df.groupby("week_key")["pv"].cumsum()
        df["weekly_vol"] = df.groupby("week_key")["volume"].cumsum()
        df["vwap_weekly"] = (
            df["weekly_pv"] /
            df["weekly_vol"].replace(0, np.nan)
        )

        # Anchored VWAP.
        anc_idx = anchor_info["index"]
        df["vwap_anchored"] = np.nan

        if anc_idx in df.index:
            part = df.loc[anc_idx:].copy()
            cum_pv = part["pv"].cumsum()
            cum_vol = part["volume"].cumsum()

            df.loc[anc_idx:, "vwap_anchored"] = (
                cum_pv / cum_vol.replace(0, np.nan)
            )

        df["vwap_session"] = df["vwap_session"].ffill().bfill()
        df["vwap_weekly"] = df["vwap_weekly"].ffill().bfill()
        df["vwap_anchored"] = df["vwap_anchored"].ffill().bfill()

        # Volume statistics.
        df["volume_ma20"] = df["volume"].rolling(
            20, min_periods=5
        ).mean()

        df["volume_ratio"] = (
            df["volume"] /
            df["volume_ma20"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # Trend.
        adx, pdi, mdi = QuantitativeEngine.calculate_adx_trend(df)
        df["adx"] = adx
        df["plus_di"] = pdi
        df["minus_di"] = mdi

        # VWAP slopes.
        for col in [
            "vwap_session",
            "vwap_weekly",
            "vwap_anchored"
        ]:
            df[f"{col}_slope_atr"] = (
                df[col].diff(3) /
                df["atr"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

@staticmethod
    def build_real_cvd( df, trades_df: Optional[pd.DataFrame], interval: str ) -> Tuple[pd.DataFrame, str]:

        df = df.copy()
        df["delta"] = np.nan
        df["cvd"] = np.nan

        if trades_df is None or trades_df.empty:
            # Approximation fallback.
            direction = np.where(
                df["close"] >= df["close"].shift(1).bfill(),
                1,
                -1
            )
            df["delta"] = df["volume"] * direction
            df["cvd"] = df["delta"].cumsum()
            return df, "APPROXIMATED"

        try:
            # Binance trade endpoint returns only recent trades.
            # Aggregate them into the same candle bucket.
            trades = trades_df.copy()
            trades["delta"] = np.where(
                trades["is_buy"],
                trades["qty"],
                -trades["qty"]
            )

            # Use candle boundaries based on the selected timeframe.
            freq_map = {
                "5m": "5min",
                "15m": "15min",
                "1h": "1h",
                "4h": "4h",
                "1d": "1D"
            }
            freq = freq_map.get(interval, "5min")

            trade_buckets = (
                trades.set_index("time")
                .resample(freq, label="left", closed="left")["delta"]
                .sum()
            )

            # Match to candle timestamps.
            delta_map = trade_buckets.reindex(
                pd.DatetimeIndex(df["timestamp"])
            )

            df["delta"] = delta_map.values

            # We only have trade-level information for a short recent window.
            # Do not pretend historical candles have real CVD.
            real_mask = df["delta"].notna()

            # Fill historical section with candle approximation only as a
            # separate component, then attach real delta for available candles.
            direction = np.where(
                df["close"] >= df["close"].shift(1).bfill(),
                1,
                -1
            )
            approx_delta = df["volume"] * direction

            combined_delta = approx_delta.astype(float)
            combined_delta[real_mask] = df.loc[real_mask, "delta"]

            df["delta"] = combined_delta
            df["cvd"] = df["delta"].cumsum()

            if real_mask.sum() >= 2:
                return df, "LIMITED_TRADE_LEVEL"

            return df, "APPROXIMATED"

        except Exception as e:
            logger.warning("CVD aggregation failed: %s", e)

            direction = np.where(
                df["close"] >= df["close"].shift(1).bfill(),
                1,
                -1
            )
            df["delta"] = df["volume"] * direction
            df["cvd"] = df["delta"].cumsum()

            return df, "APPROXIMATED"

@staticmethod
    def detect_market_regime(df):
        if df is None or len(df) < 30:
            return "RANGING"

        latest = df.iloc[-1]

        adx = latest["adx"]
        pdi = latest["plus_di"]
        mdi = latest["minus_di"]
        atr_pct = latest["atr_pct"]
        vol_ratio = latest["volume_ratio"]

        recent_high = df["high"].iloc[-21:-1].max()
        recent_low = df["low"].iloc[-21:-1].min()

        breakout_up = latest["close"] > recent_high
        breakout_down = latest["close"] < recent_low

        if breakout_up or breakout_down:
            if vol_ratio >= 1.3:
                return "BREAKOUT"

        if adx >= 25:
            return "TRENDING_BULL" if pdi > mdi else "TRENDING_BEAR"

        if atr_pct >= 2.5 or vol_ratio >= 2.0:
            return "HIGH_VOLATILITY"

        if atr_pct <= 0.6:
            return "LOW_VOLATILITY"

        return "RANGING"


# =========================================================
# 4. VWAP ANALYSIS ENGINE
# =========================================================

class VWAPAnalysisEngine:

@staticmethod
    def analyze(df) -> Dict[str, Any]:
        latest = df.iloc[-1]
        atr = max(latest["atr"], latest["close"] * 0.001)

        result = {
            "session_distance_atr": 0.0,
            "weekly_distance_atr": 0.0,
            "anchored_distance_atr": 0.0,
            "avg_distance_atr": 0.0,
            "session_slope": 0.0,
            "weekly_slope": 0.0,
            "anchored_slope": 0.0,
            "structure": "NEUTRAL",
            "reclaim": False,
            "rejection": False,
            "overextended": False,
            "score": 50.0
        }

        distances = {}

        for name, col in [
            ("session", "vwap_session"),
            ("weekly", "vwap_weekly"),
            ("anchored", "vwap_anchored")
        ]:
            v = latest[col]
            distances[name] = (latest["close"] - v) / atr

        result["session_distance_atr"] = distances["session"]
        result["weekly_distance_atr"] = distances["weekly"]
        result["anchored_distance_atr"] = distances["anchored"]

        result["avg_distance_atr"] = np.mean(list(distances.values()))

        result["session_slope"] = latest["vwap_session_slope_atr"]
        result["weekly_slope"] = latest["vwap_weekly_slope_atr"]
        result["anchored_slope"] = latest["vwap_anchored_slope_atr"]

        above_count = sum(v > 0 for v in distances.values())
        below_count = sum(v < 0 for v in distances.values())

        if above_count >= 2:
            result["structure"] = "BULLISH"
        elif below_count >= 2:
            result["structure"] = "BEARISH"

        # Recent reclaim/rejection detection.
        if len(df) >= 4:
            prev = df.iloc[-2]
            current = df.iloc[-1]

            for col in [
                "vwap_session",
                "vwap_weekly",
                "vwap_anchored"
            ]:
                if (
                    prev["close"] < prev[col] and
                    current["close"] > current[col]
                ):
                    result["reclaim"] = True

                if (
                    prev["close"] > prev[col] and
                    current["close"] < current[col]
                ):
                    result["rejection"] = True

        # Extension across all VWAPs.
        if abs(result["avg_distance_atr"]) >= 2.5:
            result["overextended"] = True

        # Location score is deliberately bounded.
        score = 50.0

        if result["structure"] == "BULLISH":
            score += 15
        elif result["structure"] == "BEARISH":
            score -= 15

        if result["reclaim"]:
            score += 15

        if result["rejection"]:
            score -= 15

        slope_avg = np.mean([
            result["session_slope"],
            result["weekly_slope"],
            result["anchored_slope"]
        ])

        score += np.clip(slope_avg * 5, -10, 10)

        if result["overextended"]:
            # Extension is not automatically bullish/bearish.
            score *= 0.85

        result["score"] = float(np.clip(score, 0, 100))
        return result


# =========================================================
# 5. POSITIONING ENGINE
# =========================================================

class PositioningEngine:

@staticmethod
    def analyze( df_spot, df_futures: Optional[pd.DataFrame] ) -> Dict[str, Any]:

        result = {
            "score": 50.0,
            "state": "NEUTRAL",
            "oi_change_pct": 0.0,
            "funding": np.nan,
            "funding_state": "UNAVAILABLE"
        }

        if df_futures is None or df_futures.empty:
            return result

        try:
            f = df_futures.dropna(
                subset=["openInterest"]
            ).copy()

            if len(f) < 2:
                return result

            latest_oi = f["openInterest"].iloc[-1]
            lookback = min(5, len(f) - 1)
            prev_oi = f["openInterest"].iloc[-1 - lookback]

            oi_change = pct_change_safe(latest_oi, prev_oi)

            price_latest = df_spot["close"].iloc[-1]
            price_prev = df_spot["close"].iloc[-1 - min(
                lookback,
                len(df_spot) - 1
            )]

            price_change = pct_change_safe(
                price_latest,
                price_prev
            )

            funding = safe_float(
                f["funding_rate"].iloc[-1]
            )

            result["oi_change_pct"] = oi_change
            result["funding"] = funding

            # Funding thresholds are intentionally conservative.
            if np.isfinite(funding):
                if funding >= 0.001:
                    funding_state = "ELEVATED_POSITIVE"
                elif funding <= -0.001:
                    funding_state = "ELEVATED_NEGATIVE"
                else:
                    funding_state = "NORMAL"
            else:
                funding_state = "UNAVAILABLE"

            result["funding_state"] = funding_state

            if oi_change > 1 and price_change > 0.1:
                state = "LONG_BUILDUP"
                score = 75
            elif oi_change > 1 and price_change < -0.1:
                state = "SHORT_BUILDUP"
                score = 25
            elif oi_change < -1 and price_change > 0.1:
                state = "SHORT_COVERING"
                score = 58
            elif oi_change < -1 and price_change < -0.1:
                state = "LONG_UNWINDING"
                score = 42
            else:
                state = "NEUTRAL"
                score = 50

            # Funding is a modifier, not an independent directional factor.
            if funding_state == "ELEVATED_POSITIVE":
                if state == "LONG_BUILDUP":
                    score -= 8
                elif state == "SHORT_BUILDUP":
                    score += 4

            if funding_state == "ELEVATED_NEGATIVE":
                if state == "SHORT_BUILDUP":
                    score += 8
                elif state == "LONG_BUILDUP":
                    score -= 4

            result["state"] = state
            result["score"] = float(np.clip(score, 0, 100))

            return result

        except Exception as e:
            logger.warning("Positioning analysis failed: %s", e)
            return result


# =========================================================
# 6. MARKET STATE ENGINE
# =========================================================

class MarketStateEngine:

@staticmethod
    def analyze( df, regime, vwap_info, positioning_info, cvd_quality ) -> Dict[str, Any]:

        latest = df.iloc[-1]

        direction_score = 50.0

        if latest["plus_di"] > latest["minus_di"]:
            direction_score += min(
                latest["adx"] * 1.0,
                40
            )
        else:
            direction_score -= min(
                latest["adx"] * 1.0,
                40
            )

        cvd_score = 50.0

        if len(df) >= 10:
            cvd_change = (
                df["cvd"].iloc[-1] -
                df["cvd"].iloc[-10]
            )
            price_change = (
                df["close"].iloc[-1] -
                df["close"].iloc[-10]
            )

            if cvd_change > 0 and price_change > 0:
                cvd_score = 75
            elif cvd_change < 0 and price_change < 0:
                cvd_score = 25
            elif cvd_change > 0 and price_change < 0:
                cvd_score = 60
            elif cvd_change < 0 and price_change > 0:
                cvd_score = 40

        if cvd_quality == "APPROXIMATED":
            cvd_score = 50 + (cvd_score - 50) * 0.35
        elif cvd_quality == "LIMITED_TRADE_LEVEL":
            cvd_score = 50 + (cvd_score - 50) * 0.75

        volume_score = float(
            np.clip(
                50 + (latest["volume_ratio"] - 1) * 25,
                20,
                90
            )
        )

        momentum_score = (
            direction_score * 0.45 +
            cvd_score * 0.35 +
            volume_score * 0.20
        )

        return {
            "regime": regime,
            "direction_score": float(np.clip(direction_score, 0, 100)),
            "flow_score": float(np.clip(cvd_score, 0, 100)),
            "volume_score": volume_score,
            "momentum_score": float(
                np.clip(momentum_score, 0, 100)
            ),
            "positioning_score": positioning_info["score"],
            "vwap_score": vwap_info["score"]
        }


# =========================================================
# 7. SETUP DETECTION ENGINE
# =========================================================

class SetupDetectionEngine:

@staticmethod
    def detect( df, market_state, vwap_info, positioning_info, htf_bias, tf_context="execution" ) -> Dict[str, Any]:

        latest = df.iloc[-1]

        volume_confirm = latest["volume_ratio"] >= 1.15
        flow_bull = market_state["flow_score"] >= 58
        flow_bear = market_state["flow_score"] <= 42

        bullish_context = (
            "BULLISH" in htf_bias or
            market_state["direction_score"] >= 60
        )

        bearish_context = (
            "BEARISH" in htf_bias or
            market_state["direction_score"] <= 40
        )

        recent_high = df["high"].iloc[-21:-1].max()
        recent_low = df["low"].iloc[-21:-1].min()

        breakout_up = latest["close"] > recent_high
        breakout_down = latest["close"] < recent_low

        setups = []

        # VWAP Reclaim
        if (
            vwap_info["reclaim"] and
            bullish_context and
            flow_bull and
            volume_confirm and
            not vwap_info["overextended"]
        ):
            setups.append(
                (
                    "VWAP_RECLAIM",
                    82,
                    "السعر استعاد VWAP مع تأكيد الحجم وتدفق الشراء."
                )
            )

        # VWAP Rejection
        if (
            vwap_info["rejection"] and
            bearish_context and
            flow_bear and
            volume_confirm
        ):
            setups.append(
                (
                    "VWAP_REJECTION",
                    82,
                    "السعر فشل في استعادة VWAP مع تأكيد ضغط البيع."
                )
            )

        # Breakout
        if (
            breakout_up and
            volume_confirm and
            flow_bull and
            bullish_context
        ):
            setups.append(
                (
                    "BREAKOUT",
                    85,
                    "كسر هيكلي صاعد مدعوم بالحجم وتدفق الشراء."
                )
            )

        if (
            breakout_down and
            volume_confirm and
            flow_bear and
            bearish_context
        ):
            setups.append(
                (
                    "BREAKDOWN",
                    85,
                    "كسر هيكلي هابط مدعوم بالحجم وتدفق البيع."
                )
            )

        # Mean Reversion only in range.
        if (
            market_state["regime"] == "RANGING" and
            abs(vwap_info["avg_distance_atr"]) >= 1.5
        ):
            if vwap_info["avg_distance_atr"] < 0:
                setups.append(
                    (
                        "MEAN_REVERSION_LONG",
                        65,
                        "السوق جانبي والسعر ممتد أسفل VWAP."
                    )
                )
            else:
                setups.append(
                    (
                        "MEAN_REVERSION_SHORT",
                        65,
                        "السوق جانبي والسعر ممتد أعلى VWAP."
                    )
                )

        # Exhaustion warning.
        exhaustion = (
            vwap_info["overextended"] and
            (
                (flow_bull and vwap_info["structure"] == "BEARISH") or
                (flow_bear and vwap_info["structure"] == "BULLISH")
            )
        )

        if exhaustion:
            setups.append(
                (
                    "EXHAUSTION",
                    55,
                    "السعر ممتد مع تعارض بين التدفق وموقع VWAP."
                )
            )

        if not setups:
            return {
                "name": "NONE",
                "score": 50.0,
                "direction": "NEUTRAL",
                "status": "NO_SETUP",
                "reason": "لا يوجد Setup مكتمل حاليًا."
            }

        setups.sort(key=lambda x: x[1], reverse=True)
        name, score, reason = setups[0]

        if "LONG" in name or name in ["VWAP_RECLAIM", "BREAKOUT"]:
            direction = "LONG"
        elif "SHORT" in name or name in ["VWAP_REJECTION", "BREAKDOWN"]:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        return {
            "name": name,
            "score": score,
            "direction": direction,
            "status": "SETUP_FORMING",
            "reason": reason
        }


# =========================================================
# 8. DATA QUALITY ENGINE
# =========================================================

class DataQualityEngine:

@staticmethod
    def calculate( spot_status, futures_status, cvd_quality ) -> Dict[str, Any]:

        score = 100
        reasons = []

        if spot_status == DataStatus.FALLBACK:
            score -= 15
            reasons.append("Spot يعمل عبر مصدر بديل.")

        if spot_status == DataStatus.UNAVAILABLE:
            score -= 50
            reasons.append("بيانات Spot غير متوفرة.")

        if futures_status == DataStatus.UNAVAILABLE:
            score -= 20
            reasons.append("بيانات OI/Funding غير متوفرة.")

        if cvd_quality == "LIMITED_TRADE_LEVEL":
            score -= 10
            reasons.append("CVD Trade-Level متاح لكن تاريخيًا محدود.")

        elif cvd_quality == "APPROXIMATED":
            score -= 25
            reasons.append("CVD تقريبي وليس Trade-Level كاملًا.")

        score = int(np.clip(score, 0, 100))

        return {
            "score": score,
            "reasons": reasons
        }


# =========================================================
# 9. MULTI-TIMEFRAME ENGINE
# =========================================================

class MultiTimeframeEngine:

@staticmethod
    def evaluate(symbol, anchor_mode):
        timeframes = ["1d", "4h", "1h", "15m", "5m"]

        scores = {}
        regimes = {}
        bias = {}

        for tf in timeframes:
            df, status, _ = MarketDataLoader.fetch_spot_ohlcv(
                symbol, tf, limit=180
            )

            if df is None or len(df) < 40:
                scores[tf] = 50.0
                regimes[tf] = "INSUFFICIENT_DATA"
                bias[tf] = "NEUTRAL"
                continue

            anchor = QuantitativeEngine.detect_smart_anchor(
                df, anchor_mode
            )
            df = QuantitativeEngine.calculate_indicators(
                df, anchor
            )

            # No historical trade-level call for every MTF frame.
            # Candle approximation is explicitly marked as such.
            df, cvd_quality = QuantitativeEngine.build_real_cvd(
                df, None, tf
            )

            regime = QuantitativeEngine.detect_market_regime(df)
            vwap = VWAPAnalysisEngine.analyze(df)

            positioning = {
                "score": 50.0,
                "state": "UNAVAILABLE"
            }

            market_state = MarketStateEngine.analyze(
                df,
                regime,
                vwap,
                positioning,
                cvd_quality
            )

            score = (
                market_state["direction_score"] * 0.30 +
                market_state["flow_score"] * 0.20 +
                market_state["positioning_score"] * 0.10 +
                market_state["vwap_score"] * 0.20 +
                market_state["volume_score"] * 0.10 +
                market_state["momentum_score"] * 0.10
            )

            scores[tf] = float(np.clip(score, 0, 100))
            regimes[tf] = regime

            if scores[tf] >= 60:
                bias[tf] = "BULLISH"
            elif scores[tf] <= 40:
                bias[tf] = "BEARISH"
            else:
                bias[tf] = "NEUTRAL"

        htf_score = (
            scores["1d"] * 0.45 +
            scores["4h"] * 0.35 +
            scores["1h"] * 0.20
        )

        if htf_score >= 62:
            htf_bias = "BULLISH"
        elif htf_score <= 38:
            htf_bias = "BEARISH"
        else:
            htf_bias = "NEUTRAL"

        execution_score = (
            scores["15m"] * 0.60 +
            scores["5m"] * 0.40
        )

        return {
            "scores": scores,
            "regimes": regimes,
            "bias": bias,
            "htf_score": htf_score,
            "htf_bias": htf_bias,
            "execution_score": execution_score
        }


# =========================================================
# 10. FINAL SCORING / SIGNAL ENGINE
# =========================================================

class FactorScoringEngine:

@staticmethod
    def score( market_state, vwap_info, positioning, setup, mtf, data_quality ) -> Dict[str, Any]:

        direction = market_state["direction_score"]
        flow = market_state["flow_score"]
        position = positioning["score"]
        location = vwap_info["score"]
        setup_score = setup["score"]

        # Direction / Flow / Positioning / Location / Trigger.
        # Setup is a trigger, so it has lower weight than context.
        raw_score = (
            direction * 0.25 +
            flow * 0.20 +
            position * 0.15 +
            location * 0.15 +
            setup_score * 0.25
        )

        # Confidence adjustment for data quality.
        confidence = (
            45 +
            abs(raw_score - 50) * 0.9
        )
        confidence *= data_quality / 100.0
        confidence = float(np.clip(confidence, 10, 95))

        # Directional score is used for final long/short interpretation.
        if setup["direction"] == "LONG":
            directional_edge = raw_score
        elif setup["direction"] == "SHORT":
            directional_edge = 100 - raw_score
        else:
            directional_edge = 50

        return {
            "global_score": float(np.round(raw_score, 1)),
            "confidence": float(np.round(confidence, 1)),
            "direction_score": direction,
            "flow_score": flow,
            "positioning_score": position,
            "location_score": location,
            "setup_score": setup_score,
            "directional_edge": directional_edge
        }


class SignalAndRiskEngine:

@staticmethod
    def classify( score, setup, mtf, data_quality ) -> Tuple[str, List[str]]:

        reasons = []

        if data_quality < 55:
            return "NO TRADE", [
                "جودة البيانات منخفضة جدًا."
            ]

        if setup["status"] == "NO_SETUP":
            return "WAIT", [
                "لا يوجد Setup مكتمل."
            ]

        if setup["name"] == "EXHAUSTION":
            return "NO TRADE", [
                setup["reason"],
                "تم منع الدخول بسبب احتمال Exhaustion."
            ]

        htf = mtf["htf_bias"]
        direction = setup["direction"]

        if direction == "LONG":
            if htf != "BULLISH":
                return "SETUP FORMING", [
                    "الـ Setup صاعد لكن اتجاه HTF غير متوافق بالكامل."
                ]

            if score["global_score"] >= 70 and score["confidence"] >= 65:
                reasons.append("اتجاه HTF صاعد.")
                reasons.append(setup["reason"])
                return "CONFIRMED LONG", reasons

            return "SETUP FORMING", [
                setup["reason"],
                "قوة الإشارة أو الثقة غير كافية للتأكيد."
            ]

        if direction == "SHORT":
            if htf != "BEARISH":
                return "SETUP FORMING", [
                    "الـ Setup هابط لكن اتجاه HTF غير متوافق بالكامل."
                ]

            if score["global_score"] <= 30 and score["confidence"] >= 65:
                reasons.append("اتجاه HTF هابط.")
                reasons.append(setup["reason"])
                return "CONFIRMED SHORT", reasons

            return "SETUP FORMING", [
                setup["reason"],
                "قوة الإشارة أو الثقة غير كافية للتأكيد."
            ]

        return "WAIT", ["لا يوجد اتجاه تنفيذي واضح."]

@staticmethod
    def calculate_risk( entry, reference, atr, is_long, capital, base_risk_pct, signal ):

        atr = max(float(atr), entry * 0.001)

        if is_long:
            sl = min(
                reference - atr * 0.5,
                entry - atr * 0.8
            )
            distance = entry - sl
        else:
            sl = max(
                reference + atr * 0.5,
                entry + atr * 0.8
            )
            distance = sl - entry

        warnings = []

        if distance <= 0:
            return None

        sl_atr = distance / atr

        if sl_atr < 0.5:
            warnings.append("SL ضيق جدًا مقارنة بالتقلب.")
        elif sl_atr > 3.5:
            warnings.append("SL واسع جدًا مقارنة بـ ATR.")

        if signal == "CONFIRMED LONG" or signal == "CONFIRMED SHORT":
            multiplier = 1.0
        elif signal == "SETUP FORMING":
            multiplier = 0.4
        else:
            multiplier = 0.0

        effective_risk_pct = base_risk_pct * multiplier
        risk_amount = capital * effective_risk_pct / 100

        units = risk_amount / distance if distance > 0 else 0
        position_value = units * entry

        if is_long:
            tp1 = entry + distance * 1.5
            tp2 = entry + distance * 3.0
        else:
            tp1 = entry - distance * 1.5
            tp2 = entry - distance * 3.0

        return {
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "distance": distance,
            "sl_atr": sl_atr,
            "risk_amount": risk_amount,
            "risk_pct": effective_risk_pct,
            "units": units,
            "position_value": position_value,
            "rr1": 1.5,
            "rr2": 3.0,
            "warnings": warnings
        }


# =========================================================
# 11. UI
# =========================================================

st.sidebar.title("⚡ AliQuantFund")
st.sidebar.caption("Market State + Setup Engine v4.0")
st.sidebar.markdown("---")

selected_symbol = st.sidebar.selectbox(
    "الأصل المالي:",
    ["BTC/USDT", "ETH/USDT", "ZEC/USDT", "XRP/USDT", "SOL/USDT"]
)

selected_tf = st.sidebar.selectbox(
    "إطار التنفيذ:",
    ["5m", "15m", "1h", "4h", "1d"],
    index=0
)

anchor_mode = st.sidebar.selectbox(
    "مرساة VWAP:",
    ["Automatic", "Swing High", "Swing Low"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 إدارة رأس المال")

capital = st.sidebar.number_input(
    "رأس المال ($):",
    min_value=1.0,
    value=100.0,
    step=10.0
)

base_risk_pct = st.sidebar.number_input(
    "أقصى مخاطرة (%):",
    min_value=0.1,
    max_value=10.0,
    value=2.0,
    step=0.5
)

st.title(
    f"📊 AliQuantFund — {selected_symbol} / {selected_tf}"
)

# =========================================================
# DATA FETCH
# =========================================================

spot_df, spot_status, spot_source = (
    MarketDataLoader.fetch_spot_ohlcv(
        selected_symbol,
        selected_tf,
        250
    )
)

futures_df, futures_status = (
    MarketDataLoader.fetch_futures_metrics(
        selected_symbol,
        selected_tf,
        100
    )
)

trades_df, _, trade_status = (
    MarketDataLoader.fetch_trade_level_orderflow(
        selected_symbol,
        1000
    )
)

if spot_df is None or spot_df.empty or len(spot_df) < 40:
    st.error(
        "❌ البيانات غير كافية لبناء تحليل موثوق."
    )
    st.stop()

# =========================================================
# CALCULATIONS
# =========================================================

anchor_info = QuantitativeEngine.detect_smart_anchor(
    spot_df,
    anchor_mode
)

spot_df = QuantitativeEngine.calculate_indicators(
    spot_df,
    anchor_info
)

spot_df, cvd_quality = QuantitativeEngine.build_real_cvd(
    spot_df,
    trades_df,
    selected_tf
)

regime = QuantitativeEngine.detect_market_regime(
    spot_df
)

vwap_info = VWAPAnalysisEngine.analyze(
    spot_df
)

positioning = PositioningEngine.analyze(
    spot_df,
    futures_df
)

mtf = MultiTimeframeEngine.evaluate(
    selected_symbol,
    anchor_mode
)

market_state = MarketStateEngine.analyze(
    spot_df,
    regime,
    vwap_info,
    positioning,
    cvd_quality
)

data_quality = DataQualityEngine.calculate(
    spot_status,
    futures_status,
    cvd_quality
)

setup = SetupDetectionEngine.detect(
    spot_df,
    market_state,
    vwap_info,
    positioning,
    mtf["htf_bias"],
    "execution"
)

score = FactorScoringEngine.score(
    market_state,
    vwap_info,
    positioning,
    setup,
    mtf,
    data_quality["score"]
)

signal, reasons = SignalAndRiskEngine.classify(
    score,
    setup,
    mtf,
    data_quality["score"]
)

# =========================================================
# STATUS BAR
# =========================================================

spot_badge = (
    "status-live"
    if spot_status == DataStatus.LIVE
    else "status-warn"
)

fut_badge = (
    "status-live"
    if futures_status == DataStatus.LIVE
    else "status-warn"
)

cvd_badge = (
    "status-live"
    if cvd_quality == "LIMITED_TRADE_LEVEL"
    else "status-warn"
)

st.markdown(
    f""" <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:15px"> <span class="{spot_badge}">Spot: {spot_status}</span> <span class="{fut_badge}">OI/Funding: {futures_status}</span> <span class="{cvd_badge}">CVD: {cvd_quality}</span> <span style="font-weight:bold">Data Quality: {data_quality["score"]}%</span> </div> """,
    unsafe_allow_html=True
)

# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================

st.markdown("### 🌐 Executive Decision Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "FINAL DECISION",
    signal,
    f"Confidence {score['confidence']:.0f}%"
)

c2.metric(
    "Global Score",
    f"{score['global_score']:.1f}/100",
    f"HTF {mtf['htf_bias']}"
)

c3.metric(
    "Market State",
    regime
)

c4.metric(
    "Active Setup",
    setup["name"]
)

c5.metric(
    "VWAP Structure",
    vwap_info["structure"],
    f"{vwap_info['avg_distance_atr']:.2f} ATR"
)

# =========================================================
# MARKET STATE
# =========================================================

st.markdown("### 🧠 Market State")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Direction",
    f"{market_state['direction_score']:.1f}"
)

m2.metric(
    "Flow / CVD",
    f"{market_state['flow_score']:.1f}"
)

m3.metric(
    "Positioning",
    f"{market_state['positioning_score']:.1f}"
)

m4.metric(
    "VWAP Location",
    f"{market_state['vwap_score']:.1f}"
)

m5.metric(
    "Volume",
    f"{market_state['volume_score']:.1f}"
)

# =========================================================
# MTF
# =========================================================

with st.expander(
    "🕐 Multi-Timeframe Context",
    expanded=True
):
    tf_cols = st.columns(5)

    for col, tf in zip(
        tf_cols,
        ["1d", "4h", "1h", "15m", "5m"]
    ):
        col.metric(
            tf,
            f"{mtf['scores'][tf]:.1f}",
            mtf["bias"][tf]
        )

    st.caption(
        f"HTF Bias: {mtf['htf_bias']} | "
        f"HTF Score: {mtf['htf_score']:.1f} | "
        f"Execution Score: {mtf['execution_score']:.1f}"
    )

# =========================================================
# FACTOR BREAKDOWN
# =========================================================

with st.expander(
    "🔬 Factor Breakdown",
    expanded=False
):
    f1, f2, f3, f4, f5 = st.columns(5)

    f1.metric(
        "Direction",
        f"{score['direction_score']:.1f}"
    )

    f2.metric(
        "Flow",
        f"{score['flow_score']:.1f}"
    )

    f3.metric(
        "Positioning",
        f"{score['positioning_score']:.1f}"
    )

    f4.metric(
        "Location",
        f"{score['location_score']:.1f}"
    )

    f5.metric(
        "Setup",
        f"{score['setup_score']:.1f}"
    )

    st.write(
        f"**OI State:** {positioning['state']} | "
        f"OI Change: {positioning['oi_change_pct']:.2f}%"
    )

    if np.isfinite(positioning["funding"]):
        st.write(
            f"**Funding:** {positioning['funding']:.6f} "
            f"({positioning['funding_state']})"
        )
    else:
        st.write("**Funding:** unavailable")

# =========================================================
# DECISION / RISK
# =========================================================

left, right = st.columns([3, 1])

with right:
    st.markdown("### 🎯 Execution")

    latest_price = float(spot_df["close"].iloc[-1])

    entry = st.number_input(
        "سعر الدخول",
        value=latest_price
    )

    is_long = signal == "CONFIRMED LONG"

    if signal == "CONFIRMED SHORT":
        is_long = False

    if is_long:
        default_reference = float(
            min(
                spot_df["vwap_session"].iloc[-1],
                spot_df["vwap_weekly"].iloc[-1],
                spot_df["vwap_anchored"].iloc[-1]
            )
        )
    else:
        default_reference = float(
            max(
                spot_df["vwap_session"].iloc[-1],
                spot_df["vwap_weekly"].iloc[-1],
                spot_df["vwap_anchored"].iloc[-1]
            )
        )

    reference = st.number_input(
        "Structural / VWAP reference",
        value=default_reference
    )

    risk = SignalAndRiskEngine.calculate_risk(
        entry,
        reference,
        float(spot_df["atr"].iloc[-1]),
        is_long,
        capital,
        base_risk_pct,
        signal
    )

    st.markdown(
        f"**Decision:** `{signal}`"
    )

    if risk:
        st.markdown(
            f""" - **Type:** {'🟢 LONG' if is_long else '🔴 SHORT'} - **SL:** `${risk['sl']:.4f}` - **TP1:** `${risk['tp1']:.4f}` - **TP2:** `${risk['tp2']:.4f}` - **SL Distance:** `{risk['sl_atr']:.2f} ATR` - **Risk:** `${risk['risk_amount']:.2f}` - **Risk %:** `{risk['risk_pct']:.2f}%` - **Units:** `{risk['units']:.6f}` - **Position Value:** `${risk['position_value']:.2f}` """
        )

        for warning in risk["warnings"]:
            st.warning(warning)

    else:
        st.info(
            "لا توجد إدارة مخاطرة قابلة للتنفيذ "
            "لأن القرار الحالي ليس صفقة مؤكدة."
        )

    st.markdown("#### 🧾 Decision Reasons")

    for reason in reasons:
        st.write(f"• {reason}")

    if data_quality["reasons"]:
        st.markdown("#### ⚠️ Data Quality")

        for reason in data_quality["reasons"]:
            st.write(f"• {reason}")

with left:

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20]
    )

    fig.add_trace(
        go.Candlestick(
            x=spot_df["timestamp"],
            open=spot_df["open"],
            high=spot_df["high"],
            low=spot_df["low"],
            close=spot_df["close"],
            name="Price"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=spot_df["timestamp"],
            y=spot_df["vwap_session"],
            mode="lines",
            name="Session VWAP",
            line=dict(color="gold", width=1.5)
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=spot_df["timestamp"],
            y=spot_df["vwap_weekly"],
            mode="lines",
            name="Weekly VWAP",
            line=dict(color="magenta", width=1.5, dash="dot")
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=spot_df["timestamp"],
            y=spot_df["vwap_anchored"],
            mode="lines",
            name=f"Anchored VWAP ({anchor_info['type']})",
            line=dict(color="cyan", width=2, dash="dash")
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=spot_df["timestamp"],
            y=spot_df["cvd"],
            mode="lines",
            name=f"CVD ({cvd_quality})",
            line=dict(color="deepskyblue", width=2)
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=spot_df["timestamp"],
            y=spot_df["delta"],
            name="Delta",
            opacity=0.45
        ),
        row=2,
        col=1
    )

    if futures_df is not None and not futures_df.empty:
        fig.add_trace(
            go.Scatter(
                x=futures_df["timestamp"],
                y=futures_df["openInterest"],
                mode="lines",
                name="Open Interest",
                line=dict(color="orange", width=2)
            ),
            row=3,
            col=1
        )

    fig.update_layout(
        title=(
            f"{selected_symbol} | {selected_tf} | "
            f"{signal} | Setup: {setup['name']}"
        ),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=760,
        margin=dict(l=10, r=10, t=45, b=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# VWAP DETAILS
# =========================================================

with st.expander(
    "📐 VWAP Analysis",
    expanded=False
):
    v1, v2, v3 = st.columns(3)

    v1.metric(
        "Session Distance",
        f"{vwap_info['session_distance_atr']:.2f} ATR"
    )

    v2.metric(
        "Weekly Distance",
        f"{vwap_info['weekly_distance_atr']:.2f} ATR"
    )

    v3.metric(
        "Anchored Distance",
        f"{vwap_info['anchored_distance_atr']:.2f} ATR"
    )

    st.write(
        f"**Reclaim:** {vwap_info['reclaim']} | "
        f"**Rejection:** {vwap_info['rejection']} | "
        f"**Overextended:** {vwap_info['overextended']}"
    )

# =========================================================
# SETUP DETAILS
# =========================================================

with st.expander(
    "🧩 Setup Detection Details",
    expanded=False
):
    st.write(f"**Setup:** {setup['name']}")
    st.write(f"**Direction:** {setup['direction']}")
    st.write(f"**Setup Score:** {setup['score']:.1f}")
    st.write(f"**Status:** {setup['status']}")
    st.write(f"**Reason:** {setup['reason']}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "⚡ AliQuantFund Quantitative Market Analysis Engine v4.0 | "
    "Market State + Setup Detection Architecture"
)
