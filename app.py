import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import traceback

# ==========================================
# LOGGING & SYSTEM CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AliQuantFundEngine")

st.set_page_config(
    page_title="AliQuantFund | Quantitative Market Analysis Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stMetric {
        background-color: #1a1e29;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #2b3245;
    }
    /* Only wrap long Arabic body text inside our own custom blocks — never
       apply break-word globally, since it shatters unbroken English words
       (like "AliQuantFund") into a vertical letter stack on narrow screens. */
    .aqf-rtl-text {
        overflow-wrap: break-word;
        white-space: normal;
    }
    div[data-testid="stSidebarNav"] {
        display: none !important;
    }
    .status-badge-live {
        background-color: #0e382c; color: #00e676; padding: 4px 10px; border-radius: 6px; font-weight: bold; border: 1px solid #00e676; display: inline-block;
    }
    .status-badge-fallback {
        background-color: #3d310d; color: #ffb300; padding: 4px 10px; border-radius: 6px; font-weight: bold; border: 1px solid #ffb300; display: inline-block;
    }
    .status-badge-bad {
        background-color: #3d0d0d; color: #ff5252; padding: 4px 10px; border-radius: 6px; font-weight: bold; border: 1px solid #ff5252; display: inline-block;
    }
    .decision-box {
        padding: 16px; border-radius: 12px; margin-bottom: 14px; border: 2px solid;
        color: #f0f2f6 !important;
    }
    .decision-box * { color: #f0f2f6 !important; }
    .decision-confirmed-long { background-color: #0e382c; border-color: #00e676; }
    .decision-confirmed-short { background-color: #3d0d0d; border-color: #ff5252; }
    .decision-setup { background-color: #1d2a3d; border-color: #4fa3ff; }
    .decision-wait { background-color: #3d310d; border-color: #ffb300; }
    .decision-no-trade { background-color: #22262f; border-color: #555c6e; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 0. SHARED CONSTANTS / STATUS ENUMS
# ==========================================

class DataStatus:
    LIVE = "LIVE"
    FALLBACK = "FALLBACK"
    UNAVAILABLE = "UNAVAILABLE"

class CVDQuality:
    REAL = "REAL TRADE-LEVEL"
    LIMITED = "LIMITED TRADE-LEVEL"
    APPROX = "APPROXIMATED"

TF_TO_MINUTES = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}


def safe_last(series: pd.Series, default=0.0):
    """Defensive accessor: returns last value of a series or a default if empty/NaN."""
    try:
        if series is None or len(series) == 0:
            return default
        val = series.iloc[-1]
        if pd.isna(val):
            return default
        return val
    except Exception:
        return default


# ==========================================
# 1. DATA STRUCTURES & DATA LAYER
# ==========================================

class MarketDataLoader:

    BYBIT_TF_MAP = {'5m': '5', '15m': '15', '1h': '60', '4h': '240', '1d': 'D'}

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_spot_ohlcv(symbol: str, interval: str, limit: int = 150) -> Tuple[Optional[pd.DataFrame], str, str]:
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}

        endpoints = [
            f"https://api1.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}",
            f"https://api3.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}",
            f"https://data-api.binance.vision/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}"
        ]

        for url in endpoints:
            try:
                res = requests.get(url, headers=headers, timeout=3)
                if res.status_code == 200:
                    data = res.json()
                    if not isinstance(data, list) or len(data) == 0:
                        continue
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
                    if len(df) < 5:
                        continue
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], DataStatus.LIVE, "Binance Spot Direct"
            except Exception as e:
                logger.info(f"Binance endpoint failed ({url}): {e}")
                continue

        try:
            bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, '5')
            url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={formatted_symbol}&interval={bybit_tf}&limit={limit}"
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                result = res.json().get('result', {}).get('list', [])
                if result:
                    df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    df = df.iloc[::-1].reset_index(drop=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms', utc=True).dt.tz_localize(None)
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
                    if len(df) >= 5:
                        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], DataStatus.FALLBACK, "Bybit Spot API"
        except Exception as e:
            logger.warning(f"Bybit spot fallback failed: {e}")

        logger.error(f"All spot data sources failed for {symbol} / {interval}")
        return None, DataStatus.UNAVAILABLE, "None"

    @staticmethod
    @st.cache_data(ttl=25)
    def fetch_futures_metrics(symbol: str, interval: str, limit: int = 50) -> Tuple[Optional[pd.DataFrame], str, Dict[str, Any]]:
        """
        Modular OI + Funding layer with two live sources, tried in order:
        1) Bybit linear futures (primary)  2) Binance Futures (fallback, e.g. when
        a symbol isn't listed/served on Bybit). Both normalize into the same
        schema so callers never need to know which source responded. Designed
        so further sources (e.g. OKX) can be appended as additional try-blocks.
        Returns: (oi_dataframe, status, funding_meta)
        """
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}
        bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, '5')
        funding_meta = {'available': False, 'current': None, 'history': []}

        # --- Source 1: Bybit linear (primary) ---
        try:
            interval_time = f"{bybit_tf}min" if interval != '1d' else '1d'
            url_oi = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={formatted_symbol}&intervalTime={interval_time}&limit={limit}"
            res_oi = requests.get(url_oi, headers=headers, timeout=4)

            url_fr = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={formatted_symbol}&limit=50"
            res_fr = requests.get(url_fr, headers=headers, timeout=4)

            if res_oi.status_code == 200:
                oi_data = res_oi.json().get('result', {}).get('list', [])
                if oi_data:
                    df_oi = pd.DataFrame(oi_data)
                    df_oi['openInterest'] = pd.to_numeric(df_oi['openInterest'], errors='coerce')
                    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'].astype(float), unit='ms', utc=True).dt.tz_localize(None)
                    df_oi = df_oi.dropna(subset=['openInterest']).iloc[::-1].reset_index(drop=True)

                    if res_fr.status_code == 200:
                        fr_list = res_fr.json().get('result', {}).get('list', [])
                        if fr_list:
                            hist = []
                            for row in fr_list:
                                try:
                                    hist.append(float(row.get('fundingRate', 0.0)))
                                except Exception:
                                    continue
                            if hist:
                                funding_meta['available'] = True
                                funding_meta['current'] = hist[0]
                                funding_meta['history'] = hist  # most-recent-first

                    if len(df_oi) >= 3:
                        return df_oi, DataStatus.LIVE, funding_meta
        except Exception as e:
            logger.warning(f"Bybit OI/Funding fetch failed: {e}")

        # --- Source 2: Binance Futures (fallback) ---
        try:
            binance_interval = interval if interval != '1d' else '1d'
            # Binance openInterestHist only supports 5m/15m/30m/1h/2h/4h/6h/12h/1d
            binance_oi_interval = interval if interval in ('5m', '15m', '1h', '4h', '1d') else '5m'
            url_oi = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={formatted_symbol}&period={binance_oi_interval}&limit={limit}"
            res_oi = requests.get(url_oi, headers=headers, timeout=4)

            url_fr = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={formatted_symbol}&limit=50"
            res_fr = requests.get(url_fr, headers=headers, timeout=4)

            if res_oi.status_code == 200:
                oi_data = res_oi.json()
                if isinstance(oi_data, list) and len(oi_data) >= 3:
                    df_oi = pd.DataFrame(oi_data)
                    df_oi['openInterest'] = pd.to_numeric(df_oi['sumOpenInterest'], errors='coerce')
                    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
                    df_oi = df_oi.dropna(subset=['openInterest']).reset_index(drop=True)

                    if res_fr.status_code == 200:
                        fr_data = res_fr.json()
                        if isinstance(fr_data, list) and fr_data:
                            hist = []
                            for row in reversed(fr_data):  # Binance returns oldest-first
                                try:
                                    hist.append(float(row.get('fundingRate', 0.0)))
                                except Exception:
                                    continue
                            if hist:
                                funding_meta['available'] = True
                                funding_meta['current'] = hist[0]
                                funding_meta['history'] = hist  # most-recent-first

                    if len(df_oi) >= 3:
                        return df_oi, DataStatus.FALLBACK, funding_meta
        except Exception as e:
            logger.warning(f"Binance Futures OI/Funding fetch failed: {e}")

        # --- Additional sources would be appended here as further try-blocks ---

        return None, DataStatus.UNAVAILABLE, funding_meta

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_recent_trades(symbol: str, limit: int = 1000) -> Tuple[Optional[pd.DataFrame], str]:
        """Raw trade-level tape used to build a genuine CVD."""
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}

        url = f"https://api1.binance.com/api/v3/trades?symbol={formatted_symbol}&limit={limit}"
        try:
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code == 200:
                trades = res.json()
                if isinstance(trades, list) and len(trades) > 0:
                    df_trades = pd.DataFrame(trades)
                    df_trades['price'] = pd.to_numeric(df_trades['price'], errors='coerce')
                    df_trades['qty'] = pd.to_numeric(df_trades['qty'], errors='coerce')
                    df_trades['time'] = pd.to_datetime(df_trades['time'], unit='ms', utc=True).dt.tz_localize(None)
                    df_trades['is_buy'] = ~df_trades['isBuyerMaker']
                    df_trades = df_trades.dropna(subset=['price', 'qty']).reset_index(drop=True)
                    if len(df_trades) > 0:
                        return df_trades, DataStatus.LIVE
        except Exception as e:
            logger.info(f"Trade tape fetch failed: {e}")

        return None, DataStatus.UNAVAILABLE


# ==========================================
# 2. DATA QUALITY ENGINE
# ==========================================

class DataQualityEngine:
    """
    Independent scoring of *how much to trust* the current analysis.
    Quality is not cosmetic — it feeds directly into Confidence in the
    Global Score layer, and every deduction is explained to the user.
    """

    @staticmethod
    def evaluate(spot_status: str, futures_status: str, funding_available: bool,
                 cvd_quality: str, candle_count: int) -> Dict[str, Any]:
        score = 100
        reasons = []

        if spot_status == DataStatus.FALLBACK:
            score -= 15
            reasons.append("مصدر بيانات السعر احتياطي (Bybit Fallback) بدل Binance المباشر.")
        elif spot_status == DataStatus.UNAVAILABLE:
            score -= 60
            reasons.append("تعذر جلب بيانات السعر من أي مصدر.")

        if futures_status == DataStatus.UNAVAILABLE:
            score -= 20
            reasons.append("بيانات Open Interest غير متاحة حالياً.")

        if not funding_available:
            score -= 8
            reasons.append("بيانات Funding Rate غير متاحة حالياً.")

        if cvd_quality == CVDQuality.APPROX:
            score -= 20
            reasons.append("CVD تقريبي بالكامل (بدون بيانات صفقات فعلية) — وزنه مخفّض.")
        elif cvd_quality == CVDQuality.LIMITED:
            score -= 10
            reasons.append("CVD حقيقي لعدد محدود فقط من الشموع الأخيرة.")

        if candle_count < 60:
            score -= 10
            reasons.append("عدد الشموع المتاحة للتحليل منخفض نسبياً.")

        score = int(max(0, min(100, score)))
        if not reasons:
            reasons.append("جميع مصادر البيانات تعمل بكامل جودتها.")

        return {'score': score, 'reasons': reasons}


# ==========================================
# 3. INDICATOR & QUANTITATIVE ENGINE LAYER
# ==========================================

class QuantitativeEngine:

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        res = tr.rolling(period, min_periods=1).mean()
        return res.fillna(df['close'] * 0.01)

    @staticmethod
    def calculate_adx_trend(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        if df is None or len(df) < 2:
            n = len(df) if df is not None else 0
            neutral = pd.Series([20.0] * n)
            return neutral, neutral, neutral

        df_copy = df.copy()
        up_move = df_copy['high'] - df_copy['high'].shift(1)
        down_move = df_copy['low'].shift(1) - df_copy['low']

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = QuantitativeEngine.calculate_atr(df_copy, period=1)
        atr = tr.rolling(period, min_periods=1).mean().replace(0, 1e-5)

        plus_di = 100 * (pd.Series(plus_dm, index=df_copy.index).rolling(period, min_periods=1).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=df_copy.index).rolling(period, min_periods=1).mean() / atr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-5)) * 100
        adx = dx.rolling(period, min_periods=1).mean()
        return adx.fillna(20.0), plus_di.fillna(20.0), minus_di.fillna(20.0)

    @staticmethod
    def detect_smart_anchor(df: pd.DataFrame, mode: str = "Automatic") -> Dict[str, Any]:
        """
        NOTE ON LOOK-AHEAD: this anchor is selected from the *full known history up
        to the current (last) candle* — it never references data beyond what is
        currently available, so it is safe for live use. It must NOT be reused
        inside a bar-by-bar backtest loop without re-computing it using only the
        data available at each simulated point in time.
        """
        if df is None or len(df) < 5:
            fallback_price = df['close'].iloc[0] if df is not None and len(df) > 0 else 0
            fallback_ts = df['timestamp'].iloc[0] if df is not None and len(df) > 0 else datetime.now()
            return {'index': 0, 'price': fallback_price, 'timestamp': fallback_ts, 'type': 'Default'}

        if mode == "Swing High":
            idx = df['high'].idxmax()
            return {'index': idx, 'price': df.loc[idx, 'high'], 'timestamp': df.loc[idx, 'timestamp'], 'type': 'Swing High'}
        elif mode == "Swing Low":
            idx = df['low'].idxmin()
            return {'index': idx, 'price': df.loc[idx, 'low'], 'timestamp': df.loc[idx, 'timestamp'], 'type': 'Swing Low'}

        vol_ma = df['volume'].rolling(20, min_periods=1).mean()
        vol_spike_mask = df['volume'] > (vol_ma * 2.2)

        if vol_spike_mask.any():
            idx = df.loc[vol_spike_mask, 'volume'].idxmax()
            anchor_type = "Major Volume Spike"
        else:
            idx = df['low'].idxmin()
            anchor_type = "Structural Pivot Low"

        return {
            'index': idx,
            'price': df.loc[idx, 'close'],
            'timestamp': df.loc[idx, 'timestamp'],
            'type': anchor_type
        }

    @staticmethod
    def calculate_indicators(df: pd.DataFrame, anchor_info: Dict[str, Any]) -> pd.DataFrame:
        """يحسب جميع المؤشرات والـ VWAPs. كل الحسابات تراكمية (expanding/rolling) فقط — لا future leakage."""
        if df is None or df.empty:
            return df

        df = df.copy()
        df['atr'] = QuantitativeEngine.calculate_atr(df, period=14)

        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['tp'] * df['volume']

        df['date'] = df['timestamp'].dt.date
        session_pv = df.groupby('date')['pv'].cumsum()
        session_vol = df.groupby('date')['volume'].cumsum()
        df['vwap_session'] = np.where(session_vol > 0, session_pv / session_vol, df['tp'])

        df['week_year'] = df['timestamp'].dt.strftime('%Y-%U')
        weekly_pv = df.groupby('week_year')['pv'].cumsum()
        weekly_vol = df.groupby('week_year')['volume'].cumsum()
        df['vwap_weekly'] = np.where(weekly_vol > 0, weekly_pv / weekly_vol, df['tp'])

        anc_idx = anchor_info['index']
        df_anc = df.loc[anc_idx:].copy()
        anc_pv = (df_anc['tp'] * df_anc['volume']).cumsum()
        anc_vol = df_anc['volume'].cumsum()

        df['vwap_anchored'] = np.nan
        df.loc[anc_idx:, 'vwap_anchored'] = np.where(anc_vol > 0, anc_pv / anc_vol, df_anc['tp'])
        df['vwap_anchored'] = df['vwap_anchored'].ffill().bfill()

        df['volume_ma20'] = df['volume'].rolling(20, min_periods=1).mean()

        return df

    @staticmethod
    def build_orderflow_cvd(df: pd.DataFrame, trades_df: Optional[pd.DataFrame],
                             interval: str) -> Tuple[pd.DataFrame, str]:
        """
        Real trade-level CVD reconstruction:
        - Buckets the raw trade tape into the *same candle boundaries* as `df`.
        - Any candle whose bucket has real trade coverage gets Delta = BuyVol - SellVol.
        - Candles without trade coverage (older than the trade tape's reach) fall
          back to an approximated candle-direction delta, weighted down by 0.5x
          so it never carries the same weight as real flow.
        - Reports whether CVD is REAL / LIMITED / APPROXIMATED based on how much
          of the *recent* window is genuinely trade-based.
        """
        df = df.copy()
        n = len(df)
        if n == 0:
            df['delta'] = []
            df['cvd'] = []
            return df, CVDQuality.APPROX

        minutes = TF_TO_MINUTES.get(interval, 5)
        candle_span = timedelta(minutes=minutes)

        # default: approximated everywhere
        prev_close = df['close'].shift(1).bfill()
        direction = np.where(df['close'] >= prev_close, 1, -1)
        approx_delta = df['volume'] * direction * 0.5  # down-weighted vs real flow
        df['delta'] = approx_delta
        df['cvd_bar_quality'] = 'APPROX'

        real_bar_count = 0

        if trades_df is not None and not trades_df.empty:
            trades_df = trades_df.sort_values('time')
            tape_start = trades_df['time'].min()

            # bucket boundary = candle open time -> open time + interval
            bucket_start = df['timestamp']
            bucket_end = df['timestamp'] + candle_span

            for i in range(n):
                b_start = bucket_start.iloc[i]
                b_end = bucket_end.iloc[i]

                # only treat as "covered" if the trade tape actually reaches back this far
                if tape_start > b_start:
                    continue

                mask = (trades_df['time'] >= b_start) & (trades_df['time'] < b_end)
                bucket_trades = trades_df.loc[mask]
                if bucket_trades.empty:
                    continue

                buy_vol = bucket_trades.loc[bucket_trades['is_buy'], 'qty'].sum()
                sell_vol = bucket_trades.loc[~bucket_trades['is_buy'], 'qty'].sum()
                df.iat[i, df.columns.get_loc('delta')] = buy_vol - sell_vol
                df.iat[i, df.columns.get_loc('cvd_bar_quality')] = 'REAL'
                real_bar_count += 1

        df['cvd'] = df['delta'].cumsum()

        # Determine overall quality based on the most recent 20 bars (the ones that matter for signals)
        recent_window = min(20, n)
        recent_quality = df['cvd_bar_quality'].iloc[-recent_window:]
        real_ratio = (recent_quality == 'REAL').mean() if recent_window > 0 else 0.0

        if real_ratio >= 0.8:
            overall_quality = CVDQuality.REAL
        elif real_ratio >= 0.25:
            overall_quality = CVDQuality.LIMITED
        else:
            overall_quality = CVDQuality.APPROX

        return df, overall_quality

    @staticmethod
    def analyze_positioning(df: pd.DataFrame, df_futures: Optional[pd.DataFrame],
                             funding_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        OI + Funding combined into a single Positioning read.
        OI alone never determines direction — it is always read jointly with price change,
        and funding is used only as a *modifier* (crowded-long / crowded-short flag),
        not as an independent duplicate score.
        """
        result = {
            'oi_available': False,
            'oi_state': 'Neutral Positioning',
            'oi_change_pct': 0.0,
            'oi_momentum': 0.0,
            'funding_available': funding_meta.get('available', False),
            'funding_current': funding_meta.get('current'),
            'funding_percentile': None,
            'funding_extreme': False,
            'crowding_flag': None,   # 'CROWDED_LONGS' / 'CROWDED_SHORTS' / None
            'positioning_score': 50.0,
        }

        if df is None or df.empty or df_futures is None or len(df_futures) < 5:
            return result

        try:
            latest_oi = df_futures['openInterest'].iloc[-1]
            prev_oi = df_futures['openInterest'].iloc[-5]
            oi_chg = (latest_oi - prev_oi) / prev_oi if prev_oi else 0.0

            # momentum: slope of OI over the available window, normalized
            oi_series = df_futures['openInterest']
            oi_momentum = (oi_series.iloc[-1] - oi_series.iloc[0]) / oi_series.iloc[0] if oi_series.iloc[0] else 0.0

            price_chg = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 and df['close'].iloc[-5] else 0.0

            result['oi_available'] = True
            result['oi_change_pct'] = round(oi_chg * 100, 2)
            result['oi_momentum'] = round(oi_momentum * 100, 2)

            if oi_chg > 0.01 and price_chg > 0:
                state, score = "Long Build-up", 78.0
            elif oi_chg > 0.01 and price_chg < 0:
                state, score = "Short Build-up", 22.0
            elif oi_chg < -0.01 and price_chg > 0:
                state, score = "Short Covering", 58.0
            elif oi_chg < -0.01 and price_chg < 0:
                state, score = "Long Unwinding", 42.0
            else:
                state, score = "Neutral Positioning", 50.0

            result['oi_state'] = state
            result['positioning_score'] = score

            # --- Funding as modifier, not an independent score ---
            history = funding_meta.get('history', [])
            current = funding_meta.get('current')
            if current is not None and len(history) >= 5:
                arr = np.array(history)
                percentile = float((arr < current).mean() * 100)
                result['funding_percentile'] = round(percentile, 1)
                is_extreme_high = percentile >= 90
                is_extreme_low = percentile <= 10
                result['funding_extreme'] = bool(is_extreme_high or is_extreme_low)

                if is_extreme_high and price_chg > 0 and oi_chg > 0:
                    result['crowding_flag'] = 'CROWDED_LONGS'
                    result['positioning_score'] = max(0.0, result['positioning_score'] - 12.0)
                elif is_extreme_low and price_chg < 0 and oi_chg > 0:
                    result['crowding_flag'] = 'CROWDED_SHORTS'
                    result['positioning_score'] = min(100.0, result['positioning_score'] + 12.0)

        except Exception as e:
            logger.warning(f"Positioning analysis failed: {e}")

        return result

    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> str:
        """Legacy simple regime tag — retained for backward compatibility with older UI pieces.
        Prefer MarketStateEngine.evaluate() for the full multi-dimension state."""
        if df is None or len(df) < 15:
            return "RANGING"

        atr = QuantitativeEngine.calculate_atr(df, 14)
        atr_pct = (safe_last(atr, df['close'].iloc[-1] * 0.01) / df['close'].iloc[-1]) * 100
        adx, plus_di, minus_di = QuantitativeEngine.calculate_adx_trend(df, 14)

        latest_adx = safe_last(adx, 20.0)
        latest_pdi = safe_last(plus_di, 20.0)
        latest_mdi = safe_last(minus_di, 20.0)

        vol_ma = df['volume'].rolling(20, min_periods=1).mean()
        is_high_vol = atr_pct > 2.5 or (df['volume'].iloc[-1] > safe_last(vol_ma, df['volume'].iloc[-1]) * 2.0)

        if latest_adx > 25:
            return "TRENDING BULL" if latest_pdi > latest_mdi else "TRENDING BEAR"
        elif is_high_vol:
            return "HIGH VOLATILITY"
        elif atr_pct < 0.6:
            return "LOW VOLATILITY"
        else:
            return "RANGING"


# ==========================================
# 4. VWAP ANALYSIS ENGINE (NEW)
# ==========================================

class VWAPAnalysisEngine:
    """
    Goes beyond raw distance/ATR. Reads structure: above/below, slope,
    reclaim/rejection events, and whether volume + CVD actually confirm
    the location — so an "overextended" move doesn't get scored as strength.
    """

    LOOKBACK = 6

    @staticmethod
    def _analyze_single(df: pd.DataFrame, vwap_col: str) -> Dict[str, Any]:
        latest = df.iloc[-1]
        atr = max(safe_last(df['atr'], latest['close'] * 0.01), 1e-9)
        vwap_now = latest.get(vwap_col, latest['close'])
        price = latest['close']

        distance_atr = (price - vwap_now) / atr
        above = price >= vwap_now

        lb = min(VWAPAnalysisEngine.LOOKBACK, len(df) - 1)
        slope = 0.0
        if lb > 0 and vwap_col in df.columns:
            past_vwap = df[vwap_col].iloc[-1 - lb]
            if past_vwap:
                slope = (vwap_now - past_vwap) / past_vwap * 100

        # Reclaim: was below N bars ago, now closes above
        reclaim = False
        rejection = False
        if lb > 0:
            past_prices = df['close'].iloc[-1 - lb:-1]
            past_vwaps = df[vwap_col].iloc[-1 - lb:-1]
            was_below = (past_prices < past_vwaps).any()
            was_above = (past_prices > past_vwaps).any()
            if was_below and above:
                reclaim = True
            if was_above and not above:
                rejection = True

        vol_confirm = df['volume'].iloc[-1] > safe_last(df['volume_ma20'], df['volume'].iloc[-1]) * 1.2

        cvd_confirm = False
        if 'cvd' in df.columns and len(df) > lb:
            cvd_delta = df['cvd'].iloc[-1] - df['cvd'].iloc[-1 - lb]
            cvd_confirm = (cvd_delta > 0) if above else (cvd_delta < 0)

        overextended = abs(distance_atr) > 2.2 and not (vol_confirm and cvd_confirm)

        if overextended:
            structure = "OVEREXTENDED"
        elif above and (reclaim or slope > 0) and vol_confirm and cvd_confirm:
            structure = "BULLISH"
        elif (not above) and (rejection or slope < 0) and vol_confirm and cvd_confirm:
            structure = "BEARISH"
        elif above:
            structure = "BULLISH_WEAK"
        else:
            structure = "BEARISH_WEAK"

        return {
            'vwap': float(vwap_now), 'above': bool(above), 'distance_atr': round(float(distance_atr), 2),
            'slope_pct': round(float(slope), 3), 'reclaim': bool(reclaim), 'rejection': bool(rejection),
            'volume_confirmation': bool(vol_confirm), 'cvd_confirmation': bool(cvd_confirm),
            'overextended': bool(overextended), 'structure': structure
        }

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty or len(df) < 8 or 'atr' not in df.columns:
            return {
                'session': {}, 'weekly': {}, 'anchored': {},
                'composite_structure': 'NEUTRAL', 'location_score': 50.0
            }

        session = VWAPAnalysisEngine._analyze_single(df, 'vwap_session')
        weekly = VWAPAnalysisEngine._analyze_single(df, 'vwap_weekly')
        anchored = VWAPAnalysisEngine._analyze_single(df, 'vwap_anchored')

        structures = [session['structure'], weekly['structure'], anchored['structure']]
        bull_votes = sum(1 for s in structures if s == 'BULLISH')
        bear_votes = sum(1 for s in structures if s == 'BEARISH')
        overext_votes = sum(1 for s in structures if s == 'OVEREXTENDED')

        if overext_votes >= 2:
            composite = "OVEREXTENDED"
        elif bull_votes >= 2:
            composite = "BULLISH"
        elif bear_votes >= 2:
            composite = "BEARISH"
        else:
            composite = "MIXED"

        avg_distance = np.mean([session['distance_atr'], weekly['distance_atr'], anchored['distance_atr']])

        if composite == "OVEREXTENDED":
            location_score = 50.0 - np.clip(avg_distance, -3, 3) * 5  # pulled toward neutral, penalized for chasing
        elif composite == "BULLISH":
            location_score = 65.0 + min(abs(avg_distance) * 8, 25.0)
        elif composite == "BEARISH":
            location_score = 35.0 - min(abs(avg_distance) * 8, 25.0)
        else:
            location_score = 50.0 + np.clip(avg_distance, -2, 2) * 5

        location_score = float(np.clip(location_score, 0.0, 100.0))

        return {
            'session': session, 'weekly': weekly, 'anchored': anchored,
            'composite_structure': composite, 'location_score': round(location_score, 1)
        }


# ==========================================
# 5. MARKET STATE ENGINE (NEW)
# ==========================================

class MarketStateEngine:
    """
    Combines trend, volatility, momentum, positioning and VWAP structure into
    a single Market State label. No single indicator can decide the state alone.
    """

    STATES = ["TRENDING_BULL", "TRENDING_BEAR", "RANGING", "HIGH_VOLATILITY",
              "LOW_VOLATILITY", "BREAKOUT", "POSSIBLE_EXHAUSTION"]

    @staticmethod
    def evaluate(df: pd.DataFrame, vwap_analysis: Dict[str, Any],
                 positioning: Dict[str, Any], cvd_quality: str) -> Dict[str, Any]:

        flags = []
        if df is None or df.empty or len(df) < 15:
            return {'state': 'RANGING', 'flags': flags, 'components': {}}

        atr = QuantitativeEngine.calculate_atr(df, 14)
        atr_pct = (safe_last(atr, df['close'].iloc[-1] * 0.01) / df['close'].iloc[-1]) * 100
        adx, plus_di, minus_di = QuantitativeEngine.calculate_adx_trend(df, 14)
        latest_adx, latest_pdi, latest_mdi = safe_last(adx, 20), safe_last(plus_di, 20), safe_last(minus_di, 20)

        vol_ma = df['volume'].rolling(20, min_periods=1).mean()
        vol_ratio = df['volume'].iloc[-1] / max(safe_last(vol_ma, df['volume'].iloc[-1]), 1e-9)
        is_high_vol = atr_pct > 2.5 or vol_ratio > 2.0
        is_low_vol = atr_pct < 0.6 and vol_ratio < 1.1

        # Momentum: recent CVD slope + price ROC
        momentum = 0.0
        if 'cvd' in df.columns and len(df) > 6:
            cvd_roc = df['cvd'].iloc[-1] - df['cvd'].iloc[-6]
            price_roc = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
            momentum = np.sign(cvd_roc) * abs(price_roc) * 100

        # Breakout: close beyond prior 20-bar range with volume expansion
        breakout_up = breakout_down = False
        if len(df) >= 21:
            prior_high = df['high'].iloc[-21:-1].max()
            prior_low = df['low'].iloc[-21:-1].min()
            breakout_up = df['close'].iloc[-1] > prior_high and vol_ratio > 1.5
            breakout_down = df['close'].iloc[-1] < prior_low and vol_ratio > 1.5

        # Exhaustion: overextended VWAP structure + weak/opposing CVD + (OI or funding extreme)
        exhaustion = False
        if vwap_analysis.get('composite_structure') == 'OVEREXTENDED':
            oi_extreme = positioning.get('oi_state') in ('Long Build-up', 'Short Build-up') and abs(positioning.get('oi_change_pct', 0)) > 3
            funding_extreme = positioning.get('funding_extreme', False)
            if oi_extreme or funding_extreme:
                exhaustion = True
                flags.append("POSSIBLE_EXHAUSTION")

        components = {
            'adx': round(float(latest_adx), 1), 'plus_di': round(float(latest_pdi), 1),
            'minus_di': round(float(latest_mdi), 1), 'atr_pct': round(float(atr_pct), 2),
            'volume_ratio': round(float(vol_ratio), 2), 'momentum': round(float(momentum), 3),
            'vwap_structure': vwap_analysis.get('composite_structure', 'MIXED')
        }

        if breakout_up or breakout_down:
            state = "BREAKOUT"
            flags.append("BREAKOUT_UP" if breakout_up else "BREAKOUT_DOWN")
        elif exhaustion:
            state = "POSSIBLE_EXHAUSTION"
        elif latest_adx > 25:
            state = "TRENDING_BULL" if latest_pdi > latest_mdi else "TRENDING_BEAR"
        elif is_high_vol:
            state = "HIGH_VOLATILITY"
        elif is_low_vol:
            state = "LOW_VOLATILITY"
        else:
            state = "RANGING"

        if cvd_quality == CVDQuality.APPROX:
            flags.append("CVD_APPROXIMATED")

        return {'state': state, 'flags': flags, 'components': components}


# ==========================================
# 6. FACTOR SCORING ENGINE (LAYERED — NO DOUBLE COUNTING)
# ==========================================

class FactorScoringEngine:
    """
    Layers: Direction / Flow / Positioning / Location / Setup / Data Quality.
    Each layer captures an independent dimension of the market so the same
    move is never counted twice under different names.
    """

    @staticmethod
    def evaluate_factors(
        df: pd.DataFrame,
        positioning: Dict[str, Any],
        cvd_quality: str,
        market_state: Dict[str, Any],
        vwap_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:

        if df is None or df.empty or 'atr' not in df.columns:
            return {
                'direction_score': 50.0, 'flow_score': 50.0, 'positioning_score': 50.0,
                'location_score': 50.0, 'raw_score': 50.0
            }

        # --- Direction: HTF-agnostic structural direction (price structure + trend) ---
        prev_10_high = df['high'].iloc[-11:-1].max() if len(df) >= 11 else df['high'].max()
        prev_10_low = df['low'].iloc[-11:-1].min() if len(df) >= 11 else df['low'].min()
        latest_close = df['close'].iloc[-1]

        if latest_close > prev_10_high:
            structure_score = 85.0
        elif latest_close < prev_10_low:
            structure_score = 15.0
        else:
            rng = max(prev_10_high - prev_10_low, 1e-5)
            structure_score = 30.0 + ((latest_close - prev_10_low) / rng) * 40.0

        adx, pdi, mdi = QuantitativeEngine.calculate_adx_trend(df)
        trend_bias = 50.0 + min(safe_last(adx, 20), 45.0) if safe_last(pdi, 20) > safe_last(mdi, 20) \
            else 50.0 - min(safe_last(adx, 20), 45.0)

        direction_score = (structure_score * 0.5) + (trend_bias * 0.5)

        # --- Flow: CVD + Volume behavior ---
        flow_score = 50.0
        if len(df) >= 10 and 'cvd' in df.columns:
            cvd_delta = df['cvd'].iloc[-1] - df['cvd'].iloc[-10]
            price_delta = df['close'].iloc[-1] - df['close'].iloc[-10]

            if cvd_delta > 0 and price_delta > 0: flow_score = 80.0
            elif cvd_delta < 0 and price_delta < 0: flow_score = 20.0
            elif cvd_delta > 0 and price_delta < 0: flow_score = 65.0
            elif cvd_delta < 0 and price_delta > 0: flow_score = 35.0

        if cvd_quality == CVDQuality.APPROX:
            flow_score = 50.0 + (flow_score - 50.0) * 0.4
        elif cvd_quality == CVDQuality.LIMITED:
            flow_score = 50.0 + (flow_score - 50.0) * 0.7

        # --- Positioning: OI + Funding modifier (already computed) ---
        positioning_score = positioning.get('positioning_score', 50.0)

        # --- Location: VWAP structure (already computed, penalizes overextension) ---
        location_score = vwap_analysis.get('location_score', 50.0)

        # Regime-aware weights
        state = market_state.get('state', 'RANGING')
        if state in ("TRENDING_BULL", "TRENDING_BEAR"):
            weights = {'direction': 0.35, 'flow': 0.25, 'positioning': 0.15, 'location': 0.25}
        elif state == "HIGH_VOLATILITY":
            weights = {'direction': 0.15, 'flow': 0.40, 'positioning': 0.20, 'location': 0.25}
        elif state == "BREAKOUT":
            weights = {'direction': 0.30, 'flow': 0.35, 'positioning': 0.15, 'location': 0.20}
        elif state == "POSSIBLE_EXHAUSTION":
            weights = {'direction': 0.20, 'flow': 0.30, 'positioning': 0.25, 'location': 0.25}
        else:  # RANGING / LOW_VOLATILITY
            weights = {'direction': 0.20, 'flow': 0.25, 'positioning': 0.15, 'location': 0.40}

        raw_score = (
            direction_score * weights['direction'] +
            flow_score * weights['flow'] +
            positioning_score * weights['positioning'] +
            location_score * weights['location']
        )

        return {
            'direction_score': round(direction_score, 1),
            'flow_score': round(flow_score, 1),
            'positioning_score': round(positioning_score, 1),
            'location_score': round(location_score, 1),
            'raw_score': round(raw_score, 1),
            'weights': weights
        }


# ==========================================
# 7. SETUP DETECTION ENGINE (NEW)
# ==========================================

class SetupDetectionEngine:
    """
    Turns 'Score = 82 -> LONG' into an actual named setup with a trigger.
    A setup only ever contributes a *bounded* bonus to the score, and only
    if it doesn't just restate what Direction/Location already said.
    """

    @staticmethod
    def detect(df: pd.DataFrame, vwap_analysis: Dict[str, Any], market_state: Dict[str, Any],
               positioning: Dict[str, Any], cvd_quality: str) -> List[Dict[str, Any]]:

        setups = []
        if df is None or df.empty or len(df) < 21:
            return setups

        state = market_state.get('state', 'RANGING')
        session = vwap_analysis.get('session', {})
        anchored = vwap_analysis.get('anchored', {})
        vol_confirm = session.get('volume_confirmation', False)
        cvd_confirm = session.get('cvd_confirmation', False)
        weighted_cvd_ok = cvd_confirm and cvd_quality != CVDQuality.APPROX

        # --- Setup 1: VWAP Reclaim ---
        if session.get('reclaim') and vol_confirm and weighted_cvd_ok and not vwap_analysis.get('composite_structure') == 'OVEREXTENDED':
            setups.append({
                'type': 'VWAP_RECLAIM', 'direction': 'LONG', 'confidence': 70,
                'reasons': ["استعادة السعر لـ Session VWAP", "تأكيد حجم التداول", "تأكيد CVD"]
            })
        # --- Setup 2: VWAP Rejection ---
        if session.get('rejection') and vol_confirm:
            cvd_supports_short = False
            if 'cvd' in df.columns and len(df) > 6:
                cvd_supports_short = (df['cvd'].iloc[-1] - df['cvd'].iloc[-6]) < 0
            if cvd_supports_short and cvd_quality != CVDQuality.APPROX:
                setups.append({
                    'type': 'VWAP_REJECTION', 'direction': 'SHORT', 'confidence': 68,
                    'reasons': ["فشل السعر في اختراق Session VWAP", "تأكيد حجم التداول", "CVD يدعم الرفض"]
                })

        # --- Setup 3: Breakout ---
        if state == "BREAKOUT":
            direction = 'LONG' if 'BREAKOUT_UP' in market_state.get('flags', []) else 'SHORT'
            oi_confirms = False
            if positioning.get('oi_available'):
                oi_confirms = (positioning['oi_change_pct'] > 0.5) if direction == 'LONG' else (positioning['oi_change_pct'] < -0.5 or positioning['oi_state'] == 'Short Build-up')
            setups.append({
                'type': 'BREAKOUT', 'direction': direction, 'confidence': 75 if oi_confirms else 62,
                'reasons': ["كسر الهيكل السعري لآخر 20 شمعة", "توسع في الحجم"] + (["تأكيد Open Interest"] if oi_confirms else ["OI لم يؤكد الحركة بعد"])
            })

        # --- Setup 4: Mean Reversion (RANGING only) ---
        if state == "RANGING" and len(df) >= 20:
            atr = max(safe_last(df['atr'], df['close'].iloc[-1] * 0.01), 1e-9)
            range_high = df['high'].iloc[-20:].max()
            range_low = df['low'].iloc[-20:].min()
            dist_to_high = (range_high - df['close'].iloc[-1]) / atr
            dist_to_low = (df['close'].iloc[-1] - range_low) / atr
            if dist_to_low < 0.6:
                setups.append({'type': 'MEAN_REVERSION', 'direction': 'LONG', 'confidence': 55,
                                'reasons': ["السعر عند حافة النطاق السفلية", "نظام السوق RANGING"]})
            elif dist_to_high < 0.6:
                setups.append({'type': 'MEAN_REVERSION', 'direction': 'SHORT', 'confidence': 55,
                                'reasons': ["السعر عند حافة النطاق العلوية", "نظام السوق RANGING"]})

        # --- Setup 5: Exhaustion (warning, not a directional entry) ---
        if market_state.get('state') == 'POSSIBLE_EXHAUSTION' or 'POSSIBLE_EXHAUSTION' in market_state.get('flags', []):
            divergence = False
            if 'cvd' in df.columns and len(df) > 10:
                price_up = df['close'].iloc[-1] > df['close'].iloc[-10]
                cvd_up = df['cvd'].iloc[-1] > df['cvd'].iloc[-10]
                divergence = price_up != cvd_up
            setups.append({
                'type': 'EXHAUSTION', 'direction': 'WARNING', 'confidence': 50,
                'reasons': ["امتداد سعري كبير عن VWAP"] + (["تباعد بين السعر وCVD"] if divergence else []) +
                           (["Funding في منطقة متطرفة"] if positioning.get('funding_extreme') else [])
            })

        return setups

    @staticmethod
    def best_setup(setups: List[Dict[str, Any]], preferred_direction: Optional[str] = None) -> Optional[Dict[str, Any]]:
        actionable = [s for s in setups if s['direction'] in ('LONG', 'SHORT')]
        if not actionable:
            return None
        if preferred_direction:
            aligned = [s for s in actionable if s['direction'] == preferred_direction]
            if aligned:
                return max(aligned, key=lambda s: s['confidence'])
        return max(actionable, key=lambda s: s['confidence'])


# ==========================================
# 8. MULTI-TIMEFRAME ENGINE
# ==========================================

class MultiTimeframeEngine:
    """
    Separated roles instead of a flat average:
    Context (1D+4H) -> Direction (1H) -> Setup (15M) -> Trigger (5M).
    """

    @staticmethod
    def _analyze_tf(symbol: str, tf: str, anchor_mode: str) -> Dict[str, Any]:
        try:
            df_spot, spot_status, _ = MarketDataLoader.fetch_spot_ohlcv(symbol, tf, limit=120)
            if df_spot is None or df_spot.empty:
                return {'ok': False}

            anchor_info = QuantitativeEngine.detect_smart_anchor(df_spot, mode=anchor_mode)
            df_calc = QuantitativeEngine.calculate_indicators(df_spot, anchor_info)
            df_calc, cvd_quality = QuantitativeEngine.build_orderflow_cvd(df_calc, None, tf)
            vwap_analysis = VWAPAnalysisEngine.analyze(df_calc)
            positioning = {'positioning_score': 50.0, 'oi_available': False, 'oi_state': 'Neutral Positioning',
                            'oi_change_pct': 0.0, 'funding_extreme': False}
            market_state = MarketStateEngine.evaluate(df_calc, vwap_analysis, positioning, cvd_quality)
            factors = FactorScoringEngine.evaluate_factors(df_calc, positioning, cvd_quality, market_state, vwap_analysis)

            adx, pdi, mdi = QuantitativeEngine.calculate_adx_trend(df_calc)
            bias = "BULLISH" if safe_last(pdi, 20) > safe_last(mdi, 20) and safe_last(adx, 20) > 18 else \
                   "BEARISH" if safe_last(mdi, 20) > safe_last(pdi, 20) and safe_last(adx, 20) > 18 else "NEUTRAL"

            return {
                'ok': True, 'df': df_calc, 'score': factors['raw_score'], 'state': market_state['state'],
                'bias': bias, 'vwap_analysis': vwap_analysis, 'market_state': market_state,
                'cvd_quality': cvd_quality
            }
        except Exception as e:
            logger.error(f"MTF analysis failed for {tf}: {e}\n{traceback.format_exc()}")
            return {'ok': False}

    @staticmethod
    def evaluate_mft(symbol: str, anchor_mode: str) -> Dict[str, Any]:
        tf_results = {}
        for tf in ['1d', '4h', '1h', '15m', '5m']:
            tf_results[tf] = MultiTimeframeEngine._analyze_tf(symbol, tf, anchor_mode)

        scores = {tf: (r['score'] if r.get('ok') else 50.0) for tf, r in tf_results.items()}
        biases = {tf: (r['bias'] if r.get('ok') else 'NEUTRAL') for tf, r in tf_results.items()}

        # --- Context (1D + 4H) ---
        context_score = (scores['1d'] * 0.6) + (scores['4h'] * 0.4)
        if context_score >= 62 and biases['1d'] != 'BEARISH':
            context_bias = "BULLISH"
        elif context_score <= 38 and biases['1d'] != 'BULLISH':
            context_bias = "BEARISH"
        else:
            context_bias = "NEUTRAL"

        # --- Direction (1H) ---
        direction_bias = biases['1h']
        direction_aligned = (context_bias != 'NEUTRAL') and (direction_bias == context_bias)

        # --- Setup (15M) ---
        setup_tf_result = tf_results['15m']
        setup_candidates = []
        if setup_tf_result.get('ok'):
            setup_candidates = SetupDetectionEngine.detect(
                setup_tf_result['df'], setup_tf_result['vwap_analysis'], setup_tf_result['market_state'],
                {'positioning_score': 50.0, 'oi_available': False, 'oi_state': 'Neutral Positioning',
                 'oi_change_pct': 0.0, 'funding_extreme': False},
                setup_tf_result['cvd_quality']
            )
        preferred_dir = 'LONG' if context_bias == 'BULLISH' else ('SHORT' if context_bias == 'BEARISH' else None)
        setup_15m = SetupDetectionEngine.best_setup(setup_candidates, preferred_dir)

        # --- Trigger (5M) ---
        trigger_tf_result = tf_results['5m']
        trigger_confirmed = False
        trigger_desc = None
        if trigger_tf_result.get('ok') and setup_15m:
            exec_state = trigger_tf_result['market_state']['state']
            exec_flags = trigger_tf_result['market_state']['flags']
            exec_vwap = trigger_tf_result['vwap_analysis'].get('session', {})
            if setup_15m['direction'] == 'LONG':
                if 'BREAKOUT_UP' in exec_flags or exec_vwap.get('reclaim'):
                    trigger_confirmed = True
                    trigger_desc = "5M Breakout Confirmation" if 'BREAKOUT_UP' in exec_flags else "5M VWAP Reclaim"
            elif setup_15m['direction'] == 'SHORT':
                if 'BREAKOUT_DOWN' in exec_flags or exec_vwap.get('rejection'):
                    trigger_confirmed = True
                    trigger_desc = "5M Breakdown Confirmation" if 'BREAKOUT_DOWN' in exec_flags else "5M VWAP Rejection"

        return {
            'tf_scores': scores, 'tf_biases': biases, 'tf_results': tf_results,
            'context_score': round(context_score, 1), 'context_bias': context_bias,
            'direction_bias': direction_bias, 'direction_aligned': direction_aligned,
            'setup_15m': setup_15m, 'trigger_confirmed': trigger_confirmed, 'trigger_desc': trigger_desc,
            # legacy fields kept for any old UI reference
            'htf_bias': f"{context_bias} 🟢" if context_bias == 'BULLISH' else (f"{context_bias} 🔴" if context_bias == 'BEARISH' else f"{context_bias} 🟡"),
            'htf_score': context_score,
            'exec_score': scores['5m']
        }


# ==========================================
# 9. SIGNAL CLASSIFICATION & RISK ENGINE
# ==========================================

class SignalAndRiskEngine:

    @staticmethod
    def build_final_decision(raw_score: float, data_quality: int, mft_res: Dict[str, Any],
                              setup: Optional[Dict[str, Any]], market_state: str) -> Dict[str, Any]:
        """
        Confidence-Adjusted Score + a real decision funnel:
        Direction -> Setup -> Trigger -> Data Quality all have to line up
        for CONFIRMED. Missing data is never silently treated as neutral —
        it's explicitly flagged and it caps the decision.
        """
        reasons = []
        context_bias = mft_res['context_bias']
        direction_aligned = mft_res['direction_aligned']
        trigger_confirmed = mft_res['trigger_confirmed']

        confidence = data_quality
        if data_quality < 60:
            reasons.append(f"⚠️ جودة بيانات منخفضة ({data_quality}%) — تم تقييد أي قرار تأكيدي.")

        setup_score_bonus = 0.0
        if setup:
            # bonus is capped and only applied if setup direction agrees with context —
            # this is the anti-double-counting guard: if Direction already implies the
            # same thing, the bonus stays small; it never stacks freely with raw_score.
            aligns = (setup['direction'] == 'LONG' and context_bias == 'BULLISH') or \
                     (setup['direction'] == 'SHORT' and context_bias == 'BEARISH')
            setup_score_bonus = min(setup['confidence'] * 0.15, 10.0) if aligns else -5.0

        confidence_adjusted_score = float(np.clip(raw_score + setup_score_bonus, 0, 100))
        # Confidence reflects data quality primarily, scaled by how decisive the score is
        # (a score sitting at 50 = no conviction = lower confidence even with perfect data)
        conviction = abs(confidence_adjusted_score - 50) / 50.0
        confidence = int(np.clip(data_quality * (0.6 + 0.4 * conviction), 10, 100))

        has_setup = setup is not None and setup['direction'] in ('LONG', 'SHORT')
        setup_matches_context = has_setup and (
            (setup['direction'] == 'LONG' and context_bias == 'BULLISH') or
            (setup['direction'] == 'SHORT' and context_bias == 'BEARISH')
        )

        if data_quality < 45:
            decision = "NO TRADE"
            reasons.append("جودة البيانات منخفضة جداً لاتخاذ قرار موثوق.")
        elif market_state == 'POSSIBLE_EXHAUSTION' and not (setup_matches_context and trigger_confirmed):
            decision = "WAIT"
            reasons.append("النظام في حالة إنهاك محتمل (Possible Exhaustion) — يفضل الانتظار.")
        elif context_bias == 'NEUTRAL' and not has_setup:
            decision = "NO TRADE"
            reasons.append("لا يوجد انحياز واضح على الأطر الكبرى ولا Setup فعّال.")
        elif setup_matches_context and trigger_confirmed and direction_aligned:
            decision = f"CONFIRMED {setup['direction']}"
            reasons.append(f"Context ({context_bias}) + Direction 1H متوافق + Setup {setup['type']} + Trigger {mft_res.get('trigger_desc')}.")
        elif setup_matches_context and trigger_confirmed and not direction_aligned:
            decision = "SETUP FORMING"
            reasons.append("يوجد Setup وTrigger لكن إطار الـ 1H غير متوافق تماماً مع الأطر الكبرى.")
        elif setup_matches_context and not trigger_confirmed:
            decision = "SETUP FORMING"
            reasons.append(f"Setup {setup['type']} تشكّل على 15M لكن لم يتأكد بعد بـ Trigger على 5M.")
        elif has_setup and not setup_matches_context:
            decision = "WAIT"
            reasons.append("يوجد Setup لكنه يعاكس اتجاه الأطر الزمنية الكبرى — بانتظار توافق أوضح.")
        else:
            decision = "WAIT"
            reasons.append("لا يوجد Setup واضح حالياً رغم وجود انحياز في الاتجاه العام.")

        # Legacy A+/A/B grading, retained as supplementary info alongside the new funnel
        if confidence_adjusted_score >= 82 and context_bias == 'BULLISH':
            grade = "A+ LONG"
        elif confidence_adjusted_score >= 68 and context_bias != 'BEARISH':
            grade = "A LONG"
        elif confidence_adjusted_score >= 58:
            grade = "B LONG"
        elif confidence_adjusted_score <= 18 and context_bias == 'BEARISH':
            grade = "A+ SHORT"
        elif confidence_adjusted_score <= 32 and context_bias != 'BULLISH':
            grade = "A SHORT"
        elif confidence_adjusted_score <= 42:
            grade = "B SHORT"
        else:
            grade = "NEUTRAL"

        return {
            'decision': decision, 'grade': grade, 'confidence': confidence,
            'raw_score': round(raw_score, 1), 'confidence_adjusted_score': round(confidence_adjusted_score, 1),
            'reasons': reasons
        }

    @staticmethod
    def calculate_smart_risk_parameters(
        entry_price: float, structural_price: Optional[float], vwap_price: Optional[float],
        atr: float, is_long: bool, capital: float, base_risk_pct: float,
        decision: str, market_state: str
    ) -> Dict[str, Any]:
        """
        SL priority: 1) structural invalidation  2) VWAP invalidation  3) ATR buffer.
        Position size is only computed after the SL distance is sanity-checked against ATR.
        """
        warnings = []
        atr = max(atr, 1e-9)
        atr_buffer = atr * 0.5

        candidates = []
        if structural_price is not None:
            candidates.append(('Structural', structural_price))
        if vwap_price is not None:
            candidates.append(('VWAP', vwap_price))

        sl_source_label = "ATR Buffer"
        if is_long:
            # pick the tightest sensible invalidation below entry, else ATR fallback
            valid = [(lbl, p) for lbl, p in candidates if p < entry_price]
            if valid:
                sl_source_label, sl_source_price = max(valid, key=lambda x: x[1])  # closest below entry
            else:
                sl_source_price = entry_price - atr * 1.5
            calculated_sl = min(sl_source_price - atr_buffer, entry_price - (atr * 0.5))
            sl_distance = entry_price - calculated_sl
            tp1 = entry_price + (sl_distance * 1.5)
            tp2 = entry_price + (sl_distance * 3.0)
        else:
            valid = [(lbl, p) for lbl, p in candidates if p > entry_price]
            if valid:
                sl_source_label, sl_source_price = min(valid, key=lambda x: x[1])
            else:
                sl_source_price = entry_price + atr * 1.5
            calculated_sl = max(sl_source_price + atr_buffer, entry_price + (atr * 0.5))
            sl_distance = calculated_sl - entry_price
            tp1 = entry_price - (sl_distance * 1.5)
            tp2 = entry_price - (sl_distance * 3.0)

        sl_distance = max(sl_distance, 1e-9)
        sl_in_atr = sl_distance / atr
        if sl_in_atr < 0.3:
            warnings.append(f"⚠️ وقف الخسارة قريب جداً ({sl_in_atr:.2f}× ATR) — قد يُضرب بضوضاء السوق.")
        elif sl_in_atr > 4.0:
            warnings.append(f"⚠️ وقف الخسارة بعيد جداً ({sl_in_atr:.2f}× ATR) — سيقلّص حجم المركز بشدة.")

        grade_multiplier = 1.0 if "CONFIRMED" in decision else (0.6 if "SETUP FORMING" in decision else (0.25 if decision == "WAIT" else 0.0))

        effective_risk_pct = base_risk_pct * grade_multiplier
        risk_amount = capital * (effective_risk_pct / 100.0)

        units = (risk_amount / sl_distance) if sl_distance > 0 else 0.0
        position_value = units * entry_price

        return {
            'entry': entry_price, 'sl': calculated_sl, 'tp1': tp1, 'tp2': tp2,
            'sl_distance': sl_distance, 'sl_in_atr': round(sl_in_atr, 2), 'sl_source': sl_source_label,
            'risk_amount': risk_amount, 'effective_risk_pct': effective_risk_pct, 'units': units,
            'position_value': position_value, 'rr_tp1': 1.5, 'rr_tp2': 3.0, 'warnings': warnings
        }


# ==========================================
# 10. STREAMLIT USER INTERFACE LAYER
# ==========================================

st.sidebar.title("⚡ AliQuantFund")
st.sidebar.caption("Institutional Market Engine v4.0 — Setup Detection")
st.sidebar.markdown("---")

selected_symbol = st.sidebar.selectbox("الأصل المالي (Symbol):", ["BTC/USDT", "ETH/USDT", "ZEC/USDT", "XRP/USDT", "SOL/USDT"])
selected_tf = st.sidebar.selectbox("الإطار الزمني للتنفيذ (Execution TF):", ["5m", "15m", "1h", "4h", "1d"], index=0)
anchor_mode = st.sidebar.selectbox("نمط مرساة الـ VWAP:", ["Automatic", "Swing High", "Swing Low"])

st.sidebar.markdown("---")
st.sidebar.subheader("📐 إدارة رأس المال")
capital = st.sidebar.number_input("رأس المال ($):", value=100.0, step=10.0)
base_risk_pct = st.sidebar.number_input("أقصى نسبة مخاطرة (%):", value=2.0, step=0.5)

st.title(f"📊 التحليل الكمي المؤسسي: {selected_symbol}")

try:
    spot_df, spot_status, spot_source = MarketDataLoader.fetch_spot_ohlcv(selected_symbol, selected_tf)
    futures_df, futures_status, funding_meta = MarketDataLoader.fetch_futures_metrics(selected_symbol, selected_tf)
    trades_df, trades_status = MarketDataLoader.fetch_recent_trades(selected_symbol)
except Exception as e:
    logger.error(f"Top-level data fetch failure: {e}\n{traceback.format_exc()}")
    spot_df, spot_status, spot_source = None, DataStatus.UNAVAILABLE, "None"
    futures_df, futures_status, funding_meta = None, DataStatus.UNAVAILABLE, {'available': False, 'current': None, 'history': []}
    trades_df, trades_status = None, DataStatus.UNAVAILABLE

if spot_df is not None and not spot_df.empty:
    try:
        anchor_info = QuantitativeEngine.detect_smart_anchor(spot_df, mode=anchor_mode)
        spot_df = QuantitativeEngine.calculate_indicators(spot_df, anchor_info)
        spot_df, cvd_quality = QuantitativeEngine.build_orderflow_cvd(spot_df, trades_df, selected_tf)

        positioning = QuantitativeEngine.analyze_positioning(spot_df, futures_df, funding_meta)
        vwap_analysis = VWAPAnalysisEngine.analyze(spot_df)
        market_state = MarketStateEngine.evaluate(spot_df, vwap_analysis, positioning, cvd_quality)
        market_regime_legacy = QuantitativeEngine.detect_market_regime(spot_df)  # kept for compatibility

        data_quality_res = DataQualityEngine.evaluate(
            spot_status, futures_status, positioning.get('funding_available', False), cvd_quality, len(spot_df)
        )
        data_quality = data_quality_res['score']

        factors = FactorScoringEngine.evaluate_factors(spot_df, positioning, cvd_quality, market_state, vwap_analysis)
        mft_res = MultiTimeframeEngine.evaluate_mft(selected_symbol, anchor_mode)

        setup_candidates = SetupDetectionEngine.detect(spot_df, vwap_analysis, market_state, positioning, cvd_quality)
        preferred_dir = 'LONG' if mft_res['context_bias'] == 'BULLISH' else ('SHORT' if mft_res['context_bias'] == 'BEARISH' else None)
        active_setup = mft_res.get('setup_15m') or SetupDetectionEngine.best_setup(setup_candidates, preferred_dir)

        final = SignalAndRiskEngine.build_final_decision(
            factors['raw_score'], data_quality, mft_res, active_setup, market_state['state']
        )

        badge_class = "status-badge-live" if spot_status == DataStatus.LIVE else "status-badge-fallback"
        futures_badge = "status-badge-live" if futures_status == DataStatus.LIVE else "status-badge-bad"
        cvd_badge = "status-badge-live" if cvd_quality == CVDQuality.REAL else (
            "status-badge-fallback" if cvd_quality == CVDQuality.LIMITED else "status-badge-bad")

        st.markdown(f"""
        <div style="display: flex; gap: 10px; margin-bottom: 15px; align-items: center; flex-wrap: wrap;">
            <span class="{badge_class}">Spot: {spot_status} ({spot_source})</span>
            <span class="{futures_badge}">Futures/OI: {futures_status}</span>
            <span class="{cvd_badge}">CVD: {cvd_quality}</span>
            <span style="font-weight: bold; color: #888;">جودة البيانات: {data_quality}%</span>
        </div>
        """, unsafe_allow_html=True)

        # --- FINAL DECISION BANNER (5-second readability) ---
        decision_class_map = {
            "CONFIRMED LONG": "decision-confirmed-long", "CONFIRMED SHORT": "decision-confirmed-short",
            "SETUP FORMING": "decision-setup", "WAIT": "decision-wait", "NO TRADE": "decision-no-trade"
        }
        dcls = decision_class_map.get(final['decision'], "decision-no-trade")
        st.markdown(f"""
        <div class="decision-box {dcls}">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:14px;">
                <div><b>MARKET STATE:</b> {market_state['state']}</div>
                <div><b>HTF BIAS:</b> {mft_res['context_bias']}</div>
                <div><b>ACTIVE SETUP:</b> {active_setup['type'] if active_setup else '—'}</div>
                <div><b>TRIGGER:</b> {mft_res.get('trigger_desc') or ('—' if not mft_res['trigger_confirmed'] else 'Confirmed')}</div>
            </div>
            <hr style="opacity:0.2;">
            <div style="font-size:1.4em; font-weight:900;">FINAL DECISION: {final['decision']}</div>
            <div>CONFIDENCE: {final['confidence']}% &nbsp;|&nbsp; DATA QUALITY: {data_quality}% &nbsp;|&nbsp; Grade: {final['grade']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🌐 اللوحة التنفيذية الموحدة (Executive Dashboard)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("القرار النهائي", final['decision'], f"Confidence: {final['confidence']}%")
        col2.metric("Raw / Adjusted Score", f"{final['raw_score']} → {final['confidence_adjusted_score']}", f"Grade: {final['grade']}")
        col3.metric("نظام السوق (State)", market_state['state'], f"ATR: {safe_last(spot_df['atr']):.2f}")
        col4.metric("مرساة الـ VWAP الحالية", f"{anchor_info['type']}", f"Price: ${anchor_info['price']:.2f}")

        st.markdown("---")

        with st.expander("🧭 تفسير القرار (Decision Explanation)", expanded=True):
            for r in final['reasons']:
                st.write(f"- {r}")
            if data_quality_res['reasons']:
                st.caption("**أسباب جودة البيانات:**")
                for r in data_quality_res['reasons']:
                    st.caption(f"  • {r}")

        with st.expander("🔬 تفكيك العوامل الكمية (Layered Factors — Anti-Double Counting)", expanded=False):
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            f_col1.metric("Direction", f"{factors['direction_score']:.1f}")
            f_col2.metric("Flow (CVD/Volume)", f"{factors['flow_score']:.1f}")
            f_col3.metric("Positioning (OI/Funding)", f"{factors['positioning_score']:.1f}")
            f_col4.metric("Location (VWAP)", f"{factors['location_score']:.1f}")
            st.caption(f"• **حالة OI:** {positioning['oi_state']} | تغيّر OI: {positioning['oi_change_pct']}% | Momentum: {positioning['oi_momentum']}%")
            if positioning.get('funding_available'):
                st.caption(f"• **Funding الحالي:** {positioning['funding_current']:.5f} | Percentile: {positioning.get('funding_percentile')}% | Extreme: {positioning['funding_extreme']}")
                if positioning.get('crowding_flag'):
                    st.caption(f"• ⚠️ **{positioning['crowding_flag']}**")
            else:
                st.caption("• Funding Rate غير متاح حالياً.")
            st.caption(f"• **بنية VWAP المركبة:** {vwap_analysis.get('composite_structure')}")

        with st.expander("🗺️ سياق الأطر الزمنية (Context → Direction → Setup → Trigger)", expanded=False):
            tf_cols = st.columns(5)
            tf_labels = {'1d': 'Context 1D', '4h': 'Context 4H', '1h': 'Direction 1H', '15m': 'Setup 15M', '5m': 'Trigger 5M'}
            for i, tf in enumerate(['1d', '4h', '1h', '15m', '5m']):
                tf_cols[i].metric(tf_labels[tf], f"{mft_res['tf_scores'][tf]:.1f}", mft_res['tf_biases'][tf])
            st.caption(f"Context Bias: **{mft_res['context_bias']}** | Direction (1H) aligned: **{mft_res['direction_aligned']}**")
            if active_setup:
                st.caption(f"Setup مختار: **{active_setup['type']} ({active_setup['direction']})** — ثقة: {active_setup['confidence']}%")
                for r in active_setup['reasons']:
                    st.caption(f"  • {r}")
            else:
                st.caption("لا يوجد Setup فعّال حالياً على 15M.")
            st.caption(f"Trigger مؤكد على 5M: **{mft_res['trigger_confirmed']}**")

        c_chart, c_risk = st.columns([3, 1])
        latest_close = spot_df['close'].iloc[-1]
        is_long_trade = "LONG" in final['decision'] or (active_setup and active_setup['direction'] == 'LONG') or factors['raw_score'] >= 50

        with c_risk:
            st.markdown("### 🎯 إشعار التنفيذ وإدارة المخاطر")
            st.write(f"**القرار:** `{final['decision']}`  |  **Grade:** `{final['grade']}`")

            entry_p = st.number_input("سعر الدخول:", value=float(latest_close))

            structural_price = float(spot_df['low'].iloc[-10:].min() if is_long_trade else spot_df['high'].iloc[-10:].max())
            vwap_price = float(spot_df['vwap_anchored'].iloc[-1]) if 'vwap_anchored' in spot_df.columns else None

            risk_params = SignalAndRiskEngine.calculate_smart_risk_parameters(
                entry_p, structural_price, vwap_price, safe_last(spot_df['atr'], entry_p * 0.01),
                is_long_trade, capital, base_risk_pct, final['decision'], market_state['state']
            )

            st.markdown(f"• **نوع الصفقة:** `{'🟢 شراء (Long)' if is_long_trade else '🔴 بيع (Short)'}`")
            st.markdown(f"• **مصدر وقف الخسارة:** `{risk_params['sl_source']}`")
            st.markdown(f"• **وقف الخسارة (SL):** `${risk_params['sl']:.2f}` ({risk_params['sl_in_atr']}× ATR)")
            st.markdown(f"• **الهدف الأول (TP1 - 1:1.5):** `${risk_params['tp1']:.2f}`")
            st.markdown(f"• **الهدف الثاني (TP2 - 1:3.0):** `${risk_params['tp2']:.2f}`")

            for w in risk_params['warnings']:
                st.warning(w)

            st.markdown("---")
            st.caption(f"• المخاطرة الفعالة: `${risk_params['risk_amount']:.2f}` ({risk_params['effective_risk_pct']:.2f}%)")
            st.caption(f"• حجم العقود (Units): `{risk_params['units']:.4f}`")
            st.caption(f"• القيمة الإجمالية للعقد: `${risk_params['position_value']:.2f}`")

        with c_chart:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.60, 0.20, 0.20]
            )

            fig.add_trace(go.Candlestick(
                x=spot_df['timestamp'], open=spot_df['open'], high=spot_df['high'],
                low=spot_df['low'], close=spot_df['close'], name='Price'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=spot_df['timestamp'], y=spot_df['vwap_session'], mode='lines',
                name='Session VWAP', line=dict(color='gold', width=1.5)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=spot_df['timestamp'], y=spot_df['vwap_weekly'], mode='lines',
                name='Weekly VWAP', line=dict(color='magenta', width=1.5, dash='dot')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=spot_df['timestamp'], y=spot_df['vwap_anchored'], mode='lines',
                name=f"Anchored VWAP ({anchor_info['type']})", line=dict(color='cyan', width=2, dash='dash')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=spot_df['timestamp'], y=spot_df['cvd'], mode='lines',
                name=f'CVD Order Flow ({cvd_quality})', line=dict(color='deepskyblue', width=2), fill='tozeroy'
            ), row=2, col=1)

            if futures_df is not None and not futures_df.empty:
                fig.add_trace(go.Scatter(
                    x=futures_df['timestamp'], y=futures_df['openInterest'], mode='lines',
                    name='Futures Open Interest (Bybit)', line=dict(color='orange', width=2)
                ), row=3, col=1)

            fig.update_layout(
                title=f"شارت {selected_symbol} - {selected_tf} (Quantitative Engine v4.0)",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=700,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Analysis pipeline failure: {e}\n{traceback.format_exc()}")
        st.error("⚠️ حدث خطأ أثناء التحليل. تم تسجيل التفاصيل في السجل (logger). حاول تحديث الصفحة أو اختيار زوج/إطار زمني آخر.")

else:
    st.error("❌ تعذر جلب البيانات الحية من خوادم السوق. يرجى التحقق من الاتصال أو محاولة اختيار زوج آخر.")

st.markdown("---")
st.caption("⚡ AliQuantFund Institutional Architecture v4.0 — Market State + Setup Detection Engine | All Rights Reserved")
