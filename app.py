# -*- coding: utf-8 -*-
"""
⚡ AliQuantFund Institutional Architecture v4.0
=================================================
Quantitative Market Analysis Engine
"""

import sys
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 0. CONFIGURATION & LOGGING SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AliQuantFund - Control Center v4.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AliQuantFund")

# -----------------------------------------------------------------------------
# 1. ENUMS & DATA STRUCTURES
# -----------------------------------------------------------------------------
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

@dataclass
class QuantitativeMetrics:
    vwap_session: float = 0.0
    vwap_weekly: float = 0.0
    vwap_monthly: float = 0.0
    vwap_anchored: Optional[float] = None
    atr_14: float = 0.0
    cvd_slope: float = 0.0
    cvd_divergence: str = "NONE"  # BULLISH, BEARISH, NONE
    oi_change_pct: float = 0.0
    oi_interpretation: str = "NEUTRAL"
    funding_rate: Optional[float] = None
    funding_bias: str = "NEUTRAL"

@dataclass
class MultiTimeframeFrame:
    tf: str
    bias: str  # BULLISH, BEARISH, NEUTRAL
    vwap_relation: str
    structure: str

@dataclass
class ScoringBreakdown:
    direction_score: float = 0.0
    flow_score: float = 0.0
    positioning_score: float = 0.0
    location_score: float = 0.0
    total_score: float = 0.0
    data_quality_pct: float = 100.0

# -----------------------------------------------------------------------------
# 2. DATA ACQUISITION & FALLBACK LAYER
# -----------------------------------------------------------------------------
class MarketDataLoader:
    
    BYBIT_TF_MAP = {
        '5m': '5',
        '15m': '15',
        '1h': '60',
        '4h': '240',
        '1d': 'D'
    }

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_klines(symbol: str, interval: str, limit: int = 500) -> Tuple[Optional[pd.DataFrame], str]:
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}

        # Primary: Binance Spot
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}"
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                data = res.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
                ])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], DataStatus.LIVE.value
        except Exception as e:
            logger.warning(f"Binance fetch failed: {e}")

        # Fallback: Bybit Spot
        try:
            bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, '5')
            url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={formatted_symbol}&interval={bybit_tf}&limit={limit}"
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                data = res.json().get('result', {}).get('list', [])
                if data:
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms', utc=True).dt.tz_localize(None)
                    df = df.iloc[::-1].reset_index(drop=True)
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], DataStatus.FALLBACK.value
        except Exception as e:
            logger.warning(f"Bybit fetch failed: {e}")

        return None, DataStatus.UNAVAILABLE.value

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_futures_metrics(symbol: str, interval: str, limit: int = 50) -> Tuple[Optional[pd.DataFrame], str, Dict[str, Any]]:
        """
        محرك المشتقات المحدث: يحاول جلب البيانات من Binance Futures أولاً (لأنها أكثر استقراراً على Streamlit) 
        ثم ينقل إلى Bybit في حال التعذر.
        """
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        funding_meta = {'available': False, 'current': None, 'history': []}

        # --- Attempt 1: Binance Futures (Primary & Most Stable on Streamlit Cloud) ---
        try:
            binance_oi_interval = interval if interval in ('5m', '15m', '1h', '4h', '1d') else '5m'
            url_oi = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={formatted_symbol}&period={binance_oi_interval}&limit={limit}"
            url_fr = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={formatted_symbol}&limit=30"

            res_oi = requests.get(url_oi, headers=headers, timeout=4)
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
                            hist = [float(row.get('fundingRate', 0.0)) for row in reversed(fr_data) if 'fundingRate' in row]
                            if hist:
                                funding_meta['available'] = True
                                funding_meta['current'] = hist[0]
                                funding_meta['history'] = hist

                    return df_oi, DataStatus.LIVE.value, funding_meta
        except Exception as e:
            logger.warning(f"Binance Futures fetch failed: {e}")

        # --- Attempt 2: Bybit Linear (Fallback) ---
        try:
            bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, '5')
            interval_time = f"{bybit_tf}" if interval != '1d' else '1d'
            url_oi = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={formatted_symbol}&intervalTime={interval_time}&limit={limit}"
            res_oi = requests.get(url_oi, headers=headers, timeout=4)

            if res_oi.status_code == 200:
                oi_data = res_oi.json().get('result', {}).get('list', [])
                if oi_data:
                    df_oi = pd.DataFrame(oi_data)
                    df_oi['openInterest'] = pd.to_numeric(df_oi['openInterest'], errors='coerce')
                    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'].astype(float), unit='ms', utc=True).dt.tz_localize(None)
                    df_oi = df_oi.dropna(subset=['openInterest']).iloc[::-1].reset_index(drop=True)
                    return df_oi, DataStatus.FALLBACK.value, funding_meta
        except Exception as e:
            logger.warning(f"Bybit OI fetch failed: {e}")

        return None, DataStatus.UNAVAILABLE.value, funding_meta

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_recent_trades(symbol: str, limit: int = 1000) -> Tuple[Optional[pd.DataFrame], str]:
        """جلب صفقات التداول مع دعم سيرفرات بديلة متعددة لمنع الحظر"""
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}

        endpoints = [
            f"https://api.binance.com/api/v3/trades?symbol={formatted_symbol}&limit={limit}",
            f"https://api1.binance.com/api/v3/trades?symbol={formatted_symbol}&limit={limit}",
            f"https://api3.binance.com/api/v3/trades?symbol={formatted_symbol}&limit={limit}",
            f"https://data-api.binance.vision/api/v3/trades?symbol={formatted_symbol}&limit={limit}"
        ]

        for url in endpoints:
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
                            return df_trades, DataStatus.LIVE.value
            except Exception:
                continue

        return None, DataStatus.UNAVAILABLE.value

# -----------------------------------------------------------------------------
# 3. QUANTITATIVE ANALYSIS ENGINES
# -----------------------------------------------------------------------------
class QuantitativeEngine:

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        df = df.copy()
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def calculate_vwap(df: pd.DataFrame, anchor_type: str = 'SESSION') -> pd.Series:
        df = df.copy()
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        pv = typical_price * df['volume']

        if anchor_type == 'SESSION':
            group = df['timestamp'].dt.date
        elif anchor_type == 'WEEKLY':
            group = df['timestamp'].dt.to_period('W')
        elif anchor_type == 'MONTHLY':
            group = df['timestamp'].dt.to_period('M')
        else:
            group = pd.Series(0, index=df.index)

        cum_pv = pv.groupby(group).cumsum()
        cum_vol = df['volume'].groupby(group).cumsum()
        return cum_pv / np.where(cum_vol == 0, 1e-9, cum_vol)

    @staticmethod
    def detect_smart_anchor(df: pd.DataFrame) -> Tuple[int, str]:
        """يكتشف تلقائياً أحدث شمعة حدث كبرى (Volatile Breakout or Max Volume Spike)"""
        if len(df) < 50:
            return 0, "Initial"
        
        df_sub = df.tail(100).copy()
        df_sub['vol_z'] = (df_sub['volume'] - df_sub['volume'].mean()) / (df_sub['volume'].std() + 1e-9)
        max_vol_idx = df_sub['vol_z'].idxmax()
        
        return int(max_vol_idx), f"Volume Spike (Z={df_sub.loc[max_vol_idx, 'vol_z']:.1f})"

    @staticmethod
    def calculate_anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
        df = df.copy()
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        pv = typical_price * df['volume']
        
        # Zero out before anchor
        pv_anchored = pv.copy()
        vol_anchored = df['volume'].copy()
        
        pv_anchored.iloc[:anchor_idx] = 0
        vol_anchored.iloc[:anchor_idx] = 0
        
        cum_pv = pv_anchored.cumsum()
        cum_vol = vol_anchored.cumsum()
        
        avwap = cum_pv / np.where(cum_vol == 0, 1e-9, cum_vol)
        avwap.iloc[:anchor_idx] = np.nan
        return avwap

    @staticmethod
    def compute_cvd_metrics(df_klines: pd.DataFrame, df_trades: Optional[pd.DataFrame]) -> Tuple[pd.Series, str, float]:
        df = df_klines.copy()
        
        if df_trades is not None and not df_trades.empty:
            # Real Trade-Tape CVD Calculation
            df_trades['signed_vol'] = np.where(df_trades['is_buy'], df_trades['qty'], -df_trades['qty'])
            # Resample to klines timeline
            trades_resampled = df_trades.set_index('time').resample('5min')['signed_vol'].sum().reindex(df['timestamp'], fill_value=0)
            cvd = trades_resampled.cumsum().reset_index(drop=True)
            cvd_type = DataStatus.LIVE.value
        else:
            # Approximated CVD (Delta Approximation using candle body/range ratio)
            candle_range = (df['high'] - df['low']).replace(0, 1e-9)
            delta_approx = df['volume'] * ((df['close'] - df['open']) / candle_range)
            cvd = delta_approx.cumsum()
            cvd_type = DataStatus.APPROXIMATED.value

        # Calculate CVD Slope & Divergence
        price_change = df['close'].iloc[-1] - df['close'].iloc[-10]
        cvd_change = cvd.iloc[-1] - cvd.iloc[-10]
        cvd_slope = (cvd.iloc[-1] - cvd.iloc[-5]) / (abs(cvd.iloc[-5]) + 1e-9)

        divergence = "NONE"
        if price_change < 0 and cvd_change > 0:
            divergence = "BULLISH_ABSORPTION"
        elif price_change > 0 and cvd_change < 0:
            divergence = "BEARISH_ABSORPTION"

        return cvd, cvd_type, cvd_slope

# -----------------------------------------------------------------------------
# 4. DECISION ENGINE & FACTOR SCORING
# -----------------------------------------------------------------------------
class MarketStateEngine:
    @staticmethod
    def classify_market_state(df: pd.DataFrame, atr: float) -> MarketState:
        close = df['close'].iloc[-1]
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        
        recent_range = df['high'].tail(10).max() - df['low'].tail(10).min()
        
        if recent_range > 3.5 * atr:
            return MarketState.VOLATILE_EXPANSION
        elif recent_range < 1.5 * atr:
            return MarketState.RANGE_COMPRESSION
        elif close > ema_20 > ema_50:
            return MarketState.TRENDING_BULL
        elif close < ema_20 < ema_50:
            return MarketState.TRENDING_BEAR
        return MarketState.RANGE_COMPRESSION

class FactorScoringEngine:
    @staticmethod
    def compute_layered_score(
        market_state: MarketState,
        df: pd.DataFrame,
        metrics: QuantitativeMetrics,
        data_status_futures: str,
        data_status_cvd: str
    ) -> ScoringBreakdown:
        
        close = df['close'].iloc[-1]
        atr = metrics.atr_14 if metrics.atr_14 > 0 else 1.0
        
        # 1. Direction Factor (25%)
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        dir_score = 0.0
        if close > ema_20 > ema_50:
            dir_score = 25.0
        elif close < ema_20 < ema_50:
            dir_score = -25.0
        elif close > ema_20:
            dir_score = 12.5
        else:
            dir_score = -12.5

        # 2. Flow Factor (CVD & Volume) (25%)
        flow_score = 0.0
        if metrics.cvd_divergence == "BULLISH_ABSORPTION":
            flow_score = 25.0
        elif metrics.cvd_divergence == "BEARISH_ABSORPTION":
            flow_score = -25.0
        else:
            flow_score = np.clip(metrics.cvd_slope * 50.0, -20.0, 20.0)

        # 3. Positioning Factor (OI & Funding) (25%)
        pos_score = 0.0
        if data_status_futures != DataStatus.UNAVAILABLE.value:
            if metrics.oi_change_pct > 2.0 and dir_score > 0:
                pos_score += 15.0  # Long Accumulation
            elif metrics.oi_change_pct > 2.0 and dir_score < 0:
                pos_score -= 15.0  # Short Accumulation
            
            if metrics.funding_rate is not None:
                if metrics.funding_rate < -0.0001:
                    pos_score += 10.0  # Short Squeeze Potential
                elif metrics.funding_rate > 0.0003:
                    pos_score -= 10.0  # Long Overheated

        # 4. Location Factor (3-VWAP Distance / ATR) (25%)
        loc_score = 0.0
        dist_session_atr = (close - metrics.vwap_session) / atr
        
        if abs(dist_session_atr) <= 0.5:
            loc_score = 25.0 if dir_score >= 0 else -25.0  # At Value Reclaim/Support
        elif dist_session_atr > 2.0:
            loc_score = -15.0  # Overextended Bull
        elif dist_session_atr < -2.0:
            loc_score = 15.0  # Overextended Bear
        else:
            loc_score = 10.0 if dist_session_atr > 0 else -10.0

        # Adjust Weights Based on Market State
        if market_state == MarketState.RANGE_COMPRESSION:
            loc_weight, flow_weight, dir_weight, pos_weight = 0.40, 0.30, 0.15, 0.15
        elif market_state in (MarketState.TRENDING_BULL, MarketState.TRENDING_BEAR):
            loc_weight, flow_weight, dir_weight, pos_weight = 0.20, 0.25, 0.35, 0.20
        else:
            loc_weight, flow_weight, dir_weight, pos_weight = 0.25, 0.25, 0.25, 0.25

        total = (dir_score * dir_weight) + (flow_score * flow_weight) + (pos_score * pos_weight) + (loc_score * loc_weight)
        
        # Calculate Data Quality Percentage
        quality = 100.0
        if data_status_futures == DataStatus.UNAVAILABLE.value:
            quality -= 30.0
        elif data_status_futures == DataStatus.FALLBACK.value:
            quality -= 10.0

        if data_status_cvd == DataStatus.APPROXIMATED.value:
            quality -= 18.0
        elif data_status_cvd == DataStatus.UNAVAILABLE.value:
            quality -= 30.0

        return ScoringBreakdown(
            direction_score=round(dir_score, 1),
            flow_score=round(flow_score, 1),
            positioning_score=round(pos_score, 1),
            location_score=round(loc_score, 1),
            total_score=round(total, 1),
            data_quality_pct=max(0.0, quality)
        )

# -----------------------------------------------------------------------------
# 5. STREAMLIT UI & DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
def render_css():
    st.markdown("""
        <style>
        /* Fix Vertical Text Collapsing */
        .stSidebar, div[data-testid="stSidebar"], div[data-testid="stSidebar"] * {
            word-break: normal !important;
            word-wrap: normal !important;
            white-space: normal !important;
        }
        .main-header {
            font-size: 26px;
            font-weight: 800;
            color: #FAFAFA;
            margin-bottom: 0px;
        }
        .metric-card {
            background-color: #1E222D;
            border: 1px solid #2A2E39;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
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
        </style>
    """, unsafe_allow_html=True)

def main():
    render_css()

    # Sidebar Navigation & Inputs
    with st.sidebar:
        st.title("⚡ AliQuantFund")
        st.caption("Institutional Control Center v4.0")
        st.markdown("---")

        symbol = st.selectbox("المطالبة المالية (Symbol)", ["BTC/USDT", "ETH/USDT", "SOL/USDT"], index=0)
        timeframe = st.selectbox("الإطار الزمني التكتيكي (Setup TF)", ["5m", "15m", "1h", "4h"], index=1)
        
        st.markdown("---")
        auto_refresh = st.checkbox("تحديث تلقائي لحظي (Auto-Refresh)", value=True)
        st.markdown("---")
        st.info("💡 يتم استخدام Binance Futures وسيرفرات متعددة كخيار أول لتفادي انقطاع البيانات على Streamlit.")

    # Data Fetching Progress
    with st.spinner("جاري الاتصال بالمحركات الكمية وجلب البيانات..."):
        df_klines, spot_status = MarketDataLoader.fetch_klines(symbol, timeframe)
        df_futures, futures_status, funding_meta = MarketDataLoader.fetch_futures_metrics(symbol, timeframe)
        df_trades, cvd_status = MarketDataLoader.fetch_recent_trades(symbol)

    if df_klines is None or df_klines.empty:
        st.error("❌ تعذر جلب بيانات الشموع الأساسية من المصادر. يرجى التحقق من الاتصال.")
        return

    # Quantitative Calculations
    atr = QuantitativeEngine.calculate_atr(df_klines).iloc[-1]
    vwap_sess = QuantitativeEngine.calculate_vwap(df_klines, 'SESSION').iloc[-1]
    vwap_week = QuantitativeEngine.calculate_vwap(df_klines, 'WEEKLY').iloc[-1]
    vwap_month = QuantitativeEngine.calculate_vwap(df_klines, 'MONTHLY').iloc[-1]
    
    anchor_idx, anchor_reason = QuantitativeEngine.detect_smart_anchor(df_klines)
    avwap_series = QuantitativeEngine.calculate_anchored_vwap(df_klines, anchor_idx)
    
    cvd_series, cvd_calc_type, cvd_slope = QuantitativeEngine.compute_cvd_metrics(df_klines, df_trades)
    
    # Metrics Container
    oi_change_pct = 0.0
    if df_futures is not None and not df_futures.empty and len(df_futures) >= 2:
        oi_start = df_futures['openInterest'].iloc[0]
        oi_end = df_futures['openInterest'].iloc[-1]
        oi_change_pct = ((oi_end - oi_start) / (oi_start + 1e-9)) * 100.0

    metrics = QuantitativeMetrics(
        vwap_session=vwap_sess,
        vwap_weekly=vwap_week,
        vwap_monthly=vwap_month,
        vwap_anchored=avwap_series.iloc[-1] if not avwap_series.isna().all() else None,
        atr_14=atr,
        cvd_slope=cvd_slope,
        oi_change_pct=oi_change_pct,
        funding_rate=funding_meta.get('current')
    )

    market_state = MarketStateEngine.classify_market_state(df_klines, atr)
    scoring = FactorScoringEngine.compute_layered_score(
        market_state, df_klines, metrics, futures_status, cvd_calc_type
    )

    # Top Status Bar
    col_head1, col_head2, col_head3, col_head4 = st.columns(4)
    with col_head1:
        st.markdown(f"### {symbol}")
        st.caption(f"Price: **${df_klines['close'].iloc[-1]:,.2f}**")
    with col_head2:
        badge = "status-badge-green" if spot_status == DataStatus.LIVE.value else "status-badge-yellow"
        st.markdown(f"**Spot Data:** <span class='{badge}'>{spot_status}</span>", unsafe_allow_html=True)
    with col_head3:
        badge = "status-badge-green" if futures_status == DataStatus.LIVE.value else ("status-badge-yellow" if futures_status == DataStatus.FALLBACK.value else "status-badge-red")
        st.markdown(f"**Futures/OI:** <span class='{badge}'>{futures_status}</span>", unsafe_allow_html=True)
    with col_head4:
        badge = "status-badge-green" if cvd_calc_type == DataStatus.LIVE.value else "status-badge-yellow"
        st.markdown(f"**CVD Engine:** <span class='{badge}'>{cvd_calc_type}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Decision Banner
    dec_col1, dec_col2 = st.columns([1, 2])
    
    with dec_col1:
        st.subheader("🎯 بطاقة القرار التنفيذي")
        score = scoring.total_score
        if score >= 15.0 and scoring.data_quality_pct >= 60:
            final_dec = "CONFIRMED LONG"
            color = "#00E676"
        elif score <= -15.0 and scoring.data_quality_pct >= 60:
            final_dec = "CONFIRMED SHORT"
            color = "#FF5252"
        else:
            final_dec = "NO TRADE / WAIT"
            color = "#FFD600"

        st.markdown(f"""
            <div style="background-color: #1E222D; border-left: 5px solid {color}; padding: 15px; border-radius: 5px;">
                <h2 style="color: {color}; margin: 0;">{final_dec}</h2>
                <p style="margin-top: 5px; color: #B2B9C7;">
                    الثقة الكمية: <b>{abs(score):.1f}%</b> | جودة البيانات: <b>{scoring.data_quality_pct:.0f}%</b>
                </p>
                <small>الحالة الهيكلية: {market_state.value}</small>
            </div>
        """, unsafe_allow_html=True)

    with dec_col2:
        with st.expander("🧩 تفكيك العوامل الكمية (Layered Scoring - Anti-Double Counting)", expanded=True):
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Direction (الاتجاه)", f"{scoring.direction_score}")
            f2.metric("Flow (CVD/السيولة)", f"{scoring.flow_score}")
            f3.metric("Positioning (OI)", f"{scoring.positioning_score}")
            f4.metric("Location (VWAP)", f"{scoring.location_score}")

    st.markdown("---")

    # Interactive Chart Section
    st.subheader(f"📈 شارت {symbol} التكتيكي المدمج مع الـ VWAP والـ CVD")
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_klines['timestamp'],
        open=df_klines['open'],
        high=df_klines['high'],
        low=df_klines['low'],
        close=df_klines['close'],
        name="Price"
    ), row=1, col=1)

    # VWAP Overlay
    fig.add_trace(go.Scatter(x=df_klines['timestamp'], y=df_klines['close'].assign(v=vwap_sess)['v'], mode='lines', name='Session VWAP', line=dict(color='#FFD600', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_klines['timestamp'], y=df_klines['close'].assign(v=vwap_week)['v'], mode='lines', name='Weekly VWAP', line=dict(color='#00E676', width=1.5, dash='dash')), row=1, col=1)

    # CVD Plot
    fig.add_trace(go.Scatter(
        x=df_klines['timestamp'], 
        y=cvd_series, 
        mode='lines', 
        name='Cumulative Volume Delta (CVD)',
        line=dict(color='#29B6F6', width=1.5)
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
