# -*- coding: utf-8 -*-
"""
⚡ AliQuantFund Institutional Architecture v4.2 - Final Master Integration
================================================------------------------
Quantitative Market Analysis Engine
- Multi-TF Hierarchy (1D/4H -> 1H -> 15M -> 5M)
- CVD Engine (Real/Approx)
- Ichimoku Cloud + 3-VWAP Suite + Smart Anchor
- Futures OI & Funding Rate Integration
- Anti-Double Counting Layered Scoring
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Tuple, Any, List

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
    page_title="AliQuantFund - Master Engine v4.2",
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

class DataStatus(Enum):
    LIVE = "LIVE (Direct)"
    FALLBACK = "FALLBACK (Secondary API)"
    APPROXIMATED = "APPROXIMATED"
    UNAVAILABLE = "UNAVAILABLE"

@dataclass
class QuantitativeMetrics:
    vwap_session: float = 0.0
    vwap_weekly: float = 0.0
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

# -----------------------------------------------------------------------------
# 2. DATA ACQUISITION LAYER (Spot, Futures OI, Recent Trades)
# -----------------------------------------------------------------------------
class MarketDataLoader:
    
    BYBIT_TF_MAP = {'5m': '5', '15m': '15', '1h': '60', '4h': '240', '1d': 'D'}

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_klines(symbol: str, interval: str, limit: int = 300) -> Tuple[Optional[pd.DataFrame], str]:
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        binance_endpoints = [
            f"https://data-api.binance.vision/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}",
            f"https://api1.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}",
            f"https://api2.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}",
            f"https://api3.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}"
        ]

        for url in binance_endpoints:
            try:
                res = requests.get(url, headers=headers, timeout=3)
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
                        ])
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = df[col].astype(float)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
                        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], DataStatus.LIVE.value
            except Exception:
                continue

        # Fallback: Bybit Spot
        try:
            bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, '5')
            url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={formatted_symbol}&interval={bybit_tf}&limit={limit}"
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                result = res.json().get('result', {}).get('list', [])
                if result:
                    df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms', utc=True).dt.tz_localize(None)
                    df = df.iloc[::-1].reset_index(drop=True)
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], DataStatus.FALLBACK.value
        except Exception as e:
            logger.warning(f"Bybit kline fetch failed: {e}")

        return None, DataStatus.UNAVAILABLE.value

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_futures_metrics(symbol: str, interval: str, limit: int = 50) -> Tuple[Optional[pd.DataFrame], str, Dict[str, Any]]:
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}
        funding_meta = {'available': False, 'current': None, 'history': []}

        # Attempt 1: Binance Futures (Direct)
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

        # Attempt 2: Bybit Linear Futures
        try:
            bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, '5')
            url_oi = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={formatted_symbol}&intervalTime={bybit_tf}&limit={limit}"
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
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}

        endpoints = [
            f"https://data-api.binance.vision/api/v3/trades?symbol={formatted_symbol}&limit={limit}",
            f"https://api1.binance.com/api/v3/trades?symbol={formatted_symbol}&limit={limit}",
            f"https://api3.binance.com/api/v3/trades?symbol={formatted_symbol}&limit={limit}"
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
# 3. QUANTITATIVE & INDICATOR ENGINE (Ichimoku + 3-VWAP + CVD)
# -----------------------------------------------------------------------------
class QuantitativeEngine:

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        df = df.copy()
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean().bfill().fillna(df['close'] * 0.01)

    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2.0
        df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2.0
        df['span_a'] = ((df['tenkan'] + df['kijun']) / 2.0).shift(26)
        df['span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2.0).shift(26)
        return df

    @staticmethod
    def calculate_vwap(df: pd.DataFrame, anchor_type: str = 'SESSION') -> pd.Series:
        df = df.copy()
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        pv = typical_price * df['volume']

        if anchor_type == 'SESSION':
            group = df['timestamp'].dt.date
        elif anchor_type == 'WEEKLY':
            group = df['timestamp'].dt.to_period('W')
        else:
            group = pd.Series(0, index=df.index)

        cum_pv = pv.groupby(group).cumsum()
        cum_vol = df['volume'].groupby(group).cumsum()
        return cum_pv / np.where(cum_vol == 0, 1e-9, cum_vol)

    @staticmethod
    def detect_smart_anchor(df: pd.DataFrame) -> Tuple[int, str]:
        if len(df) < 30:
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
    def compute_cvd_metrics(df_klines: pd.DataFrame, df_trades: Optional[pd.DataFrame]) -> Tuple[pd.Series, str, float, str]:
        df = df_klines.copy()
        
        if df_trades is not None and not df_trades.empty:
            df_trades['signed_vol'] = np.where(df_trades['is_buy'], df_trades['qty'], -df_trades['qty'])
            trades_resampled = df_trades.set_index('time').resample('5min')['signed_vol'].sum().reindex(df['timestamp'], fill_value=0)
            cvd = trades_resampled.cumsum().reset_index(drop=True)
            cvd_type = DataStatus.LIVE.value
        else:
            candle_range = (df['high'] - df['low']).replace(0, 1e-9)
            delta_approx = df['volume'] * ((df['close'] - df['open']) / candle_range)
            cvd = delta_approx.cumsum()
            cvd_type = DataStatus.APPROXIMATED.value

        price_change = df['close'].iloc[-1] - df['close'].iloc[-10] if len(df) >= 10 else 0
        cvd_change = cvd.iloc[-1] - cvd.iloc[-10] if len(cvd) >= 10 else 0
        cvd_slope = (cvd.iloc[-1] - cvd.iloc[-5]) / (abs(cvd.iloc[-5]) + 1e-9) if len(cvd) >= 5 else 0

        divergence = "NONE"
        if price_change < 0 and cvd_change > 0:
            divergence = "BULLISH_ABSORPTION"
        elif price_change > 0 and cvd_change < 0:
            divergence = "BEARISH_ABSORPTION"

        return cvd, cvd_type, cvd_slope, divergence

# -----------------------------------------------------------------------------
# 4. MULTI-TIMEFRAME ENGINE (1D/4H -> 1H -> 15M -> 5M)
# -----------------------------------------------------------------------------
class MultiTimeframeHierarchy:
    
    @staticmethod
    def evaluate_all(symbol: str) -> Dict[str, Any]:
        timeframes = ['1d', '4h', '1h', '15m', '5m']
        mft_data = {}
        
        for tf in timeframes:
            df_spot, status = MarketDataLoader.fetch_klines(symbol, tf, limit=100)
            if df_spot is not None and not df_spot.empty:
                df_calc = QuantitativeEngine.calculate_ichimoku(df_spot)
                close = df_calc['close'].iloc[-1]
                tenkan = df_calc['tenkan'].iloc[-1]
                kijun = df_calc['kijun'].iloc[-1]
                span_a = df_calc['span_a'].iloc[-1] if pd.notna(df_calc['span_a'].iloc[-1]) else close
                span_b = df_calc['span_b'].iloc[-1] if pd.notna(df_calc['span_b'].iloc[-1]) else close
                cloud_max = max(span_a, span_b)
                cloud_min = min(span_a, span_b)
                
                if close > cloud_max and tenkan > kijun:
                    bias = "BULLISH"
                    score = 80.0
                elif close < cloud_min and tenkan < kijun:
                    bias = "BEARISH"
                    score = 20.0
                else:
                    bias = "NEUTRAL"
                    score = 50.0
                
                mft_data[tf] = {'score': score, 'bias': bias, 'close': close}
            else:
                mft_data[tf] = {'score': 50.0, 'bias': "NEUTRAL", 'close': 0.0}

        # Context Bias (1D + 4H)
        context_score = (mft_data['1d']['score'] * 0.6) + (mft_data['4h']['score'] * 0.4)
        if context_score >= 65: context_bias = "BULLISH 🟢"
        elif context_score <= 35: context_bias = "BEARISH 🔴"
        else: context_bias = "NEUTRAL 🟡"

        # Direction Alignment (1H)
        direction_bias = mft_data['1h']['bias']

        # Trigger Alignment (15M & 5M)
        exec_score = (mft_data['15m']['score'] * 0.5) + (mft_data['5m']['score'] * 0.5)
        
        return {
            'scores': {tf: mft_data[tf]['score'] for tf in timeframes},
            'biases': {tf: mft_data[tf]['bias'] for tf in timeframes},
            'context_bias': context_bias,
            'direction_bias': direction_bias,
            'exec_score': exec_score
        }

# -----------------------------------------------------------------------------
# 5. FACTOR SCORING ENGINE (Anti-Double Counting)
# -----------------------------------------------------------------------------
class FactorScoringEngine:
    @staticmethod
    def compute_layered_score(
        df: pd.DataFrame,
        metrics: QuantitativeMetrics,
        mft_res: Dict[str, Any],
        data_status_futures: str,
        data_status_cvd: str
    ) -> ScoringBreakdown:
        
        close = df['close'].iloc[-1]
        atr = metrics.atr_14 if metrics.atr_14 > 0 else 1.0
        
        # 1. Direction Factor (Ichimoku + HTF Context)
        dir_score = 0.0
        if "BULLISH" in mft_res['context_bias']: dir_score += 15.0
        elif "BEARISH" in mft_res['context_bias']: dir_score -= 15.0
        
        if close > metrics.span_a and close > metrics.span_b: dir_score += 10.0
        elif close < metrics.span_a and close < metrics.span_b: dir_score -= 10.0

        # 2. Flow Factor (CVD Slope & Divergence)
        flow_score = 0.0
        if metrics.cvd_divergence == "BULLISH_ABSORPTION": flow_score = 25.0
        elif metrics.cvd_divergence == "BEARISH_ABSORPTION": flow_score = -25.0
        else: flow_score = np.clip(metrics.cvd_slope * 50.0, -20.0, 20.0)

        # 3. Positioning Factor (OI & Funding)
        pos_score = 0.0
        if data_status_futures != DataStatus.UNAVAILABLE.value:
            if metrics.oi_change_pct > 2.0 and dir_score > 0: pos_score += 15.0
            elif metrics.oi_change_pct > 2.0 and dir_score < 0: pos_score -= 15.0
            
            if metrics.funding_rate is not None:
                if metrics.funding_rate < -0.0001: pos_score += 10.0
                elif metrics.funding_rate > 0.0003: pos_score -= 10.0

        # 4. Location Factor (3-VWAP Distance / ATR)
        loc_score = 0.0
        dist_session_atr = (close - metrics.vwap_session) / atr
        
        if abs(dist_session_atr) <= 0.5: loc_score = 25.0 if dir_score >= 0 else -25.0
        elif dist_session_atr > 2.0: loc_score = -15.0
        elif dist_session_atr < -2.0: loc_score = 15.0
        else: loc_score = 10.0 if dist_session_atr > 0 else -10.0

        total = (dir_score * 0.30) + (flow_score * 0.25) + (pos_score * 0.20) + (loc_score * 0.25)
        
        quality = 100.0
        if data_status_futures == DataStatus.UNAVAILABLE.value: quality -= 25.0
        if data_status_cvd == DataStatus.APPROXIMATED.value: quality -= 15.0

        return ScoringBreakdown(
            direction_score=round(dir_score, 1),
            flow_score=round(flow_score, 1),
            positioning_score=round(pos_score, 1),
            location_score=round(loc_score, 1),
            total_score=round(total, 1),
            data_quality_pct=max(0.0, quality)
        )

# -----------------------------------------------------------------------------
# 6. STREAMLIT UI & DASHBOARD
# -----------------------------------------------------------------------------
def render_css():
    st.markdown("""
        <style>
        .stSidebar, div[data-testid="stSidebar"], div[data-testid="stSidebar"] * {
            word-break: normal !important;
            word-wrap: normal !important;
            white-space: normal !important;
        }
        .status-badge-green { background-color: #133E2B; color: #00E676; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
        .status-badge-red { background-color: #4A191B; color: #FF5252; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
        .status-badge-yellow { background-color: #3D3214; color: #FFD600; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
        </style>
    """, unsafe_allow_html=True)

def main():
    render_css()

    with st.sidebar:
        st.title("⚡ AliQuantFund")
        st.caption("Institutional Engine v4.2")
        st.markdown("---")

        symbol = st.selectbox("المطالبة المالية (Symbol)", ["BTC/USDT", "ETH/USDT", "ZEC/USDT", "SOL/USDT", "XRP/USDT"], index=0)
        timeframe = st.selectbox("الإطار الزمني للتنفيذ (Execution TF)", ["5m", "15m", "1h", "4h"], index=1)
        
        st.markdown("---")
        capital = st.number_input("رأس المال ($):", value=100.0, step=10.0)
        base_risk_pct = st.number_input("نسبة المخاطرة (%):", value=2.0, step=0.5)

    with st.spinner(f"جاري تحليل {symbol} برمجياً وتأكيد الفريمات..."):
        df_klines, spot_status = MarketDataLoader.fetch_klines(symbol, timeframe)
        df_futures, futures_status, funding_meta = MarketDataLoader.fetch_futures_metrics(symbol, timeframe)
        df_trades, cvd_status = MarketDataLoader.fetch_recent_trades(symbol)

    if df_klines is None or df_klines.empty:
        st.error("❌ تعذر جلب بيانات الشموع الأساسية من المصادر. يرجى التحقق من الاتصال.")
        return

    # Calculate Indicators
    df_calc = QuantitativeEngine.calculate_ichimoku(df_klines)
    atr = QuantitativeEngine.calculate_atr(df_calc).iloc[-1]
    
    vwap_sess_series = QuantitativeEngine.calculate_vwap(df_calc, 'SESSION')
    vwap_week_series = QuantitativeEngine.calculate_vwap(df_calc, 'WEEKLY')
    
    anchor_idx, anchor_reason = QuantitativeEngine.detect_smart_anchor(df_calc)
    avwap_series = QuantitativeEngine.calculate_anchored_vwap(df_calc, anchor_idx)
    
    cvd_series, cvd_calc_type, cvd_slope, cvd_div = QuantitativeEngine.compute_cvd_metrics(df_calc, df_trades)
    
    # Multi-Timeframe Hierarchy Evaluation
    mft_res = MultiTimeframeHierarchy.evaluate_all(symbol)
    
    oi_change_pct = 0.0
    if df_futures is not None and not df_futures.empty and len(df_futures) >= 2:
        oi_start = df_futures['openInterest'].iloc[0]
        oi_end = df_futures['openInterest'].iloc[-1]
        oi_change_pct = ((oi_end - oi_start) / (oi_start + 1e-9)) * 100.0

    metrics = QuantitativeMetrics(
        vwap_session=vwap_sess_series.iloc[-1],
        vwap_weekly=vwap_week_series.iloc[-1],
        vwap_anchored=avwap_series.iloc[-1] if not avwap_series.isna().all() else None,
        atr_14=atr,
        cvd_slope=cvd_slope,
        cvd_divergence=cvd_div,
        oi_change_pct=oi_change_pct,
        funding_rate=funding_meta.get('current'),
        tenkan=df_calc['tenkan'].iloc[-1],
        kijun=df_calc['kijun'].iloc[-1],
        span_a=df_calc['span_a'].iloc[-1] if pd.notna(df_calc['span_a'].iloc[-1]) else df_calc['close'].iloc[-1],
        span_b=df_calc['span_b'].iloc[-1] if pd.notna(df_calc['span_b'].iloc[-1]) else df_calc['close'].iloc[-1]
    )

    scoring = FactorScoringEngine.compute_layered_score(df_calc, metrics, mft_res, futures_status, cvd_calc_type)

    # Top Status Bar
    col_head1, col_head2, col_head3, col_head4 = st.columns(4)
    with col_head1:
        st.markdown(f"### {symbol}")
        st.caption(f"Price: **${df_calc['close'].iloc[-1]:,.2f}**")
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
        if score >= 12.0 and scoring.data_quality_pct >= 60 and "BULLISH" in mft_res['context_bias']:
            final_dec = "CONFIRMED LONG"
            color = "#00E676"
        elif score <= -12.0 and scoring.data_quality_pct >= 60 and "BEARISH" in mft_res['context_bias']:
            final_dec = "CONFIRMED SHORT"
            color = "#FF5252"
        else:
            final_dec = "NO TRADE / WAIT"
            color = "#FFD600"

        st.markdown(f"""
            <div style="background-color: #1E222D; border-left: 5px solid {color}; padding: 15px; border-radius: 5px;">
                <h2 style="color: {color}; margin: 0;">{final_dec}</h2>
                <p style="margin-top: 5px; color: #B2B9C7;">
                    الانحياز العام (HTF): <b>{mft_res['context_bias']}</b> | جودة البيانات: <b>{scoring.data_quality_pct:.0f}%</b>
                </p>
                <small>مستوى التوافق الكمي المركب: {score:.1f}</small>
            </div>
        """, unsafe_allow_html=True)

    with dec_col2:
        with st.expander("🗺️ سياق الأطر الزمنية المترابطة (MFT Alignment Hierarchy)", expanded=True):
            tf_c1, tf_c2, tf_c3, tf_c4, tf_c5 = st.columns(5)
            tf_c1.metric("1D", f"{mft_res['scores']['1d']:.0f}", mft_res['biases']['1d'])
            tf_c2.metric("4H", f"{mft_res['scores']['4h']:.0f}", mft_res['biases']['4h'])
            tf_c3.metric("1H", f"{mft_res['scores']['1h']:.0f}", mft_res['biases']['1h'])
            tf_c4.metric("15M", f"{mft_res['scores']['15m']:.0f}", mft_res['biases']['15m'])
            tf_c5.metric("5M", f"{mft_res['scores']['5m']:.0f}", mft_res['biases']['5m'])

    st.markdown("---")

    # Interactive Chart Section
    st.subheader(f"📈 شارت {symbol} المدمج (Ichimoku Cloud + 3-VWAP + CVD Order Flow)")
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_calc['timestamp'], open=df_calc['open'], high=df_calc['high'],
        low=df_calc['low'], close=df_calc['close'], name="Price"
    ), row=1, col=1)

    # Ichimoku Cloud Components
    fig.add_trace(go.Scatter(x=df_calc['timestamp'], y=df_calc['tenkan'], mode='lines', name='Tenkan-sen', line=dict(color='#29B6F6', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_calc['timestamp'], y=df_calc['kijun'], mode='lines', name='Kijun-sen', line=dict(color='#FF5252', width=1)), row=1, col=1)

    # VWAP Overlay
    fig.add_trace(go.Scatter(x=df_calc['timestamp'], y=vwap_sess_series, mode='lines', name='Session VWAP', line=dict(color='#FFD600', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_calc['timestamp'], y=vwap_week_series, mode='lines', name='Weekly VWAP', line=dict(color='#E040FB', width=1.5, dash='dash')), row=1, col=1)

    # CVD Plot
    fig.add_trace(go.Scatter(
        x=df_calc['timestamp'], y=cvd_series, mode='lines', 
        name=f'CVD ({cvd_calc_type})', line=dict(color='#00E676', width=1.5)
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
