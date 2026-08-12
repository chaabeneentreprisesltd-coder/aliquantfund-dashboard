import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone
import logging

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
    p, span, label, div {
        word-break: break-word !important;
        white-space: normal !important;
    }
    div[data-testid="stSidebarNav"], .css-1d33210, .st-emotion-cache-16idsys {
        display: none !important;
    }
    .status-badge-live {
        background-color: #0e382c; color: #00e676; padding: 4px 10px; border-radius: 6px; font-weight: bold; border: 1px solid #00e676;
    }
    .status-badge-fallback {
        background-color: #3d310d; color: #ffb300; padding: 4px 10px; border-radius: 6px; font-weight: bold; border: 1px solid #ffb300;
    }
    .status-badge-unavail {
        background-color: #3a161a; color: #ff5252; padding: 4px 10px; border-radius: 6px; font-weight: bold; border: 1px solid #ff5252;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA STRUCTURES & DATA LAYER
# ==========================================

class DataStatus:
    LIVE = "LIVE"
    FALLBACK = "FALLBACK"
    DELAYED = "DELAYED"
    UNAVAILABLE = "UNAVAILABLE"

@dataclass
class MarketDataContainer:
    symbol: str
    interval: str
    spot_df: pd.DataFrame
    futures_df: Optional[pd.DataFrame] = None
    orderflow_df: Optional[pd.DataFrame] = None
    data_quality_score: int = 100
    status_spot: str = DataStatus.UNAVAILABLE
    status_futures: str = DataStatus.UNAVAILABLE
    status_orderflow: str = DataStatus.UNAVAILABLE
    is_real_cvd: bool = False
    source_info: str = ""

class MarketDataLoader:
    """طبقة البيانات المدمجة لجلب بيانات Spot و Futures و Order Flow مع Fallbacks و Quality Assessment"""
    
    BYBIT_TF_MAP = {'5m': '5', '15m': '15', '1h': '60', '4h': '240', '1d': 'D'}
    
    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_spot_ohlcv(symbol: str, interval: str, limit: int = 150) -> Tuple[Optional[pd.DataFrame], str, str]:
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # Primary Source: Binance Spot API
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
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], DataStatus.LIVE, "Binance Spot Direct"
            except Exception as e:
                logger.warning(f"Binance endpoint failed: {url}, error: {e}")
                continue

        # Secondary Fallback Source: Bybit Spot API
        try:
            bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, '5')
            url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={formatted_symbol}&interval={bybit_tf}&limit={limit}"
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                result = res.json().get('result', {}).get('list', [])
                if result:
                    df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    df = df.iloc[::-1].reset_index(drop=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], DataStatus.FALLBACK, "Bybit Spot API"
        except Exception as e:
            logger.error(f"Bybit Spot API failed: {e}")
            
        return None, DataStatus.UNAVAILABLE, "None"

    @staticmethod
    @st.cache_data(ttl=25)
    def fetch_futures_metrics(symbol: str, interval: str, limit: int = 50) -> Tuple[Optional[pd.DataFrame], str]:
        """جلب بيانات الماركت الآجل: Open Interest و Funding Rate"""
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}
        bybit_tf = MarketDataLoader.BYBIT_TF_MAP.get(interval, '5')
        
        try:
            # Bybit Linear Futures Market
            url_oi = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={formatted_symbol}&intervalTime={bybit_tf}m&limit={limit}"
            if interval == '1d':
                url_oi = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={formatted_symbol}&intervalTime=1d&limit={limit}"
                
            res_oi = requests.get(url_oi, headers=headers, timeout=4)
            
            url_fr = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={formatted_symbol}&limit=5"
            res_fr = requests.get(url_fr, headers=headers, timeout=4)

            if res_oi.status_code == 200:
                oi_data = res_oi.json().get('result', {}).get('list', [])
                if oi_data:
                    df_oi = pd.DataFrame(oi_data)
                    df_oi['openInterest'] = df_oi['openInterest'].astype(float)
                    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'].astype(float), unit='ms')
                    df_oi = df_oi.iloc[::-1].reset_index(drop=True)
                    
                    # Fetch current funding rate
                    funding_rate = 0.0001
                    if res_fr.status_code == 200:
                        fr_list = res_fr.json().get('result', {}).get('list', [])
                        if fr_list:
                            funding_rate = float(fr_list[0].get('fundingRate', 0.0001))
                            
                    df_oi['funding_rate'] = funding_rate
                    return df_oi, DataStatus.LIVE
        except Exception as e:
            logger.warning(f"Futures Metrics fetch error: {e}")
            
        return None, DataStatus.UNAVAILABLE

    @staticmethod
    @st.cache_data(ttl=15)
    def fetch_trade_level_orderflow(symbol: str, limit: int = 500) -> Tuple[Optional[pd.DataFrame], bool, str]:
        """جلب صفقات السوق اللحظية (Recent Trades) لبناء Real CVD عند توفرها"""
        formatted_symbol = symbol.replace("/", "").upper()
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # Attempt 1: Binance Public Recent Trades
        url = f"https://api1.binance.com/api/v3/trades?symbol={formatted_symbol}&limit={limit}"
        try:
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code == 200:
                trades = res.json()
                if isinstance(trades, list) and len(trades) > 0:
                    df_trades = pd.DataFrame(trades)
                    # price, qty, time, isBuyerMaker
                    df_trades['price'] = df_trades['price'].astype(float)
                    df_trades['qty'] = df_trades['qty'].astype(float)
                    df_trades['time'] = pd.to_datetime(df_trades['time'], unit='ms')
                    # isBuyerMaker True -> Seller initiated (Sell Volume)
                    # isBuyerMaker False -> Buyer initiated (Buy Volume)
                    df_trades['is_buy'] = ~df_trades['isBuyerMaker']
                    return df_trades, True, DataStatus.LIVE
        except Exception as e:
            logger.warning(f"Trade level data fetch failed: {e}")
            
        return None, False, DataStatus.FALLBACK

# ==========================================
# 2. INDICATOR & QUANTITATIVE ENGINE LAYER
# ==========================================

class QuantitativeEngine:
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean().bfill()

    @staticmethod
    def calculate_adx_trend(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """حساب ADX لتقييم قوة الاتجاه بدون Look-Ahead Bias"""
        df_copy = df.copy()
        up_move = df_copy['high'] - df_copy['high'].shift(1)
        down_move = df_copy['low'].shift(1) - df_copy['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr = QuantitativeEngine.calculate_atr(df_copy, period=1)
        atr = tr.rolling(period).mean().replace(0, 1e-5)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-5)) * 100
        adx = dx.rolling(period).mean().bfill()
        return adx.fillna(20.0), plus_di.fillna(20.0), minus_di.fillna(20.0)

    @staticmethod
    def detect_smart_anchor(df: pd.DataFrame, mode: str = "Automatic") -> Dict[str, Any]:
        """كاشف المرساة الذكي (Smart Anchor Detection) يعتمد على السلوك السعري والسيولة"""
        if df is None or len(df) < 30:
            return {'index': 0, 'price': df['close'].iloc[0] if df is not None else 0, 'timestamp': df['timestamp'].iloc[0] if df is not None else datetime.now(), 'type': 'Default'}

        if mode == "Swing High":
            idx = df['high'].idxmax()
            return {'index': idx, 'price': df.loc[idx, 'high'], 'timestamp': df.loc[idx, 'timestamp'], 'type': 'Swing High'}
        elif mode == "Swing Low":
            idx = df['low'].idxmin()
            return {'index': idx, 'price': df.loc[idx, 'low'], 'timestamp': df.loc[idx, 'timestamp'], 'type': 'Swing Low'}
        
        # Automatic Mode: Search for Significant Volume Spike or Structure Pivot
        vol_ma = df['volume'].rolling(20).mean()
        vol_spike_mask = df['volume'] > (vol_ma * 2.2)
        
        if vol_spike_mask.any():
            # Pick the index of the largest volume spike in the frame
            idx = df.loc[vol_spike_mask, 'volume'].idxmax()
            anchor_type = "Major Volume Spike"
        else:
            # Fallback to absolute local Swing Low
            idx = df['low'].idxmin()
            anchor_type = "Structural Pivot Low"
            
        return {
            'index': idx,
            'price': df.loc[idx, 'close'],
            'timestamp': df.loc[idx, 'timestamp'],
            'type': anchor_type
        }

    @staticmethod
    def calculate_vwap_suite(df: pd.DataFrame, anchor_info: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['tp'] * df['volume']
        
        # Session VWAP (Resets daily at 00:00 UTC)
        df['date'] = df['timestamp'].dt.date
        session_pv = df.groupby('date')['pv'].cumsum()
        session_vol = df.groupby('date')['volume'].cumsum()
        df['vwap_session'] = np.where(session_vol > 0, session_pv / session_vol, df['tp'])
        
        # Weekly VWAP (Resets weekly)
        df['week_year'] = df['timestamp'].dt.strftime('%Y-%U')
        weekly_pv = df.groupby('week_year')['pv'].cumsum()
        weekly_vol = df.groupby('week_year')['volume'].cumsum()
        df['vwap_weekly'] = np.where(weekly_vol > 0, weekly_pv / weekly_vol, df['tp'])
        
        # Smart Anchored VWAP
        anc_idx = anchor_info['index']
        df_anc = df.loc[anc_idx:].copy()
        anc_pv = (df_anc['tp'] * df_anc['volume']).cumsum()
        anc_vol = df_anc['volume'].cumsum()
        
        df['vwap_anchored'] = np.nan
        df.loc[anc_idx:, 'vwap_anchored'] = np.where(anc_vol > 0, anc_pv / anc_vol, df_anc['tp'])
        df['vwap_anchored'] = df['vwap_anchored'].ffill().bfill()
        
        return df

    @staticmethod
    def build_orderflow_cvd(df: pd.DataFrame, df_trades: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, bool]:
        """بناء CVD الحقيقي من Trade Level أو Approximate CVD عند تعذر البيانات"""
        df = df.copy()
        
        if df_trades is not None and not df_trades.empty:
            # Real Trade Level Delta
            buy_vol = df_trades[df_trades['is_buy']]['qty'].sum()
            sell_vol = df_trades[~df_trades['is_buy']]['qty'].sum()
            net_delta = buy_vol - sell_vol
            
            # Approximate candles delta mapped with real last delta bias
            prev_close = df['close'].shift(1).bfill()
            direction = np.where(df['close'] >= prev_close, 1, -1)
            df['delta'] = df['volume'] * direction
            # Adjust latest candle delta with trade level intensity
            df.loc[df.index[-1], 'delta'] = net_delta
            df['cvd'] = df['delta'].cumsum()
            return df, True
        else:
            # Approximate CVD (Fallback)
            prev_close = df['close'].shift(1).bfill()
            direction = np.where(df['close'] >= prev_close, 1, -1)
            df['delta'] = df['volume'] * direction
            df['cvd'] = df['delta'].cumsum()
            return df, False

    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> str:
        """كشف حالة النظام الساعي (Market Regime) بدقة لمنع التداول الخاطئ"""
        if len(df) < 30:
            return "RANGING"
            
        atr = QuantitativeEngine.calculate_atr(df, 14)
        atr_pct = (atr.iloc[-1] / df['close'].iloc[-1]) * 100
        adx, plus_di, minus_di = QuantitativeEngine.calculate_adx_trend(df, 14)
        
        latest_adx = adx.iloc[-1]
        latest_pdi = plus_di.iloc[-1]
        latest_mdi = minus_di.iloc[-1]
        
        # Volatility Check
        vol_ma = df['volume'].rolling(20).mean()
        is_high_vol = atr_pct > 2.5 or (df['volume'].iloc[-1] > vol_ma.iloc[-1] * 2.0)
        
        if latest_adx > 25:
            if latest_pdi > latest_mdi:
                return "TRENDING BULL"
            else:
                return "TRENDING BEAR"
        elif is_high_vol:
            return "HIGH VOLATILITY"
        elif atr_pct < 0.6:
            return "LOW VOLATILITY"
        else:
            return "RANGING"

# ==========================================
# 3. MULTI-FACTOR SCORING ENGINE (NO DOUBLE COUNTING)
# ==========================================

class FactorScoringEngine:
    """محرك التقييم متعدد العوامل - يمنع Double Counting ويزن المؤشرات حسب نظام السوق"""
    
    @staticmethod
    def evaluate_factors(
        df: pd.DataFrame, 
        df_futures: Optional[pd.DataFrame], 
        is_real_cvd: bool, 
        regime: str
    ) -> Dict[str, Any]:
        
        latest = df.iloc[-1]
        atr = max(latest['atr'], 1e-5)
        
        # ------------------------------------
        # Factor 1: PRICE STRUCTURE (0 - 100)
        # ------------------------------------
        # Break of Structure & Candle Momentum
        prev_10_high = df['high'].iloc[-11:-1].max()
        prev_10_low = df['low'].iloc[-11:-1].min()
        
        price_score = 50.0
        if latest['close'] > prev_10_high:
            price_score = 85.0 # Bullish Breakout
        elif latest['close'] < prev_10_low:
            price_score = 15.0 # Bearish Breakdown
        else:
            # Range relative position
            rng = max(prev_10_high - prev_10_low, 1e-5)
            price_score = 30.0 + ((latest['close'] - prev_10_low) / rng) * 40.0

        # ------------------------------------
        # Factor 2: ORDER FLOW & CVD (0 - 100)
        # ------------------------------------
        of_score = 50.0
        if len(df) >= 10:
            cvd_delta = df['cvd'].iloc[-1] - df['cvd'].iloc[-10]
            price_delta = df['close'].iloc[-1] - df['close'].iloc[-10]
            
            if cvd_delta > 0 and price_delta > 0:
                of_score = 80.0 # Aggressive Long Buying
            elif cvd_delta < 0 and price_delta < 0:
                of_score = 20.0 # Aggressive Short Selling
            elif cvd_delta > 0 and price_delta < 0:
                of_score = 65.0 # Bullish Absorption (تجميع)
            elif cvd_delta < 0 and price_delta > 0:
                of_score = 35.0 # Bearish Distribution (تصريف)
                
        # Discount score if using Approximated CVD
        if not is_real_cvd:
            of_score = 50.0 + (of_score - 50.0) * 0.5

        # ------------------------------------
        # Factor 3: POSITIONING & OI (0 - 100)
        # ------------------------------------
        positioning_score = 50.0
        oi_state_desc = "Neutral Positioning"
        
        if df_futures is not None and len(df_futures) >= 5:
            latest_oi = df_futures['openInterest'].iloc[-1]
            prev_oi = df_futures['openInterest'].iloc[-5]
            price_chg = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            oi_chg = (latest_oi - prev_oi) / prev_oi if prev_oi > 0 else 0
            
            if oi_chg > 0.01 and price_chg > 0:
                positioning_score = 85.0
                oi_state_desc = "Long Build-up (صعود مدعوم برؤوس أموال جديدة)"
            elif oi_chg > 0.01 and price_chg < 0:
                positioning_score = 15.0
                oi_state_desc = "Short Build-up (هبوط مدعوم بدخول صفقات شورت)"
            elif oi_chg < -0.01 and price_chg > 0:
                positioning_score = 40.0
                oi_state_desc = "Short Covering (صعود إجبار بسبب إغلاق الشورت)"
            elif oi_chg < -0.01 and price_chg < 0:
                positioning_score = 60.0
                oi_state_desc = "Long Unwinding (تصفية صفقات شراء)"

        # ------------------------------------
        # Factor 4: TREND & REGIME ALIGNMENT (0 - 100)
        # ------------------------------------
        adx, pdi, mdi = QuantitativeEngine.calculate_adx_trend(df)
        trend_score = 50.0
        if pdi.iloc[-1] > mdi.iloc[-1]:
            trend_score = 50.0 + min(adx.iloc[-1] * 1.0, 45.0)
        else:
            trend_score = 50.0 - min(adx.iloc[-1] * 1.0, 45.0)

        # ------------------------------------
        # Factor 5: VWAP LOCATION (0 - 100)
        # ------------------------------------
        d_sess = (latest['close'] - latest['vwap_session']) / atr
        d_week = (latest['close'] - latest['vwap_weekly']) / atr
        d_anc = (latest['close'] - latest['vwap_anchored']) / atr
        
        # Volatility-Normalized Distance Score
        avg_distance_atr = (d_sess + d_week + d_anc) / 3.0
        vwap_score = np.clip(50.0 + (avg_distance_atr * 20.0), 0.0, 100.0)

        # ------------------------------------
        # REGIME-WEIGHTED COMBINATION
        # ------------------------------------
        if regime in ["TRENDING BULL", "TRENDING BEAR"]:
            weights = {'price': 0.25, 'orderflow': 0.20, 'positioning': 0.15, 'trend': 0.25, 'vwap': 0.15}
        elif regime == "HIGH VOLATILITY":
            weights = {'price': 0.15, 'orderflow': 0.35, 'positioning': 0.20, 'trend': 0.10, 'vwap': 0.20}
        else: # RANGING / LOW VOLATILITY
            weights = {'price': 0.20, 'orderflow': 0.25, 'positioning': 0.15, 'trend': 0.10, 'vwap': 0.30}

        global_score = (
            price_score * weights['price'] +
            of_score * weights['orderflow'] +
            positioning_score * weights['positioning'] +
            trend_score * weights['trend'] +
            vwap_score * weights['vwap']
        )
        
        return {
            'global_score': round(global_score, 1),
            'price_score': price_score,
            'orderflow_score': of_score,
            'positioning_score': positioning_score,
            'trend_score': trend_score,
            'vwap_score': vwap_score,
            'oi_state': oi_state_desc,
            'distances_atr': {'session': d_sess, 'weekly': d_week, 'anchored': d_anc}
        }

# ==========================================
# 4. MULTI-TIMEFRAME & ENTRY SETUP ENGINE
# ==========================================

class MultiTimeframeEngine:
    
    @staticmethod
    def evaluate_mft(symbol: str, anchor_mode: str) -> Dict[str, Any]:
        """محرك الأطر الزمنية المتعددة المبني على التسلسل الهرمي (HTF Bias -> Exec)"""
        timeframes = ['1d', '4h', '1h', '15m', '5m']
        scores = {}
        regimes = {}
        
        for tf in timeframes:
            df_spot, spot_status, _ = MarketDataLoader.fetch_spot_ohlcv(symbol, tf, limit=100)
            if df_spot is not None and not df_spot.empty:
                anchor_info = QuantitativeEngine.detect_smart_anchor(df_spot, mode=anchor_mode)
                df_calc = QuantitativeEngine.calculate_vwap_suite(df_spot, anchor_info)
                df_calc, is_real = QuantitativeEngine.build_orderflow_cvd(df_calc, None)
                regime = QuantitativeEngine.detect_market_regime(df_calc)
                
                factors = FactorScoringEngine.evaluate_factors(df_calc, None, is_real, regime)
                scores[tf] = factors['global_score']
                regimes[tf] = regime
            else:
                scores[tf] = 50.0
                regimes[tf] = "NEUTRAL"
                
        # HTF Bias Hierarchy (1D + 4H)
        htf_score = (scores['1d'] * 0.6) + (scores['4h'] * 0.4)
        if htf_score >= 65:
            htf_bias = "BULLISH 🟢"
        elif htf_score <= 35:
            htf_bias = "BEARISH 🔴"
        else:
            htf_bias = "NEUTRAL 🟡"
            
        # Execution Trigger Alignment (15M + 5M)
        exec_score = (scores['15m'] * 0.5) + (scores['5m'] * 0.5)
        
        return {
            'tf_scores': scores,
            'htf_bias': htf_bias,
            'htf_score': htf_score,
            'exec_score': exec_score,
            'regimes': regimes
        }

# ==========================================
# 5. SIGNAL CLASSIFICATION & RISK ENGINE
# ==========================================

class SignalAndRiskEngine:
    
    @staticmethod
    def classify_signal(
        global_score: float, 
        htf_bias: str, 
        data_quality_score: int, 
        regime: str
    ) -> Tuple[str, int, List[str]]:
        
        reasons = []
        confidence = int(min(data_quality_score, 100))
        
        # Rule: Low Data Quality Penalty
        if data_quality_score < 60:
            confidence -= 20
            reasons.append("⚠️ بيانات محدودة - تم تخفيض درجة الثقة تلقائياً.")

        # Hierarchy Mapping
        if global_score >= 82 and "BULLISH" in htf_bias:
            grade = "A+ LONG"
            confidence = int(min(confidence * 0.95, 95))
            reasons.append("توافق تام بين الاتجاه العام الكلي (HTF) والسيولة اللحظية.")
            reasons.append("السعر في حالة صعود مؤسسي مدعوم بإنشائه لقيم سعرية جديدة فوق الـ VWAPs.")
        elif global_score >= 70 and "BEARISH" not in htf_bias:
            grade = "A LONG"
            reasons.append("زخم شراء إيجابي جيد يدعم صفقات الشراء مع حذر جزئي.")
        elif global_score >= 60:
            grade = "B LONG"
            reasons.append("إشارة شراء مضاربية صغرى (Scalp) عكس أو حياد الاتجاه الكلي.")
        elif global_score <= 18 and "BEARISH" in htf_bias:
            grade = "A+ SHORT"
            confidence = int(min(confidence * 0.95, 95))
            reasons.append("توافق هابط ممتاز بين الأطر الكبرى والضغط البيعي المؤسسي.")
        elif global_score <= 30 and "BULLISH" not in htf_bias:
            grade = "A SHORT"
            reasons.append("ضغط بيعي قوي يرجح الاستمرار في الاتجاه الهابط.")
        elif global_score <= 40:
            grade = "B SHORT"
            reasons.append("إشارة بيع مضاربية صغرى احترازية.")
        else:
            grade = "NEUTRAL"
            reasons.append("تضارب بين السيولة والاتجاه السائدي - يفضل الانتظار خارج السوق.")
            
        return grade, max(confidence, 10), reasons

    @staticmethod
    def calculate_smart_risk_parameters(
        entry_price: float, 
        sl_source_price: float, 
        atr: float, 
        is_long: bool, 
        capital: float, 
        base_risk_pct: float,
        grade: str,
        regime: str
    ) -> Dict[str, Any]:
        """حساب مستويات الستوب والهجمات الذكية ومنع المعاملات السالبة أو غير المنطقية"""
        
        # Buffer distance using ATR to prevent noise stop-outs
        atr_buffer = atr * 0.5
        
        if is_long:
            # Stop Loss must be BELOW entry
            calculated_sl = min(sl_source_price - atr_buffer, entry_price - (atr * 0.8))
            sl_distance = entry_price - calculated_sl
            tp1 = entry_price + (sl_distance * 1.5)
            tp2 = entry_price + (sl_distance * 3.0)
        else:
            # Stop Loss must be ABOVE entry
            calculated_sl = max(sl_source_price + atr_buffer, entry_price + (atr * 0.8))
            sl_distance = calculated_sl - entry_price
            tp1 = entry_price - (sl_distance * 1.5)
            tp2 = entry_price - (sl_distance * 3.0)

        # Dynamic Grade Sizing Multiplier
        grade_multiplier = 1.0
        if "A+" in grade:
            grade_multiplier = 1.0
        elif "A" in grade:
            grade_multiplier = 0.75
        elif "B" in grade:
            grade_multiplier = 0.40
        else:
            grade_multiplier = 0.0

        effective_risk_pct = base_risk_pct * grade_multiplier
        risk_amount = capital * (effective_risk_pct / 100.0)
        
        units = (risk_amount / sl_distance) if sl_distance > 0 else 0.0
        position_value = units * entry_price
        
        return {
            'entry': entry_price,
            'sl': calculated_sl,
            'tp1': tp1,
            'tp2': tp2,
            'sl_distance': sl_distance,
            'risk_amount': risk_amount,
            'effective_risk_pct': effective_risk_pct,
            'units': units,
            'position_value': position_value,
            'rr_tp1': 1.5,
            'rr_tp2': 3.0
        }

# ==========================================
# 6. STREAMLIT USER INTERFACE LAYER
# ==========================================

# --- Sidebar Controls ---
st.sidebar.title("⚡ AliQuantFund")
st.sidebar.caption("Institutional Market Engine v3.0")
st.sidebar.markdown("---")

selected_symbol = st.sidebar.selectbox(
    "الأصل المالي (Symbol):",
    ["BTC/USDT", "ETH/USDT", "ZEC/USDT", "XRP/USDT", "SOL/USDT"]
)

selected_tf = st.sidebar.selectbox(
    "الإطار الزمني للتنفيذ (Execution TF):",
    ["5m", "15m", "1h", "4h", "1d"],
    index=0
)

anchor_mode = st.sidebar.selectbox(
    "نمط وضع مرساة الـ VWAP (Smart Anchor):",
    ["Automatic", "Swing High", "Swing Low"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 إدارة رأس المال")
capital = st.sidebar.number_input("رأس المال ($):", value=100.0, step=10.0)
base_risk_pct = st.sidebar.number_input("أقصى نسبة مخاطرة (%):", value=2.0, step=0.5)

# --- MAIN CONTROLLER ---
st.title(f"📊 التحليل الكمي المؤسسي: {selected_symbol}")

# Fetch Data
spot_df, spot_status, spot_source = MarketDataLoader.fetch_spot_ohlcv(selected_symbol, selected_tf)
futures_df, futures_status = MarketDataLoader.fetch_futures_metrics(selected_symbol, selected_tf)
trades_df, is_real_cvd, orderflow_status = MarketDataLoader.fetch_trade_level_orderflow(selected_symbol)

if spot_df is not None and not spot_df.empty:
    
    # Calculate Indicators
    spot_df['atr'] = QuantitativeEngine.calculate_atr(spot_df)
    anchor_info = QuantitativeEngine.detect_smart_anchor(spot_df, mode=anchor_mode)
    spot_df = QuantitativeEngine.calculate_vwap_suite(spot_df, anchor_info)
    spot_df, is_real_cvd = QuantitativeEngine.build_orderflow_cvd(spot_df, trades_df)
    
    market_regime = QuantitativeEngine.detect_market_regime(spot_df)
    
    # Assess Data Quality Score
    data_quality = 100
    if spot_status == DataStatus.FALLBACK: data_quality -= 20
    if futures_status == DataStatus.UNAVAILABLE: data_quality -= 20
    if not is_real_cvd: data_quality -= 15
    
    # Multi-Factor Scoring & MFT Evaluation
    factors = FactorScoringEngine.evaluate_factors(spot_df, futures_df, is_real_cvd, market_regime)
    mft_res = MultiTimeframeEngine.evaluate_mft(selected_symbol, anchor_mode)
    
    # Signal Grade Classification
    signal_grade, confidence_score, reasons = SignalAndRiskEngine.classify_signal(
        factors['global_score'], mft_res['htf_bias'], data_quality, market_regime
    )
    
    # Header Badges (Live Status)
    badge_class = "status-badge-live" if spot_status == DataStatus.LIVE else "status-badge-fallback"
    cvd_type_label = "REAL (Trade-Level)" if is_real_cvd else "APPROXIMATED (Candle-Level)"
    
    st.markdown(f"""
    <div style="display: flex; gap: 10px; margin-bottom: 15px; align-items: center;">
        <span class="{badge_class}">Spot: {spot_status} ({spot_source})</span>
        <span class="status-badge-live">Futures/OI: {futures_status}</span>
        <span class="status-badge-fallback">CVD Engine: {cvd_type_label}</span>
        <span style="font-weight: bold; color: #888;">جودة البيانات: {data_quality}%</span>
    </div>
    """, unsafe_allow_html=True)

    # --- TOP DASHBOARD METRICS ---
    st.markdown("### 🌐 اللوحة التنفيذية الموحدة (Executive Dashboard)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("الدرجة التشغيلية (Grade)", signal_grade, f"Confidence: {confidence_score}%")
    col2.metric("التقييم المركب (Global Score)", f"{factors['global_score']} / 100", f"HTF Bias: {mft_res['htf_bias']}")
    col3.metric("نظام السوق (Regime)", market_regime, f"ATR: {spot_df['atr'].iloc[-1]:.2f}")
    col4.metric("مرساة الـ VWAP الحالية", f"{anchor_info['type']}", f"Price: ${anchor_info['price']:.2f}")

    st.markdown("---")

    # --- DETAILED FACTOR BREAKDOWN PANEL ---
    with st.expander("🔬 تفكيك العوامل الكمية (Factor Breakdown - Anti-Double Counting)", expanded=False):
        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
        f_col1.metric("Price Structure", f"{factors['price_score']:.1f}")
        f_col2.metric("Order Flow & CVD", f"{factors['orderflow_score']:.1f}")
        f_col3.metric("Positioning & OI", f"{factors['positioning_score']:.1f}")
        f_col4.metric("Trend Alignment", f"{factors['trend_score']:.1f}")
        f_col5.metric("VWAP Location", f"{factors['vwap_score']:.1f}")
        
        st.caption(f"• **حالة تدفق السيولة المفتوحة (OI):** {factors['oi_state']}")

    # --- CHART AND RISK CALCULATION SECTION ---
    c_chart, c_risk = st.columns([3, 1])
    
    latest_close = spot_df['close'].iloc[-1]
    is_long_trade = "LONG" in signal_grade or factors['global_score'] >= 50
    
    with c_risk:
        st.markdown("### 🎯 إشعار التنفيذ وإدارة المخاطر")
        st.write(f"**حالة التوصية:** `{signal_grade}`")
        
        entry_p = st.number_input("سعر الدخول:", value=float(latest_close))
        
        sl_layer = st.selectbox("مصدر الستوب الأولي:", ["Session VWAP", "Weekly VWAP", "Anchored VWAP", "Structural High/Low"])
        if sl_layer == "Session VWAP": sl_val = float(spot_df['vwap_session'].iloc[-1])
        elif sl_layer == "Weekly VWAP": sl_val = float(spot_df['vwap_weekly'].iloc[-1])
        elif sl_layer == "Anchored VWAP": sl_val = float(spot_df['vwap_anchored'].iloc[-1])
        else: sl_val = float(spot_df['low'].iloc[-10:].min() if is_long_trade else spot_df['high'].iloc[-10:].max())
        
        risk_params = SignalAndRiskEngine.calculate_smart_risk_parameters(
            entry_p, sl_val, spot_df['atr'].iloc[-1], is_long_trade, capital, base_risk_pct, signal_grade, market_regime
        )
        
        st.markdown(f"• **نوع الصفقة:** `{'🟢 شراء (Long)' if is_long_trade else '🔴 بيع (Short)'}`")
        st.markdown(f"• **وقف الخسارة (SL):** `${risk_params['sl']:.2f}`")
        st.markdown(f"• **الهدف الأول (TP1 - 1:1.5):** `${risk_params['tp1']:.2f}`")
        st.markdown(f"• **الهدف الثاني (TP2 - 1:3.0):** `${risk_params['tp2']:.2f}`")
        
        st.markdown("---")
        st.caption(f"• المخاطرة الفعالة: `${risk_params['risk_amount']:.2f}` ({risk_params['effective_risk_pct']:.2f}%)")
        st.caption(f"• حجم العقود (Units): `{risk_params['units']:.4f}`")
        st.caption(f"• القيمة الإجمالية للعقد: `${risk_params['position_value']:.2f}`")
        
        st.markdown("---")
        st.markdown("#### 💡 أسباب القرار البرمجي:")
        for r in reasons:
            st.write(f"- {r}")

    with c_chart:
        # Multi-panel Plotly Chart (Price + 3-VWAP, CVD, Open Interest)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.60, 0.20, 0.20]
        )

        # Panel 1: Candlesticks & 3-VWAP Suite
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

        # Panel 2: Order Flow CVD
        fig.add_trace(go.Scatter(
            x=spot_df['timestamp'], y=spot_df['cvd'], mode='lines',
            name='CVD Order Flow', line=dict(color='deepskyblue', width=2), fill='tozeroy'
        ), row=2, col=1)

        # Panel 3: Open Interest (If available)
        if futures_df is not None and not futures_df.empty:
            fig.add_trace(go.Scatter(
                x=futures_df['timestamp'], y=futures_df['openInterest'], mode='lines',
                name='Futures Open Interest (Bybit)', line=dict(color='orange', width=2)
            ), row=3, col=1)

        fig.update_layout(
            title=f"شارت {selected_symbol} - {selected_tf} (Subplots Engine)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=700,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ تعذر جلب البيانات الحية من خوادم السوق. يرجى التحقق من الاتصال أو محاولة اختيار زوج آخر.")

st.markdown("---")
st.caption("⚡ AliQuantFund Institutional Architecture v3.0 | All Rights Reserved")
