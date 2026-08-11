import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==========================================
# 1. إعدادات الصفحة والتصميم العامة
# ==========================================
st.set_page_config(
    page_title="AliQuantFund | Multi-VWAP Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #2a2e39;
    }
    p, span, label {
        word-break: break-word;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. محرك البيانات الحركية
# ==========================================

TIMEFRAME_WEIGHTS = {
    '1d': 0.35,
    '4h': 0.25,
    '1h': 0.20,
    '15m': 0.12,
    '5m': 0.08
}

BYBIT_TF_MAP = {
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}

@st.cache_data(ttl=20)
def fetch_klines_data(symbol="BTCUSDT", interval="5m", limit=150):
    formatted_symbol = symbol.replace("/", "").upper()
    headers = {'User-Agent': 'Mozilla/5.0'}

    binance_endpoints = [
        f"https://api1.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}",
        f"https://api3.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}",
        f"https://data-api.binance.vision/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}"
    ]

    for url in binance_endpoints:
        try:
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                data = res.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df
        except:
            continue

    try:
        bybit_tf = '5' if interval == '5m' else ('15' if interval == '15m' else ('60' if interval == '1h' else ('240' if interval == '4h' else 'D')))
        bybit_url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={formatted_symbol}&interval={bybit_tf}&limit={limit}"
        res = requests.get(bybit_url, headers=headers, timeout=5)
        if res.status_code == 200:
            result = res.json().get('result', {}).get('list', [])
            if result:
                df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.iloc[::-1].reset_index(drop=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df
    except Exception:
        pass

    return None

@st.cache_data(ttl=30)
def fetch_open_interest(symbol="BTCUSDT", interval="5m", limit=30):
    formatted_symbol = symbol.replace("/", "").upper()
    headers = {'User-Agent': 'Mozilla/5.0'}
    bybit_tf = BYBIT_TF_MAP.get(interval, '5m')
    
    url = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={formatted_symbol}&intervalTime={bybit_tf}&limit={limit}"
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json().get('result', {}).get('list', [])
            if data:
                df_oi = pd.DataFrame(data)
                df_oi['openInterest'] = df_oi['openInterest'].astype(float)
                df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'].astype(float), unit='ms')
                df_oi = df_oi.iloc[::-1].reset_index(drop=True)
                return df_oi
    except Exception:
        pass
    return None

# ==========================================
# 3. حساب طبقات الـ VWAP الثلاث والـ ATR
# ==========================================

def calculate_indicators(df):
    if df is None or len(df) < 52:
        return df

    # 1. حساب مؤشر ATR (14) للتطبيب بالتذبذب
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean().bfill()

    # السعر النموذجي والسيولة
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['volume']

    # 2. Session VWAP (إعادة الضبط يومياً)
    df['date'] = df['timestamp'].dt.date
    session_pv = df.groupby('date')['pv'].cumsum()
    session_vol = df.groupby('date')['volume'].cumsum()
    df['vwap_session'] = np.where(session_vol > 0, session_pv / session_vol, df['tp'])

    # 3. Weekly VWAP (إعادة الضبط أسبوعياً)
    df['week_year'] = df['timestamp'].dt.strftime('%Y-%U')
    weekly_pv = df.groupby('week_year')['pv'].cumsum()
    weekly_vol = df.groupby('week_year')['volume'].cumsum()
    df['vwap_weekly'] = np.where(weekly_vol > 0, weekly_pv / weekly_vol, df['tp'])

    # 4. Anchored VWAP (مثبت آلياً عند أدنى قاع محلي - Swing Low)
    min_idx = df['low'].idxmin()
    df_anchored = df.loc[min_idx:].copy()
    anc_pv = (df_anchored['tp'] * df_anchored['volume']).cumsum()
    anc_vol = df_anchored['volume'].cumsum()
    
    df['vwap_anchored'] = np.nan
    df.loc[min_idx:, 'vwap_anchored'] = np.where(anc_vol > 0, anc_pv / anc_vol, df_anchored['tp'])
    df['vwap_anchored'] = df['vwap_anchored'].ffill().bfill()

    # 5. Ichimoku Cloud System
    df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
    df['span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    
    # 6. Volume Ratio
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = np.where(df['vol_ma'] > 0, df['volume'] / df['vol_ma'], 1.0)
    
    return df

def calculate_single_score(df, df_oi=None):
    if df is None or len(df) < 52:
        return 50

    latest = df.iloc[-1]
    atr = max(latest['atr'], 1e-5) # تفادي القسمة على صفر
    score = 50
    
    # 1. نظام المسافة عن الـ VWAP الموزون بالـ ATR (Max ±15 pts)
    d_session = (latest['close'] - latest['vwap_session']) / atr
    d_weekly = (latest['close'] - latest['vwap_weekly']) / atr
    d_anchored = (latest['close'] - latest['vwap_anchored']) / atr
    
    delta_session = np.clip(d_session * 2.5, -5.0, 5.0)
    delta_weekly = np.clip(d_weekly * 2.5, -5.0, 5.0)
    delta_anchored = np.clip(d_anchored * 2.5, -5.0, 5.0)
    
    score += (delta_session + delta_weekly + delta_anchored)
        
    # 2. Ichimoku Cloud
    if pd.notna(latest['span_a']) and pd.notna(latest['span_b']):
        cloud_max = max(latest['span_a'], latest['span_b'])
        cloud_min = min(latest['span_a'], latest['span_b'])
        if latest['close'] > cloud_max:
            score += 15
        elif latest['close'] < cloud_min:
            score -= 15
            
    # 3. TK Cross
    if pd.notna(latest['tenkan']) and pd.notna(latest['kijun']):
        if latest['tenkan'] > latest['kijun']:
            score += 10
        else:
            score -= 10
            
    # 4. Volume Surge
    if latest['vol_ratio'] > 1.20:
        if latest['close'] > latest['open']:
            score += 10
        elif latest['close'] < latest['open']:
            score -= 10
            
    # 5. Open Interest Logic
    if df_oi is not None and len(df_oi) >= 10:
        latest_oi = df_oi.iloc[-1]['openInterest']
        prev_oi = df_oi.iloc[-10]['openInterest']
        
        price_change_pct = ((df.iloc[-1]['close'] - df.iloc[-10]['close']) / df.iloc[-10]['close']) * 100
        oi_change_pct = ((latest_oi - prev_oi) / prev_oi) * 100
        
        if oi_change_pct > 1.0:
            if price_change_pct > 0:
                score += 10
            else:
                score -= 10
        elif oi_change_pct < -1.0:
            if price_change_pct > 0:
                score -= 5
            else:
                score += 5
            
    return int(np.clip(score, 0, 100))

def get_global_multi_tf_analysis(symbol):
    tf_scores = {}
    tf_vwaps = {}
    weighted_sum = 0.0
    
    for tf, weight in TIMEFRAME_WEIGHTS.items():
        df_raw = fetch_klines_data(symbol, interval=tf)
        df_oi = fetch_open_interest(symbol, interval=tf)
        df_calc = calculate_indicators(df_raw)
        
        score = calculate_single_score(df_calc, df_oi)
        
        tf_scores[tf] = score
        if df_calc is not None and not df_calc.empty:
            tf_vwaps[tf] = df_calc.iloc[-1]['vwap_session']
        else:
            tf_vwaps[tf] = 0.0
            
        weighted_sum += score * weight
        
    global_score = round(weighted_sum, 1)
    d_score = tf_scores.get('1d', 50)
    h4_score = tf_scores.get('4h', 50)
    
    if global_score >= 70 and d_score >= 60 and h4_score >= 60:
        master_signal = "🟢 SUPER STRONG LONG"
        status_desc = "توافق صاعد تام عبر جميع الأطر الزمنية والـ 3-VWAP Suite."
    elif global_score <= 30 and d_score <= 40 and h4_score <= 40:
        master_signal = "🔴 SUPER STRONG SHORT"
        status_desc = "توافق هابط تام عبر جميع الأطر الزمنية."
    elif global_score >= 65 and (d_score < 50 or h4_score < 50):
        master_signal = "⚠️ SCALP LONG (Counter-Trend)"
        status_desc = "صعود قصير الأجل على الصغرى عكس اتجاه اليومي."
    elif global_score <= 35 and (d_score > 50 or h4_score > 50):
        master_signal = "⚠️ SCALP SHORT (Counter-Trend)"
        status_desc = "هبوط قصير الأجل على الصغرى عكس اتجاه اليومي."
    else:
        master_signal = "🟡 NEUTRAL / CONFLICT"
        status_desc = "تضارب في الأطر الزمنية والسيولة - يفضل تقليل المخاطرة."
        
    return {
        'global_score': global_score,
        'master_signal': master_signal,
        'status_desc': status_desc,
        'tf_scores': tf_scores,
        'tf_vwaps': tf_vwaps
    }

# ==========================================
# 4. القائمة الجانبية (Sidebar)
# ==========================================

st.sidebar.title("⚡ AliQuantFund")
st.sidebar.caption("Institutional 3-VWAP Suite")
st.sidebar.markdown("---")

selected_symbol = st.sidebar.selectbox(
    "اختر العملة للتحليل العميق:",
    ["BTC/USDT", "ETH/USDT", "ZEC/USDT", "XRP/USDT"]
)

selected_tf = st.sidebar.selectbox(
    "الإطار الزمني للرسم البياني:",
    ["5m", "15m", "1h", "4h", "1d"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 خيارات رأس المال")

capital = st.sidebar.number_input("رأس المال الإجمالي ($):", value=100.0, step=10.0)
base_risk_pct = st.sidebar.number_input("المخاطرة المستهدفة القصوى (%):", value=2.0, step=0.5)

# ==========================================
# 5. الواجهة الرئيسية
# ==========================================

st.title(f"📊 التحليل الكمي المركب (3-VWAP Suite): {selected_symbol}")

global_res = get_global_multi_tf_analysis(selected_symbol)

st.markdown("### 🌐 التوصية العامة الموحدة (Multi-TF & 3-VWAP Confluence)")

g_col1, g_col2 = st.columns([1, 2])

with g_col1:
    st.metric(
        label="التقييم المركب الموحد (Global Score)",
        value=f"{global_res['global_score']} / 100",
        delta=global_res['master_signal']
    )
    st.info(f"💡 **الحالة:** {global_res['status_desc']}")

with g_col2:
    st.write("📊 **تقييم الأطر الزمنية الخمسة:**")
    tf_cols = st.columns(5)
    for idx, (tf_key, sc) in enumerate(global_res['tf_scores'].items()):
        color = "🟢" if sc >= 65 else ("🔴" if sc <= 35 else "🟡")
        tf_cols[idx].metric(label=tf_key.upper(), value=f"{sc}", delta=color)

st.markdown("---")

# --- الشارت وحاسبة المسافات ---
df_data = fetch_klines_data(selected_symbol, interval=selected_tf)
df_oi_data = fetch_open_interest(selected_symbol, interval=selected_tf)
df_calc = calculate_indicators(df_data)

if df_calc is not None and not df_calc.empty:
    latest = df_calc.iloc[-1]
    
    col_chart, col_signal = st.columns([3, 1])
    
    with col_signal:
        st.markdown("### 🎯 بطاقة المسافات وإدارة المخاطر")
        st.write(f"**الأصل الحالي:** {selected_symbol} ({selected_tf})")
        
        # عرض المسافات بوحدة الـ ATR
        atr_val = latest['atr']
        d_sess = (latest['close'] - latest['vwap_session']) / atr_val
        d_week = (latest['close'] - latest['vwap_weekly']) / atr_val
        d_anc = (latest['close'] - latest['vwap_anchored']) / atr_val
        
        st.markdown("#### 📏 المسافة عن الـ VWAPs (بـ ATR):")
        st.caption(f"• **Session VWAP:** `{d_sess:+.2f} ATR`")
        st.caption(f"• **Weekly VWAP:** `{d_week:+.2f} ATR`")
        st.caption(f"• **Anchored VWAP:** `{d_anc:+.2f} ATR`")
        
        st.markdown("---")
        
        g_score = global_res['global_score']
        if g_score >= 75 or g_score <= 25:
            risk_multiplier = 1.0
            risk_status = "🔥 توافق كامل (مخاطرة 100%)"
        elif (60 <= g_score < 75) or (25 < g_score <= 40):
            risk_multiplier = 0.5
            risk_status = "⚠️ توافق جزئي (مخاطرة 50%)"
        else:
            risk_multiplier = 0.25
            risk_status = "🛑 منطقة حيرة (مخاطرة 25%)"

        effective_risk_pct = base_risk_pct * risk_multiplier
        risk_amount = capital * (effective_risk_pct / 100)
        
        st.warning(f"**المخاطرة الديناميكية:** `{effective_risk_pct:.2f}%` (${risk_amount:.2f})\n\n*{risk_status}*")
        st.markdown("---")

        entry_price = st.number_input("سعر الدخول:", value=float(latest['close']))

        sl_source = st.selectbox(
            "مصدر وقف الخسارة (VWAP Layer):",
            ["Session VWAP", "Weekly VWAP", "Anchored VWAP", "مخصص"],
            index=0
        )

        if sl_source == "Session VWAP":
            selected_sl_val = float(latest['vwap_session'])
        elif sl_source == "Weekly VWAP":
            selected_sl_val = float(latest['vwap_weekly'])
        elif sl_source == "Anchored VWAP":
            selected_sl_val = float(latest['vwap_anchored'])
        else:
            selected_sl_val = float(latest['vwap_session'])

        sl_price = st.number_input("وقف الخسارة (SL):", value=float(selected_sl_val))

        is_long = entry_price >= sl_price
        sl_distance = abs(entry_price - sl_price)

        if is_long:
            tp1_default = entry_price + (sl_distance * 1.5)
            tp2_default = entry_price + (sl_distance * 3.0)
        else:
            tp1_default = entry_price - (sl_distance * 1.5)
            tp2_default = entry_price - (sl_distance * 3.0)

        tp1_price = st.number_input("الهدف الأول (TP1 - 1:1.5):", value=float(tp1_default))
        tp2_price = st.number_input("الهدف الثاني (TP2 - 1:3.0):", value=float(tp2_default))

        if sl_distance > 0:
            units = risk_amount / sl_distance
            pos_value = units * entry_price

            st.markdown("---")
            st.markdown(f"• **الاتجاه:** `{'🟢 شراء (Long)' if is_long else '🔴 بيع (Short)'}`")
            st.caption(f"• **حجم العقود (Units):** `{units:.4f}`")
            st.caption(f"• **قيمة العقد:** `${pos_value:.2f}`")
            st.caption(f"• **الهدف 1 (تأمين):** `${tp1_price:.2f}` (1:1.5)")
            st.caption(f"• **الهدف 2 (مؤسسي):** `${tp2_price:.2f}` (1:3.0)")

    with col_chart:
        fig = go.Figure()

        # 1. الشموع اليابانية
        fig.add_trace(go.Candlestick(
            x=df_calc['timestamp'],
            open=df_calc['open'],
            high=df_calc['high'],
            low=df_calc['low'],
            close=df_calc['close'],
            name='Price'
        ))

        # 2. خطوط الـ VWAPs الثلاثة
        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['vwap_session'],
            mode='lines', name='Session VWAP',
            line=dict(color='gold', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['vwap_weekly'],
            mode='lines', name='Weekly VWAP',
            line=dict(color='magenta', width=2, dash='dot')
        ))

        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['vwap_anchored'],
            mode='lines', name='Anchored VWAP (Swing Low)',
            line=dict(color='cyan', width=2, dash='dash')
        ))

        # 3. خطوط الإيشيموكو
        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['tenkan'],
            mode='lines', name='Tenkan-sen',
            line=dict(color='skyblue', width=1)
        ))

        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['kijun'],
            mode='lines', name='Kijun-sen',
            line=dict(color='orange', width=1)
        ))

        fig.update_layout(
            title=f"شارت {selected_symbol} - {selected_tf} (مع طبقات VWAP الثلاث)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=600,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("⚡ AliQuantFund Engine v2.0 | Volatility-Normalized Multi-Layer VWAP Suite")
