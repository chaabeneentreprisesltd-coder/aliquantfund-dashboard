import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. إعدادات الصفحة والتصميم العامة
# ==========================================
st.set_page_config(
    page_title="AliQuantFund | Institutional Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص المظهر باللغة العربية والأنماط البصرية
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
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2a2e39;
    }
    .metric-card {
        background-color: #131722;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2a2e39;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. محرك جلب البيانات والحسابات الكمية
# ==========================================

TIMEFRAME_WEIGHTS = {
    '1d': 0.35,   # الاتجاه العام والسيولة الكبرى
    '4h': 0.25,   # الهيكل الرئيسي والدعوم/المقاومات
    '1h': 0.20,   # الزخم المحلي
    '15m': 0.12,  # التهيؤ للدخول
    '5m': 0.08    # التوقيت الدقيق
}

@st.cache_data(ttl=30)
def fetch_binance_klines(symbol="BTCUSDT", interval="5m", limit=150):
    """جلب بيانات الشموع مباشرة من واجهة بينانس العامة"""
    formatted_symbol = symbol.replace("/", "").upper()
    url = f"https://api.binance.com/api/v3/klines?symbol={formatted_symbol}&interval={interval}&limit={limit}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # تحويل أنواع البيانات
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df
    except Exception as e:
        st.error(f"خطأ في جلب البيانات للزوج {symbol} على فريم {interval}: {e}")
        return None

def calculate_indicators(df):
    """حساب المؤشرات الفنية والكمية الأساسية"""
    if df is None or len(df) < 52:
        return df

    # 1. Anchored VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['tp'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # 2. Ichimoku Cloud System
    # Tenkan-sen (9)
    df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    # Kijun-sen (26)
    df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    # Senkou Span A
    df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
    # Senkou Span B (52)
    df['span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    
    # 3. Volume Moving Average (20)
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = np.where(df['vol_ma'] > 0, df['volume'] / df['vol_ma'], 1.0)
    
    return df

def calculate_single_score(df):
    """حساب التقييم المركب (0-100) لإطار زمني واحد"""
    if df is None or len(df) < 52:
        return 50

    latest = df.iloc[-1]
    score = 50
    
    # VWAP Condition
    if latest['close'] > latest['vwap']:
        score += 15
    else:
        score -= 15
        
    # Ichimoku Cloud Condition
    if pd.notna(latest['span_a']) and pd.notna(latest['span_b']):
        cloud_max = max(latest['span_a'], latest['span_b'])
        cloud_min = min(latest['span_a'], latest['span_b'])
        
        if latest['close'] > cloud_max:
            score += 15
        elif latest['close'] < cloud_min:
            score -= 15
            
    # Tenkan / Kijun Cross
    if pd.notna(latest['tenkan']) and pd.notna(latest['kijun']):
        if latest['tenkan'] > latest['kijun']:
            score += 10
        else:
            score -= 10
            
    # Volume Surge
    if latest['vol_ratio'] > 1.20:
        if latest['close'] > latest['open']:
            score += 10
        elif latest['close'] < latest['open']:
            score -= 10
            
    return int(np.clip(score, 0, 100))

def get_global_multi_tf_analysis(symbol):
    """حساب التوصية العامة الموحدة المدمجة عبر الفريمات الخمسة"""
    tf_scores = {}
    weighted_sum = 0.0
    
    for tf, weight in TIMEFRAME_WEIGHTS.items():
        df_raw = fetch_binance_klines(symbol, interval=tf)
        df_calc = calculate_indicators(df_raw)
        score = calculate_single_score(df_calc)
        
        tf_scores[tf] = score
        weighted_sum += score * weight
        
    global_score = round(weighted_sum, 1)
    d_score = tf_scores.get('1d', 50)
    h4_score = tf_scores.get('4h', 50)
    
    if global_score >= 70 and d_score >= 60 and h4_score >= 60:
        master_signal = "🟢 SUPER STRONG LONG"
        status_desc = "توافق صاعد تام عبر الفريمات الكبرى والصغرى."
    elif global_score <= 30 and d_score <= 40 and h4_score <= 40:
        master_signal = "🔴 SUPER STRONG SHORT"
        status_desc = "توافق هابط تام عبر الفريمات الكبرى والصغرى."
    elif global_score >= 65 and (d_score < 50 or h4_score < 50):
        master_signal = "⚠️ SCALP LONG (Counter-Trend)"
        status_desc = "صعود قصير الأجل على الصغرى عكس اتجاه الفريم اليومي."
    elif global_score <= 35 and (d_score > 50 or h4_score > 50):
        master_signal = "⚠️ SCALP SHORT (Counter-Trend)"
        status_desc = "هبوط قصير الأجل على الصغرى عكس اتجاه الفريم اليومي."
    else:
        master_signal = "🟡 NEUTRAL / CONFLICT"
        status_desc = "تضارب بين الأطر الزمنية - يفضل عدم الدخول."
        
    return {
        'global_score': global_score,
        'master_signal': master_signal,
        'status_desc': status_desc,
        'tf_scores': tf_scores
    }

# ==========================================
# 3. القائمة الجانبية (Sidebar & Controls)
# ==========================================

st.sidebar.title("⚡ AliQuantFund")
st.sidebar.caption("Institutional Quantitative Engine")
st.sidebar.markdown("---")

selected_symbol = st.sidebar.selectbox(
    "اختر العملة للتحليل العميق:",
    ["BTC/USDT", "ETH/USDT", "ZEC/USDT", "XRP/USDT"]
)

selected_tf = st.sidebar.selectbox(
    "الإطار الزمني للرسم البياني (Timeframe):",
    ["5m", "15m", "1h", "4h", "1d"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 حاسبة إدارة المخاطر")

capital = st.sidebar.number_input("رأس المال الإجمالي ($):", value=100.0, step=10.0)
risk_pct = st.sidebar.number_input("نسبة المخاطرة (%):", value=2.0, step=0.5)

# ==========================================
# 4. الواجهة الرئيسية (Main Dashboard)
# ==========================================

st.title(f"📊 التحليل الكمي المدمج: {selected_symbol}")

# --- القسم الأول: التوصية العامة الموحدة (Global Master Recommendation) ---
st.markdown("### 🌐 التوصية العامة الموحدة (Multi-Timeframe Master Confluence)")

global_res = get_global_multi_tf_analysis(selected_symbol)

g_col1, g_col2 = st.columns([1, 2])

with g_col1:
    st.metric(
        label="التقييم المركب الموحد (Global Score)",
        value=f"{global_res['global_score']} / 100",
        delta=global_res['master_signal']
    )
    st.info(f"💡 **الحالة:** {global_res['status_desc']}")

with g_col2:
    st.write("📊 **درجات التقييم حسب الأطر الزمنية الخمسة:**")
    tf_cols = st.columns(5)
    for idx, (tf_key, sc) in enumerate(global_res['tf_scores'].items()):
        color = "🟢" if sc >= 65 else ("🔴" if sc <= 35 else "🟡")
        tf_cols[idx].metric(label=tf_key.upper(), value=f"{sc}", delta=color)

st.markdown("---")

# --- القسم الثاني: تحليل الفريم المحدد والشارت ---
df_data = fetch_binance_klines(selected_symbol, interval=selected_tf)
df_calc = calculate_indicators(df_data)

if df_calc is not None and not df_calc.empty:
    latest = df_calc.iloc[-1]
    single_score = calculate_single_score(df_calc)
    
    col_chart, col_signal = st.columns([3, 1])
    
    with col_signal:
        st.markdown("### 🎯 بطاقة الإشارة اللحظية")
        st.write(f"**الأصل:** {selected_symbol} ({selected_tf})")
        
        st.progress(single_score / 100)
        st.write(f"**التقييم الكمي للفريم:** `{single_score}/100`")
        
        if single_score >= 65:
            st.success("🔴 التوصية: **Strong Long**")
        elif single_score <= 35:
            st.error("🔴 التوصية: **Strong Short**")
        else:
            st.warning("🟡 التوصية: **No-Trade Regime**")
            
        st.markdown("---")
        st.write(f"• **السعر الحالي:** `${latest['close']:.2f}`")
        st.write(f"• **Anchored VWAP:** `${latest['vwap']:.2f}`")
        st.write(f"• **نسبة الزخم الحجمي:** `{latest['vol_ratio']:.2f}x`")
        
        # حاسبة أسعار الدخول والستوب
        st.markdown("---")
        st.markdown("#### 🔢 الإدارة العددية للصفقة")
        entry_price = st.number_input("سعر الدخول:", value=float(latest['close']))
        sl_price = st.number_input("وقف الخسارة (SL):", value=float(latest['vwap']))
        tp_price = st.number_input("أخذ الأرباح (TP):", value=float(entry_price + (abs(entry_price - sl_price) * 2)))
        
        # معادلات حاسبة المخاطر
        risk_amount = capital * (risk_pct / 100)
        sl_distance = abs(entry_price - sl_price)
        
        if sl_distance > 0:
            units = risk_amount / sl_distance
            pos_value = units * entry_price
            rr_ratio = abs(tp_price - entry_price) / sl_distance
            
            st.caption(f"• **المخاطرة بالدولار:** `${risk_amount:.2f}`")
            st.caption(f"• **حجم الصفقة (Units):** `{units:.4f}`")
            st.caption(f"• **قيمة العقد:** `${pos_value:.2f}`")
            st.caption(f"• **نسبة العائد/المخاطرة:** `1:{rr_ratio:.2f}`")

    with col_chart:
        # رسم الشارت التفاعلي باستخدام Plotly
        fig = go.Figure()
        
        # الشموع اليابانية
        fig.add_trace(go.Candlestick(
            x=df_calc['timestamp'],
            open=df_calc['open'],
            high=df_calc['high'],
            low=df_calc['low'],
            close=df_calc['close'],
            name='Price'
        ))
        
        # خط VWAP
        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['vwap'],
            mode='lines', name='Anchored VWAP',
            line=dict(color='gold', width=2)
        ))
        
        # خطوط الإيشيموكو
        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['tenkan'],
            mode='lines', name='Tenkan-sen',
            line=dict(color='skyblue', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['kijun'],
            mode='lines', name='Kijun-sen',
            line=dict(color='orange', width=1.5)
        ))
        
        fig.update_layout(
            title=f"شارت {selected_symbol} - {selected_tf}",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=600,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(" AliQuantFund Engine v1.5 | All Quantitative Rights Reserved")

