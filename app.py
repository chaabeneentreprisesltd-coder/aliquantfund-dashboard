import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==========================================
# 1. إعدادات الصفحة والتصميم العامة
# ==========================================
st.set_page_config(
    page_title="AliQuantFund | Institutional Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص المظهر وتسهيل القراءة على الجوال
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
# 2. محرك جلب البيانات الذكي (محصن ضد الحظر)
# ==========================================

TIMEFRAME_WEIGHTS = {
    '1d': 0.35,
    '4h': 0.25,
    '1h': 0.20,
    '15m': 0.12,
    '5m': 0.08
}

BYBIT_TF_MAP = {
    '5m': '5',
    '15m': '15',
    '1h': '60',
    '4h': '240',
    '1d': 'D'
}

@st.cache_data(ttl=20)
def fetch_klines_data(symbol="BTCUSDT", interval="5m", limit=150):
    """دالة ذكية تحاول جلب البيانات من Binance وتتحول لـ Bybit عند وجود حظر 451"""
    formatted_symbol = symbol.replace("/", "").upper()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # المحاولة الأولى: سيرفرات Binance البديلة
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

    # المحاولة الثانية (Backup): خوادم Bybit
    try:
        bybit_tf = BYBIT_TF_MAP.get(interval, '5')
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
    except Exception as e:
        pass

    return None

def calculate_indicators(df):
    """حساب المؤشرات الكمية"""
    if df is None or len(df) < 52:
        return df

    # Anchored VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['tp'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Ichimoku Cloud System
    df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
    df['span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    
    # Volume Ratio
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = np.where(df['vol_ma'] > 0, df['volume'] / df['vol_ma'], 1.0)
    
    return df

def calculate_single_score(df):
    """حساب النتيجة الكمية (0-100)"""
    if df is None or len(df) < 52:
        return 50

    latest = df.iloc[-1]
    score = 50
    
    if latest['close'] > latest['vwap']:
        score += 15
    else:
        score -= 15
        
    if pd.notna(latest['span_a']) and pd.notna(latest['span_b']):
        cloud_max = max(latest['span_a'], latest['span_b'])
        cloud_min = min(latest['span_a'], latest['span_b'])
        if latest['close'] > cloud_max:
            score += 15
        elif latest['close'] < cloud_min:
            score -= 15
            
    if pd.notna(latest['tenkan']) and pd.notna(latest['kijun']):
        if latest['tenkan'] > latest['kijun']:
            score += 10
        else:
            score -= 10
            
    if latest['vol_ratio'] > 1.20:
        if latest['close'] > latest['open']:
            score += 10
        elif latest['close'] < latest['open']:
            score -= 10
            
    return int(np.clip(score, 0, 100))

def get_global_multi_tf_analysis(symbol):
    """حساب التوصية العامة الشاملة الموحدة"""
    tf_scores = {}
    weighted_sum = 0.0
    
    for tf, weight in TIMEFRAME_WEIGHTS.items():
        df_raw = fetch_klines_data(symbol, interval=tf)
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
# 3. القائمة الجانبية (Sidebar)
# ==========================================

st.sidebar.title("⚡ AliQuantFund")
st.sidebar.caption("Institutional Quantitative Engine")
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
st.sidebar.subheader("📐 حاسبة إدارة المخاطر")

capital = st.sidebar.number_input("رأس المال الإجمالي ($):", value=100.0, step=10.0)
risk_pct = st.sidebar.number_input("نسبة المخاطرة (%):", value=2.0, step=0.5)

# ==========================================
# 4. الواجهة الرئيسية
# ==========================================

st.title(f"📊 التحليل الكمي المدمج: {selected_symbol}")

# --- التوصية العامة الموحدة ---
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

# --- الشارت والتحليل الفردي ---
df_data = fetch_klines_data(selected_symbol, interval=selected_tf)
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
            st.success("🟢 التوصية: **Strong Long**")
        elif single_score <= 35:
            st.error("🔴 التوصية: **Strong Short**")
        else:
            st.warning("🟡 التوصية: **No-Trade Regime**")
            
        st.markdown("---")
        st.write(f"• **السعر الحالي:** `${latest['close']:.2f}`")
        st.write(f"• **Anchored VWAP:** `${latest['vwap']:.2f}`")
        st.write(f"• **نسبة الزخم الحجمي:** `{latest['vol_ratio']:.2f}x`")
        
        # --- حاسبة إدارة المخاطر المعدلة والتلقائية ---
        st.markdown("---")
        st.markdown("#### 🔢 الإدارة العددية للصفقة")
        
        entry_price = st.number_input("سعر الدخول:", value=float(latest['close']))
        
        # تعيين الستوب الافتراضي بناءً على التقييم الكمي (فوق أو تحت الـ VWAP)
        default_sl = float(latest['vwap'])
        sl_price = st.number_input("وقف الخسارة (SL):", value=default_sl)
        
        # تحديد اتجاه الصفقة تلقائياً
        is_long = entry_price >= sl_price
        sl_distance = abs(entry_price - sl_price)
        
        # حساب الـ TP الافتراضي الصحيح حسب الاتجاه (نسبة 1:2)
        if is_long:
            default_tp = entry_price + (sl_distance * 2)
        else:
            default_tp = entry_price - (sl_distance * 2)
            
        tp_price = st.number_input("أخذ الأرباح (TP):", value=float(default_tp))
        
        risk_amount = capital * (risk_pct / 100)
        
        if sl_distance > 0:
            units = risk_amount / sl_distance
            pos_value = units * entry_price
            
            # حساب العائد الصحيح بناءً على اتجاه الصفقة
            if is_long:
                tp_distance = tp_price - entry_price
            else:
                tp_distance = entry_price - tp_price
                
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
            
            st.markdown(f"• **نوع الصفقة:** `{'🟢 شراء (Long)' if is_long else '🔴 بيع (Short)'}`")
            st.caption(f"• **المخاطرة بالدولار:** `${risk_amount:.2f}`")
            st.caption(f"• **حجم الصفقة (Units):** `{units:.4f}`")
            st.caption(f"• **قيمة العقد الإجمالية:** `${pos_value:.2f}`")
            st.caption(f"• **نسبة العائد/المخاطرة (R:R):** `1:{rr_ratio:.2f}`")

    with col_chart:
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df_calc['timestamp'],
            open=df_calc['open'],
            high=df_calc['high'],
            low=df_calc['low'],
            close=df_calc['close'],
            name='Price'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['vwap'],
            mode='lines', name='Anchored VWAP',
            line=dict(color='gold', width=2)
        ))
        
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
st.caption("⚡ AliQuantFund Engine v1.7 | All Quantitative Rights Reserved")
