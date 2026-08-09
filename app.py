
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime

# ---------------------------------------------------------
# 1. تهيئة إعدادات الصفحة واسم التطبيق
# ---------------------------------------------------------
st.set_page_config(
    page_title="AliQuantFund | Quant Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# بيانات البوت المثبتة
TELEGRAM_BOT_TOKEN = "8766369440:AAHNpsegmHzQCxWdELXsiR8PUOyj_jPMV-g"
TELEGRAM_CHAT_ID = "6852370388"

def send_telegram_alert(message):
    token = TELEGRAM_BOT_TOKEN.strip().replace(" ", "")
    chat_id = TELEGRAM_CHAT_ID.strip().replace(" ", "")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200 and response.json().get("ok"):
            return True, "تم إرسال التنبيه بنجاح إلى تليجرام"
        else:
            return False, "فشل إرسال التنبيه إلى تليجرام"
    except Exception as e:
        return False, f"خطأ في الاتصال: {e}"

# ---------------------------------------------------------
# 2. إدارة ذاكرة النظام لتتبع الإشارات
# ---------------------------------------------------------
if "last_signal" not in st.session_state:
    st.session_state.last_signal = {}

# ---------------------------------------------------------
# 3. الشريط الجانبي (Sidebar)
# ---------------------------------------------------------
st.sidebar.title("⚡ AliQuantFund")
st.sidebar.caption("Institutional Quantitative Engine")

st.sidebar.markdown("---")

SYMBOLS_MAP = {
    "BTC/USDT": {"fsym": "BTC", "base_price": 65000.0},
    "ETH/USDT": {"fsym": "ETH", "base_price": 3500.0},
    "ZEC/USDT": {"fsym": "ZEC", "base_price": 42.0},
    "XRP/USDT": {"fsym": "XRP", "base_price": 0.58}
}

selected_display_symbol = st.sidebar.selectbox("اختر العملة للتحليل العميق:", list(SYMBOLS_MAP.keys()), index=0)

timeframe = st.sidebar.selectbox("الإطار الزمني (Timeframe):", ["5m", "15m", "1h", "4h", "1d"], index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 إعدادات التحديث")
auto_refresh = st.sidebar.checkbox("تفعيل التحديث التلقائي الشامل", value=False)
refresh_seconds = st.sidebar.slider("معدل التحديث (بالثواني):", min_value=15, max_value=120, value=30, step=5)

auto_alerts_enabled = st.sidebar.toggle("🚨 تفعيل التنبيهات الآلية التلقائية", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ اختيارات التنبيه اليدوي")

if st.sidebar.button("🔔 اختبار إرسال تنبيه تجريبي"):
    test_msg = f"⚡ *AliQuantFund - Ticker Test*\n\nالعملة: `{selected_display_symbol}`\nالإطار الزمني: `{timeframe}`\nالحالة: الاتصال بالبوت نشط وناجح 🚀"
    success, msg = send_telegram_alert(test_msg)
    if success:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

# ---------------------------------------------------------
# 4. جلب البيانات عبر واجهات آمنة الخلو من الحظر والإسقاط
# ---------------------------------------------------------
@st.cache_data(ttl=15)
def fetch_tickers_cloud_safe():
    tickers = {}
    fsyms = ",".join([info["fsym"] for info in SYMBOLS_MAP.values()])
    url = f"https://min-api.cryptocompare.com/data/pricemultifull?fsyms={fsyms}&tsyms=USDT"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            raw_data = data.get("RAW", {})
            for display_name, info in SYMBOLS_MAP.items():
                fsym = info["fsym"]
                if fsym in raw_data and "USDT" in raw_data[fsym]:
                    coin_info = raw_data[fsym]["USDT"]
                    tickers[display_name] = {
                        'price': float(coin_info.get("PRICE", info["base_price"])),
                        'change': float(coin_info.get("CHANGEPCT24HOUR", 0.0))
                    }
                else:
                    tickers[display_name] = {'price': info["base_price"], 'change': 0.0}
            return tickers
    except Exception:
        pass

    for display_name, info in SYMBOLS_MAP.items():
        tickers[display_name] = {'price': info["base_price"], 'change': 0.0}
    return tickers

@st.cache_data(ttl=20)
def fetch_market_data_cloud_safe(display_symbol, tf):
    fsym = SYMBOLS_MAP[display_symbol]["fsym"]
    base_p = SYMBOLS_MAP[display_symbol]["base_price"]
    headers = {"User-Agent": "Mozilla/5.0"}
    
    tf_mapping = {
        "5m": ("histominute", 5, "5min"),
        "15m": ("histominute", 15, "15min"),
        "1h": ("histohour", 1, "1h"),
        "4h": ("histohour", 4, "4h"),
        "1d": ("histoday", 1, "1D")
    }
    
    endpoint, aggregate, pd_freq = tf_mapping.get(tf, ("histohour", 1, "1h"))
    url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}?fsym={fsym}&tsym=USDT&limit=120&aggregate={aggregate}"
    
    df = None
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            candle_data = data.get("Data", {}).get("Data", [])
            if candle_data:
                df = pd.DataFrame(candle_data)
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df.rename(columns={'volumeto': 'volume'}, inplace=True)
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
    except Exception:
        pass

    # Safe Fallback in case of network issues
    if df is None or df.empty:
        now = pd.Timestamp.now()
        timestamps = pd.date_range(end=now, periods=120, freq=pd_freq)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': [base_p] * 120,
            'high': [base_p * 1.005] * 120,
            'low': [base_p * 0.995] * 120,
            'close': [base_p] * 120,
            'volume': [1000.0] * 120
        })

    # Anchored VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vp'] = df['typical_price'] * df['volume']
    cum_vol = df['volume'].cumsum()
    df['vwap'] = np.where(cum_vol > 0, df['vp'].cumsum() / cum_vol, df['close'])
    
    # Ichimoku Cloud
    nine_high = df['high'].rolling(window=9).max()
    nine_low = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (nine_high + nine_low) / 2

    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    
    return df

# ---------------------------------------------------------
# 5. واجهة مركز القيادة والسيطرة
# ---------------------------------------------------------
st.title("⚡ AliQuantFund (Control Center)")
st.caption(f"تحديث أخير: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

tickers_data = fetch_tickers_cloud_safe()
cols = st.columns(4)

for i, sym in enumerate(SYMBOLS_MAP.keys()):
    short_name = sym.split('/')[0]
    p = tickers_data[sym]['price']
    c = tickers_data[sym]['change']
    delta_color = "normal" if c >= 0 else "inverse"
    cols[i].metric(label=short_name, value=f"${p:,.2f}", delta=f"{c:+.2f}%", delta_color=delta_color)

st.markdown("---")

df = fetch_market_data_cloud_safe(selected_display_symbol, timeframe)

col_chart, col_signal = st.columns([2.2, 1])

with col_chart:
    st.subheader(f"📊 التحليل الكمي المدمج: {selected_display_symbol}")
    
    fig = go.Figure()
    
    # Ichimoku Cloud
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_span_a'], mode='lines', line=dict(width=0.5, color='rgba(0, 230, 118, 0.5)'), name='Span A', showlegend=False))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_span_b'], mode='lines', line=dict(width=0.5, color='rgba(255, 82, 82, 0.5)'), fill='tonexty', fillcolor='rgba(0, 230, 118, 0.08)', name='Kumo Cloud'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tenkan_sen'], mode='lines', line=dict(color='#29B6F6', width=1.5), name='Tenkan-sen'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['kijun_sen'], mode='lines', line=dict(color='#FF7043', width=1.5), name='Kijun-sen'))
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="السعر"
    ))
    
    # Anchored VWAP
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['vwap'],
        mode='lines', name='Anchored VWAP',
        line=dict(color='#FFEB3B', width=2)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

with col_signal:
    st.subheader("🎯 بطاقة الإشارة الكمية المركبة")
    
    last_price = df['close'].iloc[-1]
    last_vwap = df['vwap'].iloc[-1]
    last_tenkan = df['tenkan_sen'].iloc[-1] if not pd.isna(df['tenkan_sen'].iloc[-1]) else last_price
    last_kijun = df['kijun_sen'].iloc[-1] if not pd.isna(df['kijun_sen'].iloc[-1]) else last_price
    span_a = df['senkou_span_a'].iloc[-1] if not pd.isna(df['senkou_span_a'].iloc[-1]) else last_price
    span_b = df['senkou_span_b'].iloc[-1] if not pd.isna(df['senkou_span_b'].iloc[-1]) else last_price
    last_vol = df['volume'].iloc[-1]
    avg_vol = df['vol_ma'].iloc[-1] if not pd.isna(df['vol_ma'].iloc[-1]) else last_vol
    
    # حساب Composite Quant Score
    score = 50
    if last_price > last_vwap: score += 15
    else: score -= 15
    
    cloud_max = max(span_a, span_b)
    cloud_min = min(span_a, span_b)
    if last_price > cloud_max: score += 15
    elif last_price < cloud_min: score -= 15
    
    if last_tenkan > last_kijun: score += 10
    else: score -= 10
    
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
    if vol_ratio > 1.2:
        if last_price > df['open'].iloc[-1]: score += 10
        else: score -= 10
        
    score = int(np.clip(score, 0, 100))
    
    st.info(f"الأصل: **{selected_display_symbol}**")
    st.progress(score / 100)
    st.caption(f"التقييم المركب (Composite Quant Score): **{score}/100**")
    
    if score >= 65:
        recommendation = "🟢 Strong Long"
    elif score <= 35:
        recommendation = "🔴 Strong Short"
    else:
        recommendation = "🟡 No-Trade Regime"
        
    st.markdown(f"**التوصية:** {recommendation}")
    st.markdown("---")
    st.write(f"- السعر الحالي: `${last_price:,.2f}`")
    st.write(f"- Anchored VWAP: `${last_vwap:,.2f}`")
    st.write(f"- نسبة الزخم الحجمي: `{vol_ratio:.2f}x`")

    # ---------------------------------------------------------
    # 6. فحص وأتمتة التنبيهات التلقائية (Auto-Alert Dispatcher)
    # ---------------------------------------------------------
    if auto_alerts_enabled:
        prev_signal = st.session_state.last_signal.get(selected_display_symbol)
        
        if prev_signal != recommendation:
            st.session_state.last_signal[selected_display_symbol] = recommendation
            
            if prev_signal is not None:
                alert_text = (
                    f"🚨 *تنبيه تغيير النظام الكمي - AliQuantFund*\n\n"
                    f"📌 **الأصل:** `{selected_display_symbol}`\n"
                    f"⏱️ **الإطار الزمني:** `{timeframe}`\n"
                    f"🔄 **التغيير:** `{prev_signal}` ➡️ **{recommendation}**\n"
                    f"📊 **النتيجة الكمية (Score):** `{score}/100`\n"
                    f"💵 **السعر:** `${last_price:,.2f}`\n"
                    f"🎯 **VWAP:** `${last_vwap:,.2f}`\n"
                    f"⚡ **نسبة الفوليوم:** `{vol_ratio:.2f}x`"
                )
                success, alert_msg = send_telegram_alert(alert_text)
                if success:
                    st.toast(f"تم إرسال تنبيه آلي: {recommendation}", icon="🚀")

# ---------------------------------------------------------
# 7. حاسبة إدارة المخاطر وحجم الصفقة (Position Sizing Calculator)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🛡️ حاسبة إدارة المخاطر وحجم الصفقة (Risk Calculator)")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    capital = st.number_input("رأس مال الحساب الإجمالي ($):", min_value=10.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("نسبة المخاطرة المقبولة (%):", min_value=0.1, max_value=10.0, value=1.0, step=0.5)
    trade_type = st.radio("نوع اتجاه الصفقة:", ["Long 🟢", "Short 🔴"], horizontal=True)
    
    entry_p = st.number_input("سعر الدخول ($):", min_value=0.00001, value=float(last_price), format="%.4f")
    
    if trade_type == "Long 🟢":
        default_sl = entry_p * 0.98
        default_tp = entry_p * 1.04
    else:
        default_sl = entry_p * 1.02
        default_tp = entry_p * 0.96
        
    stop_loss = st.number_input("سعر وقف الخسارة (Stop Loss $):", min_value=0.00001, value=float(default_sl), format="%.4f")
    take_profit = st.number_input("سعر أخذ الأرباح (Take Profit $):", min_value=0.00001, value=float(default_tp), format="%.4f")

with risk_col2:
    sl_distance = abs(entry_p - stop_loss)
    tp_distance = abs(take_profit - entry_p)
    
    if sl_distance > 0:
        risk_amount = capital * (risk_pct / 100.0)
        position_units = risk_amount / sl_distance
        position_value = position_units * entry_p
        rr_ratio = tp_distance / sl_distance
        potential_profit = position_units * tp_distance
        sl_pct = (sl_distance / entry_p) * 100
        tp_pct = (tp_distance / entry_p) * 100
        
        base_asset = selected_display_symbol.split('/')[0]
        
        st.markdown("### 📊 نتائج إدارة المخاطر:")
        st.error(f"⚠️ **أقصى خسارة مسموح بها:** `${risk_amount:,.2f}` (نسبة {risk_pct}%)")
        st.info(f"🎯 **حجم الصفقة المطلوب:** `{position_units:,.4f}` {base_asset}")
        st.write(f"- **القيمة الإجمالية للصفقة (Position Value):** `${position_value:,.2f}`")
        st.write(f"- **نسبة العائد للمخاطرة (R:R Ratio):** `1:{rr_ratio:.2f}`")
        st.write(f"- **الربح المتوقع:** `${potential_profit:,.2f}` (+{tp_pct:.2f}%)")
        st.write(f"- **الخسارة عند ضرب الـ SL:** `${risk_amount:,.2f}` (-{sl_pct:.2f}%)")
        
        if rr_ratio < 1.5:
            st.warning("⚠️ نسبة العائد للمخاطرة أقل من 1:1.5. يُفضل إعادة تقييم الأهداف قبل التنفيذ.")
        else:
            st.success("✅ نسبة العائد للمخاطرة ممتازة وفق المعايير الكمية!")
    else:
        st.error("خطأ: سعر وقف الخسارة يجب ألا يكون مساوياً لسعر الدخول.")

# ---------------------------------------------------------
# 8. التحديث التلقائي للواجهة
# ---------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
