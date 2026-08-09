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
        response = requests.post(url, json=payload, timeout=15)
        res_data = response.json()
        if response.status_code == 200 and res_data.get("ok"):
            return True, "تم إرسال التنبيه بنجاح إلى تليجرام"
        else:
            desc = res_data.get("description", response.text)
            return False, f"فشل الإرسال ({response.status_code}): {desc}"
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
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "ZEC/USDT": "ZECUSDT",
    "XRP/USDT": "XRPUSDT"
}

selected_display_symbol = st.sidebar.selectbox("اختر العملة للتحليل العميق:", list(SYMBOLS_MAP.keys()), index=0)
selected_symbol = SYMBOLS_MAP[selected_display_symbol]

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
# 4. جلب البيانات المباشرة عبر HTTP Requests (مقاوم للـ Rate Limit)
# ---------------------------------------------------------
@st.cache_data(ttl=15)
def fetch_tickers_fast():
    """جلب الأسعار مباشرة عبر API خفيف وقادر على العمل على Cloud"""
    tickers = {}
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            data_dict = {item['symbol']: item for item in data}
            for display, sym in SYMBOLS_MAP.items():
                if sym in data_dict:
                    tickers[display] = {
                        'price': float(data_dict[sym]['lastPrice']),
                        'change': float(data_dict[sym]['priceChangePercent'])
                    }
                else:
                    tickers[display] = {'price': 0.0, 'change': 0.0}
            return tickers
    except Exception:
        pass

    # Backup Endpoint in case of US Cloud block
    try:
        url_alt = "https://api.bybit.com/v5/market/tickers?category=spot"
        res = requests.get(url_alt, timeout=5)
        if res.status_code == 200:
            data = res.json().get('result', {}).get('list', [])
            data_dict = {item['symbol']: item for item in data}
            for display, sym in SYMBOLS_MAP.items():
                if sym in data_dict:
                    last_p = float(data_dict[sym]['lastPrice'])
                    prev_p = float(data_dict[sym].get('prevPrice24h', last_p))
                    chg = ((last_p - prev_p) / prev_p * 100) if prev_p > 0 else 0.0
                    tickers[display] = {'price': last_p, 'change': chg}
                else:
                    tickers[display] = {'price': 0.0, 'change': 0.0}
            return tickers
    except Exception:
        pass

    for display in SYMBOLS_MAP.keys():
        tickers[display] = {'price': 0.0, 'change': 0.0}
    return tickers

@st.cache_data(ttl=15)
def fetch_market_data_fast(symbol, tf):
    """جلب الشموع مباشرة عبر HTTP دون تحميل أسواق المنصة الكلية"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit=120"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            raw_data = res.json()
            df = pd.DataFrame(raw_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'num_trades', 'tb_base', 'tb_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
        else:
            raise Exception("Primary API unreachable")
    except Exception:
        # Fallback to Bybit
        tf_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
        bybit_tf = tf_map.get(tf, "60")
        url_bybit = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={bybit_tf}&limit=120"
        res = requests.get(url_bybit, timeout=5)
        raw_data = res.json().get('result', {}).get('list', [])
        raw_data.reverse() # Bybit returns newest first
        df = pd.DataFrame(raw_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

    # Anchored VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vp'] = df['typical_price'] * df['volume']
    df['vwap'] = df['vp'].cumsum() / df['volume'].cumsum()
    
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

tickers_data = fetch_tickers_fast()
cols = st.columns(4)

for i, sym in enumerate(SYMBOLS_MAP.keys()):
    short_name = sym.split('/')[0]
    p = tickers_data[sym]['price']
    c = tickers_data[sym]['change']
    delta_color = "normal" if c >= 0 else "inverse"
    cols[i].metric(label=short_name, value=f"${p:,.2f}", delta=f"{c:+.2f}%", delta_color=delta_color)

st.markdown("---")

df = fetch_market_data_fast(selected_symbol, timeframe)

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
    last_tenkan = df['tenkan_sen'].iloc[-1]
    last_kijun = df['kijun_sen'].iloc[-1]
    span_a = df['senkou_span_a'].iloc[-1] if not pd.isna(df['senkou_span_a'].iloc[-1]) else last_price
    span_b = df['senkou_span_b'].iloc[-1] if not pd.isna(df['senkou_span_b'].iloc[-1]) else last_price
    last_vol = df['volume'].iloc[-1]
    avg_vol = df['vol_ma'].iloc[-1]
    
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
