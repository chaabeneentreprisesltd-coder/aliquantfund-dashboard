
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime

# ---------------------------------------------------------
# 1. تهيئة إعدادات الصفحة
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
# 2. إدارة ذاكرة النظام لتتبع الإشارات لجميع العملات
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
    "BTC/USDT": {"cg_id": "bitcoin", "symbol_us": "BTCUSDT"},
    "ETH/USDT": {"cg_id": "ethereum", "symbol_us": "ETHUSDT"},
    "ZEC/USDT": {"cg_id": "zcash", "symbol_us": "ZECUSDT"},
    "XRP/USDT": {"cg_id": "ripple", "symbol_us": "XRPUSDT"}
}

selected_display_symbol = st.sidebar.selectbox("اختر العملة للتحليل العميق:", list(SYMBOLS_MAP.keys()), index=0)

timeframe = st.sidebar.selectbox("الإطار الزمني (Timeframe):", ["5m", "15m", "1h", "4h", "1d"], index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 إعدادات التحديث ومراقبة السحابة")
auto_refresh = st.sidebar.checkbox("تفعيل التحديث التلقائي ومراقبة كل الأزواج", value=True)
refresh_seconds = st.sidebar.slider("معدل التحديث (بالثواني):", min_value=15, max_value=120, value=30, step=5)

auto_alerts_enabled = st.sidebar.toggle("🚨 تفعيل التنبيهات الآلية الشاملة", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ اختيارات التنبيه اليدوي")

if st.sidebar.button("🔔 اختبار إرسال تنبيه تجريبي"):
    test_msg = f"⚡ *AliQuantFund - Multi-Asset Scanner Test*\n\nالعملة: `{selected_display_symbol}`\nالإطار الزمني: `{timeframe}`\nالحالة: الاتصال الشامل بنظام التنبيهات نشط 🚀"
    success, msg = send_telegram_alert(test_msg)
    if success:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

# ---------------------------------------------------------
# 4. جلب البيانات من Binance.US / CoinGecko
# ---------------------------------------------------------
@st.cache_data(ttl=10)
def fetch_tickers_live():
    tickers = {}
    url = "https://api.binance.us/api/v3/ticker/24hr"
    
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            data_dict = {item['symbol']: item for item in data if isinstance(item, dict)}
            for display_name, info in SYMBOLS_MAP.items():
                sym = info["symbol_us"]
                if sym in data_dict:
                    tickers[display_name] = {
                        'price': float(data_dict[sym]['lastPrice']),
                        'change': float(data_dict[sym]['priceChangePercent'])
                    }
            if len(tickers) == len(SYMBOLS_MAP):
                return tickers
    except Exception:
        pass

    try:
        ids = ",".join([info["cg_id"] for info in SYMBOLS_MAP.values()])
        cg_url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true"
        res = requests.get(cg_url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            for display_name, info in SYMBOLS_MAP.items():
                cid = info["cg_id"]
                if cid in data:
                    tickers[display_name] = {
                        'price': float(data[cid].get('usd', 0.0)),
                        'change': float(data[cid].get('usd_24h_change', 0.0))
                    }
            return tickers
    except Exception:
        pass

    return {k: {'price': 0.0, 'change': 0.0} for k in SYMBOLS_MAP.keys()}

@st.cache_data(ttl=15)
def fetch_market_data_live(display_symbol, tf):
    sym_us = SYMBOLS_MAP[display_symbol]["symbol_us"]
    url = f"https://api.binance.us/api/v3/klines?symbol={sym_us}&interval={tf}&limit=120"
    
    try:
        res = requests.get(url, timeout=6)
        if res.status_code == 200:
            raw_data = res.json()
            df = pd.DataFrame(raw_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'num_trades', 'tb_base', 'tb_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
        else:
            raise Exception("Klines request failed")
    except Exception:
        cid = SYMBOLS_MAP[display_symbol]["cg_id"]
        cg_ohlc_url = f"https://api.coingecko.com/api/v3/coins/{cid}/ohlc?vs_currency=usd&days=1"
        res = requests.get(cg_ohlc_url, timeout=6)
        raw_data = res.json()
        df = pd.DataFrame(raw_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['volume'] = 1000.0
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

    # Indicators calculation
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vp'] = df['typical_price'] * df['volume']
    cum_vol = df['volume'].cumsum()
    df['vwap'] = np.where(cum_vol > 0, df['vp'].cumsum() / cum_vol, df['close'])
    
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
    
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    return df

def calculate_quant_score(df):
    last_price = df['close'].iloc[-1]
    last_vwap = df['vwap'].iloc[-1]
    last_tenkan = df['tenkan_sen'].iloc[-1] if not pd.isna(df['tenkan_sen'].iloc[-1]) else last_price
    last_kijun = df['kijun_sen'].iloc[-1] if not pd.isna(df['kijun_sen'].iloc[-1]) else last_price
    span_a = df['senkou_span_a'].iloc[-1] if not pd.isna(df['senkou_span_a'].iloc[-1]) else last_price
    span_b = df['senkou_span_b'].iloc[-1] if not pd.isna(df['senkou_span_b'].iloc[-1]) else last_price
    last_vol = df['volume'].iloc[-1]
    avg_vol = df['vol_ma'].iloc[-1] if not pd.isna(df['vol_ma'].iloc[-1]) else last_vol
    
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
    
    if score >= 65: rec = "🟢 Strong Long"
    elif score <= 35: rec = "🔴 Strong Short"
    else: rec = "🟡 No-Trade Regime"
    
    return score, rec, last_price, last_vwap, vol_ratio

# ---------------------------------------------------------
# 5. واجهة مركز القيادة والسيطرة
# ---------------------------------------------------------
st.title("⚡ AliQuantFund (Control Center)")
st.caption(f"تحديث أخير: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

tickers_data = fetch_tickers_live()
cols = st.columns(4)

for i, sym in enumerate(SYMBOLS_MAP.keys()):
    short_name = sym.split('/')[0]
    p = tickers_data[sym]['price']
    c = tickers_data[sym]['change']
    delta_color = "normal" if c >= 0 else "inverse"
    cols[i].metric(label=short_name, value=f"${p:,.2f}", delta=f"{c:+.2f}%", delta_color=delta_color)

st.markdown("---")

# ---------------------------------------------------------
# 6. فاحص جميع الأصول الشامل (Multi-Asset Auto Scanner)
# ---------------------------------------------------------
if auto_alerts_enabled:
    for sym_check in SYMBOLS_MAP.keys():
        try:
            df_check = fetch_market_data_live(sym_check, timeframe)
            score_chk, rec_chk, price_chk, vwap_chk, vol_chk = calculate_quant_score(df_check)
            prev_sig = st.session_state.last_signal.get(sym_check)
            
            if prev_sig != rec_chk:
                st.session_state.last_signal[sym_check] = rec_chk
                if prev_sig is not None:
                    alert_text = (
                        f"🚨 *تنبيه تغيير النظام الكمي - AliQuantFund*\n\n"
                        f"📌 **الأصل:** `{sym_check}`\n"
                        f"⏱️ **الإطار الزمني:** `{timeframe}`\n"
                        f"🔄 **التغيير:** `{prev_sig}` ➡️ **{rec_chk}**\n"
                        f"📊 **النتيجة الكمية (Score):** `{score_chk}/100`\n"
                        f"💵 **السعر:** `${price_chk:,.2f}`\n"
                        f"🎯 **VWAP:** `${vwap_chk:,.2f}`\n"
                        f"⚡ **نسبة الفوليوم:** `{vol_chk:.2f}x`"
                    )
                    send_telegram_alert(alert_text)
                    st.toast(f"تم إرسال تنبيه آلي لـ {sym_check}: {rec_chk}", icon="🚀")
        except Exception:
            pass

df = fetch_market_data_live(selected_display_symbol, timeframe)
score, recommendation, last_price, last_vwap, vol_ratio = calculate_quant_score(df)

col_chart, col_signal = st.columns([2.2, 1])

with col_chart:
    st.subheader(f"📊 التحليل الكمي المدمج: {selected_display_symbol}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_span_a'], mode='lines', line=dict(width=0.5, color='rgba(0, 230, 118, 0.5)'), name='Span A', showlegend=False))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_span_b'], mode='lines', line=dict(width=0.5, color='rgba(255, 82, 82, 0.5)'), fill='tonexty', fillcolor='rgba(0, 230, 118, 0.08)', name='Kumo Cloud'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tenkan_sen'], mode='lines', line=dict(color='#29B6F6', width=1.5), name='Tenkan-sen'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['kijun_sen'], mode='lines', line=dict(color='#FF7043', width=1.5), name='Kijun-sen'))
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="السعر"))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], mode='lines', name='Anchored VWAP', line=dict(color='#FFEB3B', width=2)))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), height=480)
    st.plotly_chart(fig, use_container_width=True)

with col_signal:
    st.subheader("🎯 بطاقة الإشارة الكمية المركبة")
    st.info(f"الأصل: **{selected_display_symbol}**")
    st.progress(score / 100)
    st.caption(f"التقييم المركب (Composite Quant Score): **{score}/100**")
    st.markdown(f"**التوصية:** {recommendation}")
    st.markdown("---")
    st.write(f"- السعر الحالي: `${last_price:,.2f}`")
    st.write(f"- Anchored VWAP: `${last_vwap:,.2f}`")
    st.write(f"- نسبة الزخم الحجمي: `{vol_ratio:.2f}x`")

# ---------------------------------------------------------
# 7. حاسبة إدارة المخاطر
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🛡️ حاسبة إدارة المخاطر وحجم الصفقة (Risk Calculator)")
risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    capital = st.number_input("رأس مال الحساب الإجمالي ($):", min_value=10.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("نسبة المخاطرة المقبولة (%):", min_value=0.1, max_value=10.0, value=1.0, step=0.5)
    trade_type = st.radio("نوع اتجاه الصفقة:", ["Long 🟢", "Short 🔴"], horizontal=True)
    entry_p = st.number_input("سعر الدخول ($):", min_value=0.00001, value=float(last_price), format="%.4f")
    default_sl = entry_p * 0.98 if trade_type == "Long 🟢" else entry_p * 1.02
    default_tp = entry_p * 1.04 if trade_type == "Long 🟢" else entry_p * 0.96
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

# ---------------------------------------------------------
# 8. التحديث التلقائي
# ---------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
