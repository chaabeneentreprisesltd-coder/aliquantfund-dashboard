import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==========================================
# 1. إعدادات الصفحة والهوية البصرية
# ==========================================
st.set_page_config(
    page_title="AliQuantFund | Institutional Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Design tokens ----
BG = "#0A0D13"
SURFACE = "#12161F"
SURFACE_2 = "#161B26"
BORDER = "#1F2531"
TEXT = "#E7EAF0"
TEXT_MUTED = "#8891A5"
TEAL = "#22D3B8"      # long / positive
ROSE = "#FB4A59"       # short / negative
AMBER = "#F5B942"      # neutral / scalp
TEAL_GLOW = "rgba(34, 211, 184, 0.18)"
ROSE_GLOW = "rgba(251, 74, 89, 0.18)"
AMBER_GLOW = "rgba(245, 185, 66, 0.16)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
        text-align: right;
    }}

    .stApp {{
        background:
            radial-gradient(1200px 600px at 15% -10%, rgba(34,211,184,0.06), transparent 60%),
            radial-gradient(900px 500px at 100% 0%, rgba(251,74,89,0.05), transparent 55%),
            {BG};
    }}

    p, span, label, div {{ word-break: break-word; }}

    .mono {{ font-family: 'JetBrains Mono', monospace; direction: ltr; unicode-bidi: plaintext; }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: #0B0F17;
        border-left: 1px solid {BORDER};
    }}
    div[data-testid="stSidebarNav"] {{ display: none; }}

    /* ---- Hide default chrome ---- */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* ---- Top identity bar ---- */
    .brand-bar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 18px 22px; margin-bottom: 22px;
        background: linear-gradient(135deg, {SURFACE} 0%, {SURFACE_2} 100%);
        border: 1px solid {BORDER}; border-radius: 16px;
    }}
    .brand-left {{ display: flex; align-items: center; gap: 14px; }}
    .brand-mark {{
        width: 40px; height: 40px; border-radius: 10px;
        background: linear-gradient(135deg, {TEAL}, #0E9C86);
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; box-shadow: 0 0 24px {TEAL_GLOW};
    }}
    .brand-title {{ font-weight: 900; font-size: 20px; color: {TEXT}; letter-spacing: 0.2px; }}
    .brand-sub {{ font-size: 12px; color: {TEXT_MUTED}; font-family: 'JetBrains Mono', monospace; direction: ltr; }}
    .live-dot {{
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: {TEAL}; margin-left: 6px; box-shadow: 0 0 8px {TEAL};
        animation: pulse 1.8s infinite ease-in-out;
    }}
    @keyframes pulse {{
        0% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} 100% {{ opacity: 1; }}
    }}
    .live-tag {{
        font-family: 'JetBrains Mono', monospace; font-size: 11px; color: {TEXT_MUTED};
        border: 1px solid {BORDER}; padding: 6px 12px; border-radius: 999px;
        display: flex; align-items: center; direction: ltr;
    }}

    /* ---- Section labels ---- */
    .section-label {{
        font-size: 13px; font-weight: 700; color: {TEXT_MUTED};
        text-transform: uppercase; letter-spacing: 1.2px;
        margin: 4px 0 14px 0; display: flex; align-items: center; gap: 8px;
    }}
    .section-label::after {{
        content: ""; flex: 1; height: 1px; background: {BORDER};
    }}

    /* ---- Generic card ---- */
    .qcard {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 18px 20px; margin-bottom: 14px;
    }}

    /* ---- Signal badge ---- */
    .signal-badge {{
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 16px; border-radius: 999px; font-weight: 700; font-size: 14px;
        border: 1px solid; margin-bottom: 10px;
    }}
    .sig-long {{ background: {TEAL_GLOW}; border-color: {TEAL}; color: {TEAL}; }}
    .sig-short {{ background: {ROSE_GLOW}; border-color: {ROSE}; color: {ROSE}; }}
    .sig-neutral {{ background: {AMBER_GLOW}; border-color: {AMBER}; color: {AMBER}; }}
    .sig-dot {{ width: 8px; height: 8px; border-radius: 50%; background: currentColor; }}

    /* ---- Global score card ---- */
    .score-hero {{
        background: linear-gradient(135deg, {SURFACE} 0%, {SURFACE_2} 100%);
        border: 1px solid {BORDER}; border-radius: 16px; padding: 24px 26px;
    }}
    .score-value {{
        font-family: 'JetBrains Mono', monospace; font-size: 44px; font-weight: 700;
        color: {TEXT}; line-height: 1; direction: ltr;
    }}
    .score-value span {{ font-size: 18px; color: {TEXT_MUTED}; }}
    .score-desc {{
        margin-top: 12px; padding: 12px 14px; background: rgba(255,255,255,0.03);
        border-radius: 10px; font-size: 13.5px; color: {TEXT}; border-right: 3px solid {TEAL};
    }}

    /* ---- Confluence meter (signature element) ---- */
    .conf-wrap {{ display: flex; gap: 8px; }}
    .conf-seg {{
        flex: 1; border-radius: 10px; padding: 12px 8px 10px 8px; text-align: center;
        border: 1px solid {BORDER}; background: {SURFACE_2};
    }}
    .conf-tf {{ font-size: 11px; color: {TEXT_MUTED}; font-family: 'JetBrains Mono', monospace; }}
    .conf-score {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 20px; margin: 6px 0; direction: ltr; }}
    .conf-bar-track {{ height: 4px; border-radius: 4px; background: #232838; overflow: hidden; margin-top: 4px; }}
    .conf-bar-fill {{ height: 100%; border-radius: 4px; }}

    /* ---- Metric row (signal card, price stats) ---- */
    .stat-row {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 9px 0; border-bottom: 1px solid {BORDER}; font-size: 13.5px;
    }}
    .stat-row:last-child {{ border-bottom: none; }}
    .stat-label {{ color: {TEXT_MUTED}; }}
    .stat-value {{ font-family: 'JetBrains Mono', monospace; color: {TEXT}; font-weight: 600; direction: ltr; }}

    /* ---- Score progress bar ---- */
    .qbar-track {{ height: 10px; border-radius: 6px; background: #1B2130; overflow: hidden; border: 1px solid {BORDER}; }}
    .qbar-fill {{ height: 100%; border-radius: 6px; }}

    /* ---- Trade type badge ---- */
    .trade-badge {{
        display: inline-block; padding: 6px 14px; border-radius: 8px; font-weight: 700;
        font-size: 13px; margin-top: 4px;
    }}

    /* ---- Streamlit input tightening ---- */
    div[data-testid="stNumberInput"] input {{
        font-family: 'JetBrains Mono', monospace; direction: ltr;
    }}
    .streamlit-expanderHeader {{ font-weight: 700; }}

    hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_FONT = dict(family="JetBrains Mono, Tajawal, sans-serif", color=TEXT_MUTED, size=12)

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
    except Exception:
        pass

    return None

def calculate_indicators(df):
    """حساب المؤشرات الكمية"""
    if df is None or len(df) < 52:
        return df

    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['tp'] * df['volume']).cumsum() / df['volume'].cumsum()

    df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
    df['span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)

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
        master_signal = "SUPER STRONG LONG"
        sig_class = "sig-long"
        status_desc = "توافق صاعد تام عبر الفريمات الكبرى والصغرى."
    elif global_score <= 30 and d_score <= 40 and h4_score <= 40:
        master_signal = "SUPER STRONG SHORT"
        sig_class = "sig-short"
        status_desc = "توافق هابط تام عبر الفريمات الكبرى والصغرى."
    elif global_score >= 65 and (d_score < 50 or h4_score < 50):
        master_signal = "SCALP LONG (Counter-Trend)"
        sig_class = "sig-neutral"
        status_desc = "صعود قصير الأجل على الصغرى عكس اتجاه الفريم اليومي."
    elif global_score <= 35 and (d_score > 50 or h4_score > 50):
        master_signal = "SCALP SHORT (Counter-Trend)"
        sig_class = "sig-neutral"
        status_desc = "هبوط قصير الأجل على الصغرى عكس اتجاه الفريم اليومي."
    else:
        master_signal = "NEUTRAL / CONFLICT"
        sig_class = "sig-neutral"
        status_desc = "تضارب بين الأطر الزمنية - يفضل عدم الدخول."

    return {
        'global_score': global_score,
        'master_signal': master_signal,
        'sig_class': sig_class,
        'status_desc': status_desc,
        'tf_scores': tf_scores
    }

def score_color(sc):
    if sc >= 65:
        return TEAL
    elif sc <= 35:
        return ROSE
    return AMBER

# ==========================================
# 3. القائمة الجانبية (Sidebar)
# ==========================================

st.sidebar.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,{TEAL},#0E9C86);
                display:flex;align-items:center;justify-content:center;font-size:17px;">⚡</div>
    <div>
        <div style="font-weight:900;font-size:17px;color:{TEXT};">AliQuantFund</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:{TEXT_MUTED};direction:ltr;">INSTITUTIONAL ENGINE</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin:14px 0;'>", unsafe_allow_html=True)

st.sidebar.markdown(f"<div class='section-label'>الأصل</div>", unsafe_allow_html=True)
selected_symbol = st.sidebar.selectbox(
    "اختر العملة للتحليل العميق:",
    ["BTC/USDT", "ETH/USDT", "ZEC/USDT", "XRP/USDT"],
    label_visibility="collapsed"
)

selected_tf = st.sidebar.selectbox(
    "الإطار الزمني للرسم البياني:",
    ["5m", "15m", "1h", "4h", "1d"]
)

st.sidebar.markdown("<hr style='margin:18px 0 14px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='section-label'>📐 حاسبة إدارة المخاطر</div>", unsafe_allow_html=True)

capital = st.sidebar.number_input("رأس المال الإجمالي ($):", value=100.0, step=10.0)
risk_pct = st.sidebar.number_input("نسبة المخاطرة (%):", value=2.0, step=0.5)

# ==========================================
# 4. الواجهة الرئيسية
# ==========================================

st.markdown(f"""
<div class="brand-bar">
    <div class="brand-left">
        <div class="brand-mark">📊</div>
        <div>
            <div class="brand-title">التحليل الكمي المدمج · {selected_symbol}</div>
            <div class="brand-sub">MULTI-TIMEFRAME QUANT ENGINE</div>
        </div>
    </div>
    <div class="live-tag"><span class="live-dot"></span>LIVE · {selected_tf.upper()}</div>
</div>
""", unsafe_allow_html=True)

# --- التوصية العامة الموحدة ---
st.markdown('<div class="section-label">🌐 التوصية العامة الموحدة — Multi-Timeframe Master Confluence</div>', unsafe_allow_html=True)

global_res = get_global_multi_tf_analysis(selected_symbol)
gs = global_res['global_score']
gs_color = score_color(gs)

g_col1, g_col2 = st.columns([1, 2])

with g_col1:
    st.markdown(f"""
    <div class="score-hero">
        <div class="signal-badge {global_res['sig_class']}">
            <span class="sig-dot"></span>{global_res['master_signal']}
        </div>
        <div class="score-value">{gs}<span>/100</span></div>
        <div class="qbar-track" style="margin-top:12px;">
            <div class="qbar-fill" style="width:{gs}%;background:{gs_color};"></div>
        </div>
        <div class="score-desc">💡 {global_res['status_desc']}</div>
    </div>
    """, unsafe_allow_html=True)

with g_col2:
    segs = ""
    for tf_key, sc in global_res['tf_scores'].items():
        c = score_color(sc)
        segs += f"""
        <div class="conf-seg">
            <div class="conf-tf">{tf_key.upper()}</div>
            <div class="conf-score" style="color:{c};">{sc}</div>
            <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{sc}%;background:{c};"></div></div>
        </div>
        """
    st.markdown(f"""
    <div class="qcard" style="height:100%;">
        <div style="font-size:13px;color:{TEXT_MUTED};margin-bottom:14px;">درجات التقييم حسب الأطر الزمنية الخمسة</div>
        <div class="conf-wrap">{segs}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

# --- الشارت والتحليل الفردي ---
df_data = fetch_klines_data(selected_symbol, interval=selected_tf)
df_calc = calculate_indicators(df_data)

if df_calc is not None and not df_calc.empty:
    latest = df_calc.iloc[-1]
    single_score = calculate_single_score(df_calc)
    sc_color = score_color(single_score)

    if single_score >= 65:
        sig_label, sig_class = "Strong Long", "sig-long"
    elif single_score <= 35:
        sig_label, sig_class = "Strong Short", "sig-short"
    else:
        sig_label, sig_class = "No-Trade Regime", "sig-neutral"

    col_chart, col_signal = st.columns([3, 1])

    with col_signal:
        st.markdown('<div class="section-label">🎯 بطاقة الإشارة اللحظية</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="qcard">
            <div style="font-size:13px;color:{TEXT_MUTED};margin-bottom:8px;">{selected_symbol} · {selected_tf}</div>
            <div class="signal-badge {sig_class}"><span class="sig-dot"></span>{sig_label}</div>
            <div class="qbar-track"><div class="qbar-fill" style="width:{single_score}%;background:{sc_color};"></div></div>
            <div style="text-align:left;direction:ltr;font-family:'JetBrains Mono',monospace;font-size:12px;color:{TEXT_MUTED};margin-top:6px;">{single_score}/100</div>
            <div style="margin-top:12px;">
                <div class="stat-row"><span class="stat-label">السعر الحالي</span><span class="stat-value">${latest['close']:.2f}</span></div>
                <div class="stat-row"><span class="stat-label">Anchored VWAP</span><span class="stat-value">${latest['vwap']:.2f}</span></div>
                <div class="stat-row"><span class="stat-label">نسبة الزخم الحجمي</span><span class="stat-value">{latest['vol_ratio']:.2f}x</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- حاسبة إدارة المخاطر المعدلة والتلقائية ---
        st.markdown('<div class="section-label">🔢 الإدارة العددية للصفقة</div>', unsafe_allow_html=True)

        with st.container():
            entry_price = st.number_input("سعر الدخول:", value=float(latest['close']))

            default_sl = float(latest['vwap'])
            sl_price = st.number_input("وقف الخسارة (SL):", value=default_sl)

            is_long = entry_price >= sl_price
            sl_distance = abs(entry_price - sl_price)

            if is_long:
                default_tp = entry_price + (sl_distance * 2)
            else:
                default_tp = entry_price - (sl_distance * 2)

            tp_price = st.number_input("أخذ الأرباح (TP):", value=float(default_tp))

            risk_amount = capital * (risk_pct / 100)

            if sl_distance > 0:
                units = risk_amount / sl_distance
                pos_value = units * entry_price

                if is_long:
                    tp_distance = tp_price - entry_price
                else:
                    tp_distance = entry_price - tp_price

                rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

                trade_color = TEAL if is_long else ROSE
                trade_label = "🟢 شراء (Long)" if is_long else "🔴 بيع (Short)"

                st.markdown(f"""
                <div class="qcard">
                    <span class="trade-badge" style="background:{TEAL_GLOW if is_long else ROSE_GLOW};color:{trade_color};">{trade_label}</span>
                    <div style="margin-top:10px;">
                        <div class="stat-row"><span class="stat-label">المخاطرة بالدولار</span><span class="stat-value">${risk_amount:.2f}</span></div>
                        <div class="stat-row"><span class="stat-label">حجم الصفقة (Units)</span><span class="stat-value">{units:.4f}</span></div>
                        <div class="stat-row"><span class="stat-label">قيمة العقد الإجمالية</span><span class="stat-value">${pos_value:.2f}</span></div>
                        <div class="stat-row"><span class="stat-label">نسبة العائد/المخاطرة</span><span class="stat-value">1:{rr_ratio:.2f}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_chart:
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df_calc['timestamp'],
            open=df_calc['open'],
            high=df_calc['high'],
            low=df_calc['low'],
            close=df_calc['close'],
            name='Price',
            increasing_line_color=TEAL, increasing_fillcolor=TEAL,
            decreasing_line_color=ROSE, decreasing_fillcolor=ROSE
        ))

        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['vwap'],
            mode='lines', name='Anchored VWAP',
            line=dict(color=AMBER, width=2)
        ))

        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['tenkan'],
            mode='lines', name='Tenkan-sen',
            line=dict(color='#5AC8FA', width=1.4)
        ))

        fig.add_trace(go.Scatter(
            x=df_calc['timestamp'], y=df_calc['kijun'],
            mode='lines', name='Kijun-sen',
            line=dict(color='#B48CFF', width=1.4)
        ))

        fig.update_layout(
            title=dict(text=f"{selected_symbol} · {selected_tf}", font=dict(size=15, color=TEXT, family="JetBrains Mono, monospace")),
            template="plotly_dark",
            plot_bgcolor=SURFACE,
            paper_bgcolor=SURFACE,
            font=PLOTLY_FONT,
            xaxis=dict(gridcolor=BORDER, rangeslider_visible=False),
            yaxis=dict(gridcolor=BORDER),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
            height=600,
            margin=dict(l=10, r=10, t=60, b=10)
        )

        st.markdown('<div class="qcard" style="padding:12px;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center;padding:18px 0 4px 0;color:{TEXT_MUTED};font-size:12px;
            font-family:'JetBrains Mono',monospace;direction:ltr;">
    AliQuantFund Engine v1.7 · All Quantitative Rights Reserved
</div>
""", unsafe_allow_html=True)
