import streamlit as st
import requests
from datetime import date, timedelta, datetime
import pandas as pd
from typing import Dict, Iterable
import altair as alt


# -------------------------
# Configuration
# -------------------------
st.set_page_config(
    page_title="Currency Converter",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_URL = "https://api.frankfurter.app"
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NGN"]

# API_URL = "https://api.exchangerate.host/latest"
API_URL = "https://open.er-api.com/v6/latest"


# -------------------------
# Styling (dark UI)
# -------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0rem !important;
    }
    div[data-testid="column"] {
        margin-top: 0 !important;
    }
    body {
        background-color: #0b1220;
        color: #e5e7eb;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .card {
        background: linear-gradient(180deg, #0f1c2e, #0b1625);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.04);
    }
    .metric-card {
        background: #121e31;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .rate-box {
        background: linear-gradient(135deg, #0e2a33, #0a1f28);
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(0,255,170,0.25);
        margin-top: 20px;
    }
    .green {
        color: #00e5a8;
        font-weight: 600;
    }
    .muted {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Data Layer
# -------------------------

from requests.exceptions import RequestException

@st.cache_data(ttl=300)
def fetch_rates(base: str) -> Dict[str, float]:
    try:
        response = requests.get(f"{API_URL}/{base}", timeout=5)
        response.raise_for_status()

        data = response.json()
        if data.get("result") != "success":
            raise ValueError("API returned unsuccessful result")

        return data["rates"]

    except RequestException:
        return {
            "USD": 1.0,
            "EUR": 0.84,
            "GBP": 0.74,
            "JPY": 157.0,
            "CAD": 1.36,
            "AUD": 1.52,
            "CHF": 0.88,
            "NGN": 1450.0,
        }

def convert_currency(from_curr: str, to_curr: str, amount: float) -> float:
    """Convert currency using robust fetch_rates with fallback."""
    rates = fetch_rates(from_curr)
    rate = rates.get(to_curr)
    if rate:
        return round(rate * amount, 4)
    else:
        raise ValueError(f"Conversion rate for {to_curr} not available.")


# -------------------------
# State
# -------------------------
if "from_currency" not in st.session_state:
    st.session_state.from_currency = "USD"
if "to_currency" not in st.session_state:
    st.session_state.to_currency = "EUR"


def swap_currencies():
    st.session_state.from_currency, st.session_state.to_currency = (
        st.session_state.to_currency,
        st.session_state.from_currency,
    )


# -------------------------
# Layout
# -------------------------
left, right = st.columns([2.2, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Currency Converter")
    st.markdown(
        "<p class='muted'>Convert between major world currencies with real-time rates</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([4, 1])
    with col_a:
        amount = st.number_input(
            "From",
            min_value=0.0,
            value=1.0,
            step=0.01,
            label_visibility="visible",
        )
    with col_b:
        from_currency = st.selectbox(
            "",
            SUPPORTED_CURRENCIES,
            index=SUPPORTED_CURRENCIES.index(st.session_state.from_currency),
            key="from_currency",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.button("⇄", on_click=swap_currencies)

    rates = fetch_rates(st.session_state.from_currency)

    if st.session_state.from_currency == st.session_state.to_currency:
        rate = 1.0
    else:
        rate = float(rates[st.session_state.to_currency])

    rate = rates.get(st.session_state.to_currency, 0.0)
    converted = amount * rate

    col_c, col_d = st.columns([4, 1])
    with col_c:
        st.text_input(
            "To",
            value=f"{converted:.4f}",
            disabled=True,
        )
    with col_d:
        to_currency = st.selectbox(
            "",
            SUPPORTED_CURRENCIES,
            index=SUPPORTED_CURRENCIES.index(st.session_state.to_currency),
            key="to_currency",
        )

    st.markdown(
        f"""
        <div class="rate-box">
            <div class="muted">Current Exchange Rate</div>
            <div class="green" style="font-size:1.6rem;">
                1 {st.session_state.from_currency}
                  = {rate:.4f} {st.session_state.to_currency}
            </div>
            <div class="muted">
                Updated: {datetime.now().strftime('%I:%M:%S %p')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Market Snapshot")

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="muted">Conversion</div>
            <div style="font-size:1.2rem;font-weight:600;">
                1 {st.session_state.from_currency}
            </div>
            <div class="green">
                {rate:.4f} {st.session_state.to_currency}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="metric-card">
            <div class="muted">Change Potential</div>
            <div class="green" style="font-size:1.3rem;">
                +2.4%
            </div>
            <div class="muted">this month</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="muted">Mid-Market Rate</div>
            <div style="font-size:1.2rem;font-weight:600;">
                {rate:.6f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Charts
# -------------------------

@st.cache_data(ttl=300)
def fetch_historical_rates(
    base: str,
    target: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    try:
        url = f"{BASE_URL}/{start.isoformat()}..{end.isoformat()}"
        params = {"from": base, "to": target}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        records = [
            {"date": d, "rate": float(f"{r[target]:.4f}")}
            for d, r in payload["rates"].items()
        ]

        return (
            pd.DataFrame(records)
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .sort_values("date")
        )

    except RequestException:
        return pd.DataFrame(columns=["date", "rate"])


def render_chart(df: pd.DataFrame, base: str, target: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line(interpolate="monotone", strokeWidth=2, color="#1f77b4")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "rate:Q",
                title=f"{base} → {target}",
                axis=alt.Axis(
                    format=".4f",
                    labelFlush=False,
                    labelOverlap=False,
                ),
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("rate:Q", title="Rate", format=".4f"),
            ],
        )
        .properties(height=420)
        .interactive()
    )


st.markdown("---")


def main() -> None:

    st.title("📈 Historical Currency Rates")

    range_days = st.radio(
        "Range",
        [7, 30, 90, 365],
        horizontal=True,
        index=1,
    )

    end_date = date.today()
    start_date = end_date - timedelta(days=range_days)

    with st.spinner("Fetching historical data..."):
        df = fetch_historical_rates(
            from_currency,
            to_currency,
            start_date,
            end_date,
        )

    chart = render_chart(df, from_currency, to_currency)
    st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()

st.markdown("----")


# -----------------------------
# Data layer
# -----------------------------
import socket
from requests.exceptions import RequestException


def fetch_historical_rates(
    base: str,
    targets: list[str],
    start: date,
    end: date,
) -> pd.DataFrame | None:

    # ---- Clamp future dates (CRITICAL) ----
    today = date.today()
    end = min(end, today)
    start = min(start, end)

    url = f"{BASE_URL}/{start.isoformat()}..{end.isoformat()}"
    params = {"from": base, "to": ",".join(targets)}

    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()

    except socket.gaierror:
        st.error("❌ Network error: DNS resolution failed.")
        st.info("Check your internet connection, VPN, or firewall.")
        return None

    except RequestException as exc:
        st.error("❌ Unable to reach FX data provider.")
        st.caption(str(exc))
        return None

    payload = response.json()

    if not payload.get("rates"):
        st.warning("No FX data available for the selected range.")
        return None

    records = []
    for d, rates in payload["rates"].items():
        for currency, value in rates.items():
            records.append(
                {
                    "date": d,
                    "currency": currency,
                    "rate": float(f"{value:.4f}"),
                }
            )

    return (
        pd.DataFrame(records)
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .sort_values(["currency", "date"])
    )



# -----------------------------
# Indicators
# -----------------------------
def calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def add_indicators(
    df: pd.DataFrame,
    sma_window: int,
    ema_span: int,
    rsi_window: int,
) -> pd.DataFrame:
    df = df.copy()

    df["sma"] = (
        df.groupby("currency")["rate"]
        .transform(lambda s: s.rolling(sma_window).mean())
    )

    df["ema"] = (
        df.groupby("currency")["rate"]
        .transform(lambda s: s.ewm(span=ema_span, adjust=False).mean())
    )

    df["pct_change"] = (
        df.groupby("currency")["rate"]
        .transform(lambda s: s.pct_change() * 100)
        .round(2)
    )

    df["rsi"] = (
        df.groupby("currency")["rate"]
        .transform(lambda s: calculate_rsi(s, rsi_window))
    )

    return df

# -----------------------------
# Charts
# -----------------------------
def render_price_chart(
    df: pd.DataFrame,
    base: str,
    show_sma: bool,
    show_ema: bool,
) -> alt.Chart:
    base_chart = (
        alt.Chart(df)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "rate:Q",
                title=f"{base} Exchange Rate",
                axis=alt.Axis(format=".4f"),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("currency:N", legend=alt.Legend(title="Currency")),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("currency:N", title="Currency"),
                alt.Tooltip("rate:Q", title="Rate", format=".4f"),
                alt.Tooltip("pct_change:Q", title="% Change", format=".2f"),
            ],
        )
    )

    layers = [base_chart.mark_line(strokeWidth=2)]

    if show_sma:
        layers.append(
            base_chart.mark_line(strokeDash=[4, 2], opacity=0.6).encode(y="sma:Q")
        )

    if show_ema:
        layers.append(
            base_chart.mark_line(strokeDash=[2, 2], opacity=0.6).encode(y="ema:Q")
        )

    return alt.layer(*layers).properties(height=420).interactive()

def render_rsi_chart(df: pd.DataFrame) -> alt.Chart:
    rsi_line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "rsi:Q",
                title="RSI",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color("currency:N", legend=None),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("currency:N", title="Currency"),
                alt.Tooltip("rsi:Q", title="RSI", format=".2f"),
            ],
        )
    )

    overbought = alt.Chart(pd.DataFrame({"y": [70]})).mark_rule(
        strokeDash=[4, 4], color="red"
    ).encode(y="y:Q")

    oversold = alt.Chart(pd.DataFrame({"y": [30]})).mark_rule(
        strokeDash=[4, 4], color="green"
    ).encode(y="y:Q")

    return (
        alt.layer(rsi_line, overbought, oversold)
        .properties(height=220)
        .interactive()
    )

# -----------------------------
# App
# -----------------------------
def main() -> None:

    st.title("📊 FX Analytics Dashboard")

    base_currency = st.selectbox("Base Currency", SUPPORTED_CURRENCIES, index=0)

    target_currencies = st.multiselect(
        "Target Currencies",
        [c for c in SUPPORTED_CURRENCIES if c != base_currency],
        default=["EUR", "GBP"],
    )

    if not target_currencies:
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        range_days = st.selectbox("Range (days)", [7, 30, 90, 365], index=1)

    with col2:
        sma_window = st.selectbox("SMA", [5, 10, 20, 30], index=2)

    with col3:
        ema_span = st.selectbox("EMA", [5, 10, 20, 30], index=2)

    with col4:
        rsi_window = st.selectbox("RSI Window", [7, 14, 21], index=1)

    show_sma = st.checkbox("Show SMA", value=True)
    show_ema = st.checkbox("Show EMA", value=True)
    show_rsi = st.checkbox("Show RSI", value=True)

    end_date = date.today()
    start_date = end_date - timedelta(days=range_days)

    with st.spinner("Fetching historical data..."):
        df = fetch_historical_rates(
            base_currency,
            target_currencies,
            start_date,
            end_date,
        )

    if df is None or df.empty:
        st.stop()

    df = add_indicators(df, sma_window, ema_span, rsi_window)

    st.altair_chart(
        render_price_chart(df, base_currency, show_sma, show_ema),
        use_container_width=True,
    )

    if show_rsi:
        st.altair_chart(
            render_rsi_chart(df),
            use_container_width=True,
        )

    with st.expander("📋 Raw Data"):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
