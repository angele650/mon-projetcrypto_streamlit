
# streamlit_app/tabs/ml_tab.py
# ============================================================
# CryptoBot — Tab ML Trading (PRO + STABLE)
# - Prix live Binance
# - Marché : Close + SMA20
# - Décision ML détaillée depuis API : BUY/SELL/WAIT + probas + meta
# - Règle de décision (seuils) clairement affichée
# - Fallback SMA20 si API ML indisponible
# ============================================================

from datetime import datetime, timedelta, timezone
import time
import requests
import pandas as pd
import streamlit as st

BINANCE_BASE = "https://api.binance.com"
DEFAULT_API_ML = "http://127.0.0.1:8001"


# ============================================================
# Helpers
# ============================================================
def utc_now():
    return datetime.now(timezone.utc)

def to_ms(dt):
    return int(dt.timestamp() * 1000)

def make_symbol(base, quote):
    return f"{base.upper()}{quote.upper()}"

def period_to_range(period):
    end = utc_now()
    if period == "Jour":
        start = end - timedelta(days=1)
    elif period == "Semaine":
        start = end - timedelta(days=7)
    elif period == "Mois":
        start = end - timedelta(days=30)
    else:
        start = end - timedelta(days=365)
    return start, end

def fmt_pct(x):
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"

def fmt_num(x, digits=2):
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "—"


# ============================================================
# Binance
# ============================================================
@st.cache_data(ttl=5)
def binance_price(symbol):
    r = requests.get(
        f"{BINANCE_BASE}/api/v3/ticker/price",
        params={"symbol": symbol},
        timeout=10,
    )
    r.raise_for_status()
    return float(r.json()["price"])


@st.cache_data(ttl=60)
def binance_klines(symbol, interval, start_ms, end_ms):
    # 1000 points max (suffisant pour dashboard)
    r = requests.get(
        f"{BINANCE_BASE}/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        },
        timeout=20,
    )
    r.raise_for_status()
    rows = r.json()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tb", "tq", "ignore"
        ],
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index("open_time").sort_index()
    df["sma20"] = df["close"].rolling(20).mean()
    df["ret"] = df["close"].pct_change()

    return df


# ============================================================
# ML API
# ============================================================
@st.cache_data(ttl=5)
def fetch_ml_signal(api_base, base, quote, interval):
    api_base = api_base.rstrip("/")
    url = f"{api_base}/ml/decision/{base}/{quote}/{interval}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def decide_from_probs(probs: dict, thr_wait: float, thr_action: float) -> tuple[str, str]:
    """
    Règle de décision affichable et reproductible :
    - si P(WAIT) >= thr_wait => WAIT
    - sinon si max(P(BUY), P(SELL)) >= thr_action => BUY/SELL
    - sinon => WAIT
    """
    p_buy = float(probs.get("BUY", 0.0)) if isinstance(probs, dict) else 0.0
    p_sell = float(probs.get("SELL", 0.0)) if isinstance(probs, dict) else 0.0
    p_wait = float(probs.get("WAIT", 0.0)) if isinstance(probs, dict) else 0.0

    if p_wait >= thr_wait:
        return "WAIT", f"P(WAIT)={fmt_pct(p_wait)} ≥ {fmt_pct(thr_wait)} → WAIT"
    best_action = "BUY" if p_buy >= p_sell else "SELL"
    best_p = max(p_buy, p_sell)

    if best_p >= thr_action:
        return best_action, f"max(P(BUY),P(SELL))={fmt_pct(best_p)} ≥ {fmt_pct(thr_action)} → {best_action}"
    return "WAIT", f"Signal insuffisant (best={fmt_pct(best_p)} < {fmt_pct(thr_action)}) → WAIT"


def render_decision_banner(decision: str):
    decision = str(decision).upper()
    if decision == "BUY":
        st.success("🟢 Signal : BUY")
    elif decision == "SELL":
        st.error("🔴 Signal : SELL")
    else:
        st.warning("🟡 Signal : WAIT")


def render_ml_details(payload: dict, thr_wait: float, thr_action: float):
    decision_api = str(payload.get("decision", "WAIT")).upper()
    probs = payload.get("probabilities", {}) or {}
    meta = payload.get("meta", {}) or {}

    # Probabilités normalisées en dict upper
    if isinstance(probs, dict):
        probs = {str(k).upper(): float(v) for k, v in probs.items()}

    # Décision recalculée via règle (pour expliquer clairement)
    decision_rule, rule_explain = decide_from_probs(probs, thr_wait, thr_action)

    # Bandeau (on affiche décision API, et on montre cohérence règle)
    st.markdown("## 🧠 Signal ML (détaillé)")
    render_decision_banner(decision_api)

    if decision_rule != decision_api:
        st.info(
            f"ℹ️ **Décision API** = {decision_api} | **Décision via règle** = {decision_rule}\n\n"
            "Ça peut arriver si l'API utilise une règle interne légèrement différente. "
            "Tu peux aligner les seuils côté API si besoin."
        )
    st.caption(f"Règle de décision (dashboard) : {rule_explain}")

    # Probabilités
    st.markdown("### 📊 Probabilités")
    if isinstance(probs, dict) and probs:
        prob_df = (
            pd.DataFrame([{"Action": k, "Probabilité": v} for k, v in probs.items()])
            .sort_values("Probabilité", ascending=False)
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(prob_df, use_container_width=True)
        with c2:
            st.bar_chart(prob_df.set_index("Action"))

    # Meta / features
    st.markdown("### 🧾 Contexte marché (features)")
    close = meta.get("close")
    sma20 = meta.get("close_sma20")
    diff = meta.get("diff_vs_sma20")
    tau = meta.get("tau")
    vol20 = meta.get("volatility_20")
    last_dt = meta.get("last_datetime")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Close", fmt_num(close, 2))
    k2.metric("SMA20", fmt_num(sma20, 2))
    k3.metric("Écart vs SMA20", fmt_pct(diff))
    k4.metric("Tau", fmt_num(tau, 6))
    k5.metric("Volatilité 20", fmt_pct(vol20))
    if last_dt:
        st.caption(f"Dernière bougie utilisée : **{last_dt}**")

    # Explication simple (jury + user)
    st.markdown("### 🧠 Interprétation (simple)")
    reasons = []

    p_wait = probs.get("WAIT", None) if isinstance(probs, dict) else None
    p_buy = probs.get("BUY", None) if isinstance(probs, dict) else None
    p_sell = probs.get("SELL", None) if isinstance(probs, dict) else None

    if isinstance(p_wait, (int, float)) and p_wait >= thr_wait:
        reasons.append(f"P(WAIT) est élevée ({fmt_pct(p_wait)}) → le modèle préfère attendre (incertitude).")

    if isinstance(diff, (int, float)):
        if abs(diff) < 0.01:
            reasons.append("Le prix est très proche de la SMA20 → tendance peu claire.")
        elif diff > 0:
            reasons.append("Le prix est au-dessus de la SMA20 → biais haussier.")
        else:
            reasons.append("Le prix est en dessous de la SMA20 → biais baissier.")

    if isinstance(vol20, (int, float)):
        if vol20 > 0.03:
            reasons.append("Volatilité récente élevée → plus de bruit, le modèle peut être prudent.")
        elif vol20 < 0.015:
            reasons.append("Volatilité faible → marché plat, signaux parfois moins nets.")

    if isinstance(p_buy, (int, float)) and isinstance(p_sell, (int, float)):
        reasons.append(f"Comparaison BUY vs SELL : BUY={fmt_pct(p_buy)} | SELL={fmt_pct(p_sell)}.")

    if not reasons:
        reasons.append("Le modèle combine plusieurs signaux et estime que le meilleur choix est WAIT.")

    for r in reasons:
        st.write("• " + r)

    with st.expander("🔍 Réponse brute de l’API (debug)"):
        st.json(payload)


def render_fallback(df):
    st.markdown("## 🧠 Signal (fallback SMA20)")
    if df is None or df.empty or "sma20" not in df.columns:
        st.warning("🟡 WAIT (fallback)")
        st.write("Pas assez de données pour SMA20.")
        return

    last_close = float(df["close"].iloc[-1])
    last_sma = df["sma20"].iloc[-1]

    if pd.isna(last_sma):
        st.warning("🟡 WAIT (fallback)")
        st.write("SMA20 non disponible sur la dernière bougie.")
        return

    last_sma = float(last_sma)
    if last_close > last_sma:
        st.success("🟢 BUY (fallback)")
        st.write("Prix au-dessus de la SMA20 → biais haussier simple.")
    elif last_close < last_sma:
        st.error("🔴 SELL (fallback)")
        st.write("Prix en dessous de la SMA20 → biais baissier simple.")
    else:
        st.warning("🟡 WAIT (fallback)")
        st.write("Prix égal à la SMA20 → neutre.")


# ============================================================
# Public API (attendue par member.py)
# ============================================================
def render():
    st.subheader("📈 ML Trading — Signal & Marché")

    # ---------------- Sidebar
    with st.sidebar:
        st.header("Réglages")
        base = st.selectbox("Crypto", ["BTC", "ETH", "BNB", "SOL"], index=0)
        quote = st.selectbox("Devise", ["USDT", "EUR"], index=0)
        interval = st.selectbox("Intervalle", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
        period = st.radio("Période graphe", ["Jour", "Semaine", "Mois", "Année"], index=1)

        st.markdown("---")
        st.subheader("API ML")
        api_ml = st.text_input("URL API ML", DEFAULT_API_ML)

        st.subheader("Seuils décision (dashboard)")
        thr_wait = st.slider("Seuil WAIT : P(WAIT) ≥", 0.50, 0.90, 0.60, 0.01)
        thr_action = st.slider("Seuil action : max(BUY,SELL) ≥", 0.40, 0.90, 0.55, 0.01)

        refresh = st.checkbox("Auto-refresh prix (3s)", value=False)

    symbol = make_symbol(base, quote)

    # ---------------- Explications
    with st.expander("ℹ️ Explications (pour l'utilisateur / jury)", expanded=True):
        st.markdown(
            f"""
**Ce que fait cette page**
- Récupère le **prix live** et un **historique** (Binance).
- Affiche **Close + SMA20** pour la tendance.
- Récupère un **signal ML** via l’API : BUY / SELL / WAIT.
- Affiche les **probabilités** et les **features** utilisées (meta).

**Règle de décision affichée (dashboard)**
- Si `P(WAIT) ≥ {thr_wait:.2f}` → **WAIT**
- Sinon si `max(P(BUY), P(SELL)) ≥ {thr_action:.2f}` → **BUY/SELL**
- Sinon → **WAIT**

⚠️ Le signal est informatif (pas un conseil financier).
"""
        )

    # ---------------- Prix live
    st.markdown(f"### 💰 Prix live — `{symbol}`")
    price_box = st.empty()

    def show_price_once():
        try:
            price = binance_price(symbol)
            price_box.metric(symbol, f"{price:,.2f}")
        except Exception as e:
            price_box.error(f"Erreur prix live : {e}")

    show_price_once()
    if refresh:
        st.caption("Auto-refresh actif (~60s).")
        for _ in range(20):
            time.sleep(3)
            show_price_once()

    st.divider()

    # ---------------- Données marché
    start, end = period_to_range(period)
    df = binance_klines(symbol, interval, to_ms(start), to_ms(end))
    if df.empty:
        st.warning("Aucune donnée marché.")
        return

    # ---------------- KPIs marché rapides
    last_close = float(df["close"].iloc[-1])
    first_close = float(df["close"].iloc[0])
    change_pct = (last_close / first_close - 1.0) * 100.0
    vol = float(df["ret"].std() * 100.0) if df["ret"].dropna().shape[0] > 2 else 0.0

    a, b, c = st.columns(3)
    a.metric("Variation période", f"{change_pct:+.2f}%")
    b.metric("Volatilité (std ret)", f"{vol:.2f}%")
    c.metric("Dernière bougie", df.index[-1].strftime("%Y-%m-%d %H:%M UTC"))

    # ---------------- ML détaillé
    try:
        payload = fetch_ml_signal(api_ml, base, quote, interval)
        render_ml_details(payload, thr_wait=thr_wait, thr_action=thr_action)
    except Exception as e:
        st.caption("⚠️ API ML indisponible → fallback SMA20")
        render_fallback(df)
        st.caption(f"Détail erreur API: {e}")

    st.divider()

    # ---------------- Graphique
    st.markdown("## 📊 Marché (Close + SMA20)")
    st.line_chart(df[["close", "sma20"]])

    with st.expander("Données brutes"):
        st.dataframe(df.tail(200), use_container_width=True)
