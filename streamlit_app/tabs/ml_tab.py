
# streamlit_app/tabs/ml_tab.py
# ------------------------------------------------------------
# ML TAB — robuste pour Streamlit Cloud
# - Binance price + klines avec fallback (anti 451)
# - Close + SMA20
# - Appel optionnel à une API ML (BUY/SELL/WAIT)
# ------------------------------------------------------------

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st


# ------------------------------------------------------------
# BINANCE (fallback endpoints)
# ------------------------------------------------------------
BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",  # souvent OK depuis des clouds
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CryptoBot/1.0)",
    "Accept": "application/json",
}


class BinanceHTTPError(RuntimeError):
    pass


def _get_with_fallback(path: str, params: dict | None = None, timeout: int = 15, tries: int = 2) -> requests.Response:
    last_exc: Exception | None = None

    for base in BINANCE_BASES:
        url = f"{base}{path}"
        for _ in range(tries):
            try:
                r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)

                # codes fréquents quand Binance bloque certains datacenters
                if r.status_code in (451, 403, 418):
                    last_exc = BinanceHTTPError(f"{r.status_code} for {url}")
                    break

                r.raise_for_status()
                return r
            except Exception as e:
                last_exc = e
                time.sleep(0.25)

    raise BinanceHTTPError(f"Binance request failed. Last error: {last_exc}")


@st.cache_data(ttl=30)
def binance_price(symbol: str) -> float:
    r = _get_with_fallback("/api/v3/ticker/price", params={"symbol": symbol})
    return float(r.json()["price"])


@st.cache_data(ttl=60)
def binance_klines(symbol: str, interval: str, start_ms: int | None = None, end_ms: int | None = None, limit: int = 1000):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    r = _get_with_fallback("/api/v3/klines", params=params)
    return r.json()


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _klines_to_df(klines: list) -> pd.DataFrame:
    # Binance klines format:
    # [
    #  [open_time, open, high, low, close, volume, close_time, quote_asset_volume, trades, ...],
    # ]
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    rows = []
    for k in klines:
        rows.append([k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), k[6]])

    df = pd.DataFrame(rows, columns=cols)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("Europe/Paris")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ------------------------------------------------------------
# ML API (optionnel)
# ------------------------------------------------------------
def _ml_api_url() -> str:
    # Mets ton URL d’API ML dans Streamlit Cloud > Settings > Secrets:
    # ML_API_BASE = "https://xxxxx"
    # Ou variable d'env ML_API_BASE
    secrets_base = st.secrets.get("ML_API_BASE", None) if hasattr(st, "secrets") else None
    base = secrets_base or os.getenv("ML_API_BASE", "").strip()
    return base.rstrip("/")


def fetch_ml_decision(symbol: str, currency: str, interval: str) -> dict:
    base = _ml_api_url()
    if not base:
        raise RuntimeError("ML_API_BASE non défini (secrets ou variable d'env).")

    # endpoint attendu: /ml/decision/BTC/USDT/4h
    url = f"{base}/ml/decision/{symbol}/{currency}/{interval}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def decision_rule(probas: dict) -> str:
    # règles décrites dans ton dashboard
    p_wait = float(probas.get("WAIT", 0))
    p_buy = float(probas.get("BUY", 0))
    p_sell = float(probas.get("SELL", 0))

    if p_wait >= 0.60:
        return "WAIT"
    if max(p_buy, p_sell) >= 0.55:
        return "BUY" if p_buy >= p_sell else "SELL"
    return "WAIT"


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
def render():
    st.header("📈 ML Trading — Signal & Marché")

    with st.expander("ℹ️ Explications (pour l'utilisateur / jury)", expanded=False):
        st.markdown(
            """
**Ce que fait cette page**
- Récupère le prix live et un historique (Binance).
- Affiche Close + SMA20 pour la tendance.
- Récupère un signal ML via l’API : BUY / SELL / WAIT.
- Affiche les probabilités et les features utilisées (meta).

**Règle de décision affichée (dashboard)**
- Si P(WAIT) ≥ 0.60 → WAIT  
- Sinon si max(P(BUY), P(SELL)) ≥ 0.55 → BUY/SELL  
- Sinon → WAIT

⚠️ Le signal est informatif (pas un conseil financier).
"""
        )

    # Controls
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        symbol = st.selectbox("Crypto", ["BTC", "ETH", "BNB", "SOL"], index=0)
    with col2:
        currency = st.selectbox("Devise", ["USDT"], index=0)
    with col3:
        interval = st.selectbox("Intervalle", ["1m", "15m", "1h", "4h", "1d"], index=3)

    full_symbol = f"{symbol}{currency}"

    # Time range
    period = st.selectbox("Période (historique)", ["Jour", "Semaine", "Mois", "Année"], index=1)
    now = datetime.now(timezone.utc)
    if period == "Jour":
        start = now - timedelta(days=1)
    elif period == "Semaine":
        start = now - timedelta(days=7)
    elif period == "Mois":
        start = now - timedelta(days=30)
    else:
        start = now - timedelta(days=365)

    st.subheader(f"💰 Prix live — {full_symbol}")
    try:
        price = binance_price(full_symbol)
        st.metric(label=full_symbol, value=f"{price:,.2f}")
    except Exception as e:
        st.error(f"Erreur prix live : {e}")
        st.info("Astuce : depuis Streamlit Cloud, Binance peut parfois bloquer. Le fallback essaie plusieurs endpoints.")
        st.stop()

    # Historical
    st.subheader("📉 Historique & tendance")
    try:
        kl = binance_klines(full_symbol, interval, _to_ms(start), _to_ms(now), limit=1000)
        df = _klines_to_df(kl)
        if df.empty:
            st.warning("Aucune donnée retournée par Binance.")
            st.stop()

        df["sma20"] = df["close"].rolling(20).mean()

        st.line_chart(df.set_index("datetime")[["close", "sma20"]])

        last = df.iloc[-1]
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        meta_col1.metric("Dernier close", f"{last['close']:.2f}")
        meta_col2.metric("SMA20", f"{last['sma20']:.2f}" if pd.notna(last["sma20"]) else "—")
        if pd.notna(last["sma20"]) and last["sma20"] != 0:
            meta_col3.metric("Diff vs SMA20", f"{(last['close']/last['sma20'] - 1):.2%}")
        else:
            meta_col3.metric("Diff vs SMA20", "—")

        with st.expander("Voir les dernières lignes", expanded=False):
            st.dataframe(df.tail(20), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur historique Binance : {e}")
        st.stop()

    # ML Decision
    st.subheader("🤖 Signal ML (API)")
    try:
        payload = fetch_ml_decision(symbol, currency, interval)
        probas = payload.get("probabilities", {}) or {}
        api_decision = payload.get("decision", "UNKNOWN")
        rule_decision = decision_rule(probas)

        c1, c2, c3 = st.columns(3)
        c1.metric("Décision API", str(api_decision))
        c2.metric("Décision règle dashboard", str(rule_decision))
        c3.metric("P(WAIT)", f"{float(probas.get('WAIT', 0)):.3f}")

        st.write("Probabilités :")
        st.json(probas)

        meta = payload.get("meta", {}) or {}
        if meta:
            st.write("Meta / features :")
            st.json(meta)

    except Exception as e:
        st.warning("API ML indisponible ou non configurée (l’onglet marché reste utilisable).")
        st.caption(f"Détail : {e}")
        st.info(
            "Pour activer l’API ML sur Streamlit Cloud, définis `ML_API_BASE` dans Settings → Secrets "
            "(ex: https://mon-api.exemple.com)."
        )
