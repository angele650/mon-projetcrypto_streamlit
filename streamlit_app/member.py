import streamlit as st
from tabs.ml_tab import render as render_ml

st.set_page_config(page_title="CryptoBot", layout="wide")
st.title("CryptoBot Dashboard")

tabs = st.tabs(["Accueil", "ML Trading"])

with tabs[0]:
    st.subheader("Bienvenue")
    st.write(
        "Ce dashboard affiche le marché (Binance) et un signal ML (BUY/SELL/WAIT) "
        "fourni par l'API interne. Le signal est informatif et ne constitue pas un conseil financier."
    )

with tabs[1]:
    render_ml()
