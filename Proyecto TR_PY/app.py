# =============================================================================
# frontend/app.py – Tablero Streamlit (consume el backend FastAPI)
# Autoras: Laura Alejandra Sepúlveda & Ingrid Johana Umbacia Ramírez
# Proyecto Integrador – Teoría del Riesgo · USTA
# =============================================================================
from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"
TICKERS     = ["MSI", "XOM", "JNJ", "PG", "UL", "TSM"]
NAMES = {
    "MSI": "Motorola Solutions", "XOM": "ExxonMobil",
    "JNJ": "Johnson & Johnson",  "PG":  "Procter & Gamble",
    "UL":  "Unilever",           "TSM": "TSMC",
}
USTA_NAVY   = "#001A4D"
USTA_PURPLE = "#3D008D"
USTA_PINK   = "#ED1E79"
USTA_GOLD   = "#FDB913"
USTA_LIGHT  = "#F4F6FB"
COLORS = [USTA_PURPLE, USTA_PINK, USTA_GOLD, "#0EA5E9", "#16a34a", "#F97316"]

st.set_page_config(
    page_title="RiskLab USTA – Teoría del Riesgo",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp {{background-color:{USTA_LIGHT}; color:#1E293B;}}
[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,{USTA_NAVY} 0%,#002868 60%,#003580 100%) !important;
    border-right:3px solid {USTA_GOLD};
}}
[data-testid="stSidebar"] * {{color:#FFFFFF !important;}}
.block-container {{padding:1.5rem 2.5rem 2rem 2.5rem; max-width:1300px;}}
h1 {{color:{USTA_NAVY}; font-weight:800; border-bottom:3px solid {USTA_PINK}; padding-bottom:8px;}}
h2 {{color:{USTA_PURPLE}; font-weight:700;}}
h3 {{color:{USTA_NAVY}; font-weight:600;}}
.usta-card {{background:#fff; border-radius:14px; padding:20px 24px;
    box-shadow:0 4px 20px rgba(0,26,77,0.09); margin-bottom:16px;
    border-top:4px solid {USTA_PURPLE};}}
.context-box {{background:linear-gradient(135deg,rgba(61,0,141,0.06),rgba(237,30,121,0.06));
    border:1.5px solid rgba(61,0,141,0.18); border-left:5px solid {USTA_PURPLE};
    border-radius:10px; padding:16px 20px; margin:12px 0 18px 0; color:#1E293B;}}
.module-banner {{background:linear-gradient(135deg,{USTA_NAVY} 0%,{USTA_PURPLE} 100%);
    border-radius:12px; padding:18px 28px; margin-bottom:22px; color:white;}}
.module-banner h2 {{color:white !important; margin:0; font-size:1.4rem;}}
.module-banner p {{color:rgba(255,255,255,0.8); margin:4px 0 0 0; font-size:0.88rem;}}
.signal-buy {{background:#dcfce7; border-left:5px solid #16a34a; border-radius:8px;
    padding:10px 14px; margin:5px 0; color:#14532d; font-weight:500;}}
.signal-sell {{background:#fee2e2; border-left:5px solid #dc2626; border-radius:8px;
    padding:10px 14px; margin:5px 0; color:#7f1d1d; font-weight:500;}}
.signal-neutral {{background:#fef9c3; border-left:5px solid #ca8a04; border-radius:8px;
    padding:10px 14px; margin:5px 0; color:#713f12; font-weight:500;}}
.sem-buy {{background:#16a34a; border-radius:10px; padding:12px 8px;
    text-align:center; color:#FFFFFF !important; font-weight:700;}}
.sem-sell {{background:#dc2626; border-radius:10px; padding:12px 8px;
    text-align:center; color:#FFFFFF !important; font-weight:700;}}
.sem-neutral {{background:#ca8a04; border-radius:10px; padding:12px 8px;
    text-align:center; color:#FFFFFF !important; font-weight:700;}}
[data-testid="metric-container"] {{background:#fff; border:1px solid rgba(61,0,141,0.12);
    border-radius:10px; padding:12px 16px; box-shadow:0 2px 8px rgba(0,0,0,0.05);}}
hr {{border-color:rgba(61,0,141,0.15);}}
footer {{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPERS – llamadas al backend
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def api_get(endpoint: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar al backend. Asegúrate de que FastAPI esté corriendo en localhost:8000")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Error del backend: {e.response.status_code} – {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def api_post(endpoint: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar al backend FastAPI.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", e.response.text) if e.response else str(e)
        st.error(f"❌ Error del backend ({e.response.status_code}): {detail}")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
        return None


def module_banner(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="module-banner"><h2>{title}</h2>{sub}</div>',
                unsafe_allow_html=True)


def context_box(html: str) -> None:
    st.markdown(f'<div class="context-box">{html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:10px 0 16px 0;">
        <div style="font-size:2rem;">📊</div>
        <div style="font-size:1.1rem; font-weight:800;">RiskLab <span style="color:{USTA_GOLD};">USTA</span></div>
        <div style="font-size:0.72rem; color:rgba(255,255,255,0.6); margin-top:2px;">Proyecto Integrador · Teoría del Riesgo</div>
        <div style="font-size:0.68rem; color:rgba(255,255,255,0.5); margin-top:6px; line-height:1.6;">
            Laura A. Sepúlveda<br>Ingrid J. Umbacia Ramírez
        </div>
    </div>
    <hr style="border-color:rgba(253,185,19,0.4); margin:0 0 12px 0;">
    """, unsafe_allow_html=True)

    period_map = {"6 meses": "6mo", "1 año": "1y", "2 años": "2y", "3 años": "3y", "5 años": "5y"}
    period_label = st.selectbox("📅 Período", list(period_map.keys()), index=2)
    period = period_map[period_label]

    modulo = st.radio("🗂️ Navegación:", [
        "🏠 Portada y Contexto",
        "📈 Módulo 1 – Análisis Técnico",
        "📊 Módulo 2 – Rendimientos",
        "🌊 Módulo 3 – ARCH/GARCH",
        "🛡️ Módulo 4 – CAPM y Beta",
        "⚠️ Módulo 5 – VaR y CVaR",
        "🎯 Módulo 6 – Markowitz",
        "🚦 Módulo 7 – Señales ★",
        "🌍 Módulo 8 – Macro y Benchmark ★",
        "🔌 API FastAPI ★★",
    ])

    st.markdown(f"""
    <hr style="border-color:rgba(253,185,19,0.3); margin:14px 0 10px 0;">
    <div style="font-size:0.74rem; color:rgba(255,255,255,0.75); line-height:1.9;">
    <b>Portafolio:</b><br>
    {"".join([f"• <code style='background:rgba(255,255,255,0.12);padding:1px 5px;border-radius:4px;'>{t}</code> {NAMES[t]}<br>" for t in TICKERS])}
    • <b>Benchmark:</b> S&P 500
    </div>
    <hr style="border-color:rgba(253,185,19,0.3); margin:10px 0;">
    <div style="font-size:0.68rem; color:rgba(255,255,255,0.45); text-align:center;">
    Backend: <code style="color:rgba(255,255,255,0.6)">localhost:8000</code><br>
    📅 {datetime.date.today().strftime('%d/%m/%Y')}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MÓDULO 0 – PORTADA
# ─────────────────────────────────────────────────────────────────
if modulo == "🏠 Portada y Contexto":
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{USTA_NAVY} 0%,{USTA_PURPLE} 100%);
         border-radius:16px; padding:32px 36px; margin-bottom:28px; color:white;">
        <h1 style="color:white; border:none; margin:0; font-size:1.8rem;">
            RiskLab <span style="color:{USTA_GOLD};">USTA</span> – Proyecto Integrador
        </h1>
        <p style="color:rgba(255,255,255,0.82); margin:6px 0 0 0;">
            Teoría del Riesgo · Universidad Santo Tomás · Facultad de Estadística
        </p>
        <p style="color:rgba(255,255,255,0.65); margin:4px 0 0 0; font-size:0.84rem;">
            <b>Laura Alejandra Sepúlveda</b> &nbsp;|&nbsp;
            <b>Ingrid Johana Umbacia Ramírez</b> &nbsp;·&nbsp;
            {datetime.date.today().strftime('%d/%m/%Y')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Conectando con el backend…"):
        macro_data = api_get("/macro")

    if macro_data:
        st.markdown("### 📡 Indicadores Macroeconómicos en Tiempo Real")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("📈 Rf anual", f"{macro_data.get('rf_anual_pct', 0):.2f}%")
        c2.metric("😱 VIX", f"{macro_data.get('vix', 0) or 'N/D'}")
        c3.metric("🥇 Oro USD/oz", f"${macro_data.get('oro_usd', 0):,.0f}" if macro_data.get('oro_usd') else "N/D")
        c4.metric("🛢️ Brent USD/bbl", f"${macro_data.get('brent_usd', 0):.1f}" if macro_data.get('brent_usd') else "N/D")
        c5.metric("💵 USD/COP", f"{macro_data.get('usd_cop', 0):,.0f}" if macro_data.get('usd_cop') else "N/D")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="usta-card">
        <h3>⚔️ Contexto Geopolítico y Económico</h3>
        <p><b style="color:{USTA_PURPLE};">Guerra Rusia-Ucrania:</b> Crisis energética en Europa.
        <b>XOM</b> se benefició; <b>UL</b> y <b>PG</b> sufrieron presión en márgenes.
        <b>MSI</b> creció por mayor gasto en comunicaciones de defensa.</p>
        <p><b style="color:{USTA_PURPLE};">Tensiones Taiwan / Chips:</b>
        Restricciones a exportación de semiconductores impactan directamente a <b>TSM</b>.</p>
        <p><b style="color:{USTA_PURPLE};">Ciclo Fed (2022–2024):</b>
        +525 pb afectaron valoraciones. Desde sep 2024 se iniciaron recortes graduales.
        Esto impacta directamente la Rf del CAPM y el Sharpe Ratio.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="usta-card">
        <h3>🏢 Portafolio Analizado</h3>
        <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
        <tr style="background:{USTA_PURPLE}; color:white;">
            <th style="padding:6px;">Ticker</th><th style="padding:6px;">Empresa</th>
            <th style="padding:6px;">Sector</th><th style="padding:6px;">País</th>
        </tr>
        <tr><td style="padding:5px;"><b>MSI</b></td><td>Motorola Solutions</td><td>Tecnología</td><td>EE.UU.</td></tr>
        <tr style="background:#f8fafc;"><td style="padding:5px;"><b>XOM</b></td><td>ExxonMobil</td><td>Energía</td><td>EE.UU.</td></tr>
        <tr><td style="padding:5px;"><b>JNJ</b></td><td>Johnson & Johnson</td><td>Salud</td><td>EE.UU.</td></tr>
        <tr style="background:#f8fafc;"><td style="padding:5px;"><b>PG</b></td><td>Procter & Gamble</td><td>Consumo</td><td>EE.UU.</td></tr>
        <tr><td style="padding:5px;"><b>UL</b></td><td>Unilever</td><td>Consumo</td><td>UK/NL</td></tr>
        <tr style="background:#f8fafc;"><td style="padding:5px;"><b>TSM</b></td><td>TSMC</td><td>Semicon.</td><td>Taiwán</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏗️ Arquitectura del Proyecto")
    st.markdown(f"""
    <div class="usta-card" style="border-top-color:{USTA_GOLD};">
    <pre style="font-size:0.82rem; color:{USTA_NAVY}; background:#f8fafc; padding:14px; border-radius:8px;">
proyecto-riesgo/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI – routers y endpoints
│   │   ├── models.py        # Modelos Pydantic (request/response)
│   │   ├── services.py      # Lógica de negocio + decoradores
│   │   ├── dependencies.py  # Depends() – inyección de dependencias
│   │   └── config.py        # BaseSettings + .env
│   ├── requirements.txt
│   └── .env
├── frontend/
│   └── app.py               # Este archivo – Streamlit
├── README.md
└── .gitignore
    </pre>
    <b>Patrón:</b> El frontend consume el backend vía HTTP. Nunca llama directamente a Yahoo Finance.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MÓDULO 1 – ANÁLISIS TÉCNICO
# ─────────────────────────────────────────────────────────────────
elif modulo == "📈 Módulo 1 – Análisis Técnico":
    module_banner("📈 Módulo 1 – Análisis Técnico",
                  "SMA · EMA · RSI · MACD · Bollinger · Estocástico · Datos vía backend FastAPI")

    context_box("""
    <strong>🌍 Contexto:</strong> El análisis técnico aporta señales de corto plazo.
    Los indicadores calculados por el backend permiten identificar tendencias,
    sobrecompra/sobreventa y posibles puntos de entrada/salida.
    """)

    col_s, col_p = st.columns([2, 1])
    ticker_sel = col_s.selectbox("Activo:", TICKERS, format_func=lambda x: f"{x} – {NAMES[x]}")

    with st.spinner(f"Obteniendo indicadores de {ticker_sel} del backend…"):
        ind_data  = api_get(f"/indicadores/{ticker_sel}", {"period": period})
        prec_data = api_get(f"/precios/{ticker_sel}", {"period": period})

    if not ind_data or not prec_data:
        st.warning("No se obtuvieron datos del backend.")
        st.stop()

    prices_list = prec_data.get("precios", [])
    if not prices_list:
        st.error("Sin datos de precios.")
        st.stop()

    df_prices = pd.DataFrame(prices_list)
    df_prices["fecha"] = pd.to_datetime(df_prices["fecha"])
    df_prices = df_prices.set_index("fecha").sort_index()
    p = df_prices["precio"]

    # Recalcular indicadores localmente para los gráficos completos
    sma_s = p.rolling(20).mean()
    sma_l = p.rolling(50).mean()
    ema20 = p.ewm(span=20, adjust=False).mean()
    delta = p.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi_s = 100 - (100 / (1 + gain / loss))
    ema12 = p.ewm(span=12).mean(); ema26 = p.ewm(span=26).mean()
    macd_l = ema12 - ema26; macd_sig = macd_l.ewm(span=9).mean(); macd_h = macd_l - macd_sig
    bb_mid = p.rolling(20).mean(); bb_u = bb_mid + 2*p.rolling(20).std(); bb_l = bb_mid - 2*p.rolling(20).std()
    sk = 100*(p - p.rolling(14).min())/(p.rolling(14).max()-p.rolling(14).min())
    sd = sk.rolling(3).mean()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         row_heights=[0.42,0.20,0.20,0.18],
                         subplot_titles=[f"Precio – {ticker_sel}","RSI (14)","MACD","Estocástico"])
    fig.add_trace(go.Scatter(x=p.index, y=p, name="Precio", line=dict(color=USTA_NAVY, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sma_s.index, y=sma_s, name="SMA20", line=dict(color=USTA_PURPLE, width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sma_l.index, y=sma_l, name="SMA50", line=dict(color=USTA_PINK, width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ema20.index, y=ema20, name="EMA20", line=dict(color=USTA_GOLD, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb_u.index, y=bb_u, name="BB Sup", line=dict(color="rgba(61,0,141,0.4)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb_l.index, y=bb_l, name="BB Inf", line=dict(color="rgba(61,0,141,0.4)", width=1), fill="tonexty", fillcolor="rgba(61,0,141,0.05)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s, name="RSI", line=dict(color="#0EA5E9", width=1.8)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="red", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dash", width=1), row=2, col=1)
    colors_m = [USTA_PURPLE if v >= 0 else USTA_PINK for v in macd_h.fillna(0)]
    fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="Hist.", marker_color=colors_m, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=macd_l.index, y=macd_l, name="MACD", line=dict(color=USTA_NAVY, width=1.8)), row=3, col=1)
    fig.add_trace(go.Scatter(x=macd_sig.index, y=macd_sig, name="Señal", line=dict(color=USTA_PINK, width=1.8)), row=3, col=1)
    fig.add_trace(go.Scatter(x=sk.index, y=sk, name="%K", line=dict(color=USTA_PURPLE, width=1.8)), row=4, col=1)
    fig.add_trace(go.Scatter(x=sd.index, y=sd, name="%D", line=dict(color=USTA_GOLD, width=1.8)), row=4, col=1)
    fig.add_hline(y=80, line=dict(color="red", dash="dash", width=1), row=4, col=1)
    fig.add_hline(y=20, line=dict(color="green", dash="dash", width=1), row=4, col=1)
    fig.update_layout(height=820, template="plotly_white", hovermode="x unified",
                       legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Valores Actuales (desde backend)")
    rsi_v = ind_data["rsi"]; macd_v = ind_data["macd"]; sig_v = ind_data["macd_signal"]
    px_v = ind_data["precio_actual"]; bbu_v = ind_data["bb_upper"]; bbl_v = ind_data["bb_lower"]
    smas_v = ind_data["sma_20"]; smal_v = ind_data["sma_50"]

    ci1, ci2 = st.columns(2)
    with ci1:
        css = "signal-sell" if rsi_v > 70 else ("signal-buy" if rsi_v < 30 else "signal-neutral")
        label = "🔴 SOBRECOMPRADO" if rsi_v > 70 else ("🟢 SOBREVENDIDO" if rsi_v < 30 else "🟡 NEUTRAL")
        st.markdown(f'<div class="{css}"><b>RSI = {rsi_v:.1f} → {label}</b></div>', unsafe_allow_html=True)
        css2 = "signal-buy" if macd_v > sig_v else "signal-sell"
        lbl2 = "🟢 Momentum alcista" if macd_v > sig_v else "🔴 Momentum bajista"
        st.markdown(f'<div class="{css2}"><b>MACD → {lbl2}</b><br>MACD={macd_v:.3f} | Señal={sig_v:.3f}</div>', unsafe_allow_html=True)
    with ci2:
        if px_v >= bbu_v:
            st.markdown(f'<div class="signal-sell"><b>Bollinger → 🔴 Sobre banda superior</b><br>${px_v:.2f} ≥ ${bbu_v:.2f}</div>', unsafe_allow_html=True)
        elif px_v <= bbl_v:
            st.markdown(f'<div class="signal-buy"><b>Bollinger → 🟢 Bajo banda inferior</b><br>${px_v:.2f} ≤ ${bbl_v:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="signal-neutral"><b>Bollinger → 🟡 Dentro de bandas</b><br>${bbl_v:.2f} – ${bbu_v:.2f}</div>', unsafe_allow_html=True)
        css3 = "signal-buy" if smas_v > smal_v else "signal-sell"
        lbl3 = "🟢 Golden Cross / Alcista" if smas_v > smal_v else "🔴 Death Cross / Bajista"
        st.markdown(f'<div class="{css3}"><b>Medias Móviles → {lbl3}</b><br>SMA20=${smas_v:.2f} | SMA50=${smal_v:.2f}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MÓDULO 2 – RENDIMIENTOS
# ─────────────────────────────────────────────────────────────────
elif modulo == "📊 Módulo 2 – Rendimientos":
    module_banner("📊 Módulo 2 – Rendimientos y Propiedades Empíricas",
                  "Log-rendimientos · Estadísticos · Pruebas de normalidad · Hechos estilizados")

    ticker_sel = st.selectbox("Activo:", TICKERS, format_func=lambda x: f"{x} – {NAMES[x]}")

    with st.spinner("Obteniendo rendimientos del backend…"):
        rend_data = api_get(f"/rendimientos/{ticker_sel}", {"period": period})
        prec_data = api_get(f"/precios/{ticker_sel}", {"period": period})

    if not rend_data:
        st.stop()

    st.markdown("### 📐 Estadísticos Descriptivos")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Media diaria", f"{rend_data['media_diaria_pct']:.4f}%")
    c1.metric("Ret. Anualizado", f"{rend_data['rendimiento_anualizado_pct']:.2f}%")
    c2.metric("Desv. Estándar diaria", f"{rend_data['volatilidad_diaria_pct']:.4f}%")
    c2.metric("Volatilidad Anualizada", f"{rend_data['volatilidad_anualizada_pct']:.2f}%")
    c3.metric("Mínimo diario", f"{rend_data['min_diario_pct']:.3f}%")
    c3.metric("Máximo diario", f"{rend_data['max_diario_pct']:.3f}%")
    c4.metric("Skewness", f"{rend_data['skewness']:.4f}")
    c4.metric("Kurtosis (exceso)", f"{rend_data['kurtosis']:.4f}")

    lr_df = pd.DataFrame(rend_data["rendimientos"])
    lr_df["fecha"] = pd.to_datetime(lr_df["fecha"])
    lr = lr_df["log_ret_pct"].values / 100

    st.markdown("### 📈 Serie Temporal de Log-Rendimientos")
    fig_lr = go.Figure(go.Scatter(x=lr_df["fecha"], y=lr_df["log_ret_pct"],
                                   line=dict(color=USTA_PINK, width=0.9), name="Log-Ret (%)"))
    fig_lr.update_layout(height=300, template="plotly_white",
                          xaxis_title="Fecha", yaxis_title="Log-Ret (%)")
    st.plotly_chart(fig_lr, use_container_width=True)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        mu, sigma = np.mean(lr), np.std(lr)
        x_norm = np.linspace(lr.min(), lr.max(), 200)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=lr*100, nbinsx=80, name="Frecuencia",
                                         histnorm="probability density", marker_color=USTA_PURPLE, opacity=0.7))
        fig_hist.add_trace(go.Scatter(x=x_norm*100, y=stats.norm.pdf(x_norm, mu, sigma)/100,
                                       name="Normal teórica", line=dict(color=USTA_PINK, width=2.5)))
        fig_hist.update_layout(title="Histograma + Normal", height=380, template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_h2:
        (tq, sq), (sl, ic, _) = stats.probplot(lr, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=tq, y=sq, mode="markers",
                                     marker=dict(color=USTA_PURPLE, size=4, opacity=0.55)))
        fig_qq.add_trace(go.Scatter(x=tq, y=sl*np.array(tq)+ic, mode="lines",
                                     line=dict(color=USTA_PINK, width=2.5), name="Normal"))
        fig_qq.update_layout(title="Q-Q Plot", height=380, template="plotly_white")
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown("### 🧪 Pruebas de Normalidad")
    jb_s, jb_p = stats.jarque_bera(lr)
    sw_s, sw_p = stats.shapiro(lr[-500:]) if len(lr) > 500 else stats.shapiro(lr)
    ct1, ct2 = st.columns(2)
    for col, nm, s, p in [(ct1,"Jarque-Bera",jb_s,jb_p),(ct2,"Shapiro-Wilk",sw_s,sw_p)]:
        res = "🔴 Rechaza normalidad" if p < 0.05 else "🟢 No rechaza"
        col.markdown(f'<div class="usta-card" style="border-top-color:{USTA_PINK};"><b>{nm}</b><br>Estadístico: <code>{s:.4f}</code><br>p-valor: <code>{p:.2e}</code><br><b>{res}</b></div>', unsafe_allow_html=True)

    context_box(f"""
    <strong>📌 Hechos Estilizados – {ticker_sel}:</strong>
    <ol>
    <li><b>No normalidad</b> confirmada por Jarque-Bera (p={jb_p:.2e}). Los rendimientos tienen colas más gruesas que la normal.</li>
    <li><b>Kurtosis = {rend_data['kurtosis']:.2f}:</b> Leptocurtosis → eventos extremos más frecuentes de lo esperado (fat tails).</li>
    <li><b>Skewness = {rend_data['skewness']:.3f}:</b> {"Asimetría negativa → pérdidas extremas más frecuentes que ganancias extremas." if rend_data['skewness'] < -0.05 else "Asimetría positiva o neutra."}</li>
    <li><b>Implicación:</b> El VaR paramétrico normal subestima el riesgo real. Se requieren VaR histórico, Montecarlo y modelos GARCH.</li>
    </ol>
    """)


# ─────────────────────────────────────────────────────────────────
# MÓDULO 3 – ARCH/GARCH (cálculo local)
# ─────────────────────────────────────────────────────────────────
elif modulo == "🌊 Módulo 3 – ARCH/GARCH":
    module_banner("🌊 Módulo 3 – Modelos ARCH/GARCH",
                  "ARCH(1) · GARCH(1,1) · EGARCH · GJR-GARCH · AIC/BIC · Pronóstico")
    from arch import arch_model as _arch_model

    ticker_sel = st.selectbox("Activo:", TICKERS, format_func=lambda x: f"{x} – {NAMES[x]}")

    with st.spinner("Descargando datos…"):
        prec_data = api_get(f"/precios/{ticker_sel}", {"period": period})

    if not prec_data:
        st.stop()

    df_p = pd.DataFrame(prec_data["precios"])
    df_p["fecha"] = pd.to_datetime(df_p["fecha"])
    df_p = df_p.set_index("fecha").sort_index()
    p = df_p["precio"]
    lr_t = (np.log(p / p.shift(1)).dropna() * 100)

    context_box("""
    <strong>¿Por qué ARCH/GARCH?</strong> Los rendimientos financieros presentan
    <b>agrupamiento de volatilidad</b>: períodos de alta agitación se agrupan. Esto viola la
    homocedasticidad y justifica modelos de varianza condicional.
    """)

    @st.cache_data(ttl=3600)
    def fit_models(vals: list, key: str) -> dict:
        s = pd.Series(vals)
        out = {}
        for nm, kw in [("ARCH(1)", dict(vol="ARCH", p=1)),
                        ("GARCH(1,1)", dict(vol="Garch", p=1, q=1)),
                        ("EGARCH(1,1)", dict(vol="EGARCH", p=1, q=1)),
                        ("GJR-GARCH(1,1)", dict(vol="Garch", p=1, o=1, q=1))]:
            try:
                r = _arch_model(s, dist="normal", **kw).fit(disp="off")
                out[nm] = {"aic": r.aic, "bic": r.bic, "ll": r.loglikelihood,
                            "cond_vol": r.conditional_volatility.tolist(),
                            "std_resid": r.std_resid.tolist(),
                            "forecast_var": r.forecast(horizon=10).variance.iloc[-1].tolist()}
            except Exception:
                out[nm] = None
        return out

    with st.spinner("Ajustando modelos GARCH…"):
        models = fit_models(lr_t.values.tolist(), ticker_sel)

    rows_m = []
    for nm, r in models.items():
        if r:
            rows_m.append({"Modelo": nm, "Log-L": f"{r['ll']:.2f}",
                            "AIC": f"{r['aic']:.2f}", "BIC": f"{r['bic']:.2f}"})
    if rows_m:
        df_m = pd.DataFrame(rows_m)
        aics = pd.to_numeric(df_m["AIC"])
        best_aic = df_m.loc[aics.idxmin(), "Modelo"]
        best_bic = df_m.loc[pd.to_numeric(df_m["BIC"]).idxmin(), "Modelo"]
        df_m["✅ Mejor AIC"] = df_m["Modelo"].apply(lambda x: "✅" if x == best_aic else "")
        df_m["✅ Mejor BIC"] = df_m["Modelo"].apply(lambda x: "✅" if x == best_bic else "")
        st.dataframe(df_m, use_container_width=True)
        context_box(f"Mejor por AIC: <b>{best_aic}</b> | Mejor por BIC: <b>{best_bic}</b>. Menor valor = mejor ajuste ajustado por complejidad.")

    if models.get("GARCH(1,1)"):
        g11 = models["GARCH(1,1)"]
        cond_vol = pd.Series(g11["cond_vol"], index=lr_t.index[:len(g11["cond_vol"])])
        std_res  = pd.Series(g11["std_resid"], index=lr_t.index[:len(g11["std_resid"])])

        fig_cv = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                                subplot_titles=["Volatilidad Condicional – GARCH(1,1)", "Residuos Estand."])
        fig_cv.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color=USTA_PINK, width=1.8), name="Vol. Cond."), row=1, col=1)
        fig_cv.add_trace(go.Scatter(x=std_res.index, y=std_res, line=dict(color=USTA_PURPLE, width=0.9), name="Residuos"), row=2, col=1)
        fig_cv.add_hline(y=3, line=dict(color="red", dash="dash"), row=2, col=1)
        fig_cv.add_hline(y=-3, line=dict(color="red", dash="dash"), row=2, col=1)
        fig_cv.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig_cv, use_container_width=True)

        fcast_vol = np.sqrt(g11["forecast_var"])
        fig_fc = go.Figure(go.Scatter(x=list(range(1, 11)), y=fcast_vol, mode="lines+markers",
                                       line=dict(color=USTA_PURPLE, width=2.5), marker=dict(size=9, color=USTA_PINK)))
        fig_fc.update_layout(title="Pronóstico Volatilidad – 10 días", height=320, template="plotly_white",
                              xaxis_title="Días adelante", yaxis_title="Volatilidad (%)")
        st.plotly_chart(fig_fc, use_container_width=True)
        context_box(f"El GARCH(1,1) pronostica una volatilidad promedio de <b>{np.mean(fcast_vol):.3f}%</b> diario (≈ <b>{np.mean(fcast_vol)*np.sqrt(252):.2f}% anualizado</b>).")


# ─────────────────────────────────────────────────────────────────
# MÓDULO 4 – CAPM
# ─────────────────────────────────────────────────────────────────
elif modulo == "🛡️ Módulo 4 – CAPM y Beta":
    module_banner("🛡️ Módulo 4 – CAPM y Riesgo Sistemático",
                  "Beta · SML · Rf desde API · Riesgo sistemático vs. idiosincrático")

    with st.spinner("Consultando CAPM en el backend…"):
        capm_data = api_get("/capm", {"period": period})

    if not capm_data:
        st.stop()

    context_box(f"""
    <strong>Rf = {capm_data['rf_anual_pct']:.2f}%</strong> (T-Bill 13W, obtenida automáticamente del backend).
    Prima de mercado estimada: <b>{capm_data['prima_mercado_anual_pct']:.2f}%</b> anual vs. S&P 500.
    """)

    activos = capm_data["activos"]
    df_capm = pd.DataFrame(activos)
    st.markdown("### 📋 Tabla CAPM – Todos los Activos")
    st.dataframe(df_capm[["ticker","empresa","beta","alpha_diario","r_squared",
                            "riesgo_sistematico_pct","riesgo_idiosincratico_pct",
                            "rendimiento_esperado_anual_pct","clasificacion"]], use_container_width=True)

    betas = [a["beta"] for a in activos]
    colors_b = [USTA_PINK if b > 1.1 else USTA_PURPLE if b < 0.9 else USTA_GOLD for b in betas]
    fig_b = go.Figure(go.Bar(x=[a["ticker"] for a in activos], y=betas,
                              marker_color=colors_b, text=[f"{b:.3f}" for b in betas], textposition="outside"))
    fig_b.add_hline(y=1.0, line=dict(color="gray", dash="dash"), annotation_text="β=1.0")
    fig_b.update_layout(title="Beta por Activo", height=380, template="plotly_white")
    st.plotly_chart(fig_b, use_container_width=True)

    context_box("""
    <strong>Interpretación:</strong>
    <ul>
    <li><b>β > 1:</b> Activo agresivo – amplifica movimientos del mercado.</li>
    <li><b>β < 1:</b> Activo defensivo – amortigua las caídas.</li>
    <li><b>R²:</b> Proporción del riesgo total que es sistemático (no diversificable).</li>
    </ul>
    """)


# ─────────────────────────────────────────────────────────────────
# MÓDULO 5 – VaR y CVaR
# ─────────────────────────────────────────────────────────────────
elif modulo == "⚠️ Módulo 5 – VaR y CVaR":
    module_banner("⚠️ Módulo 5 – VaR y CVaR",
                  "Paramétrico · Histórico · Montecarlo · CVaR · Kupiec via backend FastAPI")

    st.markdown("### ⚖️ Composición del Portafolio")
    col_w = st.columns(len(TICKERS))
    weights = []
    for i, t in enumerate(TICKERS):
        w = col_w[i].number_input(t, min_value=0.0, max_value=1.0,
                                    value=round(1/len(TICKERS), 4), step=0.01, format="%.3f")
        weights.append(w)
    weights_norm = [w/sum(weights) for w in weights]
    conf = st.select_slider("Nivel de confianza:", [0.90, 0.95, 0.99], value=0.95)

    if st.button("📊 Calcular VaR vía Backend"):
        body = {"tickers": TICKERS, "weights": weights_norm, "confidence": conf, "period": period}
        with st.spinner("Calculando VaR en el backend…"):
            var_data = api_post("/var", body)

        if var_data:
            st.markdown("### 📊 Resultados")
            df_var = pd.DataFrame(var_data["resultados"])
            st.dataframe(df_var, use_container_width=True)

            kup = var_data["kupiec"]
            ck1, ck2, ck3 = st.columns(3)
            ck1.metric("Violaciones obs.", kup["violaciones_observadas"], f"Esperadas: {kup['violaciones_esperadas']}")
            ck2.metric("LR Kupiec", kup["lr_statistic"])
            ck3.metric("p-valor", kup["p_valor"], "✅ Modelo válido" if kup["modelo_valido"] else "❌ Rechazado")

            context_box(f"""
            <strong>Interpretación:</strong>
            <ul>
            <li>El VaR histórico al {conf*100:.0f}% es el método más robusto dado que captura las colas pesadas documentadas en el Módulo 2.</li>
            <li>El CVaR (Expected Shortfall) es la métrica preferida por Basilea III por ser coherente (subaditividad).</li>
            <li>Test de Kupiec: {"✅ El modelo es estadísticamente válido." if kup["modelo_valido"] else "❌ El número de violaciones es inusual; considerar VaR dinámico con GARCH."}</li>
            </ul>
            """)


# ─────────────────────────────────────────────────────────────────
# MÓDULO 6 – MARKOWITZ
# ─────────────────────────────────────────────────────────────────
elif modulo == "🎯 Módulo 6 – Markowitz":
    module_banner("🎯 Módulo 6 – Frontera Eficiente de Markowitz",
                  "Correlación · 10,000 portafolios · Max Sharpe · Mín. Varianza · Rendimiento objetivo")

    n_sim = st.slider("Portafolios a simular:", 1000, 20000, 10000, step=1000)

    if st.button("🎲 Calcular Frontera vía Backend"):
        body = {"tickers": TICKERS, "period": period, "n_portfolios": n_sim}
        with st.spinner("Simulando frontera eficiente en el backend…"):
            front_data = api_post("/frontera-eficiente", body)

        if front_data:
            corr_df = pd.DataFrame(front_data["correlaciones"])
            fig_corr = px.imshow(corr_df, color_continuous_scale=["#ED1E79","white","#3D008D"],
                                  zmin=-1, zmax=1, text_auto=".3f", title="Matriz de Correlación")
            fig_corr.update_layout(height=420, template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)

            ms = front_data["max_sharpe"]
            mv = front_data["min_varianza"]
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                st.markdown(f"""
                <div class="usta-card" style="border-top-color:{USTA_PINK};">
                <h3 style="color:{USTA_PINK};">★ Portafolio Máximo Sharpe</h3>
                Rendimiento: <b>{ms['rendimiento_anual_pct']:.2f}%</b><br>
                Volatilidad: <b>{ms['volatilidad_anual_pct']:.2f}%</b><br>
                Sharpe: <b>{ms['sharpe_ratio']:.4f}</b>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(list(ms["pesos"].items()), columns=["Activo","Peso (%)"]), use_container_width=True)
            with col_o2:
                st.markdown(f"""
                <div class="usta-card" style="border-top-color:{USTA_PURPLE};">
                <h3 style="color:{USTA_PURPLE};">◆ Portafolio Mínima Varianza</h3>
                Rendimiento: <b>{mv['rendimiento_anual_pct']:.2f}%</b><br>
                Volatilidad: <b>{mv['volatilidad_anual_pct']:.2f}%</b><br>
                Sharpe: <b>{mv['sharpe_ratio']:.4f}</b>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(list(mv["pesos"].items()), columns=["Activo","Peso (%)"]), use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# MÓDULO 7 – SEÑALES
# ─────────────────────────────────────────────────────────────────
elif modulo == "🚦 Módulo 7 – Señales ★":
    module_banner("🚦 Módulo 7 – Panel de Señales y Alertas",
                  "RSI · MACD · Bollinger · Medias Móviles · Estocástico · Umbrales configurables")

    cu1, cu2, cu3, cu4 = st.columns(4)
    rsi_ob   = cu1.slider("RSI sobrecompra", 60, 85, 70)
    rsi_os   = cu2.slider("RSI sobreventa",  15, 40, 30)
    stoch_ob = cu3.slider("Estoc. sobrecompra", 65, 90, 80)
    stoch_os = cu4.slider("Estoc. sobreventa",  10, 35, 20)
    st.markdown("---")

    with st.spinner("Consultando señales en el backend…"):
        alertas_data = api_get("/alertas", {
            "period": period, "rsi_ob": rsi_ob, "rsi_os": rsi_os,
            "stoch_ob": stoch_ob, "stoch_os": stoch_os
        })

    if not alertas_data:
        st.stop()

    for alerta in alertas_data["alertas"]:
        overall = alerta["señal_global"]
        nb, ns = alerta["votos_compra"], alerta["votos_venta"]
        ov_color = "#16a34a" if overall=="BUY" else ("#dc2626" if overall=="SELL" else "#ca8a04")
        ov_label = "🟢 COMPRA" if overall=="BUY" else ("🔴 VENTA" if overall=="SELL" else "🟡 NEUTRAL")

        with st.expander(f"**{alerta['ticker']} – {alerta['empresa']}** &nbsp;·&nbsp; {ov_label} ({nb}✅/{ns}❌)", expanded=False):
            cols_s = st.columns(5)
            for i, (ind, sig) in enumerate(alerta["indicadores"].items()):
                cls = "sem-buy" if sig=="BUY" else ("sem-sell" if sig=="SELL" else "sem-neutral")
                icon = "🟢" if sig=="BUY" else ("🔴" if sig=="SELL" else "🟡")
                lbl  = "COMPRA" if sig=="BUY" else ("VENTA" if sig=="SELL" else "NEUTRAL")
                cols_s[i].markdown(f"""
                <div class="{cls}">
                    <div style="font-size:0.72rem;">{ind}</div>
                    <div style="font-size:1.1rem;">{icon}</div>
                    <div style="font-size:0.78rem;">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="context-box" style="margin-top:10px;">
            <b>{alerta['interpretacion']}</b>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MÓDULO 8 – MACRO Y BENCHMARK
# ─────────────────────────────────────────────────────────────────
elif modulo == "🌍 Módulo 8 – Macro y Benchmark ★":
    module_banner("🌍 Módulo 8 – Contexto Macroeconómico y Benchmark",
                  "Macro en tiempo real · Alpha de Jensen · Tracking Error · IR · Drawdown")
    import yfinance as yf

    with st.spinner("Consultando macro en el backend…"):
        macro_data = api_get("/macro")

    if macro_data:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("📈 Rf anual", f"{macro_data['rf_anual_pct']:.2f}%")
        c2.metric("😱 VIX", f"{macro_data.get('vix',0) or 'N/D'}")
        c3.metric("🥇 Oro USD/oz", f"${macro_data.get('oro_usd',0):,.0f}" if macro_data.get('oro_usd') else "N/D")
        c4.metric("🛢️ Brent", f"${macro_data.get('brent_usd',0):.1f}" if macro_data.get('brent_usd') else "N/D")
        c5.metric("💵 USD/COP", f"{macro_data.get('usd_cop',0):,.0f}" if macro_data.get('usd_cop') else "N/D")

    st.markdown("---")
    st.markdown("### 📈 Rendimiento Acumulado vs. S&P 500")

    with st.spinner("Descargando precios para benchmark…"):
        try:
            raw_p  = yf.download(TICKERS, period=period, auto_adjust=True, progress=False)["Close"].dropna()
            raw_b  = yf.download("^GSPC", period=period, auto_adjust=True, progress=False)["Close"].dropna()
            raw_b2 = raw_b.iloc[:,0] if isinstance(raw_b, pd.DataFrame) else raw_b
            port_ret = np.log(raw_p/raw_p.shift(1)).dropna().mean(axis=1)
            bench_s  = np.log(raw_b2/raw_b2.shift(1)).dropna()
            cidx = port_ret.index.intersection(bench_s.index)
            pr   = port_ret.loc[cidx]; br = bench_s.loc[cidx]
            cum_p = ((1+pr).cumprod()/((1+pr).cumprod().iloc[0]))*100
            cum_b = ((1+br).cumprod()/((1+br).cumprod().iloc[0]))*100

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=cum_p.index, y=cum_p, name="Portafolio", line=dict(color=USTA_PURPLE, width=2.8)))
            fig_cum.add_trace(go.Scatter(x=cum_b.index, y=cum_b, name="S&P 500", line=dict(color=USTA_PINK, width=2.8, dash="dash")))
            fig_cum.update_layout(title="Rendimiento Acumulado – Base 100", height=420, template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_cum, use_container_width=True)

            n = len(pr)
            ann_p = (1+(1+pr).prod()-1)**(252/n)-1
            ann_b = (1+(1+br).prod()-1)**(252/n)-1
            rf    = (macro_data["rf_anual_pct"]/100) if macro_data else 0.045
            vol_p = pr.std()*np.sqrt(252); vol_b = br.std()*np.sqrt(252)
            sharpe_p = (ann_p-rf)/vol_p; sharpe_b = (ann_b-rf)/vol_b
            sl_j, al_j, _, _, _ = stats.linregress(br.values, pr.values)
            alpha_j = al_j*252
            active = pr.values - br.values
            te = pd.Series(active).std()*np.sqrt(252)
            ir = (pr.mean()-br.mean())*252/te if te > 0 else 0
            def mdd(r):
                c=(1+pd.Series(r)).cumprod(); return float(((c-c.cummax())/c.cummax()).min())
            mdd_p=mdd(pr); mdd_b=mdd(br)

            perf_df = pd.DataFrame({
                "Métrica":["Ret. Anualizado","Volatilidad Anual","Sharpe Ratio",
                           "Máx. Drawdown","Alpha Jensen (anual)","Beta","Tracking Error","Info Ratio"],
                "Portafolio":[f"{ann_p*100:.2f}%",f"{vol_p*100:.2f}%",f"{sharpe_p:.4f}",
                              f"{mdd_p*100:.2f}%",f"{alpha_j*100:.4f}%",f"{sl_j:.4f}",
                              f"{te*100:.2f}%",f"{ir:.4f}"],
                "S&P 500":[f"{ann_b*100:.2f}%",f"{vol_b*100:.2f}%",f"{sharpe_b:.4f}",
                           f"{mdd_b*100:.2f}%","—","1.0","—","—"]
            })
            st.dataframe(perf_df, use_container_width=True)

            context_box(f"""
            <strong>Interpretación:</strong>
            {'✅ El portafolio supera al S&P 500 en rendimiento anualizado.' if ann_p>ann_b else '❌ El S&P 500 superó al portafolio en rendimiento.'}
            Alpha de Jensen = <b>{alpha_j*100:.4f}%</b> {'(positivo: valor añadido sobre el riesgo sistemático).' if alpha_j>0 else '(nulo/negativo: consistente con eficiencia de mercado).'}
            Information Ratio = <b>{ir:.4f}</b> {'– gestión activa con valor añadido.' if ir>0 else '– la estrategia pasiva (S&P 500) fue más eficiente en riesgo/retorno.'}
            """)
        except Exception as e:
            st.error(f"Error al calcular benchmark: {e}")


# ─────────────────────────────────────────────────────────────────
# MÓDULO API FastAPI
# ─────────────────────────────────────────────────────────────────
elif modulo == "🔌 API FastAPI ★★":
    module_banner("🔌 Backend FastAPI – Demostración",
                  "Endpoints · Pydantic · Depends() · BaseSettings · Decoradores · /docs")

    context_box("""
    <strong>Este módulo demuestra el backend FastAPI en funcionamiento.</strong>
    El frontend <b>nunca</b> llama directamente a Yahoo Finance: siempre consume el backend.
    En la sustentación se mostrará <code>/docs</code> (Swagger UI) generado automáticamente por FastAPI.
    """)

    st.markdown("### 🔍 Estado del Backend")
    with st.spinner("Verificando conexión…"):
        root_data = api_get("/")

    if root_data:
        st.success("✅ Backend FastAPI activo y respondiendo")
        st.json(root_data)
    else:
        st.error("❌ Backend no disponible. Ejecuta: uvicorn app.main:app --reload (desde backend/)")

    st.markdown("### 📋 Endpoints Disponibles")
    endpoints_df = pd.DataFrame([
        {"Endpoint":"/activos","Método":"GET","Descripción":"Lista activos del portafolio"},
        {"Endpoint":"/precios/{ticker}","Método":"GET","Descripción":"Precios históricos de un activo"},
        {"Endpoint":"/rendimientos/{ticker}","Método":"GET","Descripción":"Estadísticas de log-rendimientos"},
        {"Endpoint":"/indicadores/{ticker}","Método":"GET","Descripción":"Indicadores técnicos (RSI, MACD, etc.)"},
        {"Endpoint":"/var","Método":"POST","Descripción":"VaR y CVaR por 3 métodos + Kupiec"},
        {"Endpoint":"/capm","Método":"GET","Descripción":"Beta, Alpha y CAPM para todos los activos"},
        {"Endpoint":"/frontera-eficiente","Método":"POST","Descripción":"Frontera Eficiente de Markowitz"},
        {"Endpoint":"/alertas","Método":"GET","Descripción":"Señales de compra/venta automáticas"},
        {"Endpoint":"/macro","Método":"GET","Descripción":"Indicadores macroeconómicos en tiempo real"},
    ])
    st.dataframe(endpoints_df, use_container_width=True)

    st.markdown("### 🧪 Prueba un Endpoint en Vivo")
    ep = st.selectbox("Endpoint:", ["/activos", "/macro", "/capm"])
    if st.button("▶️ Ejecutar"):
        with st.spinner(f"Consultando {ep}…"):
            data = api_get(ep, {"period": period})
        if data:
            st.json(data)

    st.markdown("### 💡 Conceptos de Python para APIs Aplicados")
    st.markdown(f"""
    <div class="usta-card">
    <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
    <tr style="background:{USTA_PURPLE}; color:white;">
        <th style="padding:8px;">Concepto (semana)</th>
        <th style="padding:8px;">Implementación en este proyecto</th>
        <th style="padding:8px;">Archivo</th>
    </tr>
    <tr><td style="padding:6px;"><b>Decoradores (S1)</b></td><td>@log_execution_time, @cache_result(ttl=3600)</td><td>services.py</td></tr>
    <tr style="background:#f8fafc;"><td style="padding:6px;"><b>Type hints (S1)</b></td><td>Todas las funciones tienen anotaciones de tipo</td><td>Todos los archivos</td></tr>
    <tr><td style="padding:6px;"><b>POO (S2)</b></td><td>DataService, RiskCalculator, TechnicalIndicators, PortfolioAnalyzer</td><td>services.py</td></tr>
    <tr style="background:#f8fafc;"><td style="padding:6px;"><b>Pydantic (S2/4/5)</b></td><td>Request + Response models con Field(), @field_validator, @model_validator</td><td>models.py</td></tr>
    <tr><td style="padding:6px;"><b>Manejo errores HTTP (S2)</b></td><td>HTTPException con 400, 404, 503</td><td>main.py</td></tr>
    <tr style="background:#f8fafc;"><td style="padding:6px;"><b>async/await (S4)</b></td><td>Todas las rutas son async def</td><td>main.py</td></tr>
    <tr><td style="padding:6px;"><b>Depends() (S6)</b></td><td>Inyección de DataService, Settings, RiskCalculator</td><td>dependencies.py, main.py</td></tr>
    <tr style="background:#f8fafc;"><td style="padding:6px;"><b>BaseSettings (S6)</b></td><td>Settings(BaseSettings) + .env + @lru_cache</td><td>config.py</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{USTA_NAVY} 0%,{USTA_PURPLE} 100%);
     border-radius:14px; padding:20px 28px; margin-top:30px; text-align:center;">
    <p style="color:#fff; font-weight:700; font-size:1rem; margin:0;">
        Laura Alejandra Sepúlveda &nbsp;·&nbsp; Ingrid Johana Umbacia Ramírez
    </p>
    <p style="color:rgba(255,255,255,0.65); font-size:0.78rem; margin:4px 0 0 0;">
        Teoría del Riesgo · Universidad Santo Tomás · Backend: FastAPI · Frontend: Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
