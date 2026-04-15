# =============================================================================
# main.py – Aplicación FastAPI principal
# Autoras: Laura Alejandra Sepúlveda & Ingrid Johana Umbacia Ramírez
# Proyecto Integrador – Teoría del Riesgo · USTA
# =============================================================================
from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.dependencies import (
    get_data_service,
    get_portfolio_analyzer,
    get_risk_calculator,
    get_settings_dep,
)
from app.models import (
    AlertaActivo,
    AlertasResponse,
    CAPMAsset,
    CAPMResponse,
    FronteraRequest,
    FronteraResponse,
    IndicadoresResponse,
    KupiecResult,
    MacroResponse,
    PortfolioOptimo,
    PrecioItem,
    PreciosResponse,
    RendimientosResponse,
    VaRMethodResult,
    VaRRequest,
    VaRResponse,
)
from app.services import (
    DataService,
    PortfolioAnalyzer,
    RiskCalculator,
    TechnicalIndicators,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# INICIALIZACIÓN DE LA APP
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RiskLab USTA – API de Riesgo Financiero",
    description=(
        "Backend del Proyecto Integrador de Teoría del Riesgo. "
        "Proporciona endpoints para análisis técnico, rendimientos, "
        "VaR, CAPM, optimización de portafolio y señales automáticas. "
        "\n\n**Autoras:** Laura Alejandra Sepúlveda · Ingrid Johana Umbacia Ramírez"
        "\n\n**Universidad Santo Tomás · Facultad de Estadística**"
    ),
    version="1.0.0",
    contact={
        "name": "Laura Sepúlveda & Ingrid Umbacia",
        "email": "estudiantes@usta.edu.co",
    },
    license_info={"name": "Académico – USTA 2025"},
)

# CORS – permite que el frontend Streamlit consuma el backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NAMES: dict[str, str] = {
    "MSI": "Motorola Solutions",
    "XOM": "ExxonMobil",
    "JNJ": "Johnson & Johnson",
    "PG":  "Procter & Gamble",
    "UL":  "Unilever",
    "TSM": "TSMC",
}


# ─────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root() -> dict:
    """Información general de la API."""
    return {
        "proyecto": "RiskLab USTA – Teoría del Riesgo",
        "autoras": ["Laura Alejandra Sepúlveda", "Ingrid Johana Umbacia Ramírez"],
        "version": "1.0.0",
        "documentacion": "/docs",
        "endpoints_disponibles": [
            "/activos", "/precios/{ticker}", "/rendimientos/{ticker}",
            "/indicadores/{ticker}", "/var", "/capm",
            "/frontera-eficiente", "/alertas", "/macro",
        ],
    }


@app.get("/activos", tags=["Portafolio"])
async def get_activos(
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    """
    Lista los activos disponibles en el portafolio.
    Los tickers se configuran en .env o en los valores por defecto de Settings.
    """
    return {
        "tickers": settings.tickers,
        "benchmark": settings.benchmark,
        "empresas": {t: NAMES.get(t, t) for t in settings.tickers},
        "total_activos": len(settings.tickers),
    }


@app.get("/precios/{ticker}", response_model=PreciosResponse, tags=["Precios"])
async def get_precios(
    ticker: str,
    period: str = "2y",
    data_svc: DataService = Depends(get_data_service),
) -> PreciosResponse:
    """
    Retorna los precios históricos de cierre ajustados de un activo.
    - **ticker**: Símbolo del activo (ej: TSM, AAPL)
    - **period**: Período (1mo, 3mo, 6mo, 1y, 2y, 3y, 5y)
    """
    ticker = ticker.upper().strip()
    try:
        prices = data_svc.get_prices(ticker, period)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    var_diaria = None
    if len(prices) >= 2:
        var_diaria = round((float(prices.iloc[-1]) / float(prices.iloc[-2]) - 1) * 100, 4)

    return PreciosResponse(
        ticker=ticker,
        empresa=NAMES.get(ticker, ticker),
        periodo=period,
        n_observaciones=len(prices),
        precio_actual=round(float(prices.iloc[-1]), 4),
        variacion_diaria_pct=var_diaria,
        precios=[
            PrecioItem(fecha=str(idx.date()), precio=round(float(val), 4))
            for idx, val in zip(prices.index, prices.values)
        ],
    )


@app.get("/rendimientos/{ticker}", response_model=RendimientosResponse, tags=["Rendimientos"])
async def get_rendimientos(
    ticker: str,
    period: str = "2y",
    data_svc: DataService = Depends(get_data_service),
    calc: RiskCalculator = Depends(get_risk_calculator),
) -> RendimientosResponse:
    """
    Retorna estadísticas de rendimientos logarítmicos de un activo.
    Incluye media, volatilidad, skewness, kurtosis y serie histórica.
    """
    ticker = ticker.upper().strip()
    try:
        prices = data_svc.get_prices(ticker, period)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    from scipy import stats as sp_stats
    lr = calc.log_returns(prices)

    return RendimientosResponse(
        ticker=ticker,
        empresa=NAMES.get(ticker, ticker),
        periodo=period,
        media_diaria_pct=round(float(lr.mean()) * 100, 6),
        rendimiento_anualizado_pct=round(float(lr.mean()) * 252 * 100, 4),
        volatilidad_diaria_pct=round(float(lr.std()) * 100, 6),
        volatilidad_anualizada_pct=round(float(lr.std()) * (252 ** 0.5) * 100, 4),
        skewness=round(float(sp_stats.skew(lr)), 4),
        kurtosis=round(float(sp_stats.kurtosis(lr)), 4),
        min_diario_pct=round(float(lr.min()) * 100, 4),
        max_diario_pct=round(float(lr.max()) * 100, 4),
        n_observaciones=len(lr),
        rendimientos=[
            {"fecha": str(idx.date()), "log_ret_pct": round(float(val) * 100, 6)}
            for idx, val in zip(lr.index, lr.values)
        ],
    )


@app.get("/indicadores/{ticker}", response_model=IndicadoresResponse, tags=["Análisis Técnico"])
async def get_indicadores(
    ticker: str,
    period: str = "1y",
    data_svc: DataService = Depends(get_data_service),
) -> IndicadoresResponse:
    """
    Retorna indicadores técnicos calculados para el último día disponible:
    RSI, MACD, Bandas de Bollinger, SMA, EMA y Oscilador Estocástico.
    """
    ticker = ticker.upper().strip()
    try:
        prices = data_svc.get_prices(ticker, period)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    if len(prices) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insuficientes datos para calcular indicadores ({len(prices)} observaciones). Mínimo: 50.",
        )

    ind = TechnicalIndicators.all_indicators(prices)
    fecha_ultimo = str(prices.index[-1].date())

    return IndicadoresResponse(
        ticker=ticker,
        fecha_ultimo=fecha_ultimo,
        **ind,
    )


@app.post("/var", response_model=VaRResponse, tags=["Riesgo"])
async def calcular_var(
    req: VaRRequest,
    data_svc: DataService = Depends(get_data_service),
    calc: RiskCalculator = Depends(get_risk_calculator),
) -> VaRResponse:
    """
    Calcula VaR y CVaR del portafolio mediante tres métodos:
    paramétrico, histórico y Montecarlo (10,000 simulaciones).
    Incluye backtesting con el test de Kupiec.
    """
    try:
        prices = data_svc.get_multiple_prices(req.tickers, req.period)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    weights = np.array(req.weights)
    log_ret = calc.log_returns(prices)
    port_ret = log_ret.dot(weights)

    v_param, cv_param   = calc.var_parametric(port_ret, req.confidence)
    v_hist,  cv_hist    = calc.var_historical(port_ret, req.confidence)
    v_mc,    cv_mc      = calc.var_montecarlo(port_ret, req.confidence)
    kupiec_raw          = calc.kupiec_test(port_ret, v_hist, req.confidence)

    sqrt252 = 252 ** 0.5
    return VaRResponse(
        tickers=req.tickers,
        pesos=req.weights,
        nivel_confianza=req.confidence,
        resultados=[
            VaRMethodResult(metodo="Paramétrico (Normal)",
                            var_diario_pct=round(v_param*100, 4),
                            cvar_diario_pct=round(cv_param*100, 4),
                            var_anualizado_pct=round(v_param*sqrt252*100, 4)),
            VaRMethodResult(metodo="Histórico",
                            var_diario_pct=round(v_hist*100, 4),
                            cvar_diario_pct=round(cv_hist*100, 4),
                            var_anualizado_pct=round(v_hist*sqrt252*100, 4)),
            VaRMethodResult(metodo="Montecarlo (10,000 sim.)",
                            var_diario_pct=round(v_mc*100, 4),
                            cvar_diario_pct=round(cv_mc*100, 4),
                            var_anualizado_pct=round(v_mc*sqrt252*100, 4)),
        ],
        kupiec=KupiecResult(**kupiec_raw),
    )


@app.get("/capm", response_model=CAPMResponse, tags=["CAPM"])
async def get_capm(
    period: str = "2y",
    data_svc: DataService = Depends(get_data_service),
    analyzer: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
    settings: Settings = Depends(get_settings_dep),
) -> CAPMResponse:
    """
    Calcula Beta y rendimiento esperado CAPM para cada activo del portafolio.
    La tasa libre de riesgo (Rf) se obtiene automáticamente desde Yahoo Finance (^IRX).
    """
    try:
        prices    = data_svc.get_multiple_prices(settings.tickers, period)
        bench_raw = yf.download(settings.benchmark, period=period, auto_adjust=True, progress=False)
        bench_prices = bench_raw["Close"].dropna()
        if isinstance(bench_prices, pd.DataFrame):
            bench_prices = bench_prices.iloc[:, 0]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    macro      = data_svc.get_macro()
    rf_annual  = macro.get("rf_annual") or 0.045
    rf_daily   = rf_annual / 252

    results_raw = analyzer.capm(prices, bench_prices, rf_daily)
    bench_ret = np.log(bench_prices / bench_prices.shift(1)).dropna()
    prima_pct = round((float(bench_ret.mean()) - rf_daily) * 252 * 100, 4)

    return CAPMResponse(
        rf_anual_pct=round(rf_annual * 100, 4),
        prima_mercado_anual_pct=prima_pct,
        benchmark=settings.benchmark,
        activos=[CAPMAsset(**r) for r in results_raw],
    )


@app.post("/frontera-eficiente", response_model=FronteraResponse, tags=["Markowitz"])
async def get_frontera(
    req: FronteraRequest,
    data_svc: DataService = Depends(get_data_service),
    analyzer: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
) -> FronteraResponse:
    """
    Simula portafolios aleatorios y retorna la frontera eficiente de Markowitz.
    Identifica el portafolio de máximo Sharpe y el de mínima varianza.
    """
    try:
        prices = data_svc.get_multiple_prices(req.tickers, req.period)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    macro = data_svc.get_macro()
    rf    = float(macro.get("rf_annual") or 0.045)

    result = analyzer.simulate_frontier(prices, rf=rf, n_portfolios=req.n_portfolios)
    return FronteraResponse(
        tickers=req.tickers,
        n_portfolios_simulados=req.n_portfolios,
        max_sharpe=PortfolioOptimo(**result["max_sharpe"]),
        min_varianza=PortfolioOptimo(**result["min_varianza"]),
        correlaciones=result["correlaciones"],
    )


@app.get("/alertas", response_model=AlertasResponse, tags=["Señales"])
async def get_alertas(
    period: str = "1y",
    rsi_ob: int = 70,
    rsi_os: int = 30,
    stoch_ob: int = 80,
    stoch_os: int = 20,
    data_svc: DataService = Depends(get_data_service),
    analyzer: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
    settings: Settings = Depends(get_settings_dep),
) -> AlertasResponse:
    """
    Retorna señales automáticas de compra/venta para cada activo del portafolio,
    basadas en RSI, MACD, Bollinger, Medias Móviles y Estocástico.
    Los umbrales son configurables mediante parámetros de consulta.
    """
    try:
        prices = data_svc.get_multiple_prices(settings.tickers, period)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    signals = analyzer.generate_signals(prices, rsi_ob, rsi_os, stoch_ob, stoch_os)
    return AlertasResponse(
        fecha_consulta=str(datetime.date.today()),
        alertas=[AlertaActivo(**s) for s in signals],
    )


@app.get("/macro", response_model=MacroResponse, tags=["Macro"])
async def get_macro(
    data_svc: DataService = Depends(get_data_service),
) -> MacroResponse:
    """
    Retorna indicadores macroeconómicos actualizados obtenidos vía Yahoo Finance:
    Rf (T-Bill 13W), VIX, Oro, Brent, USD/COP y DXY.
    """
    try:
        macro = data_svc.get_macro()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Error al obtener datos macro: {e}")

    return MacroResponse(
        fecha_consulta=str(datetime.date.today()),
        rf_anual_pct=round(float(macro.get("rf_annual") or 0.045) * 100, 4),
        vix=macro.get("vix"),
        oro_usd=macro.get("gold"),
        brent_usd=macro.get("oil"),
        usd_cop=macro.get("usdcop"),
        dxy=macro.get("dxy"),
    )
