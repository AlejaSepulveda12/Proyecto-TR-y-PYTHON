# =============================================================================
# models.py – Modelos Pydantic de Request y Response
# Autoras: Laura Alejandra Sepúlveda & Ingrid Johana Umbacia Ramírez
# =============================================================================
from __future__ import annotations

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────

class TickerParams(BaseModel):
    """Parámetros para consultas de un activo individual."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Símbolo del activo en Yahoo Finance (ej: 'TSM', 'AAPL')",
        examples=["TSM"],
    )
    period: str = Field(
        default="2y",
        description="Período de datos: 1mo, 3mo, 6mo, 1y, 2y, 3y, 5y",
        examples=["2y"],
    )

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Convierte el ticker a mayúsculas y elimina espacios."""
        cleaned = v.strip().upper()
        if not cleaned.replace(".", "").replace("-", "").replace("^", "").isalnum():
            raise ValueError(f"Ticker inválido: '{v}'. Solo se permiten letras, números, '.', '-' y '^'.")
        return cleaned

    @field_validator("period")
    @classmethod
    def valid_period(cls, v: str) -> str:
        allowed = {"1mo", "3mo", "6mo", "1y", "2y", "3y", "5y"}
        if v not in allowed:
            raise ValueError(f"Período inválido: '{v}'. Valores permitidos: {allowed}")
        return v


class VaRRequest(BaseModel):
    """Cuerpo del POST /var – portafolio con pesos y configuración."""

    tickers: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de tickers del portafolio",
        examples=[["MSI", "XOM", "JNJ", "PG", "UL", "TSM"]],
    )
    weights: list[float] = Field(
        ...,
        min_length=1,
        description="Pesos de cada activo (deben sumar 1.0)",
        examples=[[0.2, 0.15, 0.2, 0.15, 0.15, 0.15]],
    )
    confidence: float = Field(
        default=0.95,
        ge=0.90,
        le=0.999,
        description="Nivel de confianza del VaR (entre 0.90 y 0.999)",
    )
    period: str = Field(default="2y", description="Período histórico de datos")

    @field_validator("tickers")
    @classmethod
    def tickers_uppercase(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v]

    @field_validator("weights")
    @classmethod
    def weights_positive(cls, v: list[float]) -> list[float]:
        if any(w < 0 for w in v):
            raise ValueError("Todos los pesos deben ser no negativos.")
        return v

    @model_validator(mode="after")
    def tickers_weights_same_length(self) -> "VaRRequest":
        if len(self.tickers) != len(self.weights):
            raise ValueError(
                f"La cantidad de tickers ({len(self.tickers)}) "
                f"debe coincidir con la cantidad de pesos ({len(self.weights)})."
            )
        total = sum(self.weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Los pesos deben sumar 1.0 (suman {total:.4f}). "
                "Se permite una tolerancia de ±0.01."
            )
        return self


class FronteraRequest(BaseModel):
    """Cuerpo del POST /frontera-eficiente."""

    tickers: list[str] = Field(
        ...,
        min_length=2,
        description="Lista de al menos 2 tickers para construir la frontera",
    )
    period: str = Field(default="2y")
    n_portfolios: int = Field(
        default=10_000,
        ge=1_000,
        le=50_000,
        description="Número de portafolios a simular (entre 1,000 y 50,000)",
    )

    @field_validator("tickers")
    @classmethod
    def at_least_two(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("Se necesitan al menos 2 activos para construir la frontera eficiente.")
        return [t.strip().upper() for t in v]


# ─────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────

class PrecioItem(BaseModel):
    fecha: str = Field(..., description="Fecha en formato YYYY-MM-DD")
    precio: float = Field(..., description="Precio de cierre ajustado (USD)")


class PreciosResponse(BaseModel):
    ticker: str
    empresa: str
    periodo: str
    n_observaciones: int
    precio_actual: float = Field(..., description="Último precio disponible")
    variacion_diaria_pct: Optional[float] = Field(None, description="Variación % respecto al día anterior")
    precios: list[PrecioItem]


class RendimientosResponse(BaseModel):
    ticker: str
    empresa: str
    periodo: str
    media_diaria_pct: float = Field(..., description="Log-rendimiento medio diario (%)")
    rendimiento_anualizado_pct: float
    volatilidad_diaria_pct: float
    volatilidad_anualizada_pct: float
    skewness: float = Field(..., description="Asimetría de la distribución")
    kurtosis: float = Field(..., description="Curtosis exceso (0 = normal)")
    min_diario_pct: float
    max_diario_pct: float
    n_observaciones: int
    rendimientos: list[dict]


class IndicadoresResponse(BaseModel):
    ticker: str
    fecha_ultimo: str
    rsi: float = Field(..., description="RSI (14 períodos)")
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float = Field(..., description="Banda de Bollinger superior")
    bb_middle: float
    bb_lower: float
    sma_20: float
    sma_50: float
    ema_20: float
    stoch_k: float = Field(..., description="Oscilador Estocástico %K")
    stoch_d: float = Field(..., description="Oscilador Estocástico %D")
    precio_actual: float


class CAPMAsset(BaseModel):
    ticker: str
    empresa: str
    beta: float = Field(..., description="Beta respecto al S&P 500")
    alpha_diario: float = Field(..., description="Alpha de Jensen diario")
    r_squared: float = Field(..., description="R² de la regresión")
    riesgo_sistematico_pct: float
    riesgo_idiosincratico_pct: float
    rendimiento_esperado_anual_pct: float
    clasificacion: str = Field(..., description="Agresivo / Defensivo / Neutro")


class CAPMResponse(BaseModel):
    rf_anual_pct: float = Field(..., description="Tasa libre de riesgo anualizada (%)")
    prima_mercado_anual_pct: float
    benchmark: str
    activos: list[CAPMAsset]


class VaRMethodResult(BaseModel):
    metodo: str
    var_diario_pct: float
    cvar_diario_pct: float
    var_anualizado_pct: float


class KupiecResult(BaseModel):
    violaciones_observadas: int
    violaciones_esperadas: float
    lr_statistic: float
    p_valor: float
    modelo_valido: bool


class VaRResponse(BaseModel):
    tickers: list[str]
    pesos: list[float]
    nivel_confianza: float
    resultados: list[VaRMethodResult]
    kupiec: KupiecResult


class PortfolioOptimo(BaseModel):
    tipo: str = Field(..., description="max_sharpe o min_varianza")
    rendimiento_anual_pct: float
    volatilidad_anual_pct: float
    sharpe_ratio: float
    pesos: dict[str, float] = Field(..., description="Peso (%) de cada activo")


class FronteraResponse(BaseModel):
    tickers: list[str]
    n_portfolios_simulados: int
    max_sharpe: PortfolioOptimo
    min_varianza: PortfolioOptimo
    correlaciones: dict[str, dict[str, float]]


class AlertaActivo(BaseModel):
    ticker: str
    empresa: str
    señal_global: str = Field(..., description="BUY / SELL / NEUTRAL")
    votos_compra: int
    votos_venta: int
    indicadores: dict[str, str] = Field(..., description="Señal por indicador")
    interpretacion: str


class AlertasResponse(BaseModel):
    fecha_consulta: str
    alertas: list[AlertaActivo]


class MacroResponse(BaseModel):
    fecha_consulta: str
    rf_anual_pct: float = Field(..., description="T-Bill 13W anualizado (%)")
    vix: Optional[float] = Field(None, description="VIX (índice de volatilidad implícita)")
    oro_usd: Optional[float] = Field(None, description="Oro (USD/oz)")
    brent_usd: Optional[float] = Field(None, description="Petróleo Brent (USD/barril)")
    usd_cop: Optional[float] = Field(None, description="Tipo de cambio USD/COP")
    dxy: Optional[float] = Field(None, description="Índice del Dólar (DXY)")
