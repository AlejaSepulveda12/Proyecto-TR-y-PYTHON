# =============================================================================
# dependencies.py – Inyección de dependencias con Depends()
# Autoras: Laura Alejandra Sepúlveda & Ingrid Johana Umbacia Ramírez
# =============================================================================
from functools import lru_cache
from app.config import Settings, get_settings
from app.services import DataService, RiskCalculator, TechnicalIndicators, PortfolioAnalyzer


# ── Dependencias de configuración ───────────────────────────────
def get_settings_dep() -> Settings:
    """
    Dependencia de configuración.
    Inyecta el objeto Settings en las rutas que lo necesiten.
    """
    return get_settings()


# ── Dependencias de servicios ────────────────────────────────────
@lru_cache
def get_data_service() -> DataService:
    """
    Crea una única instancia de DataService (singleton).
    El caché de lru_cache garantiza que no se creen múltiples instancias.
    """
    return DataService()


@lru_cache
def get_risk_calculator() -> RiskCalculator:
    return RiskCalculator()


@lru_cache
def get_technical_indicators() -> TechnicalIndicators:
    return TechnicalIndicators()


@lru_cache
def get_portfolio_analyzer() -> PortfolioAnalyzer:
    return PortfolioAnalyzer()
