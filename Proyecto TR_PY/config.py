# =============================================================================
# config.py – Configuración centralizada con BaseSettings + .env
# Autoras: Laura Alejandra Sepúlveda & Ingrid Johana Umbacia Ramírez
# =============================================================================
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración global del proyecto.
    Todas las variables se cargan desde el archivo .env.
    NUNCA se hardcodean API keys en el código fuente.
    """

    # ── Activos por defecto ──────────────────────────────────────
    tickers: list[str] = ["MSI", "XOM", "JNJ", "PG", "UL", "TSM"]
    benchmark: str = "^GSPC"
    rf_ticker: str = "^IRX"

    # ── Parámetros de análisis ───────────────────────────────────
    default_period: str = "2y"
    var_confidence: float = 0.95
    sma_short: int = 20
    sma_long: int = 50
    ema_period: int = 20
    n_portfolios: int = 10_000

    # ── CORS ────────────────────────────────────────────────────
    frontend_url: str = "http://localhost:8501"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Decorador @lru_cache: la instancia de Settings se crea UNA SOLA VEZ
    y se reutiliza en cada request (patrón singleton ligero).
    Esto demuestra el uso de decoradores de la Semana 1 del curso.
    """
    return Settings()
