# =============================================================================
# services.py – Lógica de negocio y cálculos financieros
# Autoras: Laura Alejandra Sepúlveda & Ingrid Johana Umbacia Ramírez
# =============================================================================
from __future__ import annotations

import datetime
import functools
import time
import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy import stats

logger = logging.getLogger(__name__)

NAMES: dict[str, str] = {
    "MSI": "Motorola Solutions",
    "XOM": "ExxonMobil",
    "JNJ": "Johnson & Johnson",
    "PG":  "Procter & Gamble",
    "UL":  "Unilever",
    "TSM": "TSMC",
}

# ─────────────────────────────────────────────────────────────────
# DECORADORES PERSONALIZADOS (Semana 1 del curso de APIs)
# ─────────────────────────────────────────────────────────────────

def log_execution_time(func: Any) -> Any:
    """
    Decorador personalizado que mide y registra el tiempo de ejecución
    de cada función de servicio. Útil para detectar cuellos de botella.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[{func.__name__}] ejecutado en {elapsed:.3f}s")
        return result
    return wrapper


def cache_result(ttl_seconds: int = 3600) -> Any:
    """
    Decorador factory con TTL (Time To Live).
    Cachea el resultado de una función por N segundos.
    Evita llamadas repetidas a APIs financieras externas.
    """
    def decorator(func: Any) -> Any:
        cache: dict[str, tuple[float, Any]] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            if key in cache:
                timestamp, value = cache[key]
                if now - timestamp < ttl_seconds:
                    logger.debug(f"[{func.__name__}] cache HIT para key={key[:40]}")
                    return value
            result = func(*args, **kwargs)
            cache[key] = (now, result)
            logger.debug(f"[{func.__name__}] cache MISS, resultado guardado")
            return result
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────
# CLASES DE SERVICIO (POO – Semana 2 del curso)
# ─────────────────────────────────────────────────────────────────

class DataService:
    """
    Servicio de acceso a datos financieros vía Yahoo Finance.
    Encapsula la lógica de descarga, limpieza y caché de datos.
    """

    @log_execution_time
    @cache_result(ttl_seconds=3600)
    def get_prices(self, ticker: str, period: str = "2y") -> pd.Series:
        """Descarga precios de cierre ajustados desde Yahoo Finance."""
        try:
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if raw.empty:
                raise ValueError(f"No se encontraron datos para el ticker '{ticker}'.")
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.dropna()
        except Exception as e:
            raise RuntimeError(f"Error al obtener datos de '{ticker}': {e}") from e

    @log_execution_time
    @cache_result(ttl_seconds=3600)
    def get_multiple_prices(self, tickers: list[str], period: str = "2y") -> pd.DataFrame:
        """Descarga precios de múltiples activos en una sola llamada."""
        try:
            raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
            prices = raw["Close"].dropna(how="all")
            missing = [t for t in tickers if t not in prices.columns]
            if missing:
                raise ValueError(f"No se encontraron datos para: {missing}")
            return prices[tickers].dropna()
        except Exception as e:
            raise RuntimeError(f"Error al obtener datos múltiples: {e}") from e

    @log_execution_time
    @cache_result(ttl_seconds=3600)
    def get_rf_rate(self) -> float:
        """Obtiene la tasa libre de riesgo (T-Bill 13W) desde Yahoo Finance."""
        try:
            h = yf.Ticker("^IRX").history(period="5d")
            if not h.empty:
                return float(h["Close"].iloc[-1]) / 100
        except Exception:
            pass
        return 0.045  # fallback

    @log_execution_time
    @cache_result(ttl_seconds=3600)
    def get_macro(self) -> dict[str, float | None]:
        """Obtiene indicadores macroeconómicos actualizados."""
        result: dict[str, float | None] = {"rf_annual": self.get_rf_rate()}
        symbols = {"vix": "^VIX", "gold": "GC=F", "oil": "BZ=F", "usdcop": "COP=X", "dxy": "DX-Y.NYB"}
        for key, sym in symbols.items():
            try:
                h = yf.Ticker(sym).history(period="5d")
                result[key] = float(h["Close"].iloc[-1]) if not h.empty else None
            except Exception:
                result[key] = None
        return result


class RiskCalculator:
    """
    Encapsula todos los cálculos de riesgo financiero:
    log-rendimientos, VaR, CVaR, backtesting Kupiec.
    """

    @staticmethod
    def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Calcula log-rendimientos a partir de precios."""
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    @log_execution_time
    def var_parametric(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
        """
        VaR paramétrico asumiendo distribución normal.
        Retorna (VaR, CVaR) como fracciones (no porcentajes).
        """
        alpha = 1 - confidence
        mu, sigma = returns.mean(), returns.std()
        z = stats.norm.ppf(alpha)
        var   = -(mu + z * sigma)
        cvar  = -(mu - sigma * stats.norm.pdf(z) / alpha)
        return var, cvar

    @staticmethod
    @log_execution_time
    def var_historical(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
        """VaR por simulación histórica."""
        alpha = 1 - confidence
        var   = -float(np.percentile(returns, alpha * 100))
        cvar  = -float(returns[returns <= -var].mean())
        return var, cvar

    @staticmethod
    @log_execution_time
    def var_montecarlo(returns: pd.Series, confidence: float = 0.95, n_sim: int = 10_000) -> tuple[float, float]:
        """VaR por simulación Montecarlo (distribución normal)."""
        alpha = 1 - confidence
        np.random.seed(42)
        sims  = np.random.normal(returns.mean(), returns.std(), n_sim)
        var   = -float(np.percentile(sims, alpha * 100))
        cvar  = -float(sims[sims <= -var].mean())
        return var, cvar

    @staticmethod
    def kupiec_test(
        returns: pd.Series, var: float, confidence: float = 0.95
    ) -> dict[str, Any]:
        """
        Test de backtesting de Kupiec (Likelihood Ratio).
        H0: El número de violaciones es consistente con el nivel de confianza.
        """
        alpha = 1 - confidence
        n = len(returns)
        violations = int((returns < -var).sum())
        p_hat = violations / n if n > 0 else 0.0

        if 0 < p_hat < 1:
            lr = 2 * (
                np.log((p_hat ** violations) * ((1 - p_hat) ** (n - violations)))
                - np.log((alpha ** violations) * ((1 - alpha) ** (n - violations)))
            )
            p_value = float(1 - stats.chi2.cdf(lr, df=1))
        else:
            lr, p_value = 0.0, 1.0

        return {
            "violaciones_observadas": violations,
            "violaciones_esperadas": round(alpha * n, 1),
            "lr_statistic": round(lr, 4),
            "p_valor": round(p_value, 4),
            "modelo_valido": p_value > 0.05,
        }


class TechnicalIndicators:
    """
    Encapsula el cálculo de indicadores técnicos.
    """

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss
        return (100 - (100 / (1 + rs))).rename("RSI")

    @staticmethod
    def macd(prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema12  = prices.ewm(span=12, adjust=False).mean()
        ema26  = prices.ewm(span=26, adjust=False).mean()
        line   = ema12 - ema26
        signal = line.ewm(span=9, adjust=False).mean()
        hist   = line - signal
        return line, signal, hist

    @staticmethod
    def bollinger(prices: pd.Series, period: int = 20, n_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid   = prices.rolling(period).mean()
        upper = mid + n_std * prices.rolling(period).std()
        lower = mid - n_std * prices.rolling(period).std()
        return upper, mid, lower

    @staticmethod
    def stochastic(prices: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series]:
        low14 = prices.rolling(period).min()
        high14 = prices.rolling(period).max()
        k = 100 * (prices - low14) / (high14 - low14)
        d = k.rolling(3).mean()
        return k, d

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(period).mean()

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    @classmethod
    def all_indicators(cls, prices: pd.Series) -> dict[str, Any]:
        """Calcula todos los indicadores y retorna el último valor de cada uno."""
        rsi_s        = cls.rsi(prices)
        macd_l, macd_sig, macd_h = cls.macd(prices)
        bb_u, bb_m, bb_l         = cls.bollinger(prices)
        stoch_k, stoch_d         = cls.stochastic(prices)
        sma20 = cls.sma(prices, 20)
        sma50 = cls.sma(prices, 50)
        ema20 = cls.ema(prices, 20)

        def last(s: pd.Series) -> float:
            return float(s.dropna().iloc[-1]) if not s.dropna().empty else 0.0

        return {
            "rsi":        last(rsi_s),
            "macd":       last(macd_l),
            "macd_signal":last(macd_sig),
            "macd_hist":  last(macd_h),
            "bb_upper":   last(bb_u),
            "bb_middle":  last(bb_m),
            "bb_lower":   last(bb_l),
            "sma_20":     last(sma20),
            "sma_50":     last(sma50),
            "ema_20":     last(ema20),
            "stoch_k":    last(stoch_k),
            "stoch_d":    last(stoch_d),
            "precio_actual": float(prices.iloc[-1]),
        }


class PortfolioAnalyzer:
    """
    Encapsula la optimización de portafolio (Markowitz),
    CAPM y generación de señales.
    """

    @staticmethod
    @log_execution_time
    def simulate_frontier(
        prices: pd.DataFrame,
        rf: float = 0.045,
        n_portfolios: int = 10_000,
    ) -> dict[str, Any]:
        """Simula n_portfolios aleatorios y encuentra los óptimos."""
        log_ret  = np.log(prices / prices.shift(1)).dropna()
        mean_ret = log_ret.mean() * 252
        cov_mat  = log_ret.cov() * 252
        n        = len(prices.columns)

        np.random.seed(42)
        rets, vols, sharpes, weights_all = [], [], [], []

        for _ in range(n_portfolios):
            w = np.random.dirichlet(np.ones(n))
            r = float(np.dot(w, mean_ret))
            v = float(np.sqrt(w @ cov_mat.values @ w))
            s = (r - rf) / v if v > 0 else 0.0
            rets.append(r); vols.append(v); sharpes.append(s); weights_all.append(w)

        idx_ms = int(np.argmax(sharpes))
        idx_mv = int(np.argmin(vols))
        tickers = list(prices.columns)

        return {
            "max_sharpe": {
                "tipo": "max_sharpe",
                "rendimiento_anual_pct": round(rets[idx_ms] * 100, 4),
                "volatilidad_anual_pct": round(vols[idx_ms] * 100, 4),
                "sharpe_ratio": round(sharpes[idx_ms], 4),
                "pesos": {t: round(float(weights_all[idx_ms][i]) * 100, 2)
                          for i, t in enumerate(tickers)},
            },
            "min_varianza": {
                "tipo": "min_varianza",
                "rendimiento_anual_pct": round(rets[idx_mv] * 100, 4),
                "volatilidad_anual_pct": round(vols[idx_mv] * 100, 4),
                "sharpe_ratio": round(sharpes[idx_mv], 4),
                "pesos": {t: round(float(weights_all[idx_mv][i]) * 100, 2)
                          for i, t in enumerate(tickers)},
            },
            "correlaciones": {
                t: {t2: round(float(log_ret.corr().loc[t, t2]), 4) for t2 in tickers}
                for t in tickers
            },
        }

    @staticmethod
    def capm(
        prices: pd.DataFrame,
        benchmark_prices: pd.Series,
        rf_daily: float,
    ) -> list[dict[str, Any]]:
        """Calcula Beta, Alpha y rendimiento esperado CAPM para cada activo."""
        log_ret   = np.log(prices / prices.shift(1)).dropna()
        bench_ret = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
        common    = log_ret.index.intersection(bench_ret.index)
        bench_arr = bench_ret.loc[common].values.flatten()
        results   = []

        for ticker in prices.columns:
            asset_arr = log_ret.loc[common, ticker].values
            slope, intercept, r_value, _, _ = stats.linregress(bench_arr, asset_arr)
            er_daily  = rf_daily + slope * (bench_arr.mean() - rf_daily)
            er_annual = er_daily * 252
            if slope > 1.1:
                clasif = "🔴 Agresivo (β>1.1)"
            elif slope < 0.9:
                clasif = "🟢 Defensivo (β<0.9)"
            else:
                clasif = "🟡 Neutro"

            results.append({
                "ticker": ticker,
                "empresa": NAMES.get(ticker, ticker),
                "beta": round(slope, 4),
                "alpha_diario": round(intercept, 6),
                "r_squared": round(r_value ** 2, 4),
                "riesgo_sistematico_pct": round(r_value ** 2 * 100, 1),
                "riesgo_idiosincratico_pct": round((1 - r_value ** 2) * 100, 1),
                "rendimiento_esperado_anual_pct": round(er_annual * 100, 4),
                "clasificacion": clasif,
            })
        return results

    @staticmethod
    def generate_signals(
        prices: pd.DataFrame,
        rsi_ob: int = 70,
        rsi_os: int = 30,
        stoch_ob: int = 80,
        stoch_os: int = 20,
    ) -> list[dict[str, Any]]:
        """Genera señales de compra/venta para cada activo."""
        ti = TechnicalIndicators()
        results = []

        for ticker in prices.columns:
            p   = prices[ticker].dropna()
            ind = ti.all_indicators(p)
            sigs: dict[str, str] = {}

            # RSI
            rv = ind["rsi"]
            sigs["RSI"] = "SELL" if rv > rsi_ob else ("BUY" if rv < rsi_os else "NEUTRAL")

            # MACD
            sigs["MACD"] = "BUY" if ind["macd"] > ind["macd_signal"] else "SELL"

            # Bollinger
            pv = ind["precio_actual"]
            sigs["Bollinger"] = ("SELL" if pv >= ind["bb_upper"]
                                 else "BUY" if pv <= ind["bb_lower"]
                                 else "NEUTRAL")

            # Medias Móviles
            sigs["Med.Moviles"] = "BUY" if ind["sma_20"] > ind["sma_50"] else "SELL"

            # Estocástico
            sk = ind["stoch_k"]
            sigs["Estocastico"] = ("SELL" if sk > stoch_ob
                                   else "BUY" if sk < stoch_os
                                   else "NEUTRAL")

            nb = sum(1 for v in sigs.values() if v == "BUY")
            ns = sum(1 for v in sigs.values() if v == "SELL")
            overall = "BUY" if nb >= 3 else ("SELL" if ns >= 3 else "NEUTRAL")

            interp = (
                f"{nb}/5 indicadores señalan COMPRA, {ns}/5 señalan VENTA. "
                + ("Consenso alcista: momentum positivo a corto plazo."
                   if overall == "BUY"
                   else "Consenso bajista: evaluar reducción de exposición."
                   if overall == "SELL"
                   else "Sin consenso: esperar confirmación de tendencia.")
            )

            results.append({
                "ticker": ticker,
                "empresa": NAMES.get(ticker, ticker),
                "señal_global": overall,
                "votos_compra": nb,
                "votos_venta": ns,
                "indicadores": sigs,
                "interpretacion": interp,
            })
        return results
