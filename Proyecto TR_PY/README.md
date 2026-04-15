# 📊 RiskLab USTA – Proyecto Integrador de Teoría del Riesgo

**Universidad Santo Tomás · Facultad de Estadística · Teoría del Riesgo**

**Autoras:** Laura Alejandra Sepúlveda · Ingrid Johana Umbacia Ramírez

---

## 🗂️ Estructura del Proyecto

```
proyecto-riesgo/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI – routers y endpoints async
│   │   ├── models.py         # Modelos Pydantic (request/response)
│   │   ├── services.py       # Lógica de negocio + decoradores personalizados
│   │   ├── dependencies.py   # Depends() – inyección de dependencias
│   │   └── config.py         # BaseSettings + variables de entorno
│   ├── requirements.txt
│   └── .env                  # Variables de entorno (NO subir a Git)
├── frontend/
│   ├── app.py                # Streamlit – consume el backend vía HTTP
│   └── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Instalación y Ejecución

### Requisitos previos
- Python 3.11+
- Git

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/proyecto-riesgo.git
cd proyecto-riesgo
```

### 2. Configurar el Backend (FastAPI)
```bash
cd backend

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Instalar dependencias
pip install -r requirements.txt

# Crear archivo .env (copiar la plantilla)
# Editar el archivo .env con tus configuraciones

# Iniciar el backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

El backend estará disponible en: `http://localhost:8000`
Documentación interactiva Swagger UI: `http://localhost:8000/docs`

### 3. Configurar el Frontend (Streamlit)
Abre una **nueva terminal**:
```bash
cd frontend

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python -m streamlit run app.py
```

El tablero estará en: `http://localhost:8501`

---

## ⚠️ IMPORTANTE: Orden de ejecución

```
Terminal 1: Backend FastAPI  →  uvicorn app.main:app --reload
Terminal 2: Frontend Streamlit  →  python -m streamlit run app.py
```

El frontend DEPENDE del backend. Si el backend no está corriendo, el frontend mostrará errores de conexión.

---

## 📡 Variables de Entorno (.env)

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `TICKERS` | Activos del portafolio | `["MSI","XOM","JNJ","PG","UL","TSM"]` |
| `BENCHMARK` | Índice de referencia | `^GSPC` (S&P 500) |
| `RF_TICKER` | Tasa libre de riesgo | `^IRX` (T-Bill 13W) |
| `DEFAULT_PERIOD` | Período histórico | `2y` |
| `VAR_CONFIDENCE` | Nivel de confianza VaR | `0.95` |
| `N_PORTFOLIOS` | Portafolios en Markowitz | `10000` |
| `FRONTEND_URL` | URL del frontend (CORS) | `http://localhost:8501` |

---

## 🔌 API REST – Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Info general de la API |
| `/activos` | GET | Lista los activos del portafolio |
| `/precios/{ticker}` | GET | Precios históricos (params: period) |
| `/rendimientos/{ticker}` | GET | Log-rendimientos y estadísticos |
| `/indicadores/{ticker}` | GET | Indicadores técnicos actuales |
| `/var` | POST | VaR + CVaR (3 métodos) + Kupiec |
| `/capm` | GET | Beta y CAPM para todos los activos |
| `/frontera-eficiente` | POST | Frontera eficiente de Markowitz |
| `/alertas` | GET | Señales de compra/venta automáticas |
| `/macro` | GET | Indicadores macroeconómicos en tiempo real |

---

## 🏗️ Conceptos de Python para APIs Implementados

| Concepto | Semana | Implementación |
|----------|--------|----------------|
| **Decoradores personalizados** | S1 | `@log_execution_time`, `@cache_result(ttl=3600)` en `services.py` |
| **Type hints** | S1 | Todas las funciones y métodos tienen anotaciones de tipo |
| **POO** | S2 | Clases `DataService`, `RiskCalculator`, `TechnicalIndicators`, `PortfolioAnalyzer` |
| **Pydantic BaseModel** | S2/4/5 | Modelos request + response con `Field()` en `models.py` |
| **@field_validator** | S4/5 | Validación personalizada de tickers, pesos y fechas |
| **@model_validator** | S5 | Validación cruzada tickers/pesos en `VaRRequest` |
| **Manejo errores HTTP** | S2 | `HTTPException` con códigos 400, 404, 503 |
| **async/await** | S4 | Todas las rutas son `async def` |
| **Depends()** | S6 | Inyección de `DataService`, `Settings`, `RiskCalculator` |
| **BaseSettings** | S6 | `Settings(BaseSettings)` con carga desde `.env` |
| **lru_cache** | S1/6 | Caché de instancias de servicios (patrón singleton) |

---

## 📊 Módulos del Tablero

| # | Módulo | Contenido | Peso rúbrica |
|---|--------|-----------|--------------|
| 0 | Portada | Contexto geopolítico, indicadores macro, arquitectura | — |
| 1 | Análisis Técnico | SMA, EMA, RSI, MACD, Bollinger, Estocástico | 10% |
| 2 | Rendimientos | Log-ret, histograma, Q-Q, boxplot, Jarque-Bera, Shapiro-Wilk | 6% |
| 3 | ARCH/GARCH | 4 modelos, AIC/BIC, diagnóstico, pronóstico 10 días | 10% |
| 4 | CAPM y Beta | Beta, SML, Rf desde API, clasificación activos | 6% |
| 5 | VaR y CVaR | Paramétrico + Histórico + Montecarlo + CVaR + Kupiec | 10% |
| 6 | Markowitz | Heatmap, 10,000 portafolios, frontera eficiente, óptimos | 10% |
| 7 | Señales ★ | Panel semáforo, 5 indicadores, umbrales configurables | 7% |
| 8 | Macro & Benchmark ★ | Alpha Jensen, TE, IR, drawdown, panel macro | 6% |
| 9 | API FastAPI ★★ | Demostración de endpoints, Pydantic, Depends | 12% |

---

## 🎁 Bonificaciones Implementadas

- ✅ **Test de Kupiec** (backtesting VaR) – endpoint `/var`
- ✅ **Optimización por rendimiento objetivo** – Módulo 6
- ✅ **Decoradores personalizados** – `@log_execution_time`, `@cache_result`
- ✅ **`@field_validator` y `@model_validator`** personalizados en todos los request models
- ✅ **`async/await`** en todas las rutas del backend
- ✅ **Documentación `/docs`** (Swagger UI) con `Field(description=...)` enriquecida

---

## 🤖 Uso de Herramientas de IA

Este proyecto utilizó **Claude (Anthropic)** como asistente de desarrollo para:
- Estructuración de la arquitectura FastAPI/Streamlit
- Implementación de fórmulas financieras (VaR, GARCH, Markowitz, CAPM)
- Generación de interpretaciones automáticas contextualizadas
- Revisión y depuración de código

**Todo el código fue revisado, comprendido y validado por las autoras.**
Cada módulo fue probado con datos reales antes de la entrega.

---

## 📚 Bibliografía

1. Moscote Flórez, O. *Elementos de estadística en riesgo financiero*. USTA, 2013.
2. Holton, G. A. *Value at Risk: Theory and Practice*. value-at-risk.net
3. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.
4. Tsay, R. S. (2010). *Analysis of Financial Time Series*. 3rd ed., Wiley.
5. Hull, J. C. (2018). *Risk Management and Financial Institutions*. 5th ed., Wiley.
6. FastAPI Documentation: https://fastapi.tiangolo.com
7. Pydantic Documentation: https://docs.pydantic.dev
8. yfinance: https://pypi.org/project/yfinance/
