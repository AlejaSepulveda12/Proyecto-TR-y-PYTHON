# 🎤 GUION DE SUSTENTACIÓN – RISKLAB USTA
## Proyecto Integrador – Teoría del Riesgo
### Laura Alejandra Sepúlveda · Ingrid Johana Umbacia Ramírez
### Universidad Santo Tomás · Facultad de Estadística

---

> **IMPORTANTE:** Este guion es una guía. En la sustentación hablen con sus propias palabras.
> El profesor evaluará que AMBAS dominen CUALQUIER sección.
> Tiempo sugerido: 15–20 minutos + 5–10 de preguntas.

---

## 🕐 APERTURA (1-2 minutos)

**Quién habla:** Cualquiera de las dos

"Buenos días/tardes, Profesor Sierra. Somos Laura Sepúlveda e Ingrid Umbacia, y vamos a presentar nuestro Proyecto Integrador de Teoría del Riesgo.

El proyecto consiste en un tablero interactivo de análisis de riesgo financiero construido con una arquitectura de dos capas: un **backend en FastAPI** que realiza todos los cálculos y consume datos de Yahoo Finance, y un **frontend en Streamlit** que presenta los resultados al usuario.

El portafolio analizado está compuesto por seis activos: Motorola Solutions (MSI), ExxonMobil (XOM), Johnson & Johnson (JNJ), Procter & Gamble (PG), Unilever (UL) y TSMC (TSM). Escogimos estos activos porque representan sectores muy distintos: tecnología, energía, salud, consumo básico y semiconductores, lo que nos permite demostrar diversificación tanto sectorial como geográfica."

---

## 🏗️ ARQUITECTURA (2-3 minutos) – CRÍTICO para la nota de FastAPI (12%)

**Abrir la terminal y mostrar que el backend corre.**
**Mostrar: http://localhost:8000/docs**

"Antes de entrar a los módulos, queremos mostrar la arquitectura del proyecto.

**El backend** está construido con FastAPI y tiene 9 endpoints. Todos los endpoints son **async def**, lo que permite manejar múltiples requests de forma eficiente. Los datos de entrada y salida están validados con **modelos Pydantic** que incluyen `Field()` con descripciones y restricciones.

Por ejemplo, en el endpoint `/var`, el request model `VaRRequest` tiene:
- Un `@field_validator` que verifica que todos los pesos sean positivos
- Un `@model_validator` que verifica que los tickers y pesos tengan la misma longitud Y que los pesos sumen 1.0 (con tolerancia ±0.01)

**La inyección de dependencias** con `Depends()` desacopla la lógica: las rutas no contienen lógica de negocio directamente, sino que reciben servicios inyectados como `DataService`, `RiskCalculator` y `PortfolioAnalyzer`.

**La configuración** usa `BaseSettings` de Pydantic Settings, que carga las variables desde el archivo `.env`. Las API keys y parámetros nunca están hardcodeados en el código.

**Los decoradores personalizados** que implementamos son:
- `@log_execution_time`: mide y registra el tiempo de ejecución de cada función de servicio
- `@cache_result(ttl=3600)`: cachea los resultados por 1 hora para no llamar a Yahoo Finance en cada request

Esta arquitectura cumple con el patrón separación frontend/backend que se trabajó en el curso de APIs."

---

## 🏠 PORTADA Y CONTEXTO (1-2 minutos)

"La portada muestra los indicadores macroeconómicos en tiempo real: la tasa libre de riesgo (Rf), el VIX, el precio del oro, el Brent y el tipo de cambio USD/COP. Estos datos los obtiene el backend automáticamente del endpoint `/macro` de nuestra API, que a su vez los consulta en Yahoo Finance.

El contexto geopolítico es relevante porque:
- La **guerra Rusia-Ucrania** benefició a XOM (energía) pero presionó a UL y PG (costos)
- Las **tensiones en Taiwan** impactan directamente a TSM
- El **ciclo de alzas de la Fed (2022-2024)** elevó la Rf, afectando el CAPM y los Sharpe Ratios"

---

## 📈 MÓDULO 1 – ANÁLISIS TÉCNICO (2 minutos)

"El Módulo 1 implementa seis indicadores técnicos obtenidos del endpoint `/indicadores/{ticker}` del backend.

**RSI (Relative Strength Index):**
- Mide el momentum de precio
- Por encima de 70: sobrecomprado (posible corrección)
- Por debajo de 30: sobrevendido (posible rebote)
- Actualmente para [TICKER] el RSI = [VALOR], que indica [INTERPRETACIÓN]

**MACD (Moving Average Convergence Divergence):**
- Cuando la línea MACD cruza al alza la señal: momentum alcista
- Cuando cruza a la baja: momentum bajista
- Histograma positivo = presión compradora dominante

**Bandas de Bollinger:**
- Miden la volatilidad relativa al precio
- Precio cerca de la banda superior = posible sobreextensión
- La anchura de las bandas indica el nivel de volatilidad actual

**Medias Móviles (Golden Cross / Death Cross):**
- Golden Cross: SMA20 supera a SMA50 → señal alcista fuerte
- Death Cross: SMA20 cae por debajo de SMA50 → señal bajista

**Oscilador Estocástico:**
- Similar al RSI: %K > 80 = sobrecomprado; %K < 20 = sobrevendido

Los parámetros son configurables interactivamente por el usuario."

---

## 📊 MÓDULO 2 – RENDIMIENTOS (2 minutos)

"El Módulo 2 caracteriza estadísticamente los rendimientos usando el endpoint `/rendimientos/{ticker}`.

**¿Por qué log-rendimientos?**
Son aditivos en el tiempo: r₁ + r₂ + ... = rendimiento acumulado. Son más estables estadísticamente para series largas.

**Estadísticos clave:**
- Media diaria de [VALOR]% → rendimiento anualizado estimado de [VALOR]%
- Volatilidad anualizada de [VALOR]%
- Skewness = [VALOR]: [asimetría negativa = pérdidas extremas más frecuentes / positiva]
- Kurtosis = [VALOR]: mayor que 0 indica colas más pesadas que la normal (leptocurtosis)

**Pruebas de normalidad:**
- **Jarque-Bera** (p=[VALOR]): rechaza/no rechaza normalidad
- **Shapiro-Wilk** (p=[VALOR]): rechaza/no rechaza normalidad

**Hechos estilizados documentados:**
1. No normalidad (colas pesadas) – el VaR normal subestima el riesgo
2. Agrupamiento de volatilidad – justifica ARCH/GARCH
3. Asimetría negativa – pérdidas extremas más frecuentes que ganancias extremas

Estos hechos justifican los métodos que usamos en los módulos 3 y 5."

---

## 🌊 MÓDULO 3 – ARCH/GARCH (2-3 minutos)

"El Módulo 3 modela la volatilidad condicional.

**¿Por qué GARCH?**
En el gráfico de rendimientos se observa claramente el *volatility clustering*: períodos de alta volatilidad tienden a agruparse. Esto viola el supuesto de varianza constante (homocedasticidad). Un modelo OLS o el VaR paramétrico simple son insuficientes.

**Comparamos 4 especificaciones:**
1. **ARCH(1)**: la varianza depende solo del error cuadrado del período anterior
2. **GARCH(1,1)**: la varianza depende del error cuadrado anterior Y de la varianza anterior. Es el más popular en la práctica
3. **EGARCH(1,1)**: captura el *leverage effect* (las malas noticias generan más volatilidad que las buenas del mismo tamaño). Esto es asimetría en la volatilidad
4. **GJR-GARCH(1,1)**: otra forma de capturar el leverage effect

**Selección por AIC/BIC:**
- AIC penaliza la complejidad moderadamente
- BIC penaliza más severamente (prefiere modelos más simples)
- El mejor modelo por [AIC/BIC] es [MODELO], con un valor de [VALOR]

**Diagnóstico de residuos:**
Si los residuos estandarizados del GARCH(1,1) tienen kurtosis cercana a 0 y media cercana a 0, el modelo captura bien la heterocedasticidad. Si el Jarque-Bera sobre residuos sigue siendo significativo, se podría considerar distribución t-Student.

**Pronóstico:**
El GARCH pronostica una volatilidad de [VALOR]% diario para los próximos 10 días, equivalente a [VALOR]% anualizado. Este pronóstico se usa como input para el VaR dinámico."

---

## 🛡️ MÓDULO 4 – CAPM (2 minutos)

"El Módulo 4 cuantifica el riesgo sistemático usando el endpoint `/capm`.

**Fórmula CAPM:**
E[R_i] = Rf + β_i × (E[R_m] - Rf)

Donde:
- Rf = [VALOR]% (T-Bill 13W, obtenida automáticamente de Yahoo Finance)
- E[R_m] - Rf = prima de riesgo de mercado ≈ [VALOR]% anual
- β_i = sensibilidad del activo i al mercado

**Clasificación de activos:**
- **β > 1.1 (Agresivo)**: TSM y MSI. Amplifican movimientos del mercado. En bull market superan al índice; en crash lo multiplican
- **β < 0.9 (Defensivo)**: JNJ, PG, UL. Amortiguan las caídas. Son activos de refugio
- **β ≈ 1 (Neutro)**: Se mueve al ritmo del mercado

**R² y descomposición del riesgo:**
- El R² indica la proporción del riesgo total que es **sistemático** (no diversificable)
- 1 - R² es el riesgo **idiosincrático** (diversificable mediante portafolio)
- XOM tiene R² bajo en ciertos períodos porque su riesgo está muy ligado al precio del petróleo, no al mercado general

**La Security Market Line (SML):**
Muestra el rendimiento esperado CAPM para cada nivel de beta. Activos sobre la SML están subvalorados (generan más rendimiento del que predice el CAPM)."

---

## ⚠️ MÓDULO 5 – VaR y CVaR (2-3 minutos)

"El Módulo 5 cuantifica la pérdida potencial del portafolio usando el endpoint `/var`.

**¿Qué es el VaR?**
El VaR al 95% diario de [VALOR]% significa que con probabilidad del 95%, la pérdida diaria del portafolio NO superará [VALOR]%.

**Tres métodos implementados:**

1. **VaR Paramétrico Normal**: asume distribución normal de rendimientos
   - Ventaja: cálculo rápido y analítico
   - Desventaja: **subestima el riesgo** porque el Módulo 2 demostró no-normalidad (colas pesadas)

2. **VaR Histórico**: usa los rendimientos reales observados, sin asumir distribución
   - Captura implícitamente las colas pesadas
   - Retrospectivo: puede no reflejar regímenes futuros

3. **VaR Montecarlo**: simulamos 10,000 escenarios de rendimiento
   - Flexible: puede incorporar distribuciones no normales y correlaciones complejas
   - El resultado depende de los supuestos de distribución

**CVaR (Expected Shortfall):**
El CVaR es la pérdida PROMEDIO en los peores escenarios más allá del VaR. Es la métrica preferida por **Basilea III** porque es una medida **coherente de riesgo** (satisface subaditividad: CVaR(A+B) ≤ CVaR(A) + CVaR(B)).

**Backtesting con Test de Kupiec:**
Verificamos si el número de días en que la pérdida superó el VaR es consistente con el nivel de confianza:
- p-valor = [VALOR]: [✅ el modelo es válido / ❌ hay demasiadas/pocas violaciones]"

---

## 🎯 MÓDULO 6 – MARKOWITZ (2 minutos)

"El Módulo 6 construye la frontera eficiente usando el endpoint `/frontera-eficiente`.

**Matriz de correlación:**
Las correlaciones bajas entre activos permiten reducir el riesgo del portafolio sin sacrificar rendimiento. UL (europea) y TSM (taiwanesa) tienen correlaciones menores con los activos americanos, aportando diversificación geográfica.

**Frontera Eficiente:**
Simulamos [N] portafolios aleatorios con pesos diferentes. Cada punto representa una combinación riesgo-rendimiento. La frontera eficiente es el contorno superior de esta nube: para cada nivel de riesgo, es el portafolio con el mayor rendimiento posible.

**Portafolio de Máximo Sharpe:**
- Rendimiento anual: [VALOR]%
- Volatilidad: [VALOR]%
- Sharpe Ratio: [VALOR]
- Composición: [VALORES]
- Este portafolio maximiza el rendimiento por unidad de riesgo asumido (ajustado por la Rf)

**Portafolio de Mínima Varianza:**
- Rendimiento: [VALOR]%, Volatilidad: [VALOR]%
- Indicado para inversores muy conservadores dispuestos a sacrificar rendimiento

**Contagio en crisis:**
Durante períodos de stress, las correlaciones aumentan (fenómeno de contagio financiero), reduciendo los beneficios de diversificación justo cuando más se necesitan."

---

## 🚦 MÓDULO 7 – SEÑALES (1-2 minutos)

"El Módulo 7 operacionaliza los indicadores técnicos como señales automáticas de compra/venta usando el endpoint `/alertas`.

**Sistema de 5 indicadores:**
Para cada activo evaluamos RSI, MACD, Bandas de Bollinger, Medias Móviles y Estocástico.

**Lógica de consenso:**
- 3 o más señales de COMPRA → señal global: 🟢 COMPRA
- 3 o más señales de VENTA → señal global: 🔴 VENTA
- Sin consenso claro → 🟡 NEUTRAL

**Umbrales configurables:**
El usuario puede ajustar los umbrales de RSI (por defecto 70/30) y estocástico (80/20) en tiempo real, y el sistema recalcula las señales automáticamente.

**Limitación importante:**
Las señales técnicas son herramientas de corto plazo. No predicen eventos fundamentales. Siempre deben complementarse con el contexto macroeconómico."

---

## 🌍 MÓDULO 8 – MACRO Y BENCHMARK (1-2 minutos)

"El Módulo 8 evalúa el desempeño del portafolio versus el S&P 500 e integra contexto macro del endpoint `/macro`.

**Métricas de desempeño:**

- **Alpha de Jensen = [VALOR]%**: rendimiento 'extra' sobre lo esperado por el beta del portafolio
  - Positivo: el portafolio genera valor añadido
  - Negativo/nulo: consistente con la hipótesis de eficiencia de mercado (Fama, 1970)

- **Tracking Error = [VALOR]%**: volatilidad de los rendimientos activos (portafolio menos benchmark)
  - TE alto → alta diferenciación respecto al benchmark

- **Information Ratio = [VALOR]**: rendimiento activo / tracking error
  - IR > 0.5: gestión activa con valor añadido significativo
  - IR < 0: la estrategia pasiva (buy & hold S&P 500) fue más eficiente

- **Máximo Drawdown = [VALOR]%**: mayor caída desde un pico histórico
  - Compara con el S&P 500 para evaluar la resiliencia del portafolio"

---

## 🔌 DEMOSTRACIÓN FASTAPI (1-2 minutos) – MUY IMPORTANTE

**Abrir el navegador en http://localhost:8000/docs**

"Aquí pueden ver la documentación interactiva generada automáticamente por FastAPI. Esta documentación se enriquece porque todos nuestros modelos Pydantic usan `Field(description=...)`.

Vamos a ejecutar el endpoint `/var` en vivo. Los modelos Pydantic validan automáticamente:
- Que los tickers sean strings válidos (el `@field_validator` los convierte a mayúsculas)
- Que los pesos sean positivos
- Que la suma de pesos sea 1.0 (el `@model_validator` hace esta validación cruzada)

Si enviamos pesos que no suman 1.0, la API responde con HTTP 422 Unprocessable Entity con un mensaje claro del error."

---

## ❓ PREGUNTAS FRECUENTES DEL PROFESOR

**P: ¿Por qué eligieron esos activos?**
R: "Buscamos diversificación sectorial (tecnología, energía, salud, consumo, semiconductores) y geográfica (EE.UU., Europa, Taiwan). La combinación de activos defensivos (JNJ, PG, UL) y agresivos (TSM, MSI) nos permite demostrar todos los conceptos del curso: betas distintos, correlaciones bajas, frontera eficiente no trivial."

**P: ¿Qué modelo GARCH seleccionaron y por qué?**
R: "Comparamos 4 especificaciones y seleccionamos según AIC y BIC. El [MODELO] fue el mejor por [AIC/BIC]. Si AIC y BIC coinciden, la elección es robusta. Optamos por [MODELO] porque [RAZÓN – menor criterio de información]. El EGARCH es preferible si hay evidencia de leverage effect (las malas noticias generan más volatilidad que las buenas)."

**P: ¿Cómo validaron los datos de entrada con Pydantic? Muestre un @field_validator.**
R: "En `models.py`, en la clase `VaRRequest`, el `@field_validator('tickers')` convierte todos los tickers a mayúsculas y elimina espacios. El `@field_validator('weights')` verifica que todos sean positivos. Y el `@model_validator(mode='after')` hace la validación cruzada: verifica que el número de tickers coincida con el número de pesos, y que la suma sea 1.0."

**P: ¿Qué dependencias inyectan con Depends()?**
R: "Inyectamos tres tipos de dependencias:
1. `DataService`: encapsula todas las llamadas a Yahoo Finance con caché
2. `Settings`: configuración desde el archivo .env (tickers, período, confianza del VaR)
3. `RiskCalculator`, `PortfolioAnalyzer`: lógica de cálculo financiero

No pusimos la lógica directamente en la ruta porque viola el principio de responsabilidad única (SRP) y hace el código difícil de testear. Con `Depends()` podemos mockear los servicios en pruebas unitarias."

**P: ¿Dónde están las API keys?**
R: "En el archivo `.env` que está excluido de Git mediante `.gitignore`. La clase `Settings(BaseSettings)` en `config.py` las carga automáticamente. Nunca aparecen en el código fuente. Si alguien clona el repositorio, debe crear su propio `.env`."

**P: ¿Qué pasa si la API externa no responde?**
R: "El backend maneja el error con `HTTPException(status_code=503, detail='...')`. El frontend Streamlit captura ese error y muestra un mensaje amigable al usuario. Además, el decorador `@cache_result(ttl=3600)` evita llamadas repetidas a la API externa en cada request."

**P: ¿Qué señales está generando el sistema hoy?**
R: "Para [TICKER], el sistema muestra [N] señales de compra y [N] de venta. La señal global es [BUY/SELL/NEUTRAL]. Esto se debe a que [RSI está en X / MACD cruzó al alza / etc.]. Sin embargo, estas son señales técnicas de corto plazo; el contexto macro (VIX en [VALOR], tensiones geopolíticas activas) sugiere cautela."

**P: ¿Cuál es el método de VaR más adecuado para este portafolio?**
R: "El Módulo 2 demostró que los rendimientos no son normales (Jarque-Bera rechaza H0). Por eso, el **VaR histórico** es el más adecuado para este portafolio: no asume distribución y captura implícitamente las colas pesadas. El VaR paramétrico subestima el riesgo. El Montecarlo es flexible pero depende de los supuestos de distribución. El test de Kupiec confirma/rechaza la validez estadística del modelo histórico."

**P: ¿Cuál es la composición del portafolio óptimo?**
R: "El portafolio de máximo Sharpe asigna mayor peso a [ACTIVOS] porque tienen el mejor balance riesgo-rendimiento ajustado por la tasa libre de riesgo. El portafolio de mínima varianza asigna más peso a [ACTIVOS] porque son los activos más defensivos (JNJ, PG) con correlaciones bajas con el resto."

**P: ¿Qué es el decorador @log_execution_time?**
R: "Es un decorador personalizado que implementamos en `services.py`. Usa `functools.wraps` para preservar los metadatos de la función original y `time.perf_counter()` para medir el tiempo de ejecución. Registra el resultado con `logging.info`. Esto permite monitorear el rendimiento de cada función de servicio sin modificar su código. Es un ejemplo clásico del patrón AOP (Aspect-Oriented Programming)."

---

## 🏁 CIERRE (30 segundos)

"Para concluir, este proyecto integra:
- **Análisis estadístico de riesgo** (GARCH, VaR, CAPM, Markowitz)
- **Buenas prácticas de ingeniería de software** (FastAPI, Pydantic, Depends, BaseSettings)
- **Datos en tiempo real** mediante Yahoo Finance
- **Visualización interactiva** en Streamlit con colores institucionales USTA

Quedamos disponibles para responder cualquier pregunta. Muchas gracias, Profesor Sierra."

---

## 📝 CHECKLIST ANTES DE LA SUSTENTACIÓN

- [ ] Backend corriendo: `uvicorn app.main:app --reload`
- [ ] Frontend corriendo: `python -m streamlit run app.py`
- [ ] `http://localhost:8000/docs` abre correctamente
- [ ] `http://localhost:8501` muestra el tablero
- [ ] Datos cargando desde Yahoo Finance
- [ ] Conocer los valores actuales de todos los indicadores del portafolio
- [ ] Conocer cuál modelo GARCH fue seleccionado y por qué
- [ ] Conocer la composición del portafolio de máximo Sharpe actual
- [ ] Poder mostrar el código de `@field_validator` en vivo
- [ ] Poder mostrar la arquitectura de `Depends()` en vivo
- [ ] README.md actualizado con instrucciones
