# 🚀 Kraken Swing Trading Bot V3 - Advanced Multi-Asset

Bot de trading automatizado de última generación con **Machine Learning**, **Multi-Asset**, **Correlación**, **Régimen Adaptativo** y **Walk-Forward Optimization**.

## 🆕 ¿Qué hay de nuevo en V3?

### ✅ **Errores Corregidos**
- **CRÍTICO**: Arreglado error `"General:Unknown method"` al cerrar posiciones
  - Ahora usa el método correcto: crear orden opuesta con `reduce_only=true`
  - Antes: `ClosePosition` (no existe en Kraken API)
  - Ahora: `AddOrder` con parámetros correctos

### 🎯 **Nuevas Funcionalidades**

#### 1. **Multi-Asset Trading**
- Opera en múltiples criptomonedas simultáneamente
- Gestión automática de portfolio
- Por defecto: BTC, ETH, ADA, SOL
- Allocación personalizable por asset

#### 2. **Gestión de Correlación**
- Evita abrir posiciones altamente correlacionadas
- Matriz de correlación en tiempo real
- Límite configurable (default: 0.7)
- Mejor diversificación del riesgo

#### 3. **Machine Learning para Swing Points**
- Validación inteligente de señales usando features técnicos
- Scoring basado en:
  - Confirmación de volumen
  - Momentum del precio
  - Posición en rango
  - Distancia de media móvil
  - Volatilidad
- Threshold de confianza configurable

#### 4. **Régimen Adaptativo**
- Detección automática de régimen de mercado:
  - **TRENDING**: Tendencias claras → Stops más amplios, TPs ambiciosos
  - **RANGING**: Lateralización → Stops ajustados, TPs conservadores
  - **VOLATILE**: Alta volatilidad → Stops muy amplios
- Parámetros de risk se ajustan dinámicamente

#### 5. **Walk-Forward Optimization**
- Backtesting más realista y robusto
- Divide histórico en ventanas de Train/Test
- Evita overfitting
- Métricas agregadas y por ventana
- Reportes detallados

---

## 📋 Requisitos

1. **Cuenta Kraken** con margen habilitado
2. **API Keys de Kraken** con permisos:
   - Query Funds ✅
   - Query Open Orders & Trades ✅
   - Create & Modify Orders ✅
   - Cancel/Close Orders ✅
3. **Bot de Telegram** (opcional)
4. **Python 3.11+** (para backtesting local)

---

## 🚀 Setup

### 1. Clonar o crear repositorio

```bash
git clone <tu-repo>
cd <tu-repo>
```

### 2. Estructura de archivos

```
tu-repo/
├── .github/
│   └── workflows/
│       └── trading-bot-v3.yml
├── kraken_bot_v3_multi_asset.py
├── backtest_v3_walkforward.py
├── requirements.txt
└── README_V3.md
```

### 3. Configurar Secrets en GitHub

**Settings → Secrets and variables → Actions**

Agregar:
- `KRAKEN_API_KEY`
- `KRAKEN_API_SECRET`
- `TELEGRAM_BOT_TOKEN` (opcional)
- `TELEGRAM_CHAT_ID` (opcional)

---

## ⚙️ Configuración

### Configuración de Assets

Edita `kraken_bot_v3_multi_asset.py`:

```python
TRADING_PAIRS = [
    TradingPair('BTC-USD', 'XBTEUR', 0.0001, 0.30),  # 30% allocación
    TradingPair('ETH-USD', 'ETHEUR', 0.001, 0.25),   # 25% allocación
    TradingPair('ADA-USD', 'ADAEUR', 10.0, 0.25),    # 25% allocación
    TradingPair('SOL-USD', 'SOLEUR', 0.01, 0.20),    # 20% allocación
]
```

**Formato:** `TradingPair(yfinance_symbol, kraken_pair, min_volume, allocation)`

### Variables de Entorno (workflow)

En `trading-bot-v3.yml`:

```yaml
# Multi-Asset
MAX_CORRELATION: '0.7'     # Correlación máxima (0.0-1.0)
MAX_POSITIONS: '3'         # Máximo posiciones simultáneas

# Trading
LEVERAGE: '3'
MIN_BALANCE: '50.0'

# Risk (base - se adaptan según régimen)
STOP_LOSS_PCT: '4.0'
TAKE_PROFIT_PCT: '8.0'
TRAILING_STOP_PCT: '2.5'

# Strategy
LOOKBACK_PERIOD: '180d'
CANDLE_INTERVAL: '1h'
USE_VOLUME_FILTER: 'true'
REGIME_LOOKBACK: '30'

# Machine Learning
USE_ML_VALIDATION: 'true'
ML_CONFIDENCE_THRESHOLD: '0.6'

# Mode
DRY_RUN: 'true'  # false = REAL
```

---

## 🎮 Uso

### GitHub Actions (Automático)

1. **Primera ejecución (simulación)**
   - Actions → Kraken Trading Bot V3 → Run workflow
   - `dry_run: true`
   - Verificar logs

2. **Activar modo REAL**
   - Editar `trading-bot-v3.yml`
   - Cambiar `DRY_RUN: 'false'`
   - Push al repositorio

3. **Ejecución automática**
   - Se ejecuta cada hora automáticamente
   - Revisa notificaciones en Telegram

### Backtesting Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar backtesting con walk-forward
python backtest_v3_walkforward.py
```

**El script genera:**
- `walkforward_results.png` - Gráficos completos
- `walkforward_windows.csv` - Métricas por ventana
- `walkforward_trades.csv` - Todos los trades

---

## 📊 Funcionalidades Avanzadas

### 1. Machine Learning

El sistema valida cada swing point con un score basado en:

```
Score = (0.3 * volume_confirmation) +
        (0.25 * momentum_clarity) +
        (0.2 * position_in_range) +
        (0.15 * distance_from_sma) +
        (0.1 * volatility_check)
```

Solo señales con score > threshold (default: 0.6) son válidas.

### 2. Régimen Adaptativo

| Régimen | Stop Loss | Take Profit | Trailing Stop |
|---------|-----------|-------------|---------------|
| **TRENDING** | +20% más amplio | +50% más ambicioso | Normal |
| **RANGING** | -20% más ajustado | -30% más conservador | -20% |
| **VOLATILE** | +50% más amplio | Normal | +30% |

### 3. Gestión de Correlación

El bot evita abrir posiciones correlacionadas:

```
Si corr(BTC, ETH) > 0.7:
    No abrir ETH si BTC ya está abierto
```

Esto mejora la diversificación y reduce el riesgo.

### 4. Walk-Forward Optimization

```
[Train 120d][Test 30d]
         [Train 120d][Test 30d]
                  [Train 120d][Test 30d]
                           ...

Step: 15 días entre ventanas
```

**Métricas reportadas:**
- Return % por ventana
- Win Rate por ventana
- Sharpe Ratio por ventana
- Drawdown por ventana
- Agregados totales

---

## 📈 Ejemplo de Resultados

### Backtest Walk-Forward (2 años)

```
═══════════════════════════════════════════════════════════
RESULTADOS AGREGADOS WALK-FORWARD
═══════════════════════════════════════════════════════════
Total ventanas:         20
Total trades:           156
Win rate promedio:      58.5%
Return promedio:        12.3%
Desv. estándar return:  8.7%
Sharpe promedio:        1.45
Profit Factor total:    2.1
Ventanas positivas:     16/20
═══════════════════════════════════════════════════════════
```

### Notificación Telegram

```
🟢 NUEVA POSICIÓN

Par: BTC-USD (XBTEUR)
Tipo: BUY
Precio: $42,350.00
Cantidad: 0.0071
Leverage: 3x

ML Confidence: 78%
Régimen: TRENDING
Fecha: 2024-12-27 15:30
```

---

## 🛠️ Troubleshooting

### Error "General:Unknown method"

✅ **Solucionado en V3**. Ahora usa `AddOrder` con `reduce_only=true`.

### Bot no abre posiciones

Posibles causas:
1. No detecta swing points → Normal, esperar
2. Correlación alta → Ajustar `MAX_CORRELATION`
3. ML confidence baja → Reducir `ML_CONFIDENCE_THRESHOLD`

### Muchas posiciones perdedoras

Ajustes recomendados:
1. Aumentar `ML_CONFIDENCE_THRESHOLD` (ej: 0.7)
2. Reducir `LEVERAGE` (ej: 2x)
3. Usar `STOP_LOSS_PCT` más ajustado

### Backtesting tarda mucho

- Reduce `LOOKBACK_PERIOD` a 1 año
- Usa menos symbols
- Aumenta `step_days` en walk-forward

---

## 🔬 Comparación de Versiones

| Feature | V1 | V2 | **V3** |
|---------|----|----|--------|
| Single Asset | ✅ | ✅ | ✅ |
| Multi-Asset | ❌ | ❌ | ✅ |
| Stop/TP/Trailing | ✅ | ✅ | ✅ |
| Volume Filter | ❌ | ✅ | ✅ |
| ML Validation | ❌ | ❌ | ✅ |
| Régimen Adaptativo | ❌ | ❌ | ✅ |
| Correlación | ❌ | ❌ | ✅ |
| Walk-Forward | ❌ | ❌ | ✅ |
| Cierre Posiciones | ❌ (bug) | ❌ (bug) | ✅ |

---

## 🎯 Estrategias Recomendadas

### Conservador (Bajo Riesgo)

```yaml
MAX_POSITIONS: '2'
LEVERAGE: '2'
STOP_LOSS_PCT: '3.0'
ML_CONFIDENCE_THRESHOLD: '0.7'
MAX_CORRELATION: '0.5'
```

### Balanceado (Default)

```yaml
MAX_POSITIONS: '3'
LEVERAGE: '3'
STOP_LOSS_PCT: '4.0'
ML_CONFIDENCE_THRESHOLD: '0.6'
MAX_CORRELATION: '0.7'
```

### Agresivo (Alto Riesgo)

```yaml
MAX_POSITIONS: '4'
LEVERAGE: '5'
STOP_LOSS_PCT: '5.0'
ML_CONFIDENCE_THRESHOLD: '0.5'
MAX_CORRELATION: '0.8'
```

---

## 📝 Notas Importantes

1. **SIEMPRE empieza en simulación** (`DRY_RUN=true`)
2. El bot puede tener rachas perdedoras (es normal)
3. No inviertas más de lo que puedes perder
4. Revisa logs y notificaciones regularmente
5. Walk-forward muestra performance realista
6. Backtests pasados NO garantizan resultados futuros

---

## 🤝 Contribuir

Pull requests bienvenidos. Para cambios mayores:

1. Fork el repositorio
2. Crea tu feature branch
3. Commit tus cambios
4. Push al branch
5. Abre un Pull Request

---

## 📄 Licencia

MIT - Usa bajo tu propio riesgo

---

## 🆘 Soporte

**¿Problemas?**

1. Revisa logs en GitHub Actions
2. Verifica Secrets están configurados
3. Asegúrate API keys tienen permisos correctos
4. Consulta Troubleshooting arriba

**¿Dudas sobre estrategia?**

- Ejecuta backtesting primero
- Analiza métricas walk-forward
- Empieza con capital pequeño

---

**🚀 Happy Trading!**

*Recuerda: Los mercados son impredecibles. Este bot es una herramienta, no una garantía de ganancias.*
