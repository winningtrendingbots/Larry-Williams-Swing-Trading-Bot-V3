"""
KRAKEN SWING BOT V3 - FIX PARA CIERRE DE POSICIONES
Distingue correctamente entre:
- Posiciones Spot (leverage=1): orden simple opuesta
- Posiciones Margin (leverage>1): orden con reduce_only
"""

import os
import time
import hmac
import hashlib
import base64
import urllib.parse
from datetime import datetime, timedelta
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import json

# ═══════════════════════════════════════════════════════════════════════════
#                        CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradingPair:
    yf_symbol: str
    kraken_pair: str
    min_volume: float
    allocation: float

class Config:
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
    KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET', '')
    KRAKEN_API_URL = 'https://api.kraken.com'
    
    TRADING_PAIRS = [
        TradingPair('BTC-USD', 'XBTEUR', 0.0001, 0.30),
        TradingPair('ETH-USD', 'ETHEUR', 0.001, 0.25),
        TradingPair('ADA-USD', 'ADAEUR', 10.0, 0.25),
        TradingPair('SOL-USD', 'SOLEUR', 0.01, 0.20),
    ]
    
    MAX_CORRELATION = float(os.getenv('MAX_CORRELATION', '0.7'))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '3'))
    LEVERAGE = int(os.getenv('LEVERAGE', '3'))
    MIN_BALANCE = float(os.getenv('MIN_BALANCE', '10.0'))
    MARGIN_SAFETY_FACTOR = 1.5
    
    REGIME_LOOKBACK = int(os.getenv('REGIME_LOOKBACK', '30'))
    BASE_STOP_LOSS = float(os.getenv('STOP_LOSS_PCT', '4.0'))
    BASE_TAKE_PROFIT = float(os.getenv('TAKE_PROFIT_PCT', '8.0'))
    BASE_TRAILING_STOP = float(os.getenv('TRAILING_STOP_PCT', '2.5'))
    MIN_PROFIT_FOR_TRAILING = float(os.getenv('MIN_PROFIT_FOR_TRAILING', '3.0'))
    
    LOOKBACK_PERIOD = os.getenv('LOOKBACK_PERIOD', '180d')
    CANDLE_INTERVAL = os.getenv('CANDLE_INTERVAL', '1h')
    USE_VOLUME_FILTER = os.getenv('USE_VOLUME_FILTER', 'true').lower() == 'true'
    
    USE_ML_VALIDATION = os.getenv('USE_ML_VALIDATION', 'true').lower() == 'true'
    ML_CONFIDENCE_THRESHOLD = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.6'))
    
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    DRY_RUN = os.getenv('DRY_RUN', 'true').lower() == 'false'


# ═══════════════════════════════════════════════════════════════════════════
#                        KRAKEN CLIENT (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

class KrakenClient:
    def __init__(self, api_key: str, api_secret: str, api_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = api_url
        self.session = requests.Session()
    
    def _sign(self, urlpath: str, data: dict) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()
    
    def _request(self, endpoint: str, data: dict = None, private: bool = False) -> dict:
        url = self.api_url + endpoint
        
        if private:
            data = data or {}
            data['nonce'] = int(time.time() * 1000)
            headers = {
                'API-Key': self.api_key,
                'API-Sign': self._sign(endpoint, data)
            }
            response = self.session.post(url, data=data, headers=headers, timeout=30)
        else:
            response = self.session.get(url, params=data, timeout=30)
        
        response.raise_for_status()
        result = response.json()
        
        if result.get('error') and len(result['error']) > 0:
            raise Exception(f"Kraken error: {result['error']}")
        
        return result.get('result', {})
    
    def get_balance(self) -> Tuple[float, str]:
        result = self._request('/0/private/Balance', private=True)
        balances = {k: float(v) for k, v in result.items()}
        
        fiat = {'ZUSD': 'USD', 'USD': 'USD', 'ZEUR': 'EUR', 'EUR': 'EUR'}
        
        for key, currency in fiat.items():
            if key in balances and balances[key] > 0:
                return balances[key], currency
        
        return 0.0, 'EUR'
    
    def get_available_margin(self) -> float:
        try:
            result = self._request('/0/private/TradeBalance', private=True)
            margin_free = float(result.get('mf', 0))
            print(f"   💰 Margen disponible: {margin_free:.2f} EUR")
            return margin_free
        except Exception as e:
            print(f"   ⚠️  Error obteniendo margen: {e}")
            balance, _ = self.get_balance()
            return balance * 0.5
    
    def get_open_positions(self) -> Dict:
        try:
            result = self._request('/0/private/OpenPositions', private=True)
            return result if result else {}
        except Exception as e:
            if "No open positions" in str(e) or "positions" not in str(e).lower():
                return {}
            raise
    
    def get_open_orders(self) -> Dict:
        try:
            result = self._request('/0/private/OpenOrders', private=True)
            return result.get('open', {})
        except:
            return {}
    
    def place_order(self, pair: str, order_type: str, volume: float, 
                   leverage: int = None, reduce_only: bool = False) -> dict:
        """
        ✅ CORREGIDO: Maneja correctamente spot vs margin
        """
        data = {
            'pair': pair,
            'type': order_type,
            'ordertype': 'market',
            'volume': str(round(volume, 8))
        }
        
        # ✅ CLAVE: Solo añadir leverage y reduce_only si leverage > 1
        is_margin_trade = leverage and leverage > 1
        
        if is_margin_trade:
            data['leverage'] = str(leverage)
            print(f"   📊 Margin trade con leverage: {leverage}x")
            
            if reduce_only:
                data['reduce_only'] = 'true'
                print(f"   🔒 reduce_only activado")
        else:
            print(f"   💱 Spot trade (sin leverage)")
        
        print(f"   📤 Orden: {data}")
        
        return self._request('/0/private/AddOrder', data=data, private=True)
    
    def close_position(self, pair: str, position_type: str, volume: float, 
                      leverage: int = None) -> dict:
        """
        ✅ CORREGIDO: Cierra correctamente según tipo de posición
        
        - Spot (leverage=1): Orden opuesta simple, SIN leverage ni reduce_only
        - Margin (leverage>1): Orden opuesta CON leverage y reduce_only
        """
        opposite_type = 'sell' if position_type == 'long' else 'buy'
        is_margin_position = leverage and leverage > 1
        
        print(f"\n   📍 Cerrando posición {'MARGIN' if is_margin_position else 'SPOT'}")
        print(f"   Leverage original: {leverage}x")
        print(f"   Tipo: {opposite_type.upper()}")
        print(f"   Volumen: {volume}")
        
        if is_margin_position:
            # Posición de margen: usar leverage y reduce_only
            return self.place_order(
                pair=pair,
                order_type=opposite_type,
                volume=volume,
                leverage=leverage,
                reduce_only=True
            )
        else:
            # Posición spot: orden simple sin parámetros adicionales
            return self.place_order(
                pair=pair,
                order_type=opposite_type,
                volume=volume,
                leverage=None,  # ✅ Sin leverage para spot
                reduce_only=False  # ✅ Sin reduce_only para spot
            )


# [RESTO DEL CÓDIGO IGUAL - RegimeDetector, MLSwingValidator, etc.]
# ... (copiar las clases RegimeDetector, MLSwingValidator, CorrelationManager, 
#      SwingDetectorV3, Telegram del código anterior)


# ═══════════════════════════════════════════════════════════════════════════
#                        POSITION MANAGER V3 (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

class PositionManagerV3:
    def __init__(self, config: Config, kraken: KrakenClient, telegram):
        self.config = config
        self.kraken = kraken
        self.telegram = telegram
        self.peak_prices = {}
        self.position_regimes = {}
    
    def check_position(self, pos_id: str, pos_data: dict, current_price: float,
                      regime_params: Dict[str, float]) -> Tuple[bool, str]:
        pos_type = pos_data.get('type', 'long')
        entry_price = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        leverage = float(pos_data.get('leverage', 1))
        
        if pos_type == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
        
        stop_loss = regime_params['stop_loss']
        take_profit = regime_params['take_profit']
        trailing_stop = regime_params['trailing_stop']
        
        print(f"   {pos_type.upper()}: entrada ${entry_price:.4f}, actual ${current_price:.4f}")
        print(f"   PnL: {pnl_pct:+.2f}% | SL: {stop_loss:.1f}% | TP: {take_profit:.1f}%")
        
        if pnl_pct <= -stop_loss:
            return True, f"🛑 Stop Loss: {pnl_pct:.2f}%"
        
        if pnl_pct >= take_profit:
            return True, f"🎯 Take Profit: {pnl_pct:.2f}%"
        
        if pnl_pct >= self.config.MIN_PROFIT_FOR_TRAILING:
            if pos_id not in self.peak_prices:
                self.peak_prices[pos_id] = current_price
            
            if pos_type == 'long' and current_price > self.peak_prices[pos_id]:
                self.peak_prices[pos_id] = current_price
            elif pos_type == 'short' and current_price < self.peak_prices[pos_id]:
                self.peak_prices[pos_id] = current_price
            
            peak = self.peak_prices[pos_id]
            if pos_type == 'long':
                peak_pnl = ((peak - entry_price) / entry_price) * 100 * leverage
            else:
                peak_pnl = ((entry_price - peak) / entry_price) * 100 * leverage
            
            drawdown = peak_pnl - pnl_pct
            
            if drawdown >= trailing_stop:
                return True, f"📉 Trailing: peak {peak_pnl:.2f}%, actual {pnl_pct:.2f}%"
        
        return False, ""
    
    def close_position(self, pair: str, pos_type: str, volume: float, 
                      reason: str, pos_data: dict, current_price: float):
        """
        ✅ CORREGIDO: Usa el nuevo método de cierre que maneja spot/margin
        """
        print(f"\n🔴 Cerrando {pair} ({pos_type})")
        print(f"   Razón: {reason}")
        
        # ✅ Obtener leverage original (crítico)
        leverage = int(float(pos_data.get('leverage', 1)))
        
        if not self.config.DRY_RUN:
            try:
                # ✅ Usar nuevo método que distingue spot/margin
                result = self.kraken.close_position(
                    pair=pair,
                    position_type=pos_type,
                    volume=volume,
                    leverage=leverage
                )
                print(f"   ✓ Posición cerrada exitosamente")
                print(f"   Resultado: {result}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ❌ Error al cerrar: {error_msg}")
                self.telegram.send(f"❌ Error cerrando {pair}: {error_msg}")
                return
        else:
            print(f"   🧪 [SIMULACIÓN - NO SE EJECUTÓ]")
        
        # Calcular PnL
        entry = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        
        if pos_type == 'long':
            pnl_pct = ((current_price - entry) / entry) * 100 * leverage
        else:
            pnl_pct = ((entry - current_price) / entry) * 100 * leverage
        
        # Notificación
        msg = f"""
🔴 <b>POSICIÓN CERRADA</b>

<b>Par:</b> {pair}
<b>Tipo:</b> {pos_type.upper()} ({'MARGIN' if leverage > 1 else 'SPOT'})
<b>Entrada:</b> ${entry:.4f}
<b>Salida:</b> ${current_price:.4f}
<b>PnL:</b> {pnl_pct:+.2f}%
<b>Leverage:</b> {leverage}x
<b>Razón:</b> {reason}
"""
        if self.config.DRY_RUN:
            msg = "🧪 <b>SIMULACIÓN</b>\n" + msg
        
        self.telegram.send(msg)

# ═══════════════════════════════════════════════════════════════════════════
#                        TRADING BOT V3
# ═══════════════════════════════════════════════════════════════════════════

class TradingBotV3:
    def __init__(self, config: Config):
        self.config = config
        self.kraken = KrakenClient(config.KRAKEN_API_KEY, config.KRAKEN_API_SECRET, 
                                   config.KRAKEN_API_URL)
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.position_mgr = PositionManagerV3(config, self.kraken, self.telegram)
    
    def get_market_data(self, symbol: str) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=self.config.LOOKBACK_PERIOD, 
                            interval=self.config.CANDLE_INTERVAL)
        
        if data.empty:
            raise Exception(f"No data for {symbol}")
        
        return data
    
    def calculate_required_margin(self, price: float, volume: float, 
                                  leverage: int) -> float:
        """
        ✅ NUEVO: Calcula el margen requerido para una posición.
        
        Margen requerido = (Precio * Volumen) / Leverage
        
        Añadimos factor de seguridad para fees y slippage.
        """
        position_size = price * volume
        margin_required = position_size / leverage
        
        # Factor de seguridad: 10% adicional para fees, slippage, etc.
        margin_with_safety = margin_required * 1.1
        
        return margin_with_safety
    
    def run(self):
        print("\n" + "="*70)
        print("KRAKEN SWING BOT V3 - MULTI-ASSET + ML + ADAPTIVE (FIXED)")
        print("="*70)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Modo: {'🧪 SIMULACIÓN' if self.config.DRY_RUN else '💰 REAL'}")
        print(f"ML Validation: {'✅' if self.config.USE_ML_VALIDATION else '❌'}")
        print("="*70)
        
        try:
            # Obtener balance y margen disponible
            balance, currency = self.kraken.get_balance()
            available_margin = self.kraken.get_available_margin()
            
            print(f"\n💰 Balance: {balance:.2f} {currency}")
            print(f"   Margen disponible: {available_margin:.2f} {currency}")
            
            if balance < self.config.MIN_BALANCE:
                print(f"⚠️  Balance insuficiente (min: {self.config.MIN_BALANCE})")
                return
            
            # Calcular margen usable (con factor de seguridad)
            usable_margin = available_margin / self.config.MARGIN_SAFETY_FACTOR
            print(f"   Margen usable (con seguridad): {usable_margin:.2f} {currency}")
            
            print("\n📊 Descargando datos multi-asset...")
            market_data = {}
            for pair in self.config.TRADING_PAIRS:
                try:
                    data = self.get_market_data(pair.yf_symbol)
                    market_data[pair.yf_symbol] = data
                    print(f"   ✓ {pair.yf_symbol}: {len(data)} velas")
                except Exception as e:
                    print(f"   ❌ {pair.yf_symbol}: {e}")
            
            if not market_data:
                print("❌ No se pudieron descargar datos")
                return
            
            print("\n🔗 Calculando correlaciones...")
            corr_matrix = CorrelationManager.calculate_correlation_matrix(
                market_data, lookback=30
            )
            
            if not corr_matrix.empty:
                print("   Matriz de correlación:")
                print(corr_matrix.round(2))
            
            print("\n📊 Verificando posiciones...")
            positions = self.kraken.get_open_positions()
            
            open_symbols = []
            total_margin_used = 0.0
            
            if positions:
                print(f"✓ {len(positions)} posición(es) abierta(s)")
                
                for pos_id, pos_data in positions.items():
                    pair = pos_data.get('pair', 'UNKNOWN')
                    
                    # Calcular margen usado por esta posición
                    pos_margin = float(pos_data.get('margin', 0))
                    total_margin_used += pos_margin
                    
                    trading_pair = next(
                        (tp for tp in self.config.TRADING_PAIRS if tp.kraken_pair == pair),
                        None
                    )
                    
                    if not trading_pair or trading_pair.yf_symbol not in market_data:
                        continue
                    
                    open_symbols.append(trading_pair.yf_symbol)
                    data = market_data[trading_pair.yf_symbol]
                    current_price = float(data['Close'].iloc[-1])
                    
                    regime = RegimeDetector.detect(data, self.config.REGIME_LOOKBACK)
                    regime_params = RegimeDetector.get_adapted_params(
                        regime, 
                        self.config.BASE_STOP_LOSS,
                        self.config.BASE_TAKE_PROFIT,
                        self.config.BASE_TRAILING_STOP
                    )
                    
                    print(f"\n   {trading_pair.yf_symbol} ({pair}) - Régimen: {regime}")
                    print(f"   Margen usado: {pos_margin:.2f} {currency}")
                    
                    should_close, reason = self.position_mgr.check_position(
                        pos_id, pos_data, current_price, regime_params
                    )
                    
                    if should_close:
                        pos_type = pos_data.get('type', 'long')
                        volume = float(pos_data.get('vol', 0))
                        self.position_mgr.close_position(
                            pair, pos_type, volume, reason, pos_data, current_price
                        )
                        total_margin_used -= pos_margin  # Liberar margen
                    else:
                        print(f"   ✓ Mantener posición")
            else:
                print("✓ No hay posiciones abiertas")
            
            print(f"\n💰 Margen usado: {total_margin_used:.2f} {currency}")
            print(f"   Margen restante: {(available_margin - total_margin_used):.2f} {currency}")
            
            if len(open_symbols) >= self.config.MAX_POSITIONS:
                print(f"\nℹ️  Máximo de posiciones alcanzado ({self.config.MAX_POSITIONS})")
                return
            
            # Calcular margen disponible para nuevas posiciones
            margin_for_new = (available_margin - total_margin_used) / self.config.MARGIN_SAFETY_FACTOR
            print(f"   Margen para nuevas posiciones: {margin_for_new:.2f} {currency}")
            
            if margin_for_new < self.config.MIN_BALANCE * 0.5:
                print(f"⚠️  Margen insuficiente para nuevas posiciones")
                return
            
            print("\n🔍 Buscando señales en activos disponibles...")
            
            signals = []
            for pair in self.config.TRADING_PAIRS:
                if pair.yf_symbol not in market_data:
                    continue
                
                if pair.yf_symbol in open_symbols:
                    print(f"   ⏭️  {pair.yf_symbol}: posición ya abierta")
                    continue
                
                data = market_data[pair.yf_symbol]
                current_price = float(data['Close'].iloc[-1])
                regime = RegimeDetector.detect(data, self.config.REGIME_LOOKBACK)
                
                detector = SwingDetectorV3(
                    data, 
                    volume_filter=self.config.USE_VOLUME_FILTER,
                    use_ml=self.config.USE_ML_VALIDATION,
                    ml_threshold=self.config.ML_CONFIDENCE_THRESHOLD
                )
                signal, signal_price, confidence = detector.get_signal()
                
                if signal:
                    # ✅ Verificar si hay suficiente margen ANTES de verificar correlación
                    # Calcular volumen tentativo
                    allocation_margin = margin_for_new * pair.allocation
                    effective_capital = allocation_margin * self.config.LEVERAGE
                    tentative_volume = effective_capital / current_price
                    
                    # Calcular margen requerido
                    required_margin = self.calculate_required_margin(
                        current_price, tentative_volume, self.config.LEVERAGE
                    )
                    
                    if required_margin > allocation_margin:
                        print(f"   ⚠️  {pair.yf_symbol}: margen insuficiente "
                              f"(necesita {required_margin:.2f}, disponible {allocation_margin:.2f})")
                        continue
                    
                    # Verificar volumen mínimo
                    if tentative_volume < pair.min_volume:
                        print(f"   ⚠️  {pair.yf_symbol}: volumen {tentative_volume:.6f} < mínimo {pair.min_volume}")
                        continue
                    
                    # Ahora sí, verificar correlación
                    can_open, max_corr = CorrelationManager.check_position_correlation(
                        open_symbols, pair.yf_symbol, corr_matrix, self.config.MAX_CORRELATION
                    )
                    
                    if can_open:
                        signals.append({
                            'pair': pair,
                            'signal': signal,
                            'price': current_price,
                            'confidence': confidence,
                            'regime': regime,
                            'data': data,
                            'volume': tentative_volume,
                            'required_margin': required_margin
                        })
                        print(f"   ✓ {pair.yf_symbol}: {signal} (conf: {confidence:.2f}, "
                              f"régimen: {regime}, corr: {max_corr:.2f}, margen: {required_margin:.2f})")
                    else:
                        print(f"   ⚠️  {pair.yf_symbol}: {signal} rechazado por correlación ({max_corr:.2f})")
                else:
                    print(f"   - {pair.yf_symbol}: sin señal")
            
            if not signals:
                print("\nℹ️  No hay señales válidas")
                return
            
            # Ordenar por confianza
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Abrir posiciones hasta el límite
            positions_to_open = min(
                len(signals), 
                self.config.MAX_POSITIONS - len(open_symbols)
            )
            
            for sig in signals[:positions_to_open]:
                self.open_position(sig, margin_for_new)
            
            print("\n✅ Ciclo completado")
            
        except Exception as e:
            msg = f"Error: {str(e)}"
            print(f"\n❌ {msg}")
            import traceback
            traceback.print_exc()
            self.telegram.send(f"❌ {msg}")
            raise
    
    def open_position(self, signal_data: Dict, available_margin: float):
        """
        ✅ MEJORADO: Abre posición con verificación de margen.
        """
        pair = signal_data['pair']
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        regime = signal_data['regime']
        volume = signal_data['volume']
        required_margin = signal_data['required_margin']
        current_price = signal_data['price']
        
        try:
            print(f"\n🟢 Abriendo {signal} en {pair.yf_symbol}")
            print(f"   Precio: ${current_price:.4f}")
            print(f"   Volumen: {volume:.8f}")
            print(f"   Leverage: {self.config.LEVERAGE}x")
            print(f"   Margen requerido: {required_margin:.2f} EUR")
            print(f"   Confianza ML: {confidence:.2f}")
            print(f"   Régimen: {regime}")
            
            if not self.config.DRY_RUN:
                order_type = 'buy' if signal == 'BUY' else 'sell'
                
                try:
                    result = self.kraken.place_order(
                        pair=pair.kraken_pair,
                        order_type=order_type,
                        volume=volume,
                        leverage=self.config.LEVERAGE,
                        reduce_only=False
                    )
                    print(f"   ✓ Ejecutada: {result}")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"   ❌ Error ejecutando orden: {error_msg}")
                    
                    # Si el error es de margen insuficiente, intentar con volumen reducido
                    if "insufficient" in error_msg.lower() and "margin" in error_msg.lower():
                        print(f"   🔄 Intentando con volumen reducido (50%)...")
                        
                        reduced_volume = volume * 0.5
                        
                        if reduced_volume >= pair.min_volume:
                            result = self.kraken.place_order(
                                pair=pair.kraken_pair,
                                order_type=order_type,
                                volume=reduced_volume,
                                leverage=self.config.LEVERAGE,
                                reduce_only=False
                            )
                            print(f"   ✓ Ejecutada con volumen reducido: {result}")
                            volume = reduced_volume  # Actualizar para el mensaje
                        else:
                            print(f"   ❌ Volumen reducido {reduced_volume:.8f} < mínimo {pair.min_volume}")
                            raise
                    else:
                        raise
            else:
                print(f"   🧪 [SIMULACIÓN]")
            
            msg = f"""
🟢 <b>NUEVA POSICIÓN</b>

<b>Par:</b> {pair.yf_symbol} ({pair.kraken_pair})
<b>Tipo:</b> {signal}
<b>Precio:</b> ${current_price:.4f}
<b>Cantidad:</b> {volume:.8f}
<b>Leverage:</b> {self.config.LEVERAGE}x
<b>Margen:</b> {required_margin:.2f} EUR

<b>ML Confidence:</b> {confidence:.2%}
<b>Régimen:</b> {regime}
<b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            if self.config.DRY_RUN:
                msg = "🧪 <b>SIMULACIÓN</b>\n" + msg
            
            self.telegram.send(msg)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error abriendo {pair.yf_symbol}: {error_msg}")
            self.telegram.send(f"❌ Error en {pair.yf_symbol}: {error_msg}")


# ═══════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    config = Config()
    
    if not config.KRAKEN_API_KEY or not config.KRAKEN_API_SECRET:
        print("❌ Faltan credenciales Kraken")
        return
    
    bot = TradingBotV3(config)
    bot.run()


if __name__ == "__main__":
    main()
      pass
