"""
KRAKEN SWING BOT V3 - MULTI-ASSET + ML + ADAPTIVE (FIXED POSITIONS COUNT)
Correcciones:
- Fix: Conteo correcto de posiciones (ignora órdenes cerradas/canceladas)
- Fix: Verificación mejorada de margen disponible
- Fix: Limpieza de posiciones fantasma
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
#                          CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradingPair:
    yf_symbol: str
    kraken_pair: str
    min_volume: float
    allocation: float

class Config:
    # Kraken
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
    KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET', '')
    KRAKEN_API_URL = 'https://api.kraken.com'
    
    # Multi-asset trading
    TRADING_PAIRS = [
        TradingPair('BTC-USD', 'XBTEUR', 0.0001, 0.30),
        TradingPair('ETH-USD', 'ETHEUR', 0.001, 0.25),
        TradingPair('ADA-USD', 'ADAEUR', 10.0, 0.25),
        TradingPair('SOL-USD', 'SOLEUR', 0.01, 0.20),
    ]
    
    MAX_CORRELATION = float(os.getenv('MAX_CORRELATION', '0.7'))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '3'))
    
    # Base trading
    LEVERAGE = int(os.getenv('LEVERAGE', '3'))
    MIN_BALANCE = float(os.getenv('MIN_BALANCE', '10.0'))
    
    MARGIN_SAFETY_FACTOR = 1.5
    
    # Adaptive risk
    REGIME_LOOKBACK = int(os.getenv('REGIME_LOOKBACK', '30'))
    
    # Risk base
    BASE_STOP_LOSS = float(os.getenv('STOP_LOSS_PCT', '4.0'))
    BASE_TAKE_PROFIT = float(os.getenv('TAKE_PROFIT_PCT', '8.0'))
    BASE_TRAILING_STOP = float(os.getenv('TRAILING_STOP_PCT', '2.5'))
    MIN_PROFIT_FOR_TRAILING = float(os.getenv('MIN_PROFIT_FOR_TRAILING', '3.0'))
    
    # Strategy
    LOOKBACK_PERIOD = os.getenv('LOOKBACK_PERIOD', '180d')
    CANDLE_INTERVAL = os.getenv('CANDLE_INTERVAL', '1h')
    USE_VOLUME_FILTER = os.getenv('USE_VOLUME_FILTER', 'true').lower() == 'true'
    
    # ML
    USE_ML_VALIDATION = os.getenv('USE_ML_VALIDATION', 'true').lower() == 'true'
    ML_CONFIDENCE_THRESHOLD = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.6'))
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Mode
    DRY_RUN = os.getenv('DRY_RUN', 'true').lower() == 'true'


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
        """Retorna (balance, currency)."""
        result = self._request('/0/private/Balance', private=True)
        balances = {k: float(v) for k, v in result.items()}
        
        fiat = {'ZUSD': 'USD', 'USD': 'USD', 'ZEUR': 'EUR', 'EUR': 'EUR'}
        
        for key, currency in fiat.items():
            if key in balances and balances[key] > 0:
                return balances[key], currency
        
        return 0.0, 'EUR'
    
    def get_available_margin(self) -> float:
        """Obtiene el margen disponible para trading."""
        try:
            result = self._request('/0/private/TradeBalance', private=True)
            margin_free = float(result.get('mf', 0))
            print(f"   💰 Margen disponible: {margin_free:.2f} EUR")
            return margin_free
        except Exception as e:
            print(f"   ⚠️ Error obteniendo margen: {e}")
            balance, _ = self.get_balance()
            return balance * 0.5
    
    def get_open_positions(self) -> Dict:
        """
        ✅ MEJORADO: Retorna posiciones consolidadas por par
        - Filtra posiciones cerradas
        - Consolida múltiples posiciones del mismo par
        """
        try:
            result = self._request('/0/private/OpenPositions', private=True)
            
            if not result:
                return {}
            
            # ✅ Consolidar posiciones por par
            consolidated = {}
            
            for pos_id, pos_data in result.items():
                vol = float(pos_data.get('vol', 0))
                vol_closed = float(pos_data.get('vol_closed', 0))
                open_vol = vol - vol_closed
                
                if open_vol <= 0:
                    continue
                
                pair = pos_data.get('pair', 'UNKNOWN')
                
                # Si ya existe posición de este par, consolidar
                if pair in consolidated:
                    # Sumar volumen
                    existing_vol = float(consolidated[pair].get('vol', 0))
                    consolidated[pair]['vol'] = str(existing_vol + open_vol)
                    
                    # Usar el cost total (aproximado)
                    existing_cost = float(consolidated[pair].get('cost', 0))
                    new_cost = float(pos_data.get('cost', 0))
                    consolidated[pair]['cost'] = str(existing_cost + new_cost)
                    
                    print(f"   🔄 Consolidando {pair}: +{open_vol:.8f} → Total: {float(consolidated[pair]['vol']):.8f}")
                else:
                    # Primera posición de este par
                    pos_data['vol'] = str(open_vol)
                    consolidated[pair] = pos_data
                    print(f"   ✓ Posición nueva: {pair} - Vol: {open_vol:.8f}")
            
            return consolidated
            
        except Exception as e:
            if "No open positions" in str(e) or "positions" not in str(e).lower():
                return {}
            raise
    
    def get_open_orders(self) -> Dict:
        """
        ✅ MEJORADO: Obtener órdenes abiertas (separado de posiciones)
        """
        try:
            result = self._request('/0/private/OpenOrders', private=True)
            orders = result.get('open', {})
            
            if orders:
                print(f"   📋 {len(orders)} orden(es) abierta(s) (no posiciones)")
                for order_id, order_data in orders.items():
                    print(f"      - {order_data.get('descr', {}).get('pair')}: {order_data.get('descr', {}).get('type')}")
            
            return orders
        except:
            return {}
    
    def place_order(self, pair: str, order_type: str, volume: float, 
                   leverage: int = None, reduce_only: bool = False) -> dict:
        """Coloca orden en Kraken."""
        data = {
            'pair': pair,
            'type': order_type,
            'ordertype': 'market',
            'volume': str(round(volume, 8))
        }
        
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
        """Cierra posición correctamente."""
        opposite_type = 'sell' if position_type == 'long' else 'buy'
        is_margin_position = leverage and leverage > 1
        
        print(f"\n   🔒 Cerrando posición {'MARGIN' if is_margin_position else 'SPOT'}")
        print(f"   Leverage original: {leverage}x")
        print(f"   Tipo: {opposite_type.upper()}")
        print(f"   Volumen: {volume}")
        
        if is_margin_position:
            return self.place_order(
                pair=pair,
                order_type=opposite_type,
                volume=volume,
                leverage=leverage,
                reduce_only=True
            )
        else:
            return self.place_order(
                pair=pair,
                order_type=opposite_type,
                volume=volume,
                leverage=None,
                reduce_only=False
            )


# ═══════════════════════════════════════════════════════════════════════════
#                        MARKET REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    @staticmethod
    def detect(data: pd.DataFrame, lookback: int = 30) -> str:
        if len(data) < lookback:
            return 'RANGING'
        
        recent = data.tail(lookback)
        returns = recent['Close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_volatility = data['Close'].pct_change().dropna().std()
        
        high_low = (recent['High'] - recent['Low']).mean()
        close_change = abs(recent['Close'].iloc[-1] - recent['Close'].iloc[0])
        trend_strength = close_change / (high_low * lookback) if high_low > 0 else 0
        
        if volatility > avg_volatility * 1.5:
            return 'VOLATILE'
        elif trend_strength > 0.5:
            return 'TRENDING'
        else:
            return 'RANGING'
    
    @staticmethod
    def get_adapted_params(regime: str, base_sl: float, base_tp: float, 
                          base_trail: float) -> Dict[str, float]:
        adaptations = {
            'TRENDING': {
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 1.5,
                'trailing_stop_multiplier': 1.0,
            },
            'RANGING': {
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 0.7,
                'trailing_stop_multiplier': 0.8,
            },
            'VOLATILE': {
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.0,
                'trailing_stop_multiplier': 1.3,
            }
        }
        
        mult = adaptations.get(regime, adaptations['RANGING'])
        
        return {
            'stop_loss': base_sl * mult['stop_loss_multiplier'],
            'take_profit': base_tp * mult['take_profit_multiplier'],
            'trailing_stop': base_trail * mult['trailing_stop_multiplier']
        }


# ═══════════════════════════════════════════════════════════════════════════
#                        ML SWING VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class MLSwingValidator:
    @staticmethod
    def calculate_features(data: pd.DataFrame, swing_idx: int) -> Dict[str, float]:
        if swing_idx < 20 or swing_idx >= len(data) - 5:
            return None
        
        window = data.iloc[swing_idx-20:swing_idx+5]
        features = {}
        
        avg_vol = window['Volume'].mean()
        swing_vol = data['Volume'].iloc[swing_idx]
        features['volume_ratio'] = swing_vol / avg_vol if avg_vol > 0 else 1.0
        
        returns = window['Close'].pct_change()
        features['momentum'] = returns.mean()
        features['volatility'] = returns.std()
        
        swing_price = data['Close'].iloc[swing_idx]
        recent_high = window['High'].max()
        recent_low = window['Low'].min()
        price_range = recent_high - recent_low
        features['price_position'] = (swing_price - recent_low) / price_range if price_range > 0 else 0.5
        
        sma_20 = window['Close'].mean()
        features['distance_from_sma'] = abs(swing_price - sma_20) / sma_20 if sma_20 > 0 else 0
        
        return features
    
    @staticmethod
    def validate_swing(data: pd.DataFrame, swing_idx: int, 
                      swing_type: str, threshold: float = 0.6) -> Tuple[bool, float]:
        features = MLSwingValidator.calculate_features(data, swing_idx)
        
        if features is None:
            return False, 0.0
        
        score = 0.0
        weights = 0.0
        
        if features['volume_ratio'] > 1.2:
            score += 0.3
        elif features['volume_ratio'] > 0.8:
            score += 0.15
        weights += 0.3
        
        if swing_type == 'LOW' and features['momentum'] < -0.001:
            score += 0.25
        elif swing_type == 'HIGH' and features['momentum'] > 0.001:
            score += 0.25
        elif abs(features['momentum']) < 0.0005:
            score += 0.125
        weights += 0.25
        
        if swing_type == 'LOW' and features['price_position'] < 0.3:
            score += 0.2
        elif swing_type == 'HIGH' and features['price_position'] > 0.7:
            score += 0.2
        weights += 0.2
        
        if features['distance_from_sma'] > 0.02:
            score += 0.15
        weights += 0.15
        
        if features['volatility'] > 0.01:
            score += 0.1
        weights += 0.1
        
        confidence = score / weights if weights > 0 else 0.0
        is_valid = confidence >= threshold
        
        return is_valid, confidence


# ═══════════════════════════════════════════════════════════════════════════
#                        CORRELATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class CorrelationManager:
    @staticmethod
    def calculate_correlation_matrix(data_dict: Dict[str, pd.DataFrame], 
                                     lookback: int = 30) -> pd.DataFrame:
        returns_dict = {}
        for symbol, data in data_dict.items():
            if len(data) >= lookback:
                returns = data['Close'].tail(lookback).pct_change().dropna()
                returns_dict[symbol] = returns
        
        if len(returns_dict) < 2:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_dict)
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    @staticmethod
    def check_position_correlation(open_positions: List[str], new_symbol: str,
                                   corr_matrix: pd.DataFrame, 
                                   max_corr: float = 0.7) -> Tuple[bool, float]:
        if corr_matrix.empty or new_symbol not in corr_matrix.columns:
            return True, 0.0
        
        max_correlation = 0.0
        
        for pos_symbol in open_positions:
            if pos_symbol in corr_matrix.columns and pos_symbol != new_symbol:
                corr = abs(corr_matrix.loc[new_symbol, pos_symbol])
                max_correlation = max(max_correlation, corr)
        
        can_open = max_correlation < max_corr
        
        return can_open, max_correlation


# ═══════════════════════════════════════════════════════════════════════════
#                        SWING DETECTOR V3
# ═══════════════════════════════════════════════════════════════════════════

def calculate_volume_ma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    return data['Volume'].rolling(window=period).mean()

class SwingDetectorV3:
    def __init__(self, data: pd.DataFrame, volume_filter: bool = True, 
                 use_ml: bool = True, ml_threshold: float = 0.6):
        self.data = data.copy()
        self.volume_filter = volume_filter
        self.use_ml = use_ml
        self.ml_threshold = ml_threshold
        self.volume_ma = calculate_volume_ma(data) if volume_filter else None
        
        self.st_highs = pd.Series(index=data.index, dtype=float)
        self.st_lows = pd.Series(index=data.index, dtype=float)
        self.int_highs = pd.Series(index=data.index, dtype=float)
        self.int_lows = pd.Series(index=data.index, dtype=float)
        self.ml_confidence = {}
    
    def _check_volume(self, i: int) -> bool:
        if not self.volume_filter or self.volume_ma is None:
            return True
        
        if pd.isna(self.volume_ma.iloc[i]):
            return True
        
        return self.data['Volume'].iloc[i] > self.volume_ma.iloc[i]
    
    def detect(self):
        highs = self.data['High'].values
        lows = self.data['Low'].values
        
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                if self._check_volume(i):
                    self.st_lows.iloc[i] = lows[i]
        
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                if self._check_volume(i):
                    self.st_highs.iloc[i] = highs[i]
        
        st_high_idx = self.st_highs.dropna().index.tolist()
        for i in range(1, len(st_high_idx) - 1):
            p, c, n = st_high_idx[i-1], st_high_idx[i], st_high_idx[i+1]
            
            if self.st_highs[c] > self.st_highs[p] and self.st_highs[c] > self.st_highs[n]:
                if self.use_ml:
                    idx = self.data.index.get_loc(c)
                    is_valid, confidence = MLSwingValidator.validate_swing(
                        self.data, idx, 'HIGH', self.ml_threshold
                    )
                    if is_valid:
                        self.int_highs[c] = self.st_highs[c]
                        self.ml_confidence[c] = confidence
                else:
                    self.int_highs[c] = self.st_highs[c]
        
        st_low_idx = self.st_lows.dropna().index.tolist()
        for i in range(1, len(st_low_idx) - 1):
            p, c, n = st_low_idx[i-1], st_low_idx[i], st_low_idx[i+1]
            
            if self.st_lows[c] < self.st_lows[p] and self.st_lows[c] < self.st_lows[n]:
                if self.use_ml:
                    idx = self.data.index.get_loc(c)
                    is_valid, confidence = MLSwingValidator.validate_swing(
                        self.data, idx, 'LOW', self.ml_threshold
                    )
                    if is_valid:
                        self.int_lows[c] = self.st_lows[c]
                        self.ml_confidence[c] = confidence
                else:
                    self.int_lows[c] = self.st_lows[c]
    
    def get_signal(self) -> Tuple[Optional[str], Optional[float], float]:
        self.detect()
        
        highs = self.int_highs.dropna()
        lows = self.int_lows.dropna()
        
        if len(highs) == 0 and len(lows) == 0:
            return None, None, 0.0
        
        last_high = highs.index[-1] if len(highs) > 0 else pd.Timestamp.min
        last_low = lows.index[-1] if len(lows) > 0 else pd.Timestamp.min
        
        if last_low > last_high:
            confidence = self.ml_confidence.get(last_low, 1.0)
            return 'BUY', lows.iloc[-1], confidence
        elif last_high > last_low:
            confidence = self.ml_confidence.get(last_high, 1.0)
            return 'SELL', highs.iloc[-1], confidence
        else:
            return None, None, 0.0


# ═══════════════════════════════════════════════════════════════════════════
#                        TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════

class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}" if token else None
    
    def send(self, message: str) -> bool:
        if not self.api_url or not self.chat_id:
            print(f"📱 {message}")
            return False
        
        try:
            if len(message) > 4000:
                message = message[:3900] + "\n..."
            
            data = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}
            response = requests.post(f"{self.api_url}/sendMessage", data=data, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
#                        POSITION MANAGER V3
# ═══════════════════════════════════════════════════════════════════════════

class PositionManagerV3:
    def __init__(self, config: Config, kraken: KrakenClient, telegram: Telegram):
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
        print(f"\n🔴 Cerrando {pair} ({pos_type})")
        print(f"   Razón: {reason}")
        
        leverage = int(float(pos_data.get('leverage', 1)))
        print(f"   Leverage original: {leverage}x")
        print(f"   Volumen a cerrar: {volume}")
        
        if not self.config.DRY_RUN:
            try:
                result = self.kraken.close_position(
                    pair, pos_type, volume, leverage
                )
                print(f"   ✓ Cerrada: {result}")
            except Exception as e:
                error_msg = str(e)
                print(f"   ❌ Error: {error_msg}")
                self.telegram.send(f"❌ Error cerrando {pair}: {error_msg}")
                return
        else:
            print(f"   🧪 [SIMULACIÓN]")
        
        entry = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        
        if pos_type == 'long':
            pnl_pct = ((current_price - entry) / entry) * 100 * leverage
        else:
            pnl_pct = ((entry - current_price) / entry) * 100 * leverage
        
        msg = f"""
🔴 <b>POSICIÓN CERRADA</b>

<b>Par:</b> {pair}
<b>Tipo:</b> {pos_type.upper()}
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
#                        TRADING BOT V3 (FIXED)
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
        position_size = price * volume
        margin_required = position_size / leverage
        margin_with_safety = margin_required * 1.1
        return margin_with_safety
    
    def run(self):
        print("\n" + "="*70)
        print("KRAKEN SWING BOT V3 - FIXED POSITIONS COUNT")
        print("="*70)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Modo: {'🧪 SIMULACIÓN' if self.config.DRY_RUN else '💰 REAL'}")
        print(f"ML Validation: {'✅' if self.config.USE_ML_VALIDATION else '❌'}")
        print("="*70)
        
        try:
            balance, currency = self.kraken.get_balance()
            available_margin = self.kraken.get_available_margin()
            
            print(f"\n💰 Balance: {balance:.2f} {currency}")
            print(f"   Margen disponible: {available_margin:.2f} {currency}")
            
            if balance < self.config.MIN_BALANCE:
                print(f"⚠️ Balance insuficiente (min: {self.config.MIN_BALANCE})")
                return
            
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
            
            # ✅ MEJORADO: Verificación más robusta de posiciones
            print("\n📊 Verificando posiciones ABIERTAS...")
            positions = self.kraken.get_open_positions()  # Ya consolidadas por par
            
            # También verificar órdenes (para debug)
            orders = self.kraken.get_open_orders()
            
            open_symbols = []
            total_margin_used = 0.0
            
            # ✅ IMPORTANTE: Ahora positions está indexado por PAR, no por position_id
            valid_position_count = len(positions)
            
            print(f"✅ {valid_position_count} posición(es) únicas (consolidadas por par)")
            
            if orders:
                print(f"📋 {len(orders)} orden(es) pendiente(s) (no se cuentan como posiciones)")
            
            if positions:
                # ✅ CAMBIO: Ahora iteramos por PAR, no por position_id
                for pair, pos_data in positions.items():
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
                    
                    # Usar el par como pos_id
                    should_close, reason = self.position_mgr.check_position(
                        pair, pos_data, current_price, regime_params
                    )
                    
                    if should_close:
                        pos_type = pos_data.get('type', 'long')
                        volume = float(pos_data.get('vol', 0))
                        self.position_mgr.close_position(
                            pair, pos_type, volume, reason, pos_data, current_price
                        )
                        total_margin_used -= pos_margin
                        valid_position_count -= 1
                    else:
                        print(f"   ✓ Mantener posición")
            else:
                print("✓ No hay posiciones abiertas")
            
            print(f"\n💰 Margen usado: {total_margin_used:.2f} {currency}")
            print(f"   Margen restante: {(available_margin - total_margin_used):.2f} {currency}")
            print(f"   Posiciones activas: {valid_position_count}/{self.config.MAX_POSITIONS}")
            
            # ✅ CRUCIAL: Usar valid_position_count en lugar de len(positions)
            if valid_position_count >= self.config.MAX_POSITIONS:
                print(f"\nℹ️ Máximo de posiciones alcanzado ({self.config.MAX_POSITIONS})")
                return
            
            margin_for_new = (available_margin - total_margin_used) / self.config.MARGIN_SAFETY_FACTOR
            print(f"   Margen para nuevas posiciones: {margin_for_new:.2f} {currency}")
            
            if margin_for_new < self.config.MIN_BALANCE * 0.5:
                print(f"⚠️ Margen insuficiente para nuevas posiciones")
                return
            
            print("\n🔍 Buscando señales en activos disponibles...")
            
            signals = []
            for pair in self.config.TRADING_PAIRS:
                if pair.yf_symbol not in market_data:
                    continue
                
                if pair.yf_symbol in open_symbols:
                    print(f"   ⭐ {pair.yf_symbol}: posición ya abierta")
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
                    allocation_margin = margin_for_new * pair.allocation
                    effective_capital = allocation_margin * self.config.LEVERAGE
                    tentative_volume = effective_capital / current_price
                    
                    required_margin = self.calculate_required_margin(
                        current_price, tentative_volume, self.config.LEVERAGE
                    )
                    
                    if required_margin > allocation_margin:
                        print(f"   ⚠️ {pair.yf_symbol}: margen insuficiente "
                              f"(necesita {required_margin:.2f}, disponible {allocation_margin:.2f})")
                        continue
                    
                    if tentative_volume < pair.min_volume:
                        print(f"   ⚠️ {pair.yf_symbol}: volumen {tentative_volume:.6f} < mínimo {pair.min_volume}")
                        continue
                    
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
                        print(f"   ⚠️ {pair.yf_symbol}: {signal} rechazado por correlación ({max_corr:.2f})")
                else:
                    print(f"   - {pair.yf_symbol}: sin señal")
            
            if not signals:
                print("\nℹ️ No hay señales válidas")
                return
            
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # ✅ USAR valid_position_count
            positions_to_open = min(
                len(signals), 
                self.config.MAX_POSITIONS - valid_position_count
            )
            
            print(f"\n🎯 Abriendo {positions_to_open} posición(es)...")
            
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
                            volume = reduced_volume
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
