"""
BACKTESTING V3 - WALK-FORWARD OPTIMIZATION
Multi-Asset + ML + Adaptive Regime + Walk-Forward Analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#                        CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    # Assets
    symbols: List[str]
    initial_capital: float = 1000.0
    max_positions: int = 3
    max_correlation: float = 0.7
    
    # Risk
    leverage: int = 3
    base_stop_loss: float = 4.0
    base_take_profit: float = 8.0
    base_trailing_stop: float = 2.5
    min_profit_trailing: float = 3.0
    
    # Strategy
    use_volume_filter: bool = True
    use_ml_validation: bool = True
    ml_threshold: float = 0.6
    regime_lookback: int = 30
    
    # Walk-Forward
    train_period_days: int = 120  # Período de entrenamiento
    test_period_days: int = 30    # Período de test
    step_days: int = 15           # Paso entre ventanas

# ═══════════════════════════════════════════════════════════════════════════
#                        REGIME DETECTOR
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
#                        ML VALIDATOR
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
    
    def get_signal_at_date(self, date) -> Tuple[Optional[str], Optional[float], float]:
        highs = self.int_highs.loc[:date].dropna()
        lows = self.int_lows.loc[:date].dropna()
        
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
#                        BACKTESTER V3
# ═══════════════════════════════════════════════════════════════════════════

class BacktesterV3:
    def __init__(self, config: BacktestConfig, market_data: Dict[str, pd.DataFrame]):
        self.config = config
        self.market_data = market_data
        
        self.capital = config.initial_capital
        self.positions = {}  # symbol -> position_data
        self.peak_prices = {}
        self.trades = []
        self.equity_curve = []
        
        # Precomputar swings
        self.detectors = {}
        print("🔍 Precomputando swing points...")
        for symbol, data in market_data.items():
            detector = SwingDetectorV3(
                data,
                volume_filter=config.use_volume_filter,
                use_ml=config.use_ml_validation,
                ml_threshold=config.ml_threshold
            )
            detector.detect()
            self.detectors[symbol] = detector
            
            highs = detector.int_highs.notna().sum()
            lows = detector.int_lows.notna().sum()
            print(f"   {symbol}: {highs} highs, {lows} lows")
    
    def run(self, start_date=None, end_date=None):
        """Ejecuta backtest en rango de fechas."""
        
        # Obtener rango común
        all_dates = []
        for data in self.market_data.values():
            all_dates.extend(data.index.tolist())
        
        all_dates = sorted(set(all_dates))
        
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        
        if not all_dates:
            return
        
        print(f"\n⏳ Ejecutando desde {all_dates[0]} hasta {all_dates[-1]}...")
        print(f"   Total días: {len(all_dates)}")
        
        for i, current_date in enumerate(all_dates):
            # Obtener precios actuales
            current_prices = {}
            for symbol, data in self.market_data.items():
                if current_date in data.index:
                    current_prices[symbol] = float(data.loc[current_date, 'Close'])
            
            if not current_prices:
                continue
            
            # Actualizar equity
            current_equity = self.capital
            for symbol, pos in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    
                    if pos['type'] == 'LONG':
                        unrealized_pnl = (current_price - pos['entry_price']) * self.config.leverage
                    else:
                        unrealized_pnl = (pos['entry_price'] - current_price) * self.config.leverage
                    
                    unrealized_pnl_dollars = (unrealized_pnl / pos['entry_price']) * pos['capital']
                    current_equity += unrealized_pnl_dollars
            
            self.equity_curve.append({
                'Date': current_date,
                'Equity': current_equity,
                'Positions': len(self.positions)
            })
            
            # Verificar stops en posiciones abiertas
            for symbol in list(self.positions.keys()):
                if symbol not in current_prices:
                    continue
                
                pos = self.positions[symbol]
                current_price = current_prices[symbol]
                
                # Obtener régimen actual
                data_up_to_date = self.market_data[symbol].loc[:current_date]
                regime = RegimeDetector.detect(data_up_to_date, self.config.regime_lookback)
                regime_params = RegimeDetector.get_adapted_params(
                    regime,
                    self.config.base_stop_loss,
                    self.config.base_take_profit,
                    self.config.base_trailing_stop
                )
                
                should_close, reason = self._check_stops(symbol, current_price, regime_params)
                
                if should_close:
                    self._close_position(symbol, current_price, current_date, reason)
            
            # Buscar nuevas señales
            if len(self.positions) < self.config.max_positions:
                self._look_for_signals(current_date, current_prices)
            
            if i % 500 == 0 and i > 0:
                progress = (i / len(all_dates)) * 100
                print(f"   {progress:.1f}% - Equity: ${current_equity:.2f}, Posiciones: {len(self.positions)}")
        
        # Cerrar posiciones finales
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                self._close_position(symbol, current_prices[symbol], all_dates[-1], "Fin backtest")
        
        print("   ✓ Backtest completado")
        return self.calculate_metrics()
    
    def _check_stops(self, symbol: str, current_price: float, 
                    regime_params: Dict[str, float]) -> Tuple[bool, str]:
        pos = self.positions[symbol]
        
        if pos['type'] == 'LONG':
            pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100 * self.config.leverage
        else:
            pnl_pct = ((pos['entry_price'] - current_price) / pos['entry_price']) * 100 * self.config.leverage
        
        # Stop Loss
        if pnl_pct <= -regime_params['stop_loss']:
            return True, f"Stop Loss: {pnl_pct:.2f}%"
        
        # Take Profit
        if pnl_pct >= regime_params['take_profit']:
            return True, f"Take Profit: {pnl_pct:.2f}%"
        
        # Trailing Stop
        if pnl_pct >= self.config.min_profit_trailing:
            peak_key = f"{symbol}_peak"
            if peak_key not in self.peak_prices:
                self.peak_prices[peak_key] = current_price
            
            if pos['type'] == 'LONG' and current_price > self.peak_prices[peak_key]:
                self.peak_prices[peak_key] = current_price
            elif pos['type'] == 'SHORT' and current_price < self.peak_prices[peak_key]:
                self.peak_prices[peak_key] = current_price
            
            peak = self.peak_prices[peak_key]
            if pos['type'] == 'LONG':
                peak_pnl = ((peak - pos['entry_price']) / pos['entry_price']) * 100 * self.config.leverage
            else:
                peak_pnl = ((pos['entry_price'] - peak) / pos['entry_price']) * 100 * self.config.leverage
            
            drawdown = peak_pnl - pnl_pct
            
            if drawdown >= regime_params['trailing_stop']:
                return True, f"Trailing: peak {peak_pnl:.2f}%, actual {pnl_pct:.2f}%"
        
        return False, ""
    
    def _close_position(self, symbol: str, exit_price: float, exit_date, reason: str):
        pos = self.positions[symbol]
        
        if pos['type'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * self.config.leverage
        else:
            pnl = (pos['entry_price'] - exit_price) * self.config.leverage
        
        pnl_pct = (pnl / pos['entry_price']) * 100
        pnl_dollars = (pnl / pos['entry_price']) * pos['capital']
        
        self.capital += pnl_dollars
        
        self.trades.append({
            'Symbol': symbol,
            'Entry_Date': pos['entry_date'],
            'Exit_Date': exit_date,
            'Type': pos['type'],
            'Entry_Price': pos['entry_price'],
            'Exit_Price': exit_price,
            'PnL_Pct': pnl_pct,
            'PnL_Dollars': pnl_dollars,
            'Capital': self.capital,
            'Exit_Reason': reason,
            'ML_Confidence': pos.get('ml_confidence', 1.0),
            'Regime': pos.get('regime', 'UNKNOWN')
        })
        
        del self.positions[symbol]
        peak_key = f"{symbol}_peak"
        if peak_key in self.peak_prices:
            del self.peak_prices[peak_key]
    
    def _look_for_signals(self, current_date, current_prices: Dict[str, float]):
        # Calcular correlación actual
        data_up_to_date = {}
        for symbol, data in self.market_data.items():
            data_up_to_date[symbol] = data.loc[:current_date]
        
        corr_matrix = CorrelationManager.calculate_correlation_matrix(
            data_up_to_date, lookback=30
        )
        
        open_symbols = list(self.positions.keys())
        
        signals = []
        
        for symbol, data in data_up_to_date.items():
            if symbol in open_symbols or symbol not in current_prices:
                continue
            
            if len(data) < 50:
                continue
            
            # Obtener señal
            signal, signal_price, confidence = self.detectors[symbol].get_signal_at_date(current_date)
            
            if signal:
                # Verificar correlación
                can_open, max_corr = CorrelationManager.check_position_correlation(
                    open_symbols, symbol, corr_matrix, self.config.max_correlation
                )
                
                if can_open:
                    regime = RegimeDetector.detect(data, self.config.regime_lookback)
                    
                    signals.append({
                        'symbol': symbol,
                        'signal': signal,
                        'confidence': confidence,
                        'regime': regime
                    })
        
        # Ordenar por confianza y abrir hasta max_positions
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        for sig in signals[:self.config.max_positions - len(self.positions)]:
            self._open_position(
                sig['symbol'],
                sig['signal'],
                current_prices[sig['symbol']],
                current_date,
                sig['confidence'],
                sig['regime']
            )
    
    def _open_position(self, symbol: str, signal: str, entry_price: float,
                      entry_date, ml_confidence: float, regime: str):
        allocation = 1.0 / self.config.max_positions
        capital = self.capital * allocation
        
        self.positions[symbol] = {
            'type': 'LONG' if signal == 'BUY' else 'SHORT',
            'entry_price': entry_price,
            'entry_date': entry_date,
            'capital': capital,
            'ml_confidence': ml_confidence,
            'regime': regime
        }
    
    def calculate_metrics(self):
        if len(self.trades) == 0:
            return {
                'Total_Trades': 0,
                'Trades_DF': pd.DataFrame(),
                'Equity_DF': pd.DataFrame()
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        winning = trades_df[trades_df['PnL_Dollars'] > 0]
        losing = trades_df[trades_df['PnL_Dollars'] <= 0]
        
        total_return = self.capital - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100
        
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
        max_dd = equity_df['Drawdown'].min() * 100
        
        returns = trades_df['PnL_Pct'].values
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252/24) if len(returns) > 1 and returns.std() > 0 else 0
        
        gross_profit = winning['PnL_Dollars'].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing['PnL_Dollars'].sum()) if len(losing) > 0 else 0.001
        
        return {
            'Total_Trades': len(trades_df),
            'Winning_Trades': len(winning),
            'Losing_Trades': len(losing),
            'Win_Rate': (len(winning) / len(trades_df) * 100) if len(trades_df) > 0 else 0,
            'Total_Return': total_return,
            'Total_Return_Pct': total_return_pct,
            'Max_Drawdown': max_dd,
            'Sharpe_Ratio': sharpe,
            'Profit_Factor': gross_profit / gross_loss,
            'Avg_Win': winning['PnL_Dollars'].mean() if len(winning) > 0 else 0,
            'Avg_Loss': losing['PnL_Dollars'].mean() if len(losing) > 0 else 0,
            'Trades_DF': trades_df,
            'Equity_DF': equity_df
        }

# ═══════════════════════════════════════════════════════════════════════════
#                        WALK-FORWARD OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardOptimizer:
    def __init__(self, config: BacktestConfig, market_data: Dict[str, pd.DataFrame]):
        self.config = config
        self.market_data = market_data
    
    def run(self):
        """Ejecuta walk-forward optimization."""
        
        print("\n" + "="*70)
        print("WALK-FORWARD OPTIMIZATION")
        print("="*70)
        print(f"Train: {self.config.train_period_days} días")
        print(f"Test: {self.config.test_period_days} días")
        print(f"Step: {self.config.step_days} días")
        print("="*70)
        
        # Obtener rango total
        all_dates = []
        for data in self.market_data.values():
            all_dates.extend(data.index.tolist())
        
        all_dates = sorted(set(all_dates))
        start_date = all_dates[0]
        end_date = all_dates[-1]
        
        print(f"Rango: {start_date} - {end_date}")
        
        # Generar ventanas
        windows = []
        current_start = start_date
        
        while current_start < end_date:
            train_end = current_start + timedelta(days=self.config.train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_period_days)
            
            if test_end > end_date:
                break
            
            windows.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_start = current_start + timedelta(days=self.config.step_days)
        
        print(f"\nTotal ventanas: {len(windows)}")
        
        # Ejecutar cada ventana
        all_results = []
        
        for i, window in enumerate(windows):
            print(f"\n{'─'*70}")
            print(f"VENTANA {i+1}/{len(windows)}")
            print(f"Train: {window['train_start'].date()} - {window['train_end'].date()}")
            print(f"Test: {window['test_start'].date()} - {window['test_end'].date()}")
            print(f"{'─'*70}")
            
            # Crear nuevo backtester para test
            bt = BacktesterV3(self.config, self.market_data)
            
            # Ejecutar solo en período de test
            metrics = bt.run(start_date=window['test_start'], end_date=window['test_end'])
            
            if metrics and metrics['Total_Trades'] > 0:
                all_results.append({
                    'Window': i+1,
                    'Test_Start': window['test_start'],
                    'Test_End': window['test_end'],
                    'Trades': metrics['Total_Trades'],
                    'Win_Rate': metrics['Win_Rate'],
                    'Return_Pct': metrics['Total_Return_Pct'],
                    'Max_DD': metrics['Max_Drawdown'],
                    'Sharpe': metrics['Sharpe_Ratio'],
                    'Profit_Factor': metrics['Profit_Factor'],
                    'Trades_DF': metrics['Trades_DF']
                })
                
                print(f"\n📊 Resultados ventana {i+1}:")
                print(f"   Trades: {metrics['Total_Trades']}")
                print(f"   Win Rate: {metrics['Win_Rate']:.2f}%")
                print(f"   Return: {metrics['Total_Return_Pct']:.2f}%")
                print(f"   Max DD: {metrics['Max_Drawdown']:.2f}%")
                print(f"   Sharpe: {metrics['Sharpe_Ratio']:.2f}")
        
        return self._aggregate_results(all_results)
    
    def _aggregate_results(self, results: List[Dict]):
        """Agrega resultados de todas las ventanas."""
        
        if not results:
            return None
        
        print("\n" + "="*70)
        print("RESULTADOS AGREGADOS WALK-FORWARD")
        print("="*70)
        
        # Métricas por ventana
        windows_df = pd.DataFrame([{
            'Window': r['Window'],
            'Start': r['Test_Start'].date(),
            'End': r['Test_End'].date(),
            'Trades': r['Trades'],
            'Win_Rate': r['Win_Rate'],
            'Return_%': r['Return_Pct'],
            'Max_DD_%': r['Max_DD'],
            'Sharpe': r['Sharpe'],
            'PF': r['Profit_Factor']
        } for r in results])
        
        print("\nResultados por ventana:")
        print(windows_df.to_string(index=False))
        
        # Agregar todos los trades
        all_trades = pd.concat([r['Trades_DF'] for r in results], ignore_index=True)
        
        # Métricas agregadas
        total_trades = len(all_trades)
        winning = all_trades[all_trades['PnL_Dollars'] > 0]
        losing = all_trades[all_trades['PnL_Dollars'] <= 0]
        
        avg_return = windows_df['Return_%'].mean()
        std_return = windows_df['Return_%'].std()
        avg_win_rate = windows_df['Win_Rate'].mean()
        avg_sharpe = windows_df['Sharpe'].mean()
        
        gross_profit = winning['PnL_Dollars'].sum()
        gross_loss = abs(losing['PnL_Dollars'].sum())
        
        print(f"\n{'='*70}")
        print("MÉTRICAS AGREGADAS")
        print(f"{'='*70}")
        print(f"Total ventanas:         {len(results)}")
        print(f"Total trades:           {total_trades}")
        print(f"Win rate promedio:      {avg_win_rate:.2f}%")
        print(f"Return promedio:        {avg_return:.2f}%")
        print(f"Desv. estándar return:  {std_return:.2f}%")
        print(f"Sharpe promedio:        {avg_sharpe:.2f}")
        print(f"Profit Factor total:    {(gross_profit/gross_loss):.2f}")
        print(f"Ventanas positivas:     {len(windows_df[windows_df['Return_%'] > 0])}/{len(windows_df)}")
        print(f"{'='*70}")
        
        return {
            'windows_df': windows_df,
            'all_trades': all_trades,
            'avg_metrics': {
                'Avg_Win_Rate': avg_win_rate,
                'Avg_Return': avg_return,
                'Std_Return': std_return,
                'Avg_Sharpe': avg_sharpe,
                'Total_Trades': total_trades,
                'Profit_Factor': gross_profit / gross_loss if gross_loss > 0 else 0
            }
        }

# ═══════════════════════════════════════════════════════════════════════════
#                        VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def plot_walkforward_results(wf_results):
    """Grafica resultados de walk-forward."""
    
    windows_df = wf_results['windows_df']
    all_trades = wf_results['all_trades']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Returns por ventana
    axes[0,0].bar(windows_df['Window'], windows_df['Return_%'], 
                  color=['g' if x > 0 else 'r' for x in windows_df['Return_%']])
    axes[0,0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0,0].set_title('Return % por Ventana', fontweight='bold')
    axes[0,0].set_xlabel('Ventana')
    axes[0,0].set_ylabel('Return %')
    axes[0,0].grid(alpha=0.3)
    
    # Win Rate por ventana
    axes[0,1].plot(windows_df['Window'], windows_df['Win_Rate'], 'o-', linewidth=2)
    axes[0,1].axhline(y=50, color='r', linestyle='--', alpha=0.5)
    axes[0,1].set_title('Win Rate por Ventana', fontweight='bold')
    axes[0,1].set_xlabel('Ventana')
    axes[0,1].set_ylabel('Win Rate %')
    axes[0,1].grid(alpha=0.3)
    
    # Sharpe por ventana
    axes[1,0].plot(windows_df['Window'], windows_df['Sharpe'], 's-', linewidth=2, color='purple')
    axes[1,0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1,0].set_title('Sharpe Ratio por Ventana', fontweight='bold')
    axes[1,0].set_xlabel('Ventana')
    axes[1,0].set_ylabel('Sharpe')
    axes[1,0].grid(alpha=0.3)
    
    # Drawdown por ventana
    axes[1,1].bar(windows_df['Window'], windows_df['Max_DD_%'], color='red', alpha=0.6)
    axes[1,1].set_title('Max Drawdown por Ventana', fontweight='bold')
    axes[1,1].set_xlabel('Ventana')
    axes[1,1].set_ylabel('Drawdown %')
    axes[1,1].grid(alpha=0.3)
    
    # Distribución PnL
    wins = all_trades[all_trades['PnL_Dollars'] > 0]['PnL_Dollars']
    losses = all_trades[all_trades['PnL_Dollars'] <= 0]['PnL_Dollars']
    axes[2,0].hist([wins, losses], bins=30, label=['Wins', 'Losses'], 
                   color=['g', 'r'], alpha=0.7)
    axes[2,0].set_title('Distribución P&L', fontweight='bold')
    axes[2,0].set_xlabel('P&L $')
    axes[2,0].legend()
    axes[2,0].grid(alpha=0.3)
    
    # Exit reasons
    exit_counts = all_trades['Exit_Reason'].value_counts()
    axes[2,1].bar(range(len(exit_counts)), exit_counts.values, color='steelblue')
    axes[2,1].set_xticks(range(len(exit_counts)))
    axes[2,1].set_xticklabels(exit_counts.index, rotation=45, ha='right')
    axes[2,1].set_title('Exit Reasons', fontweight='bold')
    axes[2,1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('walkforward_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráficos guardados: walkforward_results.png")
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════
#                        MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("BACKTESTING V3 - MULTI-ASSET + ML + WALK-FORWARD")
    print("="*70)
    
    # Configuración
    symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD']
    
    config = BacktestConfig(
        symbols=symbols,
        initial_capital=1000.0,
        max_positions=3,
        max_correlation=0.7,
        leverage=3,
        base_stop_loss=4.0,
        base_take_profit=8.0,
        base_trailing_stop=2.5,
        use_volume_filter=True,
        use_ml_validation=True,
        ml_threshold=0.6,
        regime_lookback=30,
        train_period_days=120,
        test_period_days=30,
        step_days=15
    )
    
    # Descargar datos
    print("\n📊 Descargando datos históricos...")
    market_data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", interval="1h")
            
            if not data.empty:
                market_data[symbol] = data
                print(f"   ✓ {symbol}: {len(data)} velas")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
    
    if not market_data:
        print("❌ No se pudieron descargar datos")
        return
    
    # Ejecutar walk-forward
    optimizer = WalkForwardOptimizer(config, market_data)
    wf_results = optimizer.run()
    
    if wf_results:
        plot_walkforward_results(wf_results)
        
        # Guardar resultados
        wf_results['windows_df'].to_csv('walkforward_windows.csv', index=False)
        wf_results['all_trades'].to_csv('walkforward_trades.csv', index=False)
        print("\n✓ Datos guardados: walkforward_windows.csv, walkforward_trades.csv")


if __name__ == "__main__":
    main()
