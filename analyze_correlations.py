"""
CORRELATION ANALYZER - Herramienta de análisis de correlaciones
Útil para decidir qué assets incluir en tu portfolio
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def download_data(symbols, period='90d', interval='1h'):
    """Descarga datos para múltiples símbolos."""
    print(f"📊 Descargando datos ({period}, {interval})...")
    
    data_dict = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                data_dict[symbol] = data
                print(f"   ✓ {symbol}: {len(data)} velas")
            else:
                print(f"   ⚠️  {symbol}: sin datos")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
    
    return data_dict

def calculate_correlations(data_dict, lookback_days=30):
    """Calcula matriz de correlación."""
    print(f"\n🔗 Calculando correlaciones (últimos {lookback_days} días)...")
    
    # Preparar returns
    returns_dict = {}
    for symbol, data in data_dict.items():
        if len(data) >= lookback_days * 24:  # 24 velas por día (1h)
            returns = data['Close'].tail(lookback_days * 24).pct_change().dropna()
            returns_dict[symbol] = returns
    
    if len(returns_dict) < 2:
        print("❌ Necesitas al menos 2 símbolos con datos")
        return None
    
    # Crear DataFrame y calcular correlación
    returns_df = pd.DataFrame(returns_dict)
    corr_matrix = returns_df.corr()
    
    return corr_matrix

def analyze_regime(data_dict):
    """Analiza el régimen actual de cada asset."""
    print("\n📈 Análisis de régimen actual:")
    
    regimes = {}
    
    for symbol, data in data_dict.items():
        if len(data) < 30 * 24:
            continue
        
        recent = data.tail(30 * 24)
        returns = recent['Close'].pct_change().dropna()
        
        # Volatilidad
        volatility = returns.std()
        avg_volatility = data['Close'].pct_change().dropna().std()
        
        # Tendencia
        high_low = (recent['High'] - recent['Low']).mean()
        close_change = abs(recent['Close'].iloc[-1] - recent['Close'].iloc[0])
        trend_strength = close_change / (high_low * len(recent)) if high_low > 0 else 0
        
        # Clasificación
        if volatility > avg_volatility * 1.5:
            regime = 'VOLATILE'
        elif trend_strength > 0.5:
            regime = 'TRENDING'
        else:
            regime = 'RANGING'
        
        regimes[symbol] = regime
        
        print(f"   {symbol:10s} → {regime:10s} (vol: {volatility:.4f}, trend: {trend_strength:.2f})")
    
    return regimes

def find_best_combinations(corr_matrix, max_corr=0.7, max_assets=3):
    """Encuentra mejores combinaciones de assets con baja correlación."""
    print(f"\n🎯 Mejores combinaciones (max {max_assets} assets, corr < {max_corr}):")
    
    symbols = corr_matrix.columns.tolist()
    n = len(symbols)
    
    if n < 2:
        return
    
    valid_combos = []
    
    # Generar todas las combinaciones posibles
    from itertools import combinations
    
    for r in range(2, min(max_assets + 1, n + 1)):
        for combo in combinations(symbols, r):
            # Verificar correlaciones dentro del combo
            max_corr_in_combo = 0.0
            
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    corr_val = abs(corr_matrix.loc[combo[i], combo[j]])
                    max_corr_in_combo = max(max_corr_in_combo, corr_val)
            
            if max_corr_in_combo < max_corr:
                valid_combos.append({
                    'symbols': combo,
                    'max_corr': max_corr_in_combo,
                    'size': len(combo)
                })
    
    # Ordenar por tamaño (más assets mejor) y luego por menor correlación
    valid_combos.sort(key=lambda x: (-x['size'], x['max_corr']))
    
    if not valid_combos:
        print(f"   ⚠️  No hay combinaciones con correlación < {max_corr}")
        print(f"   Sugerencia: Aumenta max_corr o elige assets menos correlacionados")
        return
    
    # Mostrar top 10
    for i, combo in enumerate(valid_combos[:10], 1):
        symbols_str = ', '.join(combo['symbols'])
        print(f"   {i}. {symbols_str:40s} (max corr: {combo['max_corr']:.3f})")

def plot_correlation_matrix(corr_matrix):
    """Visualiza matriz de correlación."""
    plt.figure(figsize=(10, 8))
    
    # Crear heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Matriz de Correlación (30 días)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: correlation_matrix.png")
    plt.show()

def plot_returns_comparison(data_dict, lookback_days=30):
    """Compara returns de diferentes assets."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Normalize prices
    normalized = {}
    for symbol, data in data_dict.items():
        if len(data) >= lookback_days * 24:
            prices = data['Close'].tail(lookback_days * 24)
            normalized[symbol] = (prices / prices.iloc[0] - 1) * 100
    
    # Plot 1: Returns normalizados
    for symbol, returns in normalized.items():
        axes[0].plot(returns.index, returns.values, label=symbol, linewidth=2)
    
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[0].set_title('Returns Normalizados (últimos 30 días)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Return %')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Volatilidad rolling
    for symbol, data in data_dict.items():
        if len(data) >= lookback_days * 24:
            returns = data['Close'].tail(lookback_days * 24).pct_change()
            rolling_vol = returns.rolling(window=24).std() * np.sqrt(24)  # Diaria
            axes[1].plot(rolling_vol.index, rolling_vol.values * 100, label=symbol, linewidth=2)
    
    axes[1].set_title('Volatilidad Rolling (24h)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volatilidad %')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('returns_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Gráfico guardado: returns_comparison.png")
    plt.show()

def main():
    print("\n" + "="*70)
    print("CORRELATION ANALYZER - Análisis de Portfolio")
    print("="*70)
    
    # Define tus assets de interés
    symbols = [
        'BTC-USD',
        'ETH-USD',
        'ADA-USD',
        'SOL-USD',
        'MATIC-USD',
        'AVAX-USD',
        'LINK-USD',
        'DOT-USD'
    ]
    
    # Parámetros
    period = '90d'      # Período de datos
    interval = '1h'     # Intervalo
    lookback_days = 30  # Días para correlación
    max_corr = 0.7      # Correlación máxima aceptable
    max_assets = 4      # Máximo assets en portfolio
    
    # 1. Descargar datos
    data_dict = download_data(symbols, period=period, interval=interval)
    
    if not data_dict:
        print("❌ No se pudieron descargar datos")
        return
    
    # 2. Calcular correlaciones
    corr_matrix = calculate_correlations(data_dict, lookback_days=lookback_days)
    
    if corr_matrix is None:
        return
    
    print("\nMatriz de Correlación:")
    print(corr_matrix.round(3))
    
    # 3. Análisis de régimen
    regimes = analyze_regime(data_dict)
    
    # 4. Encontrar mejores combinaciones
    find_best_combinations(corr_matrix, max_corr=max_corr, max_assets=max_assets)
    
    # 5. Estadísticas adicionales
    print("\n📊 Estadísticas de Correlación:")
    print(f"   Correlación media:    {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    print(f"   Correlación mediana:  {np.median(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]):.3f}")
    print(f"   Correlación máxima:   {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")
    print(f"   Correlación mínima:   {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.3f}")
    
    # 6. Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    
    # Encontrar pares muy correlacionados
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        print("\n   ⚠️  Pares muy correlacionados (>0.8) - Evita tener ambos:")
        for s1, s2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"      {s1} ↔ {s2}: {corr:.3f}")
    
    # Encontrar assets con baja correlación promedio
    avg_corr_per_asset = {}
    for symbol in corr_matrix.columns:
        other_corrs = [abs(corr_matrix.loc[symbol, other]) 
                      for other in corr_matrix.columns if other != symbol]
        avg_corr_per_asset[symbol] = np.mean(other_corrs)
    
    print("\n   ✅ Assets con menor correlación promedio (buenos para diversificar):")
    sorted_assets = sorted(avg_corr_per_asset.items(), key=lambda x: x[1])
    for symbol, avg_corr in sorted_assets[:5]:
        print(f"      {symbol}: {avg_corr:.3f}")
    
    # 7. Visualizaciones
    plot_correlation_matrix(corr_matrix)
    plot_returns_comparison(data_dict, lookback_days=lookback_days)
    
    print("\n" + "="*70)
    print("✅ Análisis completado")
    print("="*70)

if __name__ == "__main__":
    main()
