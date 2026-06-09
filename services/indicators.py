import pandas as pd
import numpy as np

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates the Exponential Moving Average (EMA) for a given series and period."""
    if len(series) < period:
        return pd.Series(index=series.index, data=np.nan)
    return series.ewm(span=period, adjust=False).mean()

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Calculates the Stochastic Oscillator (%K and %D lines).
    Returns (slow_k, slow_d).
    """
    if len(close) < k_period:
        nan_series = pd.Series(index=close.index, data=np.nan)
        return nan_series, nan_series

    # Lowest low and highest high in the last k_period candles
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()

    # Fast %K
    fast_k = 100 * ((close - low_min) / (high_max - low_min))
    # Fill division by zero cases
    fast_k = fast_k.fillna(50)

    # Slow %K (smoothed fast %K)
    slow_k = fast_k.rolling(window=smooth_k).mean()
    # Slow %D (smoothed slow %K)
    slow_d = slow_k.rolling(window=d_period).mean()

    return slow_k, slow_d

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) for a given series and period."""
    if len(close) < period:
        return pd.Series(index=close.index, data=50.0)

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()

    # Calculate exponential moving averages of gains and losses
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Average Directional Index (ADX)."""
    if len(close) < period * 2:
        return pd.Series(index=close.index, data=25.0)

    # True Range (TR)
    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

    # Directional Movement (+DM, -DM)
    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    # Smoothed TR and DM
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    # Directional Index (DX) and Average Directional Index (ADX)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.fillna(0)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx.fillna(25)

def calculate_fractals(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Calculates Williams Fractals.
    Returns (fractal_up, fractal_down) where values are either the high/low of the fractal candle or NaN.
    """
    fractal_up = pd.Series(index=high.index, data=np.nan)
    fractal_down = pd.Series(index=low.index, data=np.nan)

    # A fractal requires 2 candles before and 2 candles after (total 5 candles window)
    for i in range(2, len(high) - 2):
        # Upper fractal (Resistance point): high of center candle is greater than highs of surrounding 4 candles
        if (high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and
            high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]):
            fractal_up.iloc[i] = high.iloc[i]

        # Lower fractal (Support point): low of center candle is lower than lows of surrounding 4 candles
        if (low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and
            low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]):
            fractal_down.iloc[i] = low.iloc[i]

    return fractal_up, fractal_down

def is_doji(open_s: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, threshold: float = 0.1) -> pd.Series:
    """
    Checks if candles are Dojis (where body size is very small relative to the candle range).
    Returns a boolean Series.
    """
    body = (close - open_s).abs()
    rng = high - low
    # Avoid division by zero
    rng = rng.replace(0, 0.00001)
    
    # A doji has a body size smaller than the threshold * total high-low range
    return (body / rng) < threshold

def check_volume_filter(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Checks if the volume of the candle is above the simple moving average of volume.
    To prevent trading during dead market hours.
    """
    if len(volume) < period:
        return pd.Series(index=volume.index, data=True)
    vol_sma = volume.rolling(window=period).mean()
    return volume >= vol_sma
