import asyncio
import pandas as pd
import numpy as np
import yfinance as yfinance
import logging
from datetime import datetime, timedelta
from backend.services.quotex_websocket import QuotexWSClient, qx_candles_to_dataframe

logger = logging.getLogger("DataFetcher")

# Map standard OTC assets to Yahoo Finance tickers
YF_TICKERS = {
    "EUR/USD (OTC)": "EURUSD=X",
    "GBP/USD (OTC)": "GBPUSD=X",
    "USD/JPY (OTC)": "USDJPY=X",
    "USD/BRL (OTC)": "USDBRL=X"
}

# Base prices for assets in case we generate synthetic data
BASE_PRICES = {
    "EUR/USD (OTC)": 1.0850,
    "GBP/USD (OTC)": 1.2720,
    "USD/JPY (OTC)": 156.80,
    "USD/BRL (OTC)": 5.250
}

async def fetch_historical_data(asset: str, timeframe_minutes: int, limit: int = 150, ssid: str = None) -> pd.DataFrame:
    """
    Fetches historical candlestick data for the specified asset.
    First tries Quotex WS if ssid is provided.
    Second tries Yahoo Finance.
    Third falls back to a high-fidelity synthetic OTC candle generator.
    """
    timeframe_seconds = timeframe_minutes * 60
    
    # 1. Try Quotex WS
    if ssid:
        try:
            ws_client = QuotexWSClient(ssid=ssid)
            connected = await ws_client.connect()
            if connected:
                candles_list = await ws_client.load_candles(asset, timeframe_seconds, count=limit)
                await ws_client.disconnect()
                if candles_list:
                    df = qx_candles_to_dataframe(candles_list)
                    if not df.empty and len(df) >= 30:
                        logger.info(f"Successfully fetched {len(df)} candles from Quotex WS for {asset}")
                        return df
            else:
                await ws_client.disconnect()
        except Exception as e:
            logger.error(f"Failed to fetch data from Quotex WS: {e}")

    # 2. Try Yahoo Finance (only if not weekend or if standard market data is active)
    yf_ticker = YF_TICKERS.get(asset)
    if yf_ticker:
        try:
            # Map timeframe to Yahoo Finance interval
            interval = f"{timeframe_minutes}m"
            if timeframe_minutes == 1:
                period = "1d"
            elif timeframe_minutes == 5:
                period = "5d"
            else: # 15m
                period = "5d"
                
            # Perform download inside executor to prevent blocking the async loop
            loop = asyncio.get_event_loop()
            df_yf = await loop.run_in_executor(
                None, 
                lambda: yfinance.download(yf_ticker, period=period, interval=interval, progress=False)
            )
            
            if not df_yf.empty and len(df_yf) >= 30:
                # Format to standard OHLCV
                df_yf = df_yf.tail(limit)
                
                # Strip timezone if present
                idx = df_yf.index
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                
                df = pd.DataFrame(index=idx)
                df['Open'] = df_yf['Open'].values.astype(float)
                df['High'] = df_yf['High'].values.astype(float)
                df['Low'] = df_yf['Low'].values.astype(float)
                df['Close'] = df_yf['Close'].values.astype(float)
                df['Volume'] = df_yf['Volume'].values.astype(float)
                
                # yfinance volume is sometimes 0 for forex, fill with realistic random values if 0
                if df['Volume'].sum() == 0:
                    df['Volume'] = np.random.randint(50, 500, size=len(df))
                    
                logger.info(f"Successfully fetched {len(df)} candles from Yahoo Finance for {asset}")
                return df
                
        except Exception as e:
            logger.error(f"Yahoo Finance download failed for {asset}: {e}")

    # 3. Fallback: High-Fidelity Synthetic OTC Candlestick Generator
    logger.info(f"Falling back to high-fidelity synthetic OTC generator for {asset} ({timeframe_minutes}m)")
    return generate_synthetic_candles(asset, timeframe_minutes, limit)

def generate_synthetic_candles(asset: str, timeframe_minutes: int, limit: int = 150) -> pd.DataFrame:
    """
    Generates a realistic candlestick series for the OTC market.
    Uses a Geometric Brownian Motion model + sine waves for cycles + mean reversion 
    around support/resistance levels to generate smooth patterns suitable for technical indicators.
    """
    np.random.seed(int(datetime.now().timestamp()) % 100000)
    base_price = BASE_PRICES.get(asset, 1.0)
    
    # Establish dynamic volatility based on the asset
    volatility = 0.00015
    if "JPY" in asset:
        volatility = 0.015
    elif "INR" in asset or "BRL" in asset:
        volatility = 0.005

    # Generate timestamps ending at current time
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i * timeframe_minutes) for i in range(limit)]
    timestamps.reverse()

    prices = []
    current_price = base_price
    
    # Generate trends using a composite sine wave
    cycle_period_1 = 60 # 60 candles cycle
    cycle_period_2 = 15 # 15 candles microcycle
    
    # Support & Resistance levels to trigger bounces
    support = base_price * 0.995
    resistance = base_price * 1.005

    for idx in range(limit):
        # Calculate trend component
        trend = 0.08 * np.sin(2 * np.pi * idx / cycle_period_1) + 0.03 * np.cos(2 * np.pi * idx / cycle_period_2)
        trend_drift = trend * current_price * 0.0001
        
        # Mean reversion if price goes too close to support/resistance boundaries
        reversion_drift = 0.0
        if current_price > resistance:
            reversion_drift = -volatility * 0.5 * (current_price - resistance)
        elif current_price < support:
            reversion_drift = volatility * 0.5 * (support - current_price)

        # Geometric Brownian Motion shock
        shock = np.random.normal(0, volatility) * current_price
        
        # New Close price
        current_price = current_price + trend_drift + reversion_drift + shock
        prices.append(current_price)

    # Convert to DataFrame
    df = pd.DataFrame(index=timestamps)
    df.index.name = 'time'
    
    # Construct High, Low, Open, Close
    closes = np.array(prices)
    opens = np.zeros(limit)
    highs = np.zeros(limit)
    lows = np.zeros(limit)
    volumes = np.zeros(limit)

    opens[0] = base_price
    for i in range(1, limit):
        opens[i] = closes[i-1]

    for i in range(limit):
        # Candle range
        move = abs(closes[i] - opens[i])
        noise = move * np.random.uniform(0.1, 0.5) + (closes[i] * volatility * 0.2)
        
        # High and Low
        highs[i] = max(opens[i], closes[i]) + noise
        lows[i] = min(opens[i], closes[i]) - noise
        
        # Volume
        # Occasionally make volume very low (to simulate Doji candidates)
        if np.random.rand() < 0.08:
            volumes[i] = np.random.randint(5, 20) # Low volume
            # Close price is almost equal to open
            closes[i] = opens[i] + np.random.normal(0, closes[i] * 0.00001)
            highs[i] = max(opens[i], closes[i]) + (closes[i] * volatility * 0.05)
            lows[i] = min(opens[i], closes[i]) - (closes[i] * volatility * 0.05)
        else:
            volumes[i] = np.random.randint(100, 1000)

    df['Open'] = opens
    df['High'] = highs
    df['Low'] = lows
    df['Close'] = closes
    df['Volume'] = volumes

    return df
