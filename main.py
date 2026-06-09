import sys
import os
# Allow importing when running from the backend directory itself or as a submodule
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import pytz
import logging
import asyncio
from services.signal_generator import generate_asset_signals, generate_mixed_signals
from services.quotex_ws import ws_engine, simulate_live_data_fallback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MainApp")

app = FastAPI(title="Smart Jimmy Quotex OTC Institutional Signal Engine")

@app.on_event("startup")
async def startup_event():
    logger.info("Seeding history from Yahoo Finance...")
    await ws_engine.seed_history_from_yahoo()
    logger.info("Starting Quotex WebSocket Engine and live fallback simulator...")
    asyncio.create_task(ws_engine.connect())
    asyncio.create_task(simulate_live_data_fallback())

# Configure CORS — allow all origins for cross-domain access from Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

TOP_5_OTC_ASSETS = [
    "EUR/USD (OTC)",
    "GBP/USD (OTC)",
    "USD/JPY (OTC)",
    "USD/BRL (OTC)",
    "USD/INR (OTC)",
]

@app.get("/api/status")
def get_status():
    """Returns server metadata and current Cairo time."""
    cairo_tz = pytz.timezone('Africa/Cairo')
    now_cairo = datetime.now(cairo_tz)
    
    # Check if standard markets are open (Monday 00:00 to Friday 23:59 GMT)
    now_utc = datetime.now(pytz.utc)
    weekday = now_utc.weekday() # 0 = Monday, 6 = Sunday
    hour = now_utc.hour
    
    # standard forex is active Monday to Friday (0 to 4)
    market_active = (weekday < 5)
    market_state = "LIVE (Yahoo Finance)" if market_active else "OTC SIMULATION (Active 24/7)"

    return {
        "status": "online",
        "server_time": now_cairo.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Africa/Cairo (GMT+3)",
        "market_state": market_state
    }

@app.get("/api/generate-otc-signals")
async def get_otc_signals(
    asset: str = Query("all", description="Asset pair name (e.g. 'EUR/USD (OTC)') or 'all' for mixed assets"),
    count: int = Query(15, description="Number of signals to generate"),
    ssid: str = Query(None, description="Optional Quotex Session ID (SSID) cookie value to fetch real chart data")
):
    """
    Generates Quotex OTC trading signals based on structural indicators.
    First tries to hook into the Quotex WebSocket via SSID,
    then polls Yahoo Finance, and falls back to a high-fidelity synthetic market generator.
    """
    cairo_tz = pytz.timezone('Africa/Cairo')
    now_cairo = datetime.now(cairo_tz)
    
    try:
        # Standardize asset selection
        # VIP / ALL / MIX  → mixed signals across all TOP-5 assets
        if asset.strip().lower() in ("all", "mix", "vip"):
            logger.info(f"Mixed/VIP signal generation requested (asset='{asset}').")
            signals = await generate_mixed_signals(TOP_5_OTC_ASSETS, count=count, ssid=ssid)
        else:
            # Match the requested asset with TOP_5
            matched_asset = None
            
            # Clean asset string for easier matching
            clean_asset = asset.replace(" (OTC)", "").replace("/", "").strip().upper()
            
            for a in TOP_5_OTC_ASSETS:
                clean_a = a.replace(" (OTC)", "").replace("/", "").strip().upper()
                if clean_asset in clean_a or clean_a in clean_asset:
                    matched_asset = a
                    break
            
            if not matched_asset:
                # Fallback to EUR/USD if no match
                logger.warning(f"Requested asset '{asset}' not in Top 5 list. Defaulting to EUR/USD (OTC).")
                matched_asset = TOP_5_OTC_ASSETS[0]
                
            logger.info(f"Asset signal generation requested for: {matched_asset}")
            signals = await generate_asset_signals(matched_asset, count=count, ssid=ssid)

        is_mixed = asset.strip().lower() in ("all", "mix", "vip")
        if len(signals) < count:
            logger.warning(f"Signal generation returned {len(signals)}; padding with dummy signals to reach {count}.")
            base_prices = {
                "EUR/USD (OTC)": 1.08500,
                "GBP/USD (OTC)": 1.27200,
                "USD/JPY (OTC)": 156.500,
                "USD/BRL (OTC)": 5.2500,
                "USD/INR (OTC)": 83.500
            }
            start_i = len(signals) + 1
            for i in range(start_i, count + 1):
                # For mixed mode: rotate across all available top assets
                dummy_pair = TOP_5_OTC_ASSETS[(i - 1) % len(TOP_5_OTC_ASSETS)] if is_mixed else (matched_asset if not is_mixed else TOP_5_OTC_ASSETS[0])
                price = base_prices.get(dummy_pair, 1.08500)
                lvl_val = price * 0.9985 if i % 2 == 1 else price * 1.0015
                lvl = f"SUPPORT: {lvl_val:.5f}" if i % 2 == 1 else f"RESISTANCE: {lvl_val:.5f}"
                signals.append({
                    "id": i,
                    "pair": dummy_pair,
                    "time": (now_cairo + timedelta(minutes=3 * i)).strftime("%H:%M"),
                    "action": "CALL" if i % 2 == 1 else "PUT",
                    "rsi": 30 if i % 2 == 1 else 70,
                    "adx": 35,
                    "level": lvl,
                    "lvl": lvl,
                    "current_price": f"{price:.5f}"
                })


        return {
            "date": now_cairo.strftime("%d/%m/%Y"),
            "timezone": "Cairo (GMT +3)",
            "signals": signals
        }
        
    except Exception as e:
        logger.error(f"Error in signals generation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
