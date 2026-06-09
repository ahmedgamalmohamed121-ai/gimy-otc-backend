import asyncio
import json
import logging
import websockets
import pandas as pd
from datetime import datetime

logger = logging.getLogger("QuotexWSClient")
logging.basicConfig(level=logging.INFO)

class QuotexWSClient:
    def __init__(self, ssid: str, is_demo: bool = True):
        self.ssid = ssid
        self.is_demo = 1 if is_demo else 0
        self.ws_url = "wss://ws2.qxbroker.com/socket.io/?EIO=3&transport=websocket"
        self.websocket = None
        self.connected = False
        self.candles_cache = {}  # Format: { (asset, timeframe): [candles] }
        self.listener_task = None
        self.auth_success = False

    async def connect(self) -> bool:
        """Establishes connection and handles authentication handshake."""
        if not self.ssid:
            logger.error("No SSID provided. Cannot connect to Quotex WS.")
            return False

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Origin": "https://qxbroker.com",
            "Cookie": f"ssid={self.ssid}"
        }

        try:
            logger.info("Connecting to Quotex WebSocket...")
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Start background listener loop
            self.listener_task = asyncio.create_task(self._listen_loop())
            self.connected = True
            
            # Send Socket.io connection signal
            await self._send_raw("40")
            
            # Send Authorization payload
            auth_msg = ["authorization", {"session": self.ssid, "isDemo": self.is_demo}]
            await self._send_event(auth_msg)
            
            # Wait briefly for auth verification
            await asyncio.sleep(2.0)
            
            logger.info(f"Quotex WS Connected. Auth status: {self.auth_success}")
            return self.auth_success
            
        except Exception as e:
            logger.error(f"Error connecting to Quotex WS: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnects the WebSocket client."""
        self.connected = False
        if self.listener_task:
            self.listener_task.cancel()
        if self.websocket:
            await self.websocket.close()
            logger.info("Quotex WS Disconnected.")

    async def _send_raw(self, message: str):
        if self.websocket:
            await self.websocket.send(message)

    async def _send_event(self, event_data: list):
        """Sends a formatted Socket.IO event (type 42)."""
        payload = f"42{json.dumps(event_data)}"
        await self._send_raw(payload)

    async def load_candles(self, asset: str, timeframe: int, count: int = 100) -> list:
        """
        Sends a request to load historical candles.
        timeframe in seconds (e.g. 60, 300, 900)
        """
        if not self.connected or not self.auth_success:
            logger.warning("WS not connected/authorized. Cannot load candles.")
            return []

        # Convert standard names to Quotex OTC names if needed
        # e.g., EUR/USD (OTC) -> EURUSD_otc
        qx_asset = asset.replace("/", "").replace(" (OTC)", "").lower() + "_otc"
        if asset.endswith("_otc") or asset.endswith(" (OTC)"):
            pass
        else:
            # ensure standard OTC naming
            qx_asset = asset.replace("/", "").lower() + "_otc"

        # Request historical candles
        # Event format: 42["candles/load", {"asset": "EURUSD_otc", "timeframe": 60, "offset": 0, "size": 100}]
        # Let's request it
        event = ["candles/load", {"asset": qx_asset, "timeframe": timeframe, "offset": 0, "size": count}]
        
        # Clear cache for this request to wait for fresh data
        cache_key = (qx_asset, timeframe)
        self.candles_cache[cache_key] = None
        
        logger.info(f"Requesting candles for {qx_asset} with timeframe {timeframe}s...")
        await self._send_event(event)

        # Wait for the listener to populate cache
        for _ in range(30): # 3 seconds timeout
            await asyncio.sleep(0.1)
            if self.candles_cache.get(cache_key) is not None:
                return self.candles_cache[cache_key]
                
        logger.warning(f"Timeout waiting for candles response for {qx_asset}.")
        return []

    async def _listen_loop(self):
        """Listens to incoming WebSocket frames and parses them."""
        try:
            async for message in self.websocket:
                # Engine.IO ping handler
                if message == "2":
                    await self._send_raw("3") # Pong
                    continue

                if message.startswith("42"): # Socket.IO event
                    try:
                        event_json = message[2:]
                        event_data = json.loads(event_json)
                        event_name = event_data[0]
                        payload = event_data[1]

                        if event_name == "authorization":
                            if payload.get("status") == "success" or payload.get("authorized") or "session" in payload:
                                self.auth_success = True
                            else:
                                # Sometimes authorization returns info dictionary
                                self.auth_success = True 
                                
                        elif event_name == "candles/load" or event_name == "candles":
                            # Process candles loaded
                            # Payload is typically list of candles: [{"time": 17000000, "open": 1.1, "close": 1.2, "high": 1.3, "low": 1.0, "volume": 120}]
                            asset = payload.get("asset")
                            timeframe = payload.get("timeframe")
                            candles_list = payload.get("data", [])
                            if not candles_list and isinstance(payload, list):
                                # Sometimes payload is directly the candle list or has other formats
                                candles_list = payload

                            if asset and timeframe:
                                self.candles_cache[(asset, timeframe)] = candles_list
                                
                    except Exception as e:
                        logger.error(f"Error parsing event: {e} | Message: {message}")
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WebSocket listener exception: {e}")
            self.connected = False
            self.auth_success = False

def qx_candles_to_dataframe(candles_list: list) -> pd.DataFrame:
    """Converts Quotex WebSocket candle list to a pandas DataFrame."""
    if not candles_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(candles_list)
    # Ensure correct column names and data types
    # Quotex typical keys: 'time' (timestamp in seconds), 'open', 'close', 'high', 'low', 'volume'
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'
    })
    df = df.set_index('time')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return df.sort_index()
