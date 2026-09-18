import time
import sys
import httpx

TARGET_URL = sys.argv[1] if len(sys.argv) > 1 else "https://gridwise-bup-2026-94ht.onrender.com/health"
INTERVAL_SECONDS = 300  # Ping every 5 minutes (Render sleep timeout is 15 minutes)

print("=" * 65)
print("GRIDWISE ACTIVE HEARTBEAT & UPTIME GUARDIAN")
print(f"Target Endpoint: {TARGET_URL}")
print(f"Interval: Every {INTERVAL_SECONDS // 60} minutes")
print("Press Ctrl+C to stop.")
print("=" * 65)

while True:
    try:
        t0 = time.time()
        resp = httpx.get(TARGET_URL, timeout=30.0)
        latency = time.time() - t0
        status_text = "AWAKE & HEALTHY" if resp.status_code == 200 else f"HTTP {resp.status_code}"
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status_text} | Latency: {latency:.2f}s | Response: {resp.text.strip()[:60]}")
    except Exception as exc:
        print(f"[{time.strftime('%H:%M:%S')}] Connection Notice: {exc}")
    time.sleep(INTERVAL_SECONDS)
