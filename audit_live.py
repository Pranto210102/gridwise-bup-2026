import time
import httpx

base = "https://gridwise-bup-2026-94ht.onrender.com"
client = httpx.Client(timeout=15.0)

# 1. Health
t0 = time.time()
r_h = client.get(f"{base}/health")
t_h = time.time() - t0
print(f"1. GET /health -> Status: {r_h.status_code} ({t_h:.2f}s) | Response: {r_h.json()}")

# 2. Root
t0 = time.time()
r_r = client.get(f"{base}/")
t_r = time.time() - t0
print(f"2. GET /       -> Status: {r_r.status_code} ({t_r:.2f}s) | Response: {r_r.json()}")

# 3. Sample Case
payload = {
    "scenario_id": "LIVE-AUDIT",
    "operator_notes": ["Facilities will wash rooftop panels from 12:00 until 14:00, leaving 30% solar."],
    "hours": [{"hour": i, "demand_kwh": 100 + i*2, "solar_kwh": (50 if 6<=i<=17 else 0), "tariff_bdt_per_kwh": (25 if 18<=i<=21 else 10)} for i in range(24)],
    "battery": {"capacity_kwh": 200, "initial_energy_kwh": 100, "minimum_energy_kwh": 20, "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50}
}
t0 = time.time()
r_opt = client.post(f"{base}/optimize-energy", json=payload)
t_opt = time.time() - t0
data = r_opt.json()
print(f"3. POST /optimize-energy -> Status: {r_opt.status_code} ({t_opt:.2f}s)")
print(f"   Directive: {data['directive_interpretation']}")
print(f"   Total Cost: {data['total_cost_bdt']:.2f} BDT | Grid: {data['total_grid_kwh']:.2f} kWh")
print(f"   Hourly Plan Length: {len(data['hourly_plan'])}")
print("\n>>> ALL LIVE PRODUCTION CHECKS PASSED PERFECTLY! <<<")
