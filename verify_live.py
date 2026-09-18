import json
import httpx

with open("hard_test_pack.json", "r", encoding="utf-8") as f:
    cases = json.load(f)["cases"]

url = "https://gridwise-bup-2026-94ht.onrender.com/optimize-energy"
print(f"Testing live URL: {url}\n")

with httpx.Client(timeout=30.0) as client:
    for c in cases:
        cid = c["id"]
        resp = client.post(url, json=c["input"])
        d = resp.json()
        print(f"=== {cid}: {c['label']} (Status: {resp.status_code}) ===")
        for di in d["directive_interpretation"]:
            n_idx = di["note_index"]
            dtype = di["directive_type"]
            applies = di["applies"]
            adj = di["structured_adjustment"]
            exp = c["expected_directive_interpretation"][n_idx]
            match = (dtype == exp["directive_type"] and applies == exp["applies"])
            print(f"  Note {n_idx}: [{dtype}] applies={applies} | Adj: {adj} | Match={match}")
        print(f"  Cost: {d['total_cost_bdt']:.2f} BDT | Grid: {d['total_grid_kwh']:.2f} kWh | Peak: {d['peak_grid_kwh']:.2f} kWh")
        print(f"  Summary: {d['plan_summary']}\n")
