import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns status 200 with status='ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


def test_invalid_request_handling():
    """Verify malformed requests return HTTP 400."""
    # Empty payload
    response = client.post("/optimize-energy", json={})
    assert response.status_code == 400

    # Incomplete hours (only 2 hours instead of 24)
    bad_payload = {
        "scenario_id": "TEST-FAIL",
        "operator_notes": ["Note 1"],
        "hours": [
            {"hour": 0, "demand_kwh": 50, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
            {"hour": 1, "demand_kwh": 50, "solar_kwh": 0, "tariff_bdt_per_kwh": 5}
        ],
        "battery": {
            "capacity_kwh": 100,
            "initial_energy_kwh": 50,
            "minimum_energy_kwh": 20,
            "max_charge_kwh_per_hour": 25,
            "max_discharge_kwh_per_hour": 25
        }
    }
    response = client.post("/optimize-energy", json=bad_payload)
    assert response.status_code == 400


@pytest.fixture(scope="session")
def sample_cases():
    with open("sample_cases.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def test_all_sample_cases_end_to_end(sample_cases):
    """
    Validates end-to-end processing across all 10 public sample cases:
    - Status code 200 OK
    - Exact response schema
    - Physical consistency & energy balance
    - Battery bounds & end-of-day neutrality
    - Recalculated cost matching expected optimal reference
    """
    for case in sample_cases:
        case_id = case["id"]
        inp = case["input"]
        exp = case["expected_output"]

        resp = client.post("/optimize-energy", json=inp)
        assert resp.status_code == 200, f"Failed on {case_id}: {resp.text}"

        data = resp.json()
        assert data["scenario_id"] == inp["scenario_id"]

        # 1. Directive checks
        directives = data["directive_interpretation"]
        assert len(directives) == len(inp["operator_notes"])
        for idx, d in enumerate(directives):
            assert d["note_index"] == idx
            if d["directive_type"] == "no_op":
                assert d["applies"] is False
                assert d["structured_adjustment"] is None
            else:
                assert d["applies"] is True
                assert d["structured_adjustment"] is not None
                assert "hours" in d["structured_adjustment"]

        # 2. Hourly plan checks
        plan = data["hourly_plan"]
        assert len(plan) == 24
        battery_cfg = inp["battery"]
        hours_in = inp["hours"]

        curr_energy = battery_cfg["initial_energy_kwh"]
        recalc_grid = 0.0
        recalc_cost = 0.0
        recalc_peak = 0.0

        for h_entry in plan:
            h = h_entry["hour"]
            dem = hours_in[h]["demand_kwh"]
            tar = hours_in[h]["tariff_bdt_per_kwh"]

            g = h_entry["grid_kwh"]
            s = h_entry["solar_used_kwh"]
            action = h_entry["battery_action"]
            b_kwh = h_entry["battery_kwh"]
            e_after = h_entry["battery_energy_after_kwh"]

            # Energy balance: grid + solar + discharge = demand + charge
            disch = b_kwh if action == "discharge" else 0.0
            chg = b_kwh if action == "charge" else 0.0
            lhs = g + s + disch
            rhs = dem + chg
            assert abs(lhs - rhs) <= 0.02, f"Hour {h} energy balance failed: {lhs} vs {rhs}"

            # Battery state update
            if action == "charge":
                curr_energy += b_kwh
            elif action == "discharge":
                curr_energy -= b_kwh
            assert abs(curr_energy - e_after) <= 0.02, f"Hour {h} battery transition failed"

            # Battery capacity bound
            assert e_after <= battery_cfg["capacity_kwh"] + 0.01, f"Hour {h} exceeded battery capacity"

            recalc_grid += g
            recalc_cost += g * tar
            recalc_peak = max(recalc_peak, g)

        # 3. End-of-day neutrality check
        assert abs(plan[-1]["battery_energy_after_kwh"] - battery_cfg["initial_energy_kwh"]) <= 0.02

        # 4. Totals match recalculated values
        assert abs(data["total_grid_kwh"] - recalc_grid) <= 0.05
        assert abs(data["total_cost_bdt"] - recalc_cost) <= 0.05
        assert abs(data["peak_grid_kwh"] - recalc_peak) <= 0.05

        # 5. Cost matches optimal benchmark
        exp_cost = exp["total_cost_bdt"]
        assert abs(data["total_cost_bdt"] - exp_cost) <= 0.05, (
            f"Case {case_id} cost mismatch: Got {data['total_cost_bdt']}, Expected {exp_cost}"
        )
