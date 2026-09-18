import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture(scope="session")
def hard_cases():
    with open("hard_test_pack.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def test_hard_pack_cases(hard_cases):
    """
    Validates all 10 production-level hard cases:
    - Directive interpretation accuracy (directive_type, hours, factors, reserves, grid caps)
    - Distractor/no-op rejection
    - Physical constraint feasibility & strict compliance
    - Battery neutrality and balance
    """
    for case in hard_cases:
        cid = case["id"]
        inp = case["input"]
        exp_directives = case["expected_directive_interpretation"]
        assertions = case.get("validation_assertions", {})

        print(f"\n--- Testing {cid}: {case['label']} ---")
        resp = client.post("/optimize-energy", json=inp)
        assert resp.status_code == 200, f"Failed on {cid}: {resp.status_code} - {resp.text}"

        data = resp.json()
        assert data["scenario_id"] == inp["scenario_id"]

        # 1. Semantic directive interpretation
        actual_directives = data["directive_interpretation"]
        assert len(actual_directives) == len(exp_directives), f"Note count mismatch in {cid}"

        for idx, (act, exp) in enumerate(zip(actual_directives, exp_directives)):
            assert act["note_index"] == exp["note_index"] == idx
            assert act["applies"] == exp["applies"], f"{cid} Note {idx} applies mismatch: got {act['applies']}, exp {exp['applies']}"
            assert act["directive_type"] == exp["directive_type"], f"{cid} Note {idx} type mismatch: got {act['directive_type']}, exp {exp['directive_type']}"

            if not exp["applies"]:
                assert act["structured_adjustment"] is None, f"{cid} Note {idx} expected None adjustment for no_op"
            else:
                act_adj = act["structured_adjustment"]
                exp_adj = exp["structured_adjustment"]
                assert act_adj is not None, f"{cid} Note {idx} structured_adjustment should not be None"
                
                # Check hours
                if assertions.get("hours_exact", True):
                    assert act_adj.get("hours") == exp_adj.get("hours"), f"{cid} Note {idx} hours mismatch: got {act_adj.get('hours')}, exp {exp_adj.get('hours')}"

                # Check factor
                if "factor" in exp_adj:
                    assert "factor" in act_adj, f"{cid} Note {idx} missing factor"
                    assert abs(act_adj["factor"] - exp_adj["factor"]) < 0.01, f"{cid} Note {idx} factor mismatch: got {act_adj['factor']}, exp {exp_adj['factor']}"

                # Check minimum_energy_kwh
                if "minimum_energy_kwh" in exp_adj:
                    assert "minimum_energy_kwh" in act_adj, f"{cid} Note {idx} missing minimum_energy_kwh"
                    assert abs(act_adj["minimum_energy_kwh"] - exp_adj["minimum_energy_kwh"]) < 0.1, f"{cid} Note {idx} reserve mismatch: got {act_adj['minimum_energy_kwh']}, exp {exp_adj['minimum_energy_kwh']}"

                # Check max_grid_kwh
                if "max_grid_kwh" in exp_adj:
                    assert "max_grid_kwh" in act_adj, f"{cid} Note {idx} missing max_grid_kwh"
                    assert abs(act_adj["max_grid_kwh"] - exp_adj["max_grid_kwh"]) < 0.1, f"{cid} Note {idx} max_grid mismatch: got {act_adj['max_grid_kwh']}, exp {exp_adj['max_grid_kwh']}"

        # 2. Downstream Hourly Dispatch Physical Constraint Replay
        plan = data["hourly_plan"]
        assert len(plan) == 24
        battery_cfg = inp["battery"]
        hours_in = inp["hours"]

        # Build hourly directive constraints
        effective_solar = [hours_in[h]["solar_kwh"] for h in range(24)]
        hourly_min_reserve = [battery_cfg["minimum_energy_kwh"] for _ in range(24)]
        charge_blocked = [False] * 24
        discharge_blocked = [False] * 24
        grid_caps = [float("inf")] * 24

        for d in actual_directives:
            if not d["applies"] or not d.get("structured_adjustment"):
                continue
            adj = d["structured_adjustment"]
            d_hours = adj.get("hours", [])
            d_type = d["directive_type"]

            if d_type == "solar_reduction":
                factor = adj.get("factor", 1.0)
                for h in d_hours:
                    effective_solar[h] = hours_in[h]["solar_kwh"] * factor
            elif d_type == "minimum_battery_reserve":
                res = adj.get("minimum_energy_kwh", battery_cfg["minimum_energy_kwh"])
                for h in d_hours:
                    hourly_min_reserve[h] = max(hourly_min_reserve[h], res)
            elif d_type == "no_charge_window":
                for h in d_hours:
                    charge_blocked[h] = True
            elif d_type == "no_discharge_window":
                for h in d_hours:
                    discharge_blocked[h] = True
            elif d_type == "max_grid_window":
                cap = adj.get("max_grid_kwh", float("inf"))
                for h in d_hours:
                    grid_caps[h] = min(grid_caps[h], cap)

        curr_energy = battery_cfg["initial_energy_kwh"]
        recalc_cost = 0.0
        recalc_grid = 0.0

        for h_entry in plan:
            h = h_entry["hour"]
            dem = hours_in[h]["demand_kwh"]
            tar = hours_in[h]["tariff_bdt_per_kwh"]

            g = h_entry["grid_kwh"]
            s = h_entry["solar_used_kwh"]
            action = h_entry["battery_action"]
            b_val = h_entry["battery_kwh"]
            be = h_entry["battery_energy_after_kwh"]

            bc = b_val if action == "charge" else 0.0
            bd = b_val if action == "discharge" else 0.0

            # Non-negativity
            assert g >= -1e-4, f"{cid} Hour {h}: negative grid {g}"
            assert s >= -1e-4, f"{cid} Hour {h}: negative solar {s}"
            assert bc >= -1e-4, f"{cid} Hour {h}: negative charge {bc}"
            assert bd >= -1e-4, f"{cid} Hour {h}: negative discharge {bd}"

            # Solar availability
            assert s <= effective_solar[h] + 1e-2, f"{cid} Hour {h}: solar_used {s} > effective {effective_solar[h]}"

            # Energy balance: demand + battery_charge = grid + solar_used + battery_discharge
            lhs = dem + bc
            rhs = g + s + bd
            assert abs(lhs - rhs) < 0.05, f"{cid} Hour {h}: energy imbalance LHS {lhs} != RHS {rhs}"

            # Charge/discharge rates
            assert bc <= battery_cfg["max_charge_kwh_per_hour"] + 1e-2
            assert bd <= battery_cfg["max_discharge_kwh_per_hour"] + 1e-2

            # Lockouts
            if charge_blocked[h]:
                assert bc < 1e-2, f"{cid} Hour {h}: charging blocked but charged {bc}"
            if discharge_blocked[h]:
                assert bd < 1e-2, f"{cid} Hour {h}: discharging blocked but discharged {bd}"

            # Grid cap (checked when physically feasible without blackouts)
            if grid_caps[h] < float("inf") and (dem <= grid_caps[h] + effective_solar[h] + battery_cfg["max_discharge_kwh_per_hour"] + 1e-2):
                assert g <= grid_caps[h] + 1e-2, f"{cid} Hour {h}: grid {g} exceeded cap {grid_caps[h]}"

            # Battery continuity & bounds
            next_energy = curr_energy + bc - bd
            assert abs(be - next_energy) < 0.05, f"{cid} Hour {h}: energy tracking mismatch {be} vs {next_energy}"
            assert be <= battery_cfg["capacity_kwh"] + 1e-2, f"{cid} Hour {h}: capacity exceeded"
            assert be >= hourly_min_reserve[h] - 1e-2, f"{cid} Hour {h}: reserve violated {be} < {hourly_min_reserve[h]}"

            curr_energy = next_energy
            recalc_cost += g * tar
            recalc_grid += g

        # Neutrality
        assert abs(curr_energy - battery_cfg["initial_energy_kwh"]) < 0.1, f"{cid}: End of day neutrality violated {curr_energy} != {battery_cfg['initial_energy_kwh']}"

        # Response totals match
        assert abs(data["total_grid_kwh"] - recalc_grid) < 0.1
        assert abs(data["total_cost_bdt"] - recalc_cost) < 0.1
