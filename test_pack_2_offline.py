import json
from app.llm_interpreter import _rule_based_fallback_parser
from app.guardrails import apply_guardrails
from app.schemas import BatteryInput

with open("hard_test_pack_2.json", "r", encoding="utf-8") as f:
    cases = json.load(f)["cases"]

print("=== Testing 10 NEW Hard Cases (HARD-11 to HARD-20) in Zero-LLM Mode ===\n")
pass_count = 0

for c in cases:
    cid = c["id"]
    notes = c["input"]["operator_notes"]
    battery = BatteryInput(**c["input"]["battery"])
    raw = _rule_based_fallback_parser(notes, battery)
    actual = apply_guardrails(raw, notes, battery)
    exp = c["expected_directive_interpretation"]

    passed = True
    diffs = []
    for idx, (act, e) in enumerate(zip(actual, exp)):
        act_dict = act.model_dump()
        dtype = act_dict["directive_type"]
        applies = act_dict["applies"]
        
        if dtype != e["directive_type"] or applies != e["applies"]:
            passed = False
            diffs.append(f"Note {idx}: type={dtype} (applies={applies}) vs exp={e['directive_type']} (applies={e['applies']})")
        elif e["applies"]:
            act_adj = act_dict["structured_adjustment"] or {}
            exp_adj = e["structured_adjustment"] or {}
            
            if act_adj.get("hours") != exp_adj.get("hours"):
                passed = False
                diffs.append(f"Note {idx} hours: got {act_adj.get('hours')} vs exp {exp_adj.get('hours')}")
            if "factor" in exp_adj and abs(act_adj.get("factor", -1) - exp_adj["factor"]) > 0.01:
                passed = False
                diffs.append(f"Note {idx} factor: got {act_adj.get('factor')} vs exp {exp_adj['factor']}")
            if "minimum_energy_kwh" in exp_adj and abs(act_adj.get("minimum_energy_kwh", -1) - exp_adj["minimum_energy_kwh"]) > 0.1:
                passed = False
                diffs.append(f"Note {idx} reserve: got {act_adj.get('minimum_energy_kwh')} vs exp {exp_adj['minimum_energy_kwh']}")
            if "max_grid_kwh" in exp_adj and abs(act_adj.get("max_grid_kwh", -1) - exp_adj["max_grid_kwh"]) > 0.1:
                passed = False
                diffs.append(f"Note {idx} grid cap: got {act_adj.get('max_grid_kwh')} vs exp {exp_adj['max_grid_kwh']}")

    if passed:
        pass_count += 1
        print(f"[PASS] {cid}: {c['label']}")
    else:
        print(f"[FAIL] {cid}: {c['label']}")
        for d in diffs:
            print(f"    - {d}")

print(f"\nFinal Offline Score: {pass_count}/{len(cases)} PASS")
