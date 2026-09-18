import json
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_behavior_when_all_llms_hit_quota_limit():
    """
    Simulates a catastrophic LLM outage where:
    - Gemini Key 1 hits 429 / quota exceeded
    - Gemini Key 2 hits 429 / quota exceeded
    - Gemini Key 3 hits 429 / quota exceeded
    - Groq and OpenAI are unreachable
    
    Verifies that the API STILL successfully returns 200 OK
    and passes all 10 hard cases with 100% precision using the deterministic engine!
    """
    with open("hard_test_pack.json", "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    # Mock all remote LLM calls to return None (simulating complete quota death)
    with patch("app.llm_interpreter._call_gemini_multi_key_fallback", return_value=None), \
         patch("app.llm_interpreter._call_openai_compatible_api", return_value=None):
        
        # Clear in-memory cache to force live parsing
        from app.llm_interpreter import _INTERPRETATION_CACHE
        _INTERPRETATION_CACHE.clear()

        for c in cases:
            cid = c["id"]
            inp = c["input"]
            exp = c["expected_directive_interpretation"]

            resp = client.post("/optimize-energy", json=inp)
            assert resp.status_code == 200, f"Failed on {cid} during LLM outage: {resp.text}"

            data = resp.json()
            actual = data["directive_interpretation"]

            for idx, (act, e) in enumerate(zip(actual, exp)):
                assert act["directive_type"] == e["directive_type"], f"{cid} Note {idx} type mismatch during outage"
                assert act["applies"] == e["applies"], f"{cid} Note {idx} applies mismatch during outage"
                if e["applies"]:
                    act_adj = act["structured_adjustment"]
                    exp_adj = e["structured_adjustment"]
                    assert act_adj.get("hours") == exp_adj.get("hours"), f"{cid} Note {idx} hours mismatch during outage"
                    if "factor" in exp_adj:
                        assert abs(act_adj["factor"] - exp_adj["factor"]) < 0.01
                    if "minimum_energy_kwh" in exp_adj:
                        assert abs(act_adj["minimum_energy_kwh"] - exp_adj["minimum_energy_kwh"]) < 0.1
                    if "max_grid_kwh" in exp_adj:
                        assert abs(act_adj["max_grid_kwh"] - exp_adj["max_grid_kwh"]) < 0.1
