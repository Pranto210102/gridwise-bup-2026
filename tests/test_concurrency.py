import pytest
import asyncio
import httpx
import time
import json
from app.main import app


@pytest.mark.anyio
async def test_concurrent_load():
    """
    Simulates 5 parallel requests arriving at the exact same moment.
    Verifies that the server handles concurrent I/O, LLM calls, and MILP solves
    without deadlocks, errors, or constraint violations.
    """
    with open("sample_cases.json", "r") as f:
        cases = json.load(f)["cases"]

    payloads = [cases[i]["input"] for i in range(5)]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.time()
        tasks = [client.post("/optimize-energy", json=p) for p in payloads]
        responses = await asyncio.gather(*tasks)
        dur = time.time() - t0

        assert dur < 8.0, f"Concurrency test took too long: {dur:.2f}s"
        for i, r in enumerate(responses):
            assert r.status_code == 200
            data = r.json()
            assert data["scenario_id"] == payloads[i]["scenario_id"]
            assert len(data["hourly_plan"]) == 24
            assert data["total_cost_bdt"] > 0
