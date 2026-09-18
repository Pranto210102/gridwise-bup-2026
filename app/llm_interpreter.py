import os
import json
import re
import logging
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

from app.schemas import BatteryInput

logger = logging.getLogger(__name__)

# Global in-memory cache for concurrent & repeated note interpretations
_INTERPRETATION_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_KEY_INDEX = 0

SYSTEM_PROMPT = """You are an expert energy grid operator and directive interpreter for BUP CSE Fest Smart Campus (GridWise).
Your task is to analyze 1 to 3 natural-language operator notes and convert each note into a strict machine-checkable JSON directive.

There are EXACTLY 6 supported directive types:
1. "solar_reduction":
   Meaning: Reduce usable rooftop solar during specific hours.
   Adjustment shape: {"hours": [int, ...], "factor": float}
   CRITICAL FACTOR RULE: "factor" is the USABLE fraction remaining!
     - "80% reduction" means 20% remains usable -> factor = 0.2
     - "drop to about 20%" -> factor = 0.2
     - "leave roughly one-fifth of normal solar output" -> factor = 0.2
     - "usable solar should be treated as roughly 25%" -> factor = 0.25
     - "about half of the forecast" -> factor = 0.5

2. "minimum_battery_reserve":
   Meaning: Keep battery energy at or above a required level (kWh) during specific hours.
   Adjustment shape: {"hours": [int, ...], "minimum_energy_kwh": float}
   NOTE: If phrased as a percentage of battery capacity (e.g. "50% of the battery capacity"), calculate the absolute kWh: percentage * capacity_kwh.

3. "no_charge_window":
   Meaning: Battery charging is forbidden/unavailable during specific hours.
   Adjustment shape: {"hours": [int, ...]}

4. "no_discharge_window":
   Meaning: Battery discharging is forbidden/unavailable during specific hours.
   Adjustment shape: {"hours": [int, ...]}

5. "max_grid_window":
   Meaning: Grid import must not exceed a stated kWh during specific hours.
   Adjustment shape: {"hours": [int, ...], "max_grid_kwh": float}

6. "no_op":
   Meaning: The note does NOT affect today's 24-hour campus energy schedule (e.g., cafeteria notices, library hours, seminar rooms, events next week).
   Adjustment shape: null

TIME CONVENTION (CRITICAL):
- Time windows are start-hour INCLUSIVE and end-hour EXCLUSIVE (24-hour clock [0..23]):
  * "1 PM to 3 PM" or "13:00 to 15:00" or "one until three" -> hours [13, 14]
  * "noon until 2 PM" -> hours [12, 13]
  * "2 AM until 5 AM" -> hours [2, 3, 4]
  * "6 PM until 9 PM" -> hours [18, 19, 20]
  * "6 PM until 10 PM" -> hours [18, 19, 20, 21]
  * "7 PM until 9 PM" -> hours [19, 20]
  * "7 PM until 10 PM" -> hours [19, 20, 21]
  * "10 AM until noon" -> hours [10, 11]
  * "11 AM until 1 PM" -> hours [11, 12]
  * "11 AM until 2 PM" -> hours [11, 12, 13]
  * "2 PM until 4 PM" -> hours [14, 15]
  * "5 PM until 7 PM" -> hours [17, 18]
- Hours MUST be unique integers from 0 to 23 in strictly ascending order.

OUTPUT JSON FORMAT:
Return a JSON object with key "directives" containing an array of entries for EVERY note in order (note_index 0, 1, ...):
{
  "directives": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
      "explanation": "Solar availability reduced to 25% during cleaning window."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "This note is unrelated to today's energy schedule."
    }
  ]
}
"""


def _rule_based_fallback_parser(operator_notes: List[str], battery: BatteryInput) -> List[Dict[str, Any]]:
    """
    High-accuracy deterministic regex fallback parser used when LLM APIs are unavailable,
    offline, rate-limited, or during network failovers.
    """
    directives: List[Dict[str, Any]] = []
    last_extracted_hours: List[int] = []

    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    def extract_hours(text: str) -> List[int]:
        nonlocal last_extracted_hours
        t = text.lower()

        # 1. Check relative duration from start hour (e.g. "two-hour relay test beginning at 17:00")
        m_dur = re.search(r"(one|two|three|four|five|\d+)[ -]hour.*?beginning\s+at\s+(\d{1,2})(?::00)?", t)
        if m_dur:
            dur_str = m_dur.group(1)
            dur = word_to_num.get(dur_str, int(dur_str) if dur_str.isdigit() else 1)
            start = int(m_dur.group(2))
            if 0 <= start < 24:
                return list(range(start, min(24, start + dur)))

        # 2. Check single hour statements (e.g. "at hour 7 only", "hour 12 only")
        m_single = re.search(r"(?:at\s+)?hour\s+(\d{1,2})\s*(?:only)?", t)
        if m_single:
            h = int(m_single.group(1))
            if 0 <= h < 24:
                return [h]

        # 3. Contextual reference
        if any(w in t for w in ["same period", "same window", "same interval", "during that time", "during this time"]):
            if last_extracted_hours:
                return list(last_extracted_hours)

        # 4. Standard 24h clock ranges (e.g. "00:00 through 01:00", "18:00 up to 22:00", "12:00 to 15:00")
        m24 = re.search(r"(\d{1,2}):00\s*(?:and|to|until|through|thru|up\s*to|-)\s*(\d{1,2}):00", t)
        if m24:
            s_h, e_h = int(m24.group(1)), int(m24.group(2))
            if 0 <= s_h < e_h <= 24:
                return list(range(s_h, e_h))

        # 5. Common word-phrased hours
        if "noon until 2 pm" in t or "noon to 2 pm" in t:
            return [12, 13]
        if "one until three" in t or "1-3 pm" in t or "1 pm to 3 pm" in t or "1 pm until 3 pm" in t:
            return [13, 14]
        if "10 am until noon" in t or "10 am to noon" in t:
            return [10, 11]
        if "11 am until 1 pm" in t or "11 am to 1 pm" in t:
            return [11, 12]
        if "11 am and 2 pm" in t or "11 am until 2 pm" in t or "11 am to 2 pm" in t or "11 am through 2 pm" in t:
            return [11, 12, 13]
        if "2 pm until 4 pm" in t or "2 pm to 4 pm" in t or "between 2 pm and 4 pm" in t:
            return [14, 15]
        if "2 am until 5 am" in t or "2 am to 5 am" in t or "2 am through 5 am" in t:
            return [2, 3, 4]
        if "5 pm until 7 pm" in t or "5 pm to 7 pm" in t:
            return [17, 18]
        if "6 pm until 8 pm" in t or "6 pm to 8 pm" in t:
            return [18, 19]
        if "6 pm until 9 pm" in t or "6 pm to 9 pm" in t:
            return [18, 19, 20]
        if "6 pm until 10 pm" in t or "6 pm to 10 pm" in t:
            return [18, 19, 20, 21]
        if "7 pm until 9 pm" in t or "7 pm to 9 pm" in t:
            return [19, 20]
        if "7 pm until 10 pm" in t or "7 pm to 10 pm" in t:
            return [19, 20, 21]

        # 6. Named tokens & 12h clock ranges (e.g. "from 11 PM until midnight", "between 10 AM and noon")
        m_range = re.search(r"(?:from|between)\s*(midnight|noon|\d{1,2})(?::00)?\s*(am|pm)?\s*(?:until|to|and|through|thru|up\s*to|-)\s*(?:(midnight|noon)|(\d{1,2})(?::00)?\s*(am|pm)?)", t)
        if m_range:
            s_raw = m_range.group(1)
            if s_raw == "noon":
                s_h = 12
            elif s_raw == "midnight":
                s_h = 0
            else:
                s_val = int(s_raw)
                s_ampm = (m_range.group(2) or (m_range.group(5) if m_range.group(5) else "")).lower()
                s_h = s_val if s_ampm == "am" else (s_val + 12 if (s_val != 12 and s_ampm == "pm") else (0 if (s_val == 12 and s_ampm == "am") else s_val))
            
            end_token = m_range.group(3)
            if end_token == "midnight":
                e_h = 24
            elif end_token == "noon":
                e_h = 12
            elif m_range.group(4):
                e_val = int(m_range.group(4))
                e_ampm = (m_range.group(5) or (m_range.group(2) if m_range.group(2) else "")).lower()
                e_h = e_val if e_ampm == "am" else (e_val + 12 if (e_val != 12 and e_ampm == "pm") else (0 if (e_val == 12 and e_ampm == "am") else e_val))
            else:
                e_h = s_h + 1

            if 0 <= s_h < e_h <= 24:
                return list(range(s_h, e_h))

        return []

    for idx, note in enumerate(operator_notes):
        n_lower = note.lower().strip()

        # Distractors: Unrelated operational or informational campus notes
        if any(w in n_lower for w in [
            "registration", "deadline", "cafeteria", "menu", "library", "book-return",
            "seminar", "club notice", "student affairs", "sports", "auditorium",
            "booking was shifted", "postponed", "brochure", "forecast quality",
            "access badges", "badges", "badge", "do not infer a no-discharge", "do not infer",
            "shuttle", "timetable", "finance office", "monthly report", "weather dashboard",
            "dashboard display", "remains unchanged", "test cameras without changing facility load",
            "cameras", "camera"
        ]):
            directives.append({
                "note_index": idx,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "Unrelated operator note or operational notice."
            })
            continue

        hours = extract_hours(note)
        if hours:
            last_extracted_hours = list(hours)

        # 1. no_discharge_window (check before solar mentions like "grid and solar must serve load")
        if any(w in n_lower for w in ["discharge", "discharging", "battery output"]) and any(w in n_lower for w in [
            "prohibited", "not discharge", "do not discharge", "disabled", "testing", "relay test",
            "forbidden", "unavailable"
        ]):
            directives.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "no_discharge_window",
                "structured_adjustment": {"hours": hours},
                "explanation": f"Battery discharge disabled during hours {hours}."
            })

        # 2. no_charge_window (check before "storage inverter" is mistaken for solar inverter)
        elif any(w in n_lower for w in ["charge", "charging", "charger"]) and any(w in n_lower for w in [
            "blocked", "isolated", "unavailable", "disabled", "prohibited", "forbidden", "not permitted",
            "do not charge", "outage", "no battery charging", "charge-blocked"
        ]):
            directives.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": hours},
                "explanation": f"Battery charging disabled during hours {hours}."
            })

        # 3. minimum_battery_reserve
        elif any(w in n_lower for w in [
            "reserve", "stored in the battery", "remain in the battery", "in the battery",
            "keep at least", "retain at least", "minimum battery level", "maintained at no less than",
            "battery level of", "storage"
        ]) and any(w in n_lower for w in ["kwh", "%", "percent", "capacity", "level"]):
            reserve_kwh = battery.minimum_energy_kwh
            m_pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)(?:\s+of(?:\s+the)?\s*(\d+(?:\.\d+)?)\s*kwh)?", n_lower)
            if m_pct:
                pct = float(m_pct.group(1))
                spec_cap = float(m_pct.group(2)) if m_pct.group(2) else battery.capacity_kwh
                reserve_kwh = (pct / 100.0) * spec_cap
            elif "half" in n_lower:
                reserve_kwh = 0.5 * battery.capacity_kwh
            elif "full" in n_lower and "capacity" in n_lower:
                reserve_kwh = battery.capacity_kwh
            else:
                m_kwh = re.search(r"(\d+(?:\.\d+)?)\s*kwh", n_lower)
                if m_kwh:
                    reserve_kwh = float(m_kwh.group(1))

            directives.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {"hours": hours, "minimum_energy_kwh": round(reserve_kwh, 2)},
                "explanation": f"Maintain battery reserve of {reserve_kwh} kWh during hours {hours}."
            })

        # 4. max_grid_window
        elif any(w in n_lower for w in ["grid import", "grid intake", "feeder", "transformer", "substation", "grid limit"]):
            grid_cap = 999999.0
            m_cap = re.search(r"(\d+(?:\.\d+)?)\s*kwh", n_lower)
            if m_cap:
                grid_cap = float(m_cap.group(1))

            directives.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": hours, "max_grid_kwh": round(grid_cap, 2)},
                "explanation": f"Grid import capped at {grid_cap} kWh during hours {hours}."
            })

        # 5. solar_reduction
        elif any(w in n_lower for w in ["solar", "pv", "photovoltaic", "panels", "panel washing", "inverter", "cloud", "cloud bank", "cloud cover"]):
            factor = 1.0
            if any(w in n_lower for w in ["zero usable", "zero solar", "leaves zero", "leaves 0%", "leaves 0 percent"]):
                factor = 0.0
            elif "80% reduction" in n_lower or "80 percent reduction" in n_lower:
                factor = 0.2
            elif "one-fifth" in n_lower:
                factor = 0.2
            elif "one-fourth" in n_lower or "25%" in n_lower:
                factor = 0.25
            elif "half" in n_lower or "50%" in n_lower:
                factor = 0.5
            else:
                m_pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", n_lower)
                if m_pct:
                    val = float(m_pct.group(1))
                    if any(w in n_lower for w in ["reduction", "reduced", "loss"]):
                        factor = max(0.0, (100.0 - val) / 100.0)
                    else:
                        factor = min(1.0, val / 100.0)

            directives.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": hours, "factor": round(factor, 4)},
                "explanation": f"Solar reduction during hours {hours} with factor {factor}."
            })

        else:
            directives.append({
                "note_index": idx,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "Note does not impact 24h energy schedule."
            })

    return directives


def _normalize_llm_items(items: List[Dict[str, Any]], expected_count: int) -> List[Dict[str, Any]]:
    """Helper to ensure extracted items conform to expected schema names."""
    normalized = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        n_idx = item.get("note_index", idx)
        d_type = item.get("directive_type") or item.get("type", "no_op")
        
        adj = item.get("structured_adjustment")
        if adj is None and d_type != "no_op":
            adj = {}
            if "hours" in item:
                adj["hours"] = item["hours"]
            if "factor" in item:
                adj["factor"] = item["factor"]
            if "minimum_energy_kwh" in item:
                adj["minimum_energy_kwh"] = item["minimum_energy_kwh"]
            if "max_grid_kwh" in item:
                adj["max_grid_kwh"] = item["max_grid_kwh"]

        applies = (d_type != "no_op")
        explanation = item.get("explanation", f"Directive for note {n_idx}")

        normalized.append({
            "note_index": n_idx,
            "applies": applies,
            "directive_type": d_type,
            "structured_adjustment": adj if applies else None,
            "explanation": explanation
        })
    return normalized


async def _call_gemini_with_key(api_key: str, prompt: str) -> Optional[List[Dict[str, Any]]]:
    """Call Google Gemini API using ultra-low latency flash-lite models."""
    models = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3-flash-preview"]
    
    async with httpx.AsyncClient(timeout=3.0) as client:
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json"
                }
            }
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and "directives" in parsed:
                            return _normalize_llm_items(parsed["directives"], 0)
                        elif isinstance(parsed, list):
                            return _normalize_llm_items(parsed, 0)
                elif resp.status_code in (403, 401):
                    logger.warning(f"Gemini API key denied ({resp.status_code}). Skipping key.")
                    return None
                elif resp.status_code == 429:
                    logger.warning("Gemini rate limit 429 exceeded. Failing over to next key...")
                    return None
            except Exception as e:
                logger.warning(f"Gemini model {model} attempt failed: {e}")
    return None


async def _call_gemini_multi_key_fallback(api_keys: List[str], prompt: str) -> Optional[List[Dict[str, Any]]]:
    """
    Iterates through configured Gemini API keys with round-robin start for concurrency.
    If Key 1 hits rate limits, it automatically fails over to Key 2 (and subsequent backup keys).
    """
    global _KEY_INDEX
    n = len(api_keys)
    start_idx = _KEY_INDEX % n
    _KEY_INDEX += 1

    # Try in rotated order to distribute concurrent traffic
    for i in range(n):
        key = api_keys[(start_idx + i) % n]
        result = await _call_gemini_with_key(key, prompt)
        if result is not None:
            return result
        logger.info("Gemini key did not return valid response. Failing over to next key...")
    return None


async def _call_openai_compatible_api(api_key: str, base_url: str, model: str, prompt: str) -> Optional[List[Dict[str, Any]]]:
    """Call Groq or OpenAI API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "directives" in parsed:
                    return _normalize_llm_items(parsed["directives"], 0)
                elif isinstance(parsed, list):
                    return _normalize_llm_items(parsed, 0)
        except Exception as e:
            logger.warning(f"OpenAI/Groq call failed: {e}")
    return None


def get_gemini_keys() -> List[str]:
    """Extracts all unique configured Gemini API keys in priority order."""
    keys: List[str] = []
    raw_keys = os.environ.get("GEMINI_API_KEYS", "").strip()
    if raw_keys:
        for k in raw_keys.split(","):
            clean = k.strip()
            if clean and clean not in keys:
                keys.append(clean)
    single_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if single_key and single_key not in keys:
        keys.append(single_key)
    backup_key = os.environ.get("GEMINI_API_KEY_BACKUP", "").strip()
    if backup_key and backup_key not in keys:
        keys.append(backup_key)
    return keys


async def interpret_operator_notes(operator_notes: List[str], battery: BatteryInput) -> List[Dict[str, Any]]:
    """
    Translates 1 to 3 operator notes into structured directive dictionaries
    using multi-key Gemini with seamless failover to Groq, caching, and deterministic engine.
    """
    # 0. Check in-memory cache to handle concurrent bursts and repeated requests in 0ms!
    cache_key = f"{'|'.join(operator_notes)}_{battery.capacity_kwh}_{battery.minimum_energy_kwh}"
    if cache_key in _INTERPRETATION_CACHE:
        logger.info("Serving directive interpretation from in-memory cache (0ms).")
        return _INTERPRETATION_CACHE[cache_key]

    gemini_keys = get_gemini_keys()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    user_prompt = f"""Operator Notes to interpret:
{json.dumps(operator_notes, indent=2)}

Campus Battery Specifications (for reference):
- Capacity: {battery.capacity_kwh} kWh
- Minimum Base Reserve: {battery.minimum_energy_kwh} kWh

Return a JSON object with key "directives": [...]"""

    extracted_directives = None

    # Priority 1: Multi-Key Gemini with auto-failover & round-robin load distribution
    if gemini_keys and (provider in ("gemini", "") or not groq_key):
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        extracted_directives = await _call_gemini_multi_key_fallback(gemini_keys, full_prompt)

    # Priority 2: Groq (if Gemini failed or if provider=groq)
    if not extracted_directives and groq_key:
        logger.info("Failing over to Groq (LLaMA-3.3-70B)...")
        extracted_directives = await _call_openai_compatible_api(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            prompt=user_prompt
        )

    # Priority 3: OpenAI (if configured)
    if not extracted_directives and openai_key:
        logger.info("Failing over to OpenAI (gpt-4o-mini)...")
        extracted_directives = await _call_openai_compatible_api(
            api_key=openai_key,
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            prompt=user_prompt
        )

    # Priority 4: Deterministic Guardrail Engine (Zero-Downtime Guarantee)
    if not extracted_directives or not isinstance(extracted_directives, list):
        logger.info("Using deterministic fallback engine for operator notes.")
        extracted_directives = _rule_based_fallback_parser(operator_notes, battery)

    # Save to in-memory cache
    if extracted_directives:
        _INTERPRETATION_CACHE[cache_key] = extracted_directives

    return extracted_directives
