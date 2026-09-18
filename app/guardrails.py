from typing import List, Dict, Any, Optional
from app.schemas import DirectiveInterpretation, DirectiveType, BatteryInput


ALLOWED_DIRECTIVES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op"
}


def sanitize_hours(raw_hours: Any) -> List[int]:
    """Sanitize, deduplicate, filter to 0..23, and sort in ascending order."""
    if not isinstance(raw_hours, list):
        return []
    valid = []
    for h in raw_hours:
        try:
            h_int = int(round(float(h)))
            if 0 <= h_int <= 23:
                valid.append(h_int)
        except (ValueError, TypeError):
            continue
    return sorted(list(set(valid)))


def validate_and_repair_directive(
    raw_dict: Dict[str, Any],
    expected_index: int,
    original_note: str,
    battery: BatteryInput
) -> DirectiveInterpretation:
    """
    Deterministically validates and repairs an extracted directive interpretation
    to guarantee compliance with official GridWise constraints.
    """
    raw_type = str(raw_dict.get("directive_type", "no_op")).strip()
    if raw_type not in ALLOWED_DIRECTIVES:
        raw_type = "no_op"

    explanation = str(raw_dict.get("explanation", "")).strip()
    if not explanation:
        explanation = f"Interpretation for note {expected_index}."

    if raw_type == "no_op":
        return DirectiveInterpretation(
            note_index=expected_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation=explanation
        )

    # For non-no_op directives
    adj = raw_dict.get("structured_adjustment")
    if not isinstance(adj, dict):
        # Invalid adjustment shape -> safe fallback to no_op
        return DirectiveInterpretation(
            note_index=expected_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation="Invalid adjustment shape, treated as no_op."
        )

    hours = sanitize_hours(adj.get("hours"))
    if not hours:
        # Directives require valid hours; if none found, fallback to no_op
        return DirectiveInterpretation(
            note_index=expected_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation="No valid time window identified, treated as no_op."
        )

    repaired_adj: Dict[str, Any] = {"hours": hours}

    if raw_type == "solar_reduction":
        raw_factor = adj.get("factor")
        try:
            factor = float(raw_factor)
        except (ValueError, TypeError):
            factor = 1.0
        
        # Guardrail: factor must be fraction [0.0, 1.0]
        if factor > 1.0:
            if factor <= 100.0:
                # If LLM passed percentage remaining or reduced
                factor = factor / 100.0
            else:
                factor = 1.0
        factor = max(0.0, min(1.0, factor))
        repaired_adj["factor"] = round(factor, 4)

    elif raw_type == "minimum_battery_reserve":
        raw_val = adj.get("minimum_energy_kwh")
        try:
            reserve = float(raw_val)
        except (ValueError, TypeError):
            reserve = battery.minimum_energy_kwh

        # Guardrail: If expressed as ratio <= 1.0 and capacity > 1.0, convert
        if reserve <= 1.0 and battery.capacity_kwh > 1.0:
            reserve = reserve * battery.capacity_kwh
        # Must not exceed capacity, cannot be negative
        reserve = max(0.0, min(battery.capacity_kwh, reserve))
        repaired_adj["minimum_energy_kwh"] = round(reserve, 2)

    elif raw_type == "max_grid_window":
        raw_val = adj.get("max_grid_kwh")
        try:
            grid_cap = float(raw_val)
        except (ValueError, TypeError):
            grid_cap = 999999.0
        grid_cap = max(0.0, grid_cap)
        repaired_adj["max_grid_kwh"] = round(grid_cap, 2)

    elif raw_type in ("no_charge_window", "no_discharge_window"):
        # only hours needed
        pass

    return DirectiveInterpretation(
        note_index=expected_index,
        applies=True,
        directive_type=raw_type,
        structured_adjustment=repaired_adj,
        explanation=explanation
    )


def apply_guardrails(
    raw_interpretations: List[Dict[str, Any]],
    operator_notes: List[str],
    battery: BatteryInput
) -> List[DirectiveInterpretation]:
    """
    Ensures exactly len(operator_notes) entries are returned,
    ordered by note_index 0..N-1, with valid types and adjustment schemas.
    """
    total_notes = len(operator_notes)
    by_index: Dict[int, Dict[str, Any]] = {}

    for item in raw_interpretations:
        if isinstance(item, dict):
            idx = item.get("note_index")
            if isinstance(idx, int) and 0 <= idx < total_notes:
                by_index[idx] = item

    final_list: List[DirectiveInterpretation] = []
    for i in range(total_notes):
        raw_item = by_index.get(i, {})
        entry = validate_and_repair_directive(
            raw_item,
            expected_index=i,
            original_note=operator_notes[i],
            battery=battery
        )
        final_list.append(entry)

    return final_list
