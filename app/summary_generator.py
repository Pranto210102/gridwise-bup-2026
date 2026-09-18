from typing import List
from app.schemas import DirectiveInterpretation, HourlyPlanEntry, BatteryInput


def generate_plan_summary(
    directives: List[DirectiveInterpretation],
    hourly_plan: List[HourlyPlanEntry],
    battery: BatteryInput,
    total_cost_bdt: float,
    peak_grid_kwh: float
) -> str:
    """
    Generates a concise, professional, human-readable summary of the 24-hour energy strategy.
    """
    applied_directives = [d for d in directives if d.applies]
    ignored_notes = [d for d in directives if not d.applies]

    notes_desc = []
    grid_cap_val = None
    if applied_directives:
        d_types = [d.directive_type for d in applied_directives]
        for d in applied_directives:
            if d.directive_type == "max_grid_window" and d.structured_adjustment:
                cap = d.structured_adjustment.get("max_grid_kwh")
                if cap is not None:
                    grid_cap_val = min(grid_cap_val, cap) if grid_cap_val is not None else cap
        notes_desc.append(f"incorporates operational directives ({', '.join(d_types)})")
    if ignored_notes:
        notes_desc.append(f"ignores {len(ignored_notes)} unrelated operator note(s)")

    charge_hours = [p.hour for p in hourly_plan if p.battery_action == "charge"]
    discharge_hours = [p.hour for p in hourly_plan if p.battery_action == "discharge"]

    actions_desc = []
    if charge_hours:
        actions_desc.append(f"charges during low-tariff/solar hours {charge_hours}")
    if discharge_hours:
        actions_desc.append(f"discharges during peak-tariff hours {discharge_hours}")

    summary_parts = []
    if notes_desc:
        summary_parts.append(f"Optimal 24-hour dispatch plan that {' and '.join(notes_desc)}.")
    else:
        summary_parts.append("Optimal 24-hour dispatch plan without active operational constraints.")

    if actions_desc:
        summary_parts.append(f"Battery strategically {' and '.join(actions_desc)}.")

    if grid_cap_val is not None:
        if peak_grid_kwh <= grid_cap_val + 0.05:
            peak_text = f"respects feeder import limits with a peak grid intake of {peak_grid_kwh:.1f} kWh"
        else:
            peak_text = f"manages feeder capacity constraints with a peak grid intake of {peak_grid_kwh:.1f} kWh"
    else:
        peak_text = f"achieves a peak grid intake of {peak_grid_kwh:.1f} kWh"

    summary_parts.append(
        f"Preserves active battery reserves, {peak_text}, "
        f"restores initial state of charge to {battery.initial_energy_kwh:.1f} kWh at end of day, "
        f"and achieves a minimized total grid electricity cost of {total_cost_bdt:.2f} BDT."
    )

    return " ".join(summary_parts)
