from typing import List, Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op"
]

BatteryAction = Literal["charge", "discharge", "idle"]


class HourInput(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Unique integer from 0 to 23")
    demand_kwh: float = Field(..., ge=0, description="Campus demand that must be supplied in this hour")
    solar_kwh: float = Field(..., ge=0, description="Base solar energy available before operator-note adjustments")
    tariff_bdt_per_kwh: float = Field(..., ge=0, description="Grid electricity price for this hour")


class BatteryInput(BaseModel):
    capacity_kwh: float = Field(..., gt=0, description="Maximum energy the battery can store")
    initial_energy_kwh: float = Field(..., ge=0, description="Battery energy at the start of hour 0")
    minimum_energy_kwh: float = Field(..., ge=0, description="Base reserve level the battery must never go below")
    max_charge_kwh_per_hour: float = Field(..., ge=0, description="Maximum energy that may be added in one hour")
    max_discharge_kwh_per_hour: float = Field(..., ge=0, description="Maximum energy that may be removed in one hour")

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            raise ValueError("initial_energy_kwh cannot be less than minimum_energy_kwh")
        return self


class ScenarioRequest(BaseModel):
    scenario_id: str = Field(..., min_length=1, description="Unique scenario identifier")
    operator_notes: List[str] = Field(..., min_length=1, max_length=3, description="1 to 3 natural-language operator notes")
    hours: List[HourInput] = Field(..., min_length=24, max_length=24, description="Hourly profile for exactly 24 hours")
    battery: BatteryInput

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, v: List[HourInput]):
        if len(v) != 24:
            raise ValueError("Exactly 24 hours required")
        seen_hours = [h.hour for h in v]
        if seen_hours != list(range(24)):
            raise ValueError("hours array must contain unique hours from 0 to 23 in ascending order")
        return v


class DirectiveInterpretation(BaseModel):
    note_index: int = Field(..., ge=0, description="Zero-based index of operator note")
    applies: bool = Field(..., description="true for applicable directive, false only for no_op")
    directive_type: DirectiveType
    structured_adjustment: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Structured adjustment object, or null only for no_op"
    )
    explanation: str = Field(..., description="Short explanation of the interpretation")

    @model_validator(mode="after")
    def validate_applies_and_adjustment(self):
        if self.directive_type == "no_op":
            if self.applies is not False:
                self.applies = False
            self.structured_adjustment = None
        else:
            if self.applies is not True:
                self.applies = True
        return self


class HourlyPlanEntry(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float = Field(..., ge=0, description="Grid energy purchased in this hour")
    solar_used_kwh: float = Field(..., ge=0, description="Solar energy used in this hour")
    battery_action: BatteryAction
    battery_kwh: float = Field(..., ge=0, description="Magnitude of battery action, 0 when idle")
    battery_energy_after_kwh: float = Field(..., ge=0, description="Battery energy after this hour")


class OptimizationResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
