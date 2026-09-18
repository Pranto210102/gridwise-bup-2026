import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.schemas import (
    ScenarioRequest,
    OptimizationResponse,
    HealthResponse
)
from app.llm_interpreter import interpret_operator_notes
from app.guardrails import apply_guardrails
from app.optimizer import solve_energy_schedule
from app.summary_generator import generate_plan_summary

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("gridwise")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GridWise Energy Optimization API service started.")
    yield
    logger.info("GridWise Energy Optimization API service shutting down.")


app = FastAPI(
    title="GridWise — Smart Campus Energy Optimization API",
    description="LLM-Assisted Operator Directive Interpretation & 24-Hour Energy Scheduling",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for judging harness and web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return clean HTTP 400 for malformed or structurally invalid input."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Malformed JSON or structurally invalid request", "errors": exc.errors()}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Controlled HTTP 500 error handler.
    Strictly prevents leaking API keys, secrets, tokens, or raw stack traces.
    """
    logger.error(f"Internal server error processing {request.url.path}: {type(exc).__name__}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred while processing the energy schedule."}
    )


@app.get("/", tags=["Readiness"])
async def get_root():
    """Root endpoint to guarantee 200 OK for any default cloud health pings."""
    return {"status": "ok", "service": "GridWise Smart Campus Energy Optimization"}


@app.get("/healthz", response_model=HealthResponse, tags=["Readiness"])
@app.get("/health", response_model=HealthResponse, tags=["Readiness"])
async def get_health():
    """Readiness endpoint for the judging harness and cloud orchestrators."""
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizationResponse, tags=["Optimization"])
async def optimize_energy(req: ScenarioRequest):
    """
    Main endpoint supporting high concurrency:
    1. Interprets natural-language operator notes using LLM with caching and multi-key rotation
    2. Deterministically validates and repairs directives via guardrails
    3. Executes 24-hour MILP mathematical optimization in worker thread
    4. Returns exact machine-checkable interpretation and optimal hourly plan
    """
    # Step 1: LLM interpretation of operator notes
    raw_directives = await interpret_operator_notes(
        operator_notes=req.operator_notes,
        battery=req.battery
    )

    # Step 2: Deterministic guardrails and validation
    validated_directives = apply_guardrails(
        raw_interpretations=raw_directives,
        operator_notes=req.operator_notes,
        battery=req.battery
    )

    # Step 3: Mathematical MILP optimization offloaded to thread to prevent event-loop blocking
    hourly_plan, total_grid, total_cost, peak_grid = await asyncio.to_thread(
        solve_energy_schedule,
        hours=req.hours,
        battery=req.battery,
        directives=validated_directives
    )

    # Step 4: Strategy explanation summary
    plan_summary = generate_plan_summary(
        directives=validated_directives,
        hourly_plan=hourly_plan,
        battery=req.battery,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid
    )

    return OptimizationResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=validated_directives,
        hourly_plan=hourly_plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=plan_summary
    )
