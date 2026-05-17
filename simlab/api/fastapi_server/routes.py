"""SIMLAB FastAPI route definitions.

All endpoints mirror the MCP tool interface, returning JSON-serializable dicts.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simlab.core.constants.physical import CONSTANTS
from simlab.core.engine.simlab_core import SimLabCore
from simlab.core.schemas.experiment import ExperimentRequest

router = APIRouter()
_core = SimLabCore()


# --- Request models for shortcut endpoints ---

class MathRequest(BaseModel):
    domain_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)


class PhysicsRequest(BaseModel):
    simulation_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)


class ChemistryRequest(BaseModel):
    simulation_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)


class MaterialsRequest(BaseModel):
    simulation_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)


class AtomicRequest(BaseModel):
    simulation_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)


class NuclearRequest(BaseModel):
    simulation_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)


class DataRequest(BaseModel):
    query_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class NvidiaRequest(BaseModel):
    simulation_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)


class ControlRequest(BaseModel):
    analysis_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)


class VisualizeRequest(BaseModel):
    viz_type: str
    data: dict[str, Any] = Field(default_factory=dict)


class ReportRequest(BaseModel):
    experiment_result: dict[str, Any]
    format: str = "markdown"


class SafetyRequest(BaseModel):
    domain: str
    simulation_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


# --- Routes ---

@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "SIMLAB API",
        "version": "0.1.0",
    }


@router.get("/constants")
async def get_constants() -> dict:
    """Return all physical constants."""
    return {
        "constants": CONSTANTS.as_dict(),
        "units": {
            "G": "m^3 kg^-1 s^-2",
            "c": "m/s",
            "h": "J·s",
            "hbar": "J·s",
            "k_B": "J/K",
            "N_A": "mol^-1",
            "R": "J mol^-1 K^-1",
            "e": "C",
            "epsilon_0": "F/m",
            "mu_0": "H/m",
            "m_e": "kg",
            "m_p": "kg",
            "sigma_SB": "W m^-2 K^-4",
            "g_earth": "m/s^2",
            "g_moon": "m/s^2",
            "g_mars": "m/s^2",
        },
    }


@router.post("/experiment")
async def run_experiment(request: ExperimentRequest) -> dict:
    """Run a full SIMLAB experiment.

    Pass an ExperimentRequest body. Returns an ExperimentResult dict.
    """
    try:
        result = _core.run_experiment(request)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/math")
async def solve_math(request: MathRequest) -> dict:
    """Solve a math problem.

    domain_type examples: 'solve_equation', 'differentiate', 'integrate',
    'eigenvalues', 'solve_ode', 'monte_carlo', 'descriptive_stats'
    """
    try:
        result = _core.solve_math(request.domain_type, request.parameters, request.outputs)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/physics")
async def simulate_physics(request: PhysicsRequest) -> dict:
    """Run a physics simulation.

    simulation_type examples: 'projectile_motion', 'pendulum', 'spring_mass',
    'electric_field', 'heat_transfer', 'carnot_efficiency'
    """
    try:
        result = _core.simulate_physics(
            request.simulation_type, request.parameters,
            request.outputs, request.environment
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chemistry")
async def simulate_chemistry(request: ChemistryRequest) -> dict:
    """Run a chemistry simulation.

    simulation_type examples: 'molecule_analysis', 'first_order',
    'michaelis_menten', 'equilibrium'
    """
    try:
        result = _core.simulate_chemistry(
            request.simulation_type, request.parameters, request.outputs
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/materials")
async def simulate_materials(request: MaterialsRequest) -> dict:
    """Run a materials science simulation.

    simulation_type examples: 'create_lattice', 'stress_strain_curve',
    'elastic_deformation'
    """
    try:
        result = _core.simulate_materials(
            request.simulation_type, request.parameters, request.outputs
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/atomic")
async def simulate_atomic(request: AtomicRequest) -> dict:
    """Run an atomic or element-building simulation.

    simulation_type examples: 'electron_config', 'create_element',
    'hydrogen_energy_levels', 'shell_diagram'
    """
    try:
        result = _core.simulate_atomic(
            request.simulation_type, request.parameters, request.outputs
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/nuclear")
async def simulate_nuclear(request: NuclearRequest) -> dict:
    """Run a nuclear physics simulation.

    simulation_type examples: 'analyze_nucleus', 'fusion_energy',
    'fission_energy', 'decay', 'nuclear_chart'
    """
    try:
        result = _core.simulate_nuclear(
            request.simulation_type, request.parameters, request.outputs
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/data")
async def query_data(request: DataRequest) -> dict:
    """Query scientific data sources.

    query_type examples: 'arxiv_search', 'arxiv_paper', 'pubchem_search'
    """
    try:
        result = _core.query_data(request.query_type, request.parameters)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/control")
async def analyze_control(request: ControlRequest) -> dict:
    """Run a control-systems analysis.

    analysis_type examples: 'transfer_function', 'pid_controller',
    'step_response', 'bode'
    """
    try:
        result = _core.analyze_control(
            request.analysis_type, request.parameters, request.outputs
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/nvidia")
async def simulate_nvidia(request: NvidiaRequest) -> dict:
    """Run an NVIDIA GPU-accelerated simulation.

    simulation_type examples: 'alchemi_relax', 'alchemi_md', 'alchemi_batch_screen',
    'warp_diffusion', 'warp_allen_cahn_field', 'warp_diffusion_3d',
    'nemo_train_fno', 'nemo_sampling_plan', 'cupy_diffusion_sweep', 'gpu_diagnostics'
    """
    try:
        result = _core.simulate_nvidia(
            request.simulation_type, request.parameters, request.outputs
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/gpu")
async def gpu_status() -> dict:
    """Return GPU device info, VRAM usage, and NVIDIA backend availability.

    Reports all detected CUDA devices and whether warp, nvalchemiops, cupy,
    cupynumeric, physicsnemo, and torch are installed.
    """
    try:
        from simlab.engines.materials.nvidia_backends import GPUDiagnosticsBackend
        return GPUDiagnosticsBackend().gpu_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/visualize")
async def generate_visualization(request: VisualizeRequest) -> dict:
    """Generate a standalone visualization.

    viz_type examples: 'plot_function', 'plot_data', 'electric_field', 'molecule'
    Returns {"plot_b64": "<base64 PNG>"}
    """
    try:
        from simlab.engines.visualization.plots import PlotEngine
        from simlab.engines.visualization.vector_fields import VectorFieldEngine
        from simlab.engines.visualization.molecule_viewer import MoleculeViewer

        data = request.data
        viz_type = request.viz_type

        plot_engine = PlotEngine()
        vf_engine = VectorFieldEngine()
        mol_viewer = MoleculeViewer()

        if viz_type == "plot_function":
            b64 = plot_engine.plot_function(
                data["expr"], data.get("x_range", [-10, 10]), title=data.get("title", "")
            )
        elif viz_type == "plot_data":
            b64 = plot_engine.plot_data(
                data["x"], data["y"],
                title=data.get("title", ""),
                plot_type=data.get("plot_type", "line"),
            )
        elif viz_type == "trajectory":
            b64 = plot_engine.plot_trajectory(data["x"], data["y"], title=data.get("title", ""))
        elif viz_type == "time_series":
            b64 = plot_engine.plot_time_series(
                data["t"], data["y_arrs"], data["labels"], title=data.get("title", "")
            )
        elif viz_type == "electric_field":
            b64 = vf_engine.plot_electric_field(
                data["charges"], data["positions"],
                grid_extent=data.get("grid_extent", 5.0),
            )
        elif viz_type == "magnetic_field":
            b64 = vf_engine.plot_magnetic_field(
                data["wire_data"], grid_extent=data.get("grid_extent", 5.0)
            )
        elif viz_type == "molecule":
            b64 = mol_viewer.render_2d_molecule(
                data["smiles"], size=tuple(data.get("size", [400, 300]))
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown viz_type: {viz_type!r}")

        return {"plot_b64": b64, "viz_type": viz_type}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/report")
async def export_report(request: ReportRequest) -> dict:
    """Export an experiment result as a formatted report.

    format: 'markdown', 'html', or 'json'
    Returns {"report": "...", "format": "..."}
    """
    try:
        from simlab.core.schemas.experiment import ExperimentResult
        from simlab.engines.visualization.reports import ReportEngine

        result = ExperimentResult(**request.experiment_result)
        engine = ReportEngine()
        fmt = request.format

        if fmt == "markdown":
            report = engine.generate_markdown_report(result)
        elif fmt == "html":
            report = engine.generate_html_report(result)
        elif fmt == "json":
            report = engine.export_json(result)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown format: {fmt!r}")

        return {"report": report, "format": fmt}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/safety")
async def check_safety(request: SafetyRequest) -> dict:
    """Check if a simulation request is safe before running it."""
    try:
        req = ExperimentRequest(
            domain=request.domain,
            type=request.simulation_type,
            parameters=request.parameters,
        )
        validation = _core._validator.validate_experiment(req)
        return validation.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/experiments")
async def list_experiments(domain: str | None = None) -> dict:
    """List all available experiment types, optionally filtered by domain."""
    available = _core.available_experiments(domain)
    by_domain: dict[str, list[str]] = {}
    for dom, typ in available:
        by_domain.setdefault(dom, []).append(typ)
    return {"experiments": by_domain, "total": len(available)}
