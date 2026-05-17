"""SIMLAB MCP Server.

Exposes SIMLAB capabilities as MCP tools using the FastMCP pattern.
Run with: python -m simlab.mcp.simlab_mcp_server.server
Or via: simlab-mcp
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from simlab.core.engine.simlab_core import SimLabCore
from simlab.core.schemas.experiment import ExperimentRequest

mcp = FastMCP("SIMLAB")
_core = SimLabCore()


@mcp.tool()
async def run_experiment(request: dict) -> dict:
    """Run a SIMLAB experiment.

    Pass a dict matching the ExperimentRequest schema:
    - domain: 'math' | 'physics' | 'chemistry' | 'materials' | 'atomic' |
      'nuclear' | 'data' | 'control' | 'hybrid'
    - type: experiment type (e.g. 'projectile_motion', 'solve_equation')
    - parameters: dict of domain-specific parameters
    - outputs: list of desired outputs, e.g. ['plot', 'json']
    - environment: optional environment overrides
    - solver: optional solver preference (default 'auto')

    Returns an ExperimentResult dict with results, plots (base64 PNG), warnings, errors.
    """
    try:
        req = ExperimentRequest(**request)
        result = _core.run_experiment(req)
        return result.model_dump()
    except Exception as exc:
        return {"status": "error", "errors": [str(exc)]}


@mcp.tool()
async def solve_math(
    domain_type: str,
    parameters: dict,
    outputs: list[str] | None = None,
) -> dict:
    """Solve a math problem.

    Parameters
    ----------
    domain_type:
        One of: 'solve_equation', 'differentiate', 'integrate', 'simplify',
        'expand', 'factor', 'critical_points', 'taylor_series',
        'solve_ode', 'solve_linear', 'eigenvalues', 'svd', 'determinant',
        'matrix_multiply', 'rank', 'optimize', 'maximize',
        'monte_carlo', 'descriptive_stats', 'fit_distribution', 'hypothesis_test'
    parameters:
        Domain-specific parameters. Examples:
        - solve_equation: {"equation": "x**2 - 4", "variable": "x"}
        - differentiate: {"expression": "sin(x)*x**2", "variable": "x", "order": 1}
        - integrate: {"expression": "x**2", "limits": [0, 1]}
        - eigenvalues: {"matrix": [[1,2],[3,4]]}
        - solve_ode: {"equation": "-0.5*y", "y0": 1.0, "t_span": [0, 10]}
    outputs:
        Optional list: ['plot'] to include visualization PNG.

    Returns an ExperimentResult dict.
    """
    result = _core.solve_math(domain_type, parameters, outputs or [])
    return result.model_dump()


@mcp.tool()
async def simulate_physics(
    simulation_type: str,
    parameters: dict,
    outputs: list[str] | None = None,
    environment: dict | None = None,
) -> dict:
    """Run a physics simulation.

    Parameters
    ----------
    simulation_type:
        One of: 'projectile_motion', 'pendulum', 'spring_mass', 'collision',
        'gravity', 'circular_motion', 'electric_field', 'coulomb_force',
        'magnetic_field', 'capacitance', 'heat_transfer', 'ideal_gas',
        'entropy', 'carnot_efficiency'
    parameters:
        Simulation parameters. Examples:
        - projectile_motion: {"v0": 25.0, "angle_deg": 45.0}
        - pendulum: {"length": 1.0, "theta0_deg": 30.0, "t_span": 10}
        - spring_mass: {"mass": 1.0, "k": 10.0, "x0": 0.1, "v0": 0.0}
        - electric_field: {"charges": [1e-9, -1e-9], "positions": [[0,0],[2,0]]}
        - heat_transfer: {"T_hot": 400, "T_cold": 300, "k_conductivity": 50, "area": 0.01, "thickness": 0.005}
    environment:
        Optional overrides like {"g": 1.62} for lunar gravity.
    outputs:
        Optional list: ['plot'] to include trajectory/time-series plots.

    Returns an ExperimentResult dict.
    """
    result = _core.simulate_physics(simulation_type, parameters, outputs or [], environment or {})
    return result.model_dump()


@mcp.tool()
async def simulate_chemistry(
    simulation_type: str,
    parameters: dict,
    outputs: list[str] | None = None,
) -> dict:
    """Run a chemistry simulation.

    Parameters
    ----------
    simulation_type:
        One of: 'molecule_analysis', 'molecular_descriptors', 'parse_smiles',
        'molecule_image', '3d_coordinates',
        'first_order', 'second_order', 'consecutive', 'michaelis_menten',
        'equilibrium', 'reaction_kinetics'
    parameters:
        Examples:
        - molecule_analysis: {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}
        - first_order: {"k": 0.25, "A0": 1.0, "t_span": 20}
        - michaelis_menten: {"Vmax": 1.0, "Km": 0.5, "S0": 2.0}
    outputs:
        Optional list: ['plot'] for kinetics time series.

    Returns an ExperimentResult dict.
    """
    result = _core.simulate_chemistry(simulation_type, parameters, outputs or [])
    return result.model_dump()


@mcp.tool()
async def simulate_materials(
    simulation_type: str,
    parameters: dict,
    outputs: list[str] | None = None,
) -> dict:
    """Run a materials science simulation.

    Parameters
    ----------
    simulation_type:
        One of: 'create_lattice', 'fcc_lattice', 'bcc_lattice', 'simple_cubic',
        'stress_test', 'youngs_modulus', 'elastic_deformation', 'stress_strain_curve'
    parameters:
        Examples:
        - create_lattice: {"lattice_type": "fcc", "a": 3.52, "n_cells": 2}
        - elastic_deformation: {"E": 200e9, "force": 10000, "area": 0.01, "L0": 1.0}
        - stress_strain_curve: {"E": 200e9, "yield_stress": 250e6, "ultimate_stress": 400e6}
    outputs:
        Optional list: ['plot'] for stress-strain curves.

    Returns an ExperimentResult dict.
    """
    result = _core.simulate_materials(simulation_type, parameters, outputs or [])
    return result.model_dump()


@mcp.tool()
async def simulate_atomic(
    simulation_type: str,
    parameters: dict,
    outputs: list[str] | None = None,
) -> dict:
    """Run an atomic or element-building simulation.

    simulation_type examples: 'electron_config', 'create_element',
    'hydrogen_energy_levels', 'radial_normalization', 'shell_diagram',
    'orbital_diagram'
    """
    result = _core.simulate_atomic(simulation_type, parameters, outputs or [])
    return result.model_dump()


@mcp.tool()
async def simulate_nuclear(
    simulation_type: str,
    parameters: dict,
    outputs: list[str] | None = None,
) -> dict:
    """Run a nuclear physics simulation.

    simulation_type examples: 'analyze_nucleus', 'fusion_energy',
    'fission_energy', 'decay', 'nuclear_chart'
    """
    result = _core.simulate_nuclear(simulation_type, parameters, outputs or [])
    return result.model_dump()


@mcp.tool()
async def query_data(query_type: str, parameters: dict) -> dict:
    """Query scientific data sources.

    query_type examples: 'arxiv_search', 'arxiv_paper', 'arxiv_references',
    'pubchem_search', 'pubchem_by_cid'
    """
    result = _core.query_data(query_type, parameters)
    return result.model_dump()


@mcp.tool()
async def analyze_control(
    analysis_type: str,
    parameters: dict,
    outputs: list[str] | None = None,
) -> dict:
    """Run a control-systems analysis.

    analysis_type examples: 'transfer_function', 'pid_controller',
    'step_response', 'bode', 'root_locus'
    """
    result = _core.analyze_control(analysis_type, parameters, outputs or [])
    return result.model_dump()


@mcp.tool()
async def generate_visualization(viz_type: str, data: dict) -> dict:
    """Generate a standalone visualization.

    Parameters
    ----------
    viz_type:
        One of: 'plot_function', 'plot_data', 'vector_field', 'molecule', 'trajectory',
        'time_series', 'phase_portrait', 'histogram', 'electric_field', 'magnetic_field'
    data:
        Visualization-specific parameters. Examples:
        - plot_function: {"expr": "sin(x)*exp(-0.1*x)", "x_range": [-10, 10], "title": "Damped Sine"}
        - plot_data: {"x": [1,2,3], "y": [1,4,9], "title": "My Data", "plot_type": "scatter"}
        - trajectory: {"x": [...], "y": [...], "title": "Projectile"}
        - time_series: {"t": [...], "y_arrs": [[...], [...]], "labels": ["A", "B"]}
        - electric_field: {"charges": [1e-9, -1e-9], "positions": [[0,0],[2,0]]}
        - molecule: {"smiles": "C1=CC=CC=C1"}

    Returns {"plot_b64": "<base64 PNG string>"}
    """
    _REQUIRED_KEYS: dict[str, list[str]] = {
        "plot_function": ["expr"],
        "plot_data": ["x", "y"],
        "trajectory": ["x", "y"],
        "time_series": ["t", "y_arrs", "labels"],
        "phase_portrait": ["x", "y"],
        "histogram": ["data"],
        "vector_field": ["X", "U", "V"],
        "electric_field": ["charges", "positions"],
        "magnetic_field": ["wire_data"],
        "molecule": ["smiles"],
        "molecule_grid": ["smiles_list"],
    }

    if viz_type not in _REQUIRED_KEYS:
        return {"error": f"Unknown viz_type: {viz_type!r}. Valid types: {list(_REQUIRED_KEYS)}"}

    missing = [k for k in _REQUIRED_KEYS[viz_type] if k not in data]
    if missing:
        return {"error": f"viz_type={viz_type!r} missing required keys: {missing}"}

    try:
        from simlab.engines.visualization.plots import PlotEngine
        from simlab.engines.visualization.vector_fields import VectorFieldEngine
        from simlab.engines.visualization.molecule_viewer import MoleculeViewer

        plot_engine = PlotEngine()
        vf_engine = VectorFieldEngine()
        mol_viewer = MoleculeViewer()

        if viz_type == "plot_function":
            b64 = plot_engine.plot_function(
                data["expr"],
                data.get("x_range", [-10, 10]),
                title=data.get("title", ""),
            )
        elif viz_type == "plot_data":
            b64 = plot_engine.plot_data(
                data["x"], data["y"],
                title=data.get("title", ""),
                xlabel=data.get("xlabel", "x"),
                ylabel=data.get("ylabel", "y"),
                plot_type=data.get("plot_type", "line"),
            )
        elif viz_type == "trajectory":
            b64 = plot_engine.plot_trajectory(
                data["x"], data["y"],
                title=data.get("title", "Trajectory"),
            )
        elif viz_type == "time_series":
            b64 = plot_engine.plot_time_series(
                data["t"], data["y_arrs"], data["labels"],
                title=data.get("title", ""),
            )
        elif viz_type == "phase_portrait":
            b64 = plot_engine.plot_phase_portrait(
                data["x"], data["y"],
                title=data.get("title", "Phase Portrait"),
            )
        elif viz_type == "histogram":
            b64 = plot_engine.plot_histogram(
                data["data"],
                bins=data.get("bins", 30),
                title=data.get("title", "Histogram"),
            )
        elif viz_type == "vector_field":
            b64 = vf_engine.plot_vector_field(
                data["X"], data["U"], data["V"],
                Y=data.get("Y"),
                title=data.get("title", "Vector Field"),
            )
        elif viz_type == "electric_field":
            b64 = vf_engine.plot_electric_field(
                data["charges"], data["positions"],
                grid_extent=data.get("grid_extent", 5.0),
            )
        elif viz_type == "magnetic_field":
            b64 = vf_engine.plot_magnetic_field(
                data["wire_data"],
                grid_extent=data.get("grid_extent", 5.0),
            )
        elif viz_type == "molecule":
            b64 = mol_viewer.render_2d_molecule(
                data["smiles"],
                size=tuple(data.get("size") or [400, 300]),
            )
        elif viz_type == "molecule_grid":
            b64 = mol_viewer.render_molecule_grid(
                data["smiles_list"],
                legends=data.get("legends"),
            )

        return {"plot_b64": b64, "viz_type": viz_type}

    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
async def export_report(experiment_result: dict, format: str = "markdown") -> str:
    """Export an experiment result as a formatted report.

    Parameters
    ----------
    experiment_result:
        An ExperimentResult dict (as returned by other SIMLAB tools).
    format:
        'markdown', 'html', or 'json'

    Returns the report as a string.
    """
    from simlab.core.schemas.experiment import ExperimentResult
    from simlab.engines.visualization.reports import ReportEngine

    try:
        result = ExperimentResult(**experiment_result)
        engine = ReportEngine()
        if format == "markdown":
            return engine.generate_markdown_report(result)
        elif format == "html":
            return engine.generate_html_report(result)
        elif format == "json":
            return engine.export_json(result)
        else:
            return f"Unknown format: {format!r}. Use 'markdown', 'html', or 'json'."
    except Exception as exc:
        return f"Error generating report: {exc}"


@mcp.tool()
async def check_safety(
    domain: str,
    simulation_type: str,
    parameters: dict,
) -> dict:
    """Check if a simulation request is safe and valid before running it.

    Returns a ValidationResult dict with:
    - valid: bool — whether the request can proceed
    - blocked: bool — whether it is outright blocked
    - warnings: list of warning messages
    - errors: list of error/validation failure messages
    - checks_passed: list of passing check names

    Parameters
    ----------
    domain:          experiment domain ('math', 'physics', 'chemistry', etc.)
    simulation_type: experiment type
    parameters:      parameter dict
    """
    try:
        req = ExperimentRequest(
            domain=domain,
            type=simulation_type,
            parameters=parameters,
        )
        validation = _core._validator.validate_experiment(req)
        return validation.model_dump()
    except Exception as exc:
        return {"valid": False, "blocked": False, "errors": [str(exc)], "warnings": []}


def main():
    """Entry point for simlab-mcp CLI."""
    mcp.run()


if __name__ == "__main__":
    main()
