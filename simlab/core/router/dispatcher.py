"""Experiment dispatcher.

Routes ExperimentRequest objects to the appropriate engine method based on
(domain, type) key, then optionally attaches visualizations.
"""

from __future__ import annotations

from typing import Any, Callable

from simlab.core.schemas.experiment import ExperimentRequest, ExperimentResult

# --- Engine lazy imports wrapped in callables to avoid import cost at startup ---

def _get_symbolic():
    from simlab.engines.math.symbolic import SymbolicEngine
    return SymbolicEngine()

def _get_calculus():
    from simlab.engines.math.calculus import CalculusEngine
    return CalculusEngine()

def _get_linear_algebra():
    from simlab.engines.math.linear_algebra import LinearAlgebraEngine
    return LinearAlgebraEngine()

def _get_ode_solver():
    from simlab.engines.math.ode_solver import ODESolver
    return ODESolver()

def _get_optimization():
    from simlab.engines.math.optimization import OptimizationEngine
    return OptimizationEngine()

def _get_statistics():
    from simlab.engines.math.statistics import StatisticsEngine
    return StatisticsEngine()

def _get_classical():
    from simlab.engines.physics.classical import ClassicalMechanicsEngine
    return ClassicalMechanicsEngine()

def _get_em():
    from simlab.engines.physics.electromagnetism import ElectromagnetismEngine
    return ElectromagnetismEngine()

def _get_thermo():
    from simlab.engines.physics.thermodynamics import ThermodynamicsEngine
    return ThermodynamicsEngine()

def _get_rdkit():
    from simlab.engines.chemistry.rdkit_backend import RDKitEngine
    return RDKitEngine()

def _get_kinetics():
    from simlab.engines.chemistry.kinetics import KineticsEngine
    return KineticsEngine()

def _get_lattice():
    from simlab.engines.materials.lattice import LatticeEngine
    return LatticeEngine()

def _get_stress_strain():
    from simlab.engines.materials.stress_strain import StressStrainEngine
    return StressStrainEngine()

def _get_pymatgen():
    from simlab.engines.materials.pymatgen_backend import PymatgenEngine
    return PymatgenEngine()

def _get_materials_gpu_workflow():
    from simlab.engines.materials.gpu_workflow import MaterialsGPUWorkflowEngine
    return MaterialsGPUWorkflowEngine()

def _get_alchemi():
    from simlab.engines.materials.nvidia_backends import ALCHEMIBackend
    return ALCHEMIBackend()

def _get_warp():
    from simlab.engines.materials.nvidia_backends import WarpBackend
    return WarpBackend()

def _get_physicsnemo():
    from simlab.engines.materials.nvidia_backends import PhysicsNeMoBackend
    return PhysicsNeMoBackend()

def _get_cupy():
    from simlab.engines.materials.nvidia_backends import CuPyBackend
    return CuPyBackend()

def _get_diffusion():
    from simlab.engines.materials.diffusion import DiffusionEngine
    return DiffusionEngine()

def _get_phase_transformation():
    from simlab.engines.materials.phase_transformation import PhaseTransformationEngine
    return PhaseTransformationEngine()

def _get_oxidation_stability():
    from simlab.engines.materials.oxidation_stability import OxidationStabilityEngine
    return OxidationStabilityEngine()

def _get_forging():
    from simlab.engines.materials.forging import ForgingEngine
    return ForgingEngine()

def _get_casting():
    from simlab.engines.materials.casting import CastingEngine
    return CastingEngine()

def _get_electron_config():
    from simlab.engines.atomic.electron_config import ElectronConfigEngine
    return ElectronConfigEngine()

def _get_hydrogen_orbitals():
    from simlab.engines.atomic.hydrogen_orbitals import HydrogenOrbitalEngine
    return HydrogenOrbitalEngine()

def _get_ase():
    from simlab.engines.atomic.ase_backend import ASECrystalEngine
    return ASECrystalEngine()

def _get_element_builder():
    from simlab.engines.nuclear.element_builder import ElementBuilder
    return ElementBuilder()

def _get_nuclear():
    from simlab.engines.nuclear.nuclear_engine import NuclearEngine
    return NuclearEngine()

def _get_decay():
    from simlab.engines.nuclear.decay import DecayEngine
    return DecayEngine()

def _get_arxiv():
    from simlab.engines.data.arxiv_engine import ArXivEngine
    return ArXivEngine()

def _get_pubchem():
    from simlab.engines.data.pubchem_engine import PubChemEngine
    return PubChemEngine()

def _get_control():
    from simlab.engines.physics.control_systems import ControlSystemsEngine
    return ControlSystemsEngine()

def _get_fluid():
    from simlab.engines.physics.fluid_dynamics import FluidDynamicsEngine
    return FluidDynamicsEngine()

def _get_plot_engine():
    from simlab.engines.visualization.plots import PlotEngine
    return PlotEngine()

def _get_vector_field_engine():
    from simlab.engines.visualization.vector_fields import VectorFieldEngine
    return VectorFieldEngine()

def _get_gpu_diagnostics():
    from simlab.engines.materials.nvidia_backends import GPUDiagnosticsBackend
    return GPUDiagnosticsBackend()

def _get_calphad():
    from simlab.engines.materials.calphad_backend import CALPHADBackend
    import os, importlib.util
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "engines", "materials", "materials_config.yaml"
    )
    db_path = None
    try:
        if importlib.util.find_spec("yaml") is not None:
            import yaml
            with open(os.path.normpath(cfg_path)) as f:
                cfg = yaml.safe_load(f) or {}
            db_path = (cfg.get("calphad") or {}).get("database_path") or None
    except Exception:
        pass
    return CALPHADBackend(database_path=db_path)

def _get_dft():
    from simlab.engines.materials.dft_backend import DFTBackend
    return DFTBackend(calculator="emt")

def _get_mlip():
    from simlab.engines.materials.mlip_backend import MLIPBackend
    return MLIPBackend()

def _get_qdgeometry():
    from simlab.engines.qdgeometry.primitives import (
        linear_extrusion, revolution, sweep, fillet_box,
        boolean_union, boolean_diff, total_constraint_loss,
    )
    from simlab.engines.qdgeometry import mapelites as _me

    from simlab.engines.qdgeometry import sail as _sail
    from simlab.engines.qdgeometry import grammar as _grammar
    from simlab.engines.qdgeometry import jax_primitives as _jaxp
    from simlab.engines.qdgeometry import neural_implicit as _nimpl

    class _QDGeometryRouter:
        """Thin adapter so the dispatcher can call qdgeometry functions by name."""

        # ── MAP-Elites ─────────────────────────────────────────────────────
        def mapelites_alloy_search(self, **kwargs):
            return _me.mapelites_alloy_search(**kwargs)

        def mapelites_geometry_search(self, **kwargs):
            return _me.mapelites_geometry_search(**kwargs)

        # ── Primitives ─────────────────────────────────────────────────────
        def geometry_primitive(self, primitive: str = "linear_extrusion", **kwargs):
            fn = {
                "linear_extrusion": linear_extrusion,
                "revolution": revolution,
                "sweep": sweep,
                "fillet_box": fillet_box,
            }.get(primitive)
            if fn is None:
                raise ValueError(f"Unknown primitive {primitive!r}")
            geom = fn(**{k: v for k, v in kwargs.items() if k != "primitive"})
            return geom.to_dict()

        def constraint_loss(self, primitive: str = "linear_extrusion",
                            params: dict | None = None,
                            constraint_targets: dict | None = None):
            fn = {
                "linear_extrusion": linear_extrusion,
                "revolution": revolution,
                "sweep": sweep,
                "fillet_box": fillet_box,
            }.get(primitive)
            if fn is None:
                raise ValueError(f"Unknown primitive {primitive!r}")
            geom = fn(**(params or {}))
            ct = constraint_targets or {}
            return total_constraint_loss(
                geom,
                target_volume_m3=ct.get("target_volume_m3"),
                target_aspect_ratio=ct.get("target_aspect_ratio"),
                min_wall_m=ct.get("min_wall_m"),
                taper_angle_deg=ct.get("taper_angle_deg"),
                min_draft_deg=ct.get("min_draft_deg", 1.0),
            )

        # ── Gradient-descent optimiser (JAX autodiff) ──────────────────────
        def gradient_descent(self, **kwargs):
            return _jaxp.gradient_descent_geometry(**kwargs)

        # ── SAIL (surrogate-assisted illumination) ─────────────────────────
        def sail_geometry_search(self, **kwargs):
            return _sail.sail_geometry_search(**kwargs)

        def sail_alloy_search(self, **kwargs):
            return _sail.sail_alloy_search(**kwargs)

        # ── Grammar-guided GP ──────────────────────────────────────────────
        def grammar_guided_search(self, **kwargs):
            return _grammar.grammar_guided_search(**kwargs)

        # ── Neural implicit field ──────────────────────────────────────────
        def fit_neural_implicit(self, **kwargs):
            return _nimpl.fit_to_geometry(**kwargs)

        # ── GPU multi-physics blade simulation (PyTorch) ──────────────────
        def blade_simulation(self, **kwargs):
            from simlab.engines.qdgeometry.torch_gpu_physics import gpu_blade_simulation
            result = gpu_blade_simulation(**kwargs)
            result.pop("_fields", None)
            return result

        # ── Warp @wp.kernel FD thermal solver ─────────────────────────────
        def warp_blade_thermal(self, **kwargs):
            from simlab.engines.qdgeometry.warp_blade import warp_blade_thermal
            return warp_blade_thermal(**kwargs)

        # ── FNO2d surrogate: generate training data ────────────────────────
        def fno_generate_data(self, **kwargs):
            from simlab.engines.qdgeometry.fno_surrogate import generate_training_data
            return generate_training_data(**kwargs)

        # ── FNO2d surrogate: train ─────────────────────────────────────────
        def fno_train(self, **kwargs):
            from simlab.engines.qdgeometry.fno_surrogate import (
                generate_training_data, train_fno
            )
            dataset = kwargs.pop("dataset", None)
            if dataset is None:
                n_samples = kwargs.pop("n_samples", 100)
                dataset = generate_training_data(n_samples=n_samples,
                                                 verbose=True, **{
                    k: v for k, v in kwargs.items()
                    if k in ("solver", "nx", "ny", "seed", "n_holes_range")
                })
            train_kwargs = {k: v for k, v in kwargs.items()
                           if k in ("epochs", "batch_size", "lr", "pde_weight",
                                    "modes", "width", "n_layers", "dropout",
                                    "val_split", "verbose")}
            model, history = train_fno(dataset, **train_kwargs)
            # Store model in result dict (not serialisable but usable in-process)
            history["_model"] = model
            return history

        # ── CalculiX FEM validation ────────────────────────────────────────
        def calculix_validate(self, **kwargs):
            from simlab.engines.qdgeometry.calculix_validate import validate_fd_vs_fem
            return validate_fd_vs_fem(**kwargs)

        def calculix_thermal(self, **kwargs):
            from simlab.engines.qdgeometry.calculix_validate import calculix_thermal
            return calculix_thermal(**kwargs)

    return _QDGeometryRouter()


# ---------------------------------------------------------------------------
# Routing table: (domain, type) -> (engine_factory, method_name, param_map)
#
# param_map: optional dict mapping method kwarg names to parameter dict keys.
#            If None, parameters dict is passed as **kwargs directly.
# ---------------------------------------------------------------------------

RoutingEntry = tuple[Callable, str, dict[str, str] | None]

_ROUTING_TABLE: dict[tuple[str, str], RoutingEntry] = {
    # --- Math: Symbolic ---
    ("math", "solve_equation"):   (_get_symbolic, "solve_equation",   {"equation_str": "equation"}),
    ("math", "simplify"):         (_get_symbolic, "simplify_expression", {"expr_str": "expression"}),
    ("math", "differentiate"):    (_get_symbolic, "differentiate",    {"expr_str": "expression"}),
    ("math", "integrate"):        (_get_symbolic, "integrate",        {"expr_str": "expression"}),
    ("math", "expand"):           (_get_symbolic, "expand_expression",{"expr_str": "expression"}),
    ("math", "factor"):           (_get_symbolic, "factor_expression",{"expr_str": "expression"}),
    ("math", "critical_points"):  (_get_symbolic, "find_critical_points", {"expr_str": "expression"}),
    ("math", "taylor_series"):    (_get_symbolic, "taylor_series",    {"expr_str": "expression"}),
    # --- Math: ODE ---
    ("math", "solve_ode"):        (_get_ode_solver, "solve_ode",      {"f_str": "equation"}),
    ("math", "solve_system_ode"): (_get_ode_solver, "solve_system",   None),
    ("math", "solve_2nd_order"):  (_get_ode_solver, "solve_second_order", None),
    # --- Math: Linear Algebra ---
    ("math", "solve_linear"):     (_get_linear_algebra, "solve_linear_system", None),
    ("math", "eigenvalues"):      (_get_linear_algebra, "compute_eigenvalues", {"A": "matrix"}),
    ("math", "svd"):              (_get_linear_algebra, "compute_svd",         {"A": "matrix"}),
    ("math", "determinant"):      (_get_linear_algebra, "compute_determinant", {"A": "matrix"}),
    ("math", "inverse"):          (_get_linear_algebra, "compute_inverse",     {"A": "matrix"}),
    ("math", "matrix_multiply"):  (_get_linear_algebra, "matrix_multiply",     None),
    ("math", "rank"):             (_get_linear_algebra, "compute_rank",        {"A": "matrix"}),
    # --- Math: Optimization ---
    ("math", "optimize"):         (_get_optimization, "minimize_function", {"f_str": "expression", "x0": "x0"}),
    ("math", "maximize"):         (_get_optimization, "maximize_function", {"f_str": "expression", "x0": "x0"}),
    ("math", "fit_curve"):        (_get_optimization, "fit_curve",        None),
    ("math", "minimize_interval"):(_get_optimization, "find_minimum_on_interval", {"f_str": "expression"}),
    # --- Math: Statistics ---
    ("math", "monte_carlo"):      (_get_statistics, "run_monte_carlo",    {"f_str": "expression"}),
    ("math", "descriptive_stats"):(_get_statistics, "descriptive_stats",  None),
    ("math", "fit_distribution"): (_get_statistics, "fit_distribution",   None),
    ("math", "hypothesis_test"):  (_get_statistics, "hypothesis_test",    None),
    # --- Physics: Classical ---
    ("physics", "projectile_motion"):    (_get_classical, "simulate_projectile_motion",   None),
    ("physics", "projectile_drag"):      (_get_classical, "simulate_projectile_with_drag", None),
    ("physics", "relativistic"):         (_get_classical, "relativistic_kinematics",       None),
    ("physics", "pendulum"):             (_get_classical, "simulate_pendulum",              None),
    ("physics", "spring_mass"):          (_get_classical, "simulate_spring_mass",           None),
    ("physics", "collision"):            (_get_classical, "simulate_collision",             None),
    ("physics", "gravity"):              (_get_classical, "simulate_gravity",               None),
    ("physics", "circular_motion"):      (_get_classical, "simulate_circular_motion",      None),
    # --- Physics: Fluid Dynamics ---
    ("physics", "bernoulli"):            (_get_fluid, "bernoulli",        None),
    ("physics", "pipe_flow"):            (_get_fluid, "pipe_flow",        None),
    ("physics", "reynolds_number"):      (_get_fluid, "reynolds_number",  None),
    ("physics", "terminal_velocity"):    (_get_fluid, "terminal_velocity",None),
    ("physics", "drag_force"):           (_get_fluid, "drag_force",       None),
    # --- Physics: Thermodynamics ---
    ("physics", "van_der_waals"):        (_get_thermo, "van_der_waals_gas",         None),
    # --- Physics: Electromagnetism ---
    ("physics", "electric_field"):   (_get_em, "simulate_electric_field", None),
    ("physics", "coulomb_force"):    (_get_em, "coulomb_force",           None),
    ("physics", "magnetic_field"):   (_get_em, "simulate_magnetic_field", None),
    ("physics", "capacitance"):      (_get_em, "calculate_capacitance",   None),
    # --- Physics: Thermodynamics ---
    ("physics", "heat_transfer"):    (_get_thermo, "simulate_heat_transfer",   None),
    ("physics", "ideal_gas"):        (_get_thermo, "ideal_gas_law",            None),
    ("physics", "entropy"):          (_get_thermo, "calculate_entropy_change", None),
    ("physics", "carnot_efficiency"):(_get_thermo, "carnot_efficiency",        None),
    # --- Chemistry: RDKit ---
    ("chemistry", "molecule_analysis"):   (_get_rdkit, "analyze_molecule",        {"smiles": "smiles"}),
    ("chemistry", "molecular_descriptors"):(_get_rdkit,"calculate_descriptors",    {"smiles": "smiles"}),
    ("chemistry", "parse_smiles"):         (_get_rdkit, "parse_smiles",            {"smiles": "smiles"}),
    ("chemistry", "molecule_image"):       (_get_rdkit, "generate_2d_image",       {"smiles": "smiles"}),
    ("chemistry", "3d_coordinates"):       (_get_rdkit, "generate_3d_coordinates", {"smiles": "smiles"}),
    # --- Chemistry: Kinetics ---
    ("chemistry", "first_order"):       (_get_kinetics, "simulate_first_order",       None),
    ("chemistry", "second_order"):      (_get_kinetics, "simulate_second_order",      None),
    ("chemistry", "consecutive"):       (_get_kinetics, "simulate_consecutive_reactions", None),
    ("chemistry", "michaelis_menten"):  (_get_kinetics, "simulate_michaelis_menten",  None),
    ("chemistry", "equilibrium"):       (_get_kinetics, "simulate_equilibrium",       None),
    ("chemistry", "reaction_kinetics"): (_get_kinetics, "simulate_first_order",       None),
    # --- Materials: Lattice ---
    ("materials", "create_lattice"):   (_get_lattice, "_dispatch_lattice", None),  # custom dispatch
    ("materials", "fcc_lattice"):      (_get_lattice, "create_fcc_lattice", None),
    ("materials", "bcc_lattice"):      (_get_lattice, "create_bcc_lattice", None),
    ("materials", "simple_cubic"):     (_get_lattice, "create_simple_cubic", None),
    # --- Materials: Stress/Strain ---
    ("materials", "stress_test"):          (_get_stress_strain, "simulate_elastic_deformation", None),
    ("materials", "youngs_modulus"):       (_get_stress_strain, "calculate_youngs_modulus",     None),
    ("materials", "elastic_deformation"):  (_get_stress_strain, "simulate_elastic_deformation", None),
    ("materials", "stress_strain_curve"):  (_get_stress_strain, "stress_strain_curve",          None),
    ("materials", "elastic_plastic"):      (_get_stress_strain, "elastic_plastic_deformation",  None),
    ("materials", "ramberg_osgood"):       (_get_stress_strain, "ramberg_osgood",               None),
    # --- Materials: Pymatgen optional backend ---
    ("materials", "element_properties"): (_get_pymatgen, "element_properties", None),
    ("materials", "compare_elements"):   (_get_pymatgen, "compare_elements",   None),
    ("materials", "build_structure"):    (_get_pymatgen, "build_structure",    None),
    ("materials", "common_structure"):   (_get_pymatgen, "analyze_common_structure", {"formula": "formula"}),
    ("materials", "phase_diagram"):      (_get_pymatgen, "plot_phase_diagram_binary", None),
    ("materials", "property_trends"):    (_get_pymatgen, "plot_element_property_trends", None),
    ("materials", "oxidation_states"):   (_get_pymatgen, "oxidation_state_analysis", {"formula": "formula"}),
    # --- NVIDIA: ALCHEMI atomistic simulation ---
    ("nvidia", "alchemi_relax"):               (_get_alchemi, "relax_structure",      None),
    ("nvidia", "alchemi_md"):                  (_get_alchemi, "run_md",               None),
    ("nvidia", "alchemi_mlip"):                (_get_alchemi, "mlip_energy_forces",   None),
    ("nvidia", "alchemi_batch_screen"):        (_get_alchemi, "batch_alloy_screen",   None),
    # --- NVIDIA: Warp GPU kernels ---
    ("nvidia", "warp_diffusion"):              (_get_warp, "run_diffusion_field",     None),
    ("nvidia", "warp_allen_cahn"):             (_get_warp, "allen_cahn_grain_growth", None),
    # --- NVIDIA: PhysicsNeMo surrogate ---
    ("nvidia", "nemo_train_fno"):              (_get_physicsnemo, "train_fno_surrogate",       None),
    ("nvidia", "nemo_sampling_plan"):          (_get_physicsnemo, "generate_training_data_plan", None),
    # --- NVIDIA: CuPy/cuPyNumeric ---
    ("nvidia", "cupy_diffusion_sweep"):        (_get_cupy, "parameter_sweep_diffusion", None),
    # --- Materials: GPU alloy discovery workflow ---
    ("materials", "gpu_backend_status"):       (_get_materials_gpu_workflow, "integration_status",    None),
    ("materials", "alloy_property_prediction"):(_get_materials_gpu_workflow, "alloy_property_prediction", None),
    ("materials", "degradation_prediction"):   (_get_materials_gpu_workflow, "degradation_prediction", None),
    ("materials", "microstructure_diffusion"): (_get_materials_gpu_workflow, "microstructure_diffusion", None),
    ("materials", "surrogate_model_plan"):     (_get_materials_gpu_workflow, "surrogate_model_plan",  None),
    # --- Materials: CALPHAD ---
    ("materials", "calphad_phase_equilibrium"):(_get_materials_gpu_workflow, "calphad_phase_equilibrium", None),
    ("materials", "calphad_phase_scan"):       (_get_materials_gpu_workflow, "calphad_phase_scan",    None),
    ("materials", "calphad_binary_diagram"):   (_get_materials_gpu_workflow, "calphad_binary_diagram",None),
    ("materials", "tcp_phase_check"):          (_get_materials_gpu_workflow, "tcp_phase_check",       None),
    ("materials", "oxide_phase_equilibrium"):  (_get_materials_gpu_workflow, "oxide_phase_equilibrium", None),
    # --- Materials: DFT ---
    ("materials", "dft_formation_energy"):     (_get_materials_gpu_workflow, "dft_formation_energy",  None),
    ("materials", "dft_relaxation"):           (_get_materials_gpu_workflow, "dft_relaxation",        None),
    ("materials", "generate_sqs"):             (_get_materials_gpu_workflow, "generate_sqs",          None),
    # --- Materials: MLIP ---
    ("materials", "mlip_energy_forces"):       (_get_materials_gpu_workflow, "mlip_energy_forces",    None),
    ("materials", "mlip_batch_screen"):        (_get_materials_gpu_workflow, "mlip_batch_screen",     None),
    ("materials", "mlip_relax"):               (_get_materials_gpu_workflow, "mlip_relax",            None),
    # --- Materials: Diffusion ---
    ("materials", "diffusion_profile"):        (_get_diffusion, "diffusion_profile",       None),
    ("materials", "steady_state_flux"):        (_get_diffusion, "steady_state_flux",       None),
    ("materials", "arrhenius_diffusivity"):    (_get_diffusion, "arrhenius_diffusivity",   None),
    ("materials", "darken_interdiffusion"):    (_get_diffusion, "darken_interdiffusion",   None),
    ("materials", "grain_boundary_diffusion"): (_get_diffusion, "grain_boundary_diffusion",None),
    ("materials", "process_window_map"):       (_get_diffusion, "process_window_map",      None),
    # --- Materials: Phase Transformation ---
    ("materials", "jmak_kinetics"):            (_get_phase_transformation, "jmak_kinetics",            None),
    ("materials", "jmak_temperature_series"):  (_get_phase_transformation, "jmak_temperature_series",  None),
    ("materials", "ttt_diagram"):              (_get_phase_transformation, "ttt_diagram",              None),
    ("materials", "grain_growth"):             (_get_phase_transformation, "grain_growth",             None),
    ("materials", "precipitation_kinetics"):   (_get_phase_transformation, "precipitation_kinetics",   None),
    # --- Materials: Oxidation and Chemical Stability ---
    ("materials", "oxidation_kinetics"):       (_get_oxidation_stability, "oxidation_kinetics",    None),
    ("materials", "ellingham_diagram"):        (_get_oxidation_stability, "ellingham_diagram",     None),
    ("materials", "pilling_bedworth_ratio"):   (_get_oxidation_stability, "pilling_bedworth_ratio",None),
    ("materials", "corrosion_risk"):           (_get_oxidation_stability, "corrosion_risk",        None),
    ("materials", "scc_risk_index"):           (_get_oxidation_stability, "scc_risk_index",        None),
    # --- Materials: Forging ---
    ("materials", "flow_stress"):             (_get_forging, "flow_stress",               None),
    ("materials", "hall_petch"):              (_get_forging, "hall_petch",                None),
    ("materials", "dynamic_recrystallization"): (_get_forging, "dynamic_recrystallization", None),
    ("materials", "forging_force"):           (_get_forging, "forging_force",             None),
    ("materials", "processing_map"):          (_get_forging, "processing_map",            None),
    ("materials", "strain_rate_sensitivity"): (_get_forging, "strain_rate_sensitivity",   None),
    # --- Materials: Casting ---
    ("materials", "solidification_time"):     (_get_casting, "solidification_time",       None),
    ("materials", "cooling_curve"):           (_get_casting, "cooling_curve",             None),
    ("materials", "dendrite_arm_spacing"):    (_get_casting, "dendrite_arm_spacing",      None),
    ("materials", "scheil_microsegregation"): (_get_casting, "scheil_microsegregation",   None),
    ("materials", "niyama_criterion"):        (_get_casting, "niyama_criterion",          None),
    ("materials", "hot_tearing"):             (_get_casting, "hot_tearing",               None),
    ("materials", "mould_filling"):           (_get_casting, "mould_filling",             None),
    # Forging — process-specific
    ("materials", "open_die_pass_schedule"): (_get_forging, "open_die_pass_schedule",    None),
    ("materials", "closed_die_analysis"):    (_get_forging, "closed_die_analysis",       None),
    # Casting — process-specific
    ("materials", "investment_casting"):     (_get_casting, "investment_casting",        None),
    ("materials", "die_casting"):            (_get_casting, "die_casting",               None),
    ("materials", "application_case"):       (_get_casting, "application_case",          None),
    # --- Atomic ---
    ("atomic", "electron_config"):          (_get_electron_config, "compute_configuration",       {"Z": "Z"}),
    ("atomic", "compare_elements"):         (_get_electron_config, "compare_elements",            None),
    ("atomic", "shell_diagram"):            (_get_electron_config, "plot_shell_diagram",          {"Z": "Z"}),
    ("atomic", "orbital_diagram"):          (_get_electron_config, "plot_orbital_diagram",        {"Z": "Z"}),
    ("atomic", "ionization_trend"):         (_get_electron_config, "plot_ionization_energy_trend", None),
    ("atomic", "hydrogen_energy_levels"):   (_get_hydrogen_orbitals, "energy_levels",             None),
    ("atomic", "hydrogen_energy_diagram"):  (_get_hydrogen_orbitals, "plot_energy_levels",        None),
    ("atomic", "radial_probability"):       (_get_hydrogen_orbitals, "plot_radial_probability",   None),
    ("atomic", "radial_normalization"):     (_get_hydrogen_orbitals, "radial_probability_integral", None),
    ("atomic", "orbital_2d"):               (_get_hydrogen_orbitals, "plot_orbital_2d",           None),
    ("atomic", "radial_comparison"):        (_get_hydrogen_orbitals, "plot_radial_comparison",    None),
    ("atomic", "create_element"):           (_get_element_builder, "create_element",              None),
    ("atomic", "fusion_to_element"):        (_get_element_builder, "fusion_to_element",           None),
    ("atomic", "compare_isotopes"):         (_get_element_builder, "compare_isotopes",            None),
    ("atomic", "effective_nuclear_charge"): (_get_electron_config, "effective_nuclear_charge",    {"Z": "Z"}),
    ("atomic", "slater_ionization"):        (_get_electron_config, "slater_ionization_energy",    {"Z": "Z"}),
    # --- Atomic: ASE optional backend ---
    ("atomic", "bulk_crystal"):      (_get_ase, "build_bulk_crystal",          None),
    ("atomic", "molecule"):          (_get_ase, "build_molecule",              {"name": "name"}),
    ("atomic", "crystal_plot"):      (_get_ase, "plot_crystal_structure",      None),
    ("atomic", "molecule_plot"):     (_get_ase, "plot_molecule",               {"name": "name"}),
    ("atomic", "compare_crystals"):  (_get_ase, "compare_crystal_structures",  {"symbol": "symbol"}),
    ("atomic", "surface_slab"):      (_get_ase, "surface_slab",                None),
    ("atomic", "ase_element_data"):  (_get_ase, "get_element_ase_data",        {"symbol": "symbol"}),
    # --- Nuclear ---
    ("nuclear", "analyze_nucleus"):   (_get_nuclear, "analyze_nucleus",       None),
    ("nuclear", "binding_energy_curve"):(_get_nuclear, "binding_energy_curve", None),
    ("nuclear", "nuclear_chart"):     (_get_nuclear, "nuclear_chart",         None),
    ("nuclear", "fusion_energy"):     (_get_nuclear, "fusion_energy",         None),
    ("nuclear", "fission_energy"):    (_get_nuclear, "fission_energy",        None),
    ("nuclear", "decay"):             (_get_decay, "simulate_decay",          None),
    ("nuclear", "decay_chain"):       (_get_decay, "simulate_decay_chain",    None),
    ("nuclear", "decay_plot"):        (_get_decay, "plot_decay",              None),
    ("nuclear", "decay_chain_plot"):  (_get_decay, "plot_decay_chain",        None),
    ("nuclear", "alpha_decay"):          (_get_decay,    "alpha_decay_product",   None),
    ("nuclear", "beta_minus"):          (_get_decay,    "beta_minus_product",    None),
    ("nuclear", "beta_plus"):           (_get_decay,    "beta_plus_product",     None),
    ("nuclear", "separation_energy"):   (_get_nuclear,  "separation_energy",     None),
    ("nuclear", "alpha_halflife"):      (_get_nuclear,  "alpha_decay_halflife",  None),
    # --- Data ---
    ("data", "arxiv_search"):       (_get_arxiv, "search",                         None),
    ("data", "arxiv_paper"):        (_get_arxiv, "get_paper",                      {"arxiv_id": "arxiv_id"}),
    ("data", "arxiv_references"):   (_get_arxiv, "find_references_for_experiment", None),
    ("data", "pubchem_search"):     (_get_pubchem, "search_compound",              {"name": "name"}),
    ("data", "pubchem_by_cid"):     (_get_pubchem, "get_by_cid",                   {"cid": "cid"}),
    ("data", "pubchem_synonyms"):   (_get_pubchem, "get_synonyms",                 {"name": "name"}),
    ("data", "pubchem_safety"):     (_get_pubchem, "get_safety",                   {"cid": "cid"}),
    ("data", "pubchem_similar"):    (_get_pubchem, "similar_compounds",            {"cid": "cid"}),
    ("data", "pubchem_substructure"):(_get_pubchem, "substructure_search",         {"smiles": "smiles"}),
    # --- Control systems optional backend ---
    ("control", "transfer_function"): (_get_control, "transfer_function",       None),
    ("control", "pid_controller"):    (_get_control, "pid_controller",          None),
    ("control", "step_response"):     (_get_control, "plot_step_response",      None),
    ("control", "bode"):              (_get_control, "plot_bode",               None),
    ("control", "root_locus"):        (_get_control, "plot_root_locus",         None),
    ("control", "nyquist"):           (_get_control, "plot_nyquist",            None),
    ("control", "closed_loop"):        (_get_control, "closed_loop_analysis",    None),
    ("control", "design_pid"):         (_get_control, "design_pid_for_plant",    None),
    # --- QD Geometry: MAP-Elites ---
    ("qdgeometry", "mapelites_alloy_search"):    (_get_qdgeometry, "mapelites_alloy_search",    None),
    ("qdgeometry", "mapelites_geometry_search"): (_get_qdgeometry, "mapelites_geometry_search", None),
    # --- QD Geometry: primitives ---
    ("qdgeometry", "geometry_primitive"):        (_get_qdgeometry, "geometry_primitive",        None),
    ("qdgeometry", "constraint_loss"):           (_get_qdgeometry, "constraint_loss",           None),
    # --- QD Geometry: gradient descent (JAX autodiff) ---
    ("qdgeometry", "gradient_descent"):          (_get_qdgeometry, "gradient_descent",          None),
    # --- QD Geometry: SAIL ---
    ("qdgeometry", "sail_geometry_search"):      (_get_qdgeometry, "sail_geometry_search",      None),
    ("qdgeometry", "sail_alloy_search"):         (_get_qdgeometry, "sail_alloy_search",         None),
    # --- QD Geometry: Grammar-Guided GP ---
    ("qdgeometry", "grammar_guided_search"):     (_get_qdgeometry, "grammar_guided_search",     None),
    # --- QD Geometry: Neural Implicit ---
    ("qdgeometry", "fit_neural_implicit"):       (_get_qdgeometry, "fit_neural_implicit",       None),
    # --- QD Geometry: GPU multi-physics blade simulation ---
    ("qdgeometry", "blade_simulation"):          (_get_qdgeometry, "blade_simulation",          None),
    # --- QD Geometry: Warp kernel thermal solver ---
    ("qdgeometry", "warp_blade_thermal"):        (_get_qdgeometry, "warp_blade_thermal",        None),
    # --- QD Geometry: FNO2d surrogate ---
    ("qdgeometry", "fno_generate_data"):         (_get_qdgeometry, "fno_generate_data",         None),
    ("qdgeometry", "fno_train"):                 (_get_qdgeometry, "fno_train",                 None),
    # --- QD Geometry: CalculiX FEM validation ---
    ("qdgeometry", "calculix_validate"):         (_get_qdgeometry, "calculix_validate",         None),
    ("qdgeometry", "calculix_thermal"):          (_get_qdgeometry, "calculix_thermal",          None),
    # --- Hybrid ---
    ("hybrid", "reaction_kinetics"):   (_get_kinetics, "simulate_first_order", None),
    ("hybrid", "consecutive_kinetics"):(_get_kinetics, "simulate_consecutive_reactions", None),
    # --- Materials: GPU-backed microstructure experiments ---
    ("materials", "allen_cahn_simulation"): (_get_materials_gpu_workflow, "allen_cahn_simulation", None),
    ("materials", "diffusion_3d"):          (_get_materials_gpu_workflow, "diffusion_3d",          None),
    ("materials", "gpu_diagnostics"):       (_get_materials_gpu_workflow, "gpu_diagnostics",       None),
    # --- NVIDIA: additional Warp kernels ---
    ("nvidia", "warp_allen_cahn_field"): (_get_warp,           "run_allen_cahn_field", None),
    ("nvidia", "warp_diffusion_3d"):     (_get_warp,           "run_diffusion_3d",     None),
    ("nvidia", "gpu_diagnostics"):       (_get_gpu_diagnostics, "gpu_status",          None),
}


class ExperimentDispatcher:
    """Routes ExperimentRequest to the correct engine and returns ExperimentResult."""

    def dispatch(self, req: ExperimentRequest) -> ExperimentResult:
        """Dispatch request to appropriate engine, attach visualizations, return result."""
        key = (req.domain, req.type)
        entry = _ROUTING_TABLE.get(key)

        if entry is None:
            return ExperimentResult(
                experiment_id=req.experiment_id,
                domain=req.domain,
                type=req.type,
                status="error",
                errors=[
                    f"No handler registered for domain={req.domain!r}, type={req.type!r}. "
                    f"Available types: {[k for k in _ROUTING_TABLE if k[0] == req.domain]}"
                ],
            )

        engine_factory, method_name, param_map = entry

        try:
            engine = engine_factory()

            # Special case: create_lattice dispatches by lattice_type parameter
            if req.domain == "materials" and req.type == "create_lattice":
                return self._dispatch_lattice(req, engine)

            # Build kwargs: environment defaults (e.g. g) then explicit parameters override
            params = {**dict(req.environment), **dict(req.parameters)}
            if param_map:
                for method_key, param_key in param_map.items():
                    if param_key in params:
                        params[method_key] = params.pop(param_key)

            method = getattr(engine, method_name)
            raw_result = method(**params)

            # Methods that return a raw base64 PNG string (e.g. binding_energy_curve,
            # nuclear_chart) go directly into plots, not results.
            if isinstance(raw_result, str):
                return ExperimentResult(
                    experiment_id=req.experiment_id,
                    domain=req.domain,
                    type=req.type,
                    status="success",
                    results={},
                    plots=[raw_result],
                    metadata={"solver": req.solver, "environment": req.environment},
                )

            if isinstance(raw_result, (int, float)):
                raw_result = {"value": raw_result}

            # Attach visualizations
            plots = self._generate_plots(req, raw_result)

            return ExperimentResult(
                experiment_id=req.experiment_id,
                domain=req.domain,
                type=req.type,
                status="success",
                results=raw_result,
                plots=plots,
                metadata={"solver": req.solver, "environment": req.environment},
            )

        except Exception as exc:
            return ExperimentResult(
                experiment_id=req.experiment_id,
                domain=req.domain,
                type=req.type,
                status="error",
                errors=[str(exc)],
            )

    def _dispatch_lattice(self, req: ExperimentRequest, engine) -> ExperimentResult:
        """Handle create_lattice with lattice_type parameter routing."""
        params = {**dict(req.environment), **dict(req.parameters)}
        lattice_type = params.pop("lattice_type", "fcc").lower()

        method_map = {
            "fcc": engine.create_fcc_lattice,
            "bcc": engine.create_bcc_lattice,
            "sc": engine.create_simple_cubic,
            "simple_cubic": engine.create_simple_cubic,
        }
        method = method_map.get(lattice_type)
        if method is None:
            return ExperimentResult(
                experiment_id=req.experiment_id,
                domain=req.domain,
                type=req.type,
                status="error",
                errors=[f"Unknown lattice_type: {lattice_type!r}. Use 'fcc', 'bcc', or 'sc'."],
            )

        raw = method(**params)
        return ExperimentResult(
            experiment_id=req.experiment_id,
            domain=req.domain,
            type=req.type,
            status="success",
            results=raw,
            metadata={"lattice_type": lattice_type},
        )

    def _generate_plots(self, req: ExperimentRequest, results: dict) -> list[str]:
        """Generate plots based on req.outputs and available result arrays."""
        if "plot" not in req.outputs and "all" not in req.outputs:
            return []

        try:
            plot_engine = _get_plot_engine()
        except Exception:
            return []

        plots: list[str] = []

        try:
            if req.domain == "materials":
                self._generate_materials_plots(req, results, plot_engine, plots)

            # Trajectory plot (physics projectile / pendulum)
            if "x" in results and "y" in results and isinstance(results["x"], list):
                p = plot_engine.plot_trajectory(
                    results["x"], results["y"],
                    title=f"{req.type.replace('_', ' ').title()} Trajectory",
                )
                plots.append(p)

            # Time series (ODE, kinetics, spring-mass, etc.)
            elif "t" in results and ("y" in results or "A" in results):
                t = results["t"]
                series = []
                labels = []
                for key in ("y", "A", "B", "C", "S", "x", "theta_rad"):
                    if key in results and isinstance(results[key], list):
                        series.append(results[key])
                        labels.append(key)
                if series:
                    p = plot_engine.plot_time_series(
                        t, series, labels,
                        title=f"{req.type.replace('_', ' ').title()}",
                        xlabel="t",
                    )
                    plots.append(p)

            # Stress-strain curve
            elif "strain" in results and "stress" in results:
                p = plot_engine.plot_data(
                    results["strain"], results["stress"],
                    title="Stress-Strain Curve",
                    xlabel="Strain",
                    ylabel="Stress [Pa]",
                )
                plots.append(p)

        except Exception:
            pass  # Visualization is optional; don't fail the main result

        return plots

    def _generate_materials_plots(
        self, req: ExperimentRequest, results: dict, plot_engine, plots: list[str]
    ) -> None:
        """Attach focused visualizations for alloy/degradation workflows."""
        if req.type == "alloy_property_prediction":
            composition = results.get("normalized_composition", {})
            if composition:
                plots.append(
                    plot_engine.plot_bar_labels(
                        list(composition.keys()),
                        [float(v) for v in composition.values()],
                        title="Alloy Composition",
                        ylabel="Fraction",
                        color="#607D8B",
                    )
                )

            estimates = results.get("property_estimates", {})
            score_keys = ["phase_stability_score_raw", "oxidation_resistance_score_raw"]
            if all(key in estimates for key in score_keys):
                plots.append(
                    plot_engine.plot_bar_labels(
                        ["Phase stability", "Oxidation resistance"],
                        [float(estimates[key]) for key in score_keys],
                        title="Alloy Screening Scores",
                        ylabel="Score",
                        color="#4CAF50",
                    )
                )

        elif req.type == "degradation_prediction":
            estimates = results.get("degradation_estimates", {})
            risk_keys = ["corrosion_risk_score", "hydrogen_embrittlement_risk_score"]
            if all(key in estimates for key in risk_keys):
                plots.append(
                    plot_engine.plot_bar_labels(
                        ["Corrosion", "Hydrogen embrittlement"],
                        [float(estimates[key]) for key in risk_keys],
                        title="Degradation Risk Scores",
                        ylabel="Risk score",
                        color="#D84315",
                    )
                )

            curve = results.get("oxidation_curve", {})
            if "time_hours" in curve and "oxide_thickness_um" in curve:
                plots.append(
                    plot_engine.plot_data(
                        curve["time_hours"],
                        curve["oxide_thickness_um"],
                        title="Oxidation Growth Proxy",
                        xlabel="Exposure [h]",
                        ylabel="Oxide thickness [um]",
                        color="#795548",
                    )
                )

        elif req.type == "microstructure_diffusion":
            field = results.get("field")
            if field:
                plots.append(
                    plot_engine.plot_heatmap(
                        field,
                        title="Diffusion Field",
                        xlabel="Grid y",
                        ylabel="Grid x",
                        colorbar_label="Concentration",
                        cmap="magma",
                    )
                )

            summary = results.get("field_summary", {})
            centerline = summary.get("centerline")
            if centerline:
                plots.append(
                    plot_engine.plot_data(
                        list(range(len(centerline))),
                        centerline,
                        title="Diffusion Centerline",
                        xlabel="Grid x",
                        ylabel="Concentration",
                        color="#00897B",
                    )
                )

        elif req.type == "surrogate_model_plan":
            split = results.get("dataset_split", {})
            if split:
                labels = ["Train", "Validation", "Test"]
                plots.append(
                    plot_engine.plot_bar_labels(
                        labels,
                        [float(split[k.lower()]) for k in labels],
                        title="Surrogate Dataset Split",
                        ylabel="Samples",
                        color="#5E35B1",
                    )
                )

        elif req.type in ("diffusion_profile", "arrhenius_diffusivity",
                          "jmak_kinetics", "grain_growth", "ttt_diagram",
                          "oxidation_kinetics", "ellingham_diagram"):
            self._generate_materials_focused_plots(req, results, plots)

    def _generate_materials_focused_plots(
        self, req: ExperimentRequest, results: dict, plots: list[str]
    ) -> None:
        """Generate inline plots for the metallurgy-focused engines."""
        try:
            if req.type == "diffusion_profile":
                x_mm = [v * 1000.0 for v in results.get("x_m", [])]
                C = results.get("concentration", [])
                if x_mm and C:
                    import io, base64, matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(x_mm, C, color="#1565C0", linewidth=2.0)
                    ax.axhline(results.get("case_threshold", 0.5),
                               color="#D84315", linestyle="--", linewidth=1.2,
                               label=f"Case threshold")
                    cd = results.get("case_depth_mm", 0.0)
                    ax.axvline(cd, color="#2E7D32", linestyle=":",
                               linewidth=1.5, label=f"Case depth {cd:.3f} mm")
                    ax.set_xlabel("Depth [mm]"); ax.set_ylabel("Concentration")
                    ax.set_title("Diffusion Profile"); ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3); fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig); buf.seek(0)
                    plots.append(base64.b64encode(buf.read()).decode("ascii"))

            elif req.type == "arrhenius_diffusivity":
                import io, base64, matplotlib.pyplot as plt
                import numpy as np
                T = np.array(results.get("temperatures_K", []))
                D = np.array(results.get("diffusivities_m2_s", []))
                if len(T) and len(D):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.semilogy(1000.0 / T, D, color="#6A1B9A", linewidth=2.0)
                    ax.set_xlabel("1000/T [K⁻¹]"); ax.set_ylabel("D [m²/s]")
                    ax.set_title("Arrhenius Diffusivity"); ax.grid(True, which="both", alpha=0.3)
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig); buf.seek(0)
                    plots.append(base64.b64encode(buf.read()).decode("ascii"))

            elif req.type == "jmak_kinetics":
                import io, base64, matplotlib.pyplot as plt
                t = results.get("t_s", [])
                X = results.get("transformed_fraction_X", [])
                if t and X:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(t, X, color="#1976D2", linewidth=2.0)
                    ax.set_xlabel("Time [s]"); ax.set_ylabel("Transformed fraction X")
                    ax.set_title("JMAK Transformation Kinetics"); ax.grid(True, alpha=0.3)
                    ax.set_ylim(-0.02, 1.05); fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig); buf.seek(0)
                    plots.append(base64.b64encode(buf.read()).decode("ascii"))

            elif req.type == "grain_growth":
                import io, base64, matplotlib.pyplot as plt
                t = results.get("t_s", [])
                d = results.get("grain_diameter_um", [])
                if t and d:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(t, d, color="#7B1FA2", linewidth=2.0)
                    ax.set_xlabel("Time [s]"); ax.set_ylabel("Grain diameter [µm]")
                    ax.set_title("Grain Growth"); ax.grid(True, alpha=0.3); fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig); buf.seek(0)
                    plots.append(base64.b64encode(buf.read()).decode("ascii"))

            elif req.type == "ttt_diagram":
                import io, base64, matplotlib.pyplot as plt
                T_arr = results.get("temperatures_K", [])
                t_start = results.get("t_start_s", [])
                t_finish = results.get("t_finish_s", [])
                if T_arr and t_start and t_finish:
                    fig, ax = plt.subplots(figsize=(8, 7))
                    ax.semilogx(t_start, T_arr, color="#1565C0", linewidth=2.0, label="Start")
                    ax.semilogx(t_finish, T_arr, color="#B71C1C", linewidth=2.0, label="Finish")
                    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
                    ax.set_title(f"TTT Diagram — {results.get('transformation', '')}"); ax.legend()
                    ax.grid(True, which="both", alpha=0.3); ax.invert_yaxis(); fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig); buf.seek(0)
                    plots.append(base64.b64encode(buf.read()).decode("ascii"))

            elif req.type == "oxidation_kinetics":
                import io, base64, matplotlib.pyplot as plt
                t = results.get("t_s", [])
                x = results.get("oxide_thickness_um", [])
                if t and x:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(t, x, color="#B71C1C", linewidth=2.0)
                    ax.set_xlabel("Time [s]"); ax.set_ylabel("Oxide thickness [µm]")
                    ax.set_title(f"Oxidation Kinetics ({results.get('regime', '')})"); ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig); buf.seek(0)
                    plots.append(base64.b64encode(buf.read()).decode("ascii"))

            elif req.type == "ellingham_diagram":
                import io, base64, matplotlib.pyplot as plt, numpy as np
                T_arr = np.array(results.get("temperatures_K", []))
                colors = plt.cm.tab20.colors
                if len(T_arr):
                    fig, ax = plt.subplots(figsize=(11, 7))
                    for i, rxn in enumerate(results.get("reactions", [])):
                        ax.plot(T_arr, rxn["dG_kJ_molO2"],
                                color=colors[i % len(colors)], linewidth=1.8,
                                label=rxn["reaction"])
                    ax.set_xlabel("Temperature [K]"); ax.set_ylabel("ΔG° [kJ/mol O₂]")
                    ax.set_title("Ellingham-Richardson Diagram")
                    ax.legend(fontsize=7, loc="upper right")
                    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                    ax.grid(True, alpha=0.25); fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                    plt.close(fig); buf.seek(0)
                    plots.append(base64.b64encode(buf.read()).decode("ascii"))
        except Exception:
            pass

    def available_types(self, domain: str | None = None) -> list[tuple[str, str]]:
        """Return all registered (domain, type) tuples, optionally filtered."""
        if domain:
            return [k for k in _ROUTING_TABLE if k[0] == domain]
        return list(_ROUTING_TABLE.keys())
