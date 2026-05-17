"""SIMLAB safety and parameter validator.

Performs pre-flight checks on experiment requests before dispatching to engines.
"""

from __future__ import annotations

from simlab.core.constants.physical import CONSTANTS
from simlab.core.schemas.experiment import ExperimentRequest, ValidationResult

# Known domain -> required parameter sets
_REQUIRED_PARAMS: dict[tuple[str, str], list[str]] = {
    ("physics", "projectile_motion"): ["v0", "angle_deg"],
    ("physics", "projectile_drag"):   ["v0", "angle_deg", "mass", "diameter"],
    ("physics", "relativistic"):      [],   # requires exactly one of velocity/momentum/kinetic_energy
    ("physics", "pendulum"): ["length", "theta0_deg"],
    ("physics", "spring_mass"): ["mass", "k", "x0", "v0"],
    ("physics", "collision"): ["m1", "v1", "m2", "v2"],
    ("physics", "gravity"): ["m1", "m2", "r"],
    ("physics", "circular_motion"): ["mass", "radius", "angular_velocity"],
    ("physics", "electric_field"): ["charges", "positions"],
    ("physics", "coulomb_force"): ["q1", "q2", "r"],
    ("physics", "magnetic_field"): ["current", "wire_positions"],
    ("physics", "capacitance"): ["geometry"],
    ("physics", "heat_transfer"): ["T_hot", "T_cold", "k_conductivity", "area", "thickness"],
    ("physics", "ideal_gas"): [],  # at least 3 of P, V, n, T
    ("physics", "entropy"): ["n_moles", "T1", "T2", "Cp"],
    ("physics", "carnot_efficiency"): ["T_hot", "T_cold"],
    ("physics", "van_der_waals"):     [],   # requires gas or a+b, plus 3 of P/V/n/T
    ("physics", "bernoulli"):         ["rho"],
    ("physics", "pipe_flow"):         ["delta_P", "radius", "length", "viscosity", "rho"],
    ("physics", "reynolds_number"):   ["rho", "velocity", "length", "viscosity"],
    ("physics", "terminal_velocity"): ["mass", "diameter", "rho_fluid"],
    ("physics", "drag_force"):        ["velocity", "diameter", "Cd"],
    ("math", "solve_equation"): ["equation"],
    ("math", "differentiate"): ["expression"],
    ("math", "integrate"): ["expression"],
    ("math", "simplify"): ["expression"],
    ("math", "factor"): ["expression"],
    ("math", "expand"): ["expression"],
    ("math", "taylor_series"): ["expression"],
    ("math", "critical_points"): ["expression"],
    ("math", "solve_ode"): ["equation", "y0", "t_span"],
    ("math", "solve_system_ode"): ["equations", "y0", "t_span"],
    ("math", "solve_2nd_order"): [
        "coeff_a", "coeff_b", "coeff_c", "y0", "yp0", "t_span"
    ],
    ("math", "eigenvalues"): ["matrix"],
    ("math", "solve_linear"): ["A", "b"],
    ("math", "svd"): ["matrix"],
    ("math", "determinant"): ["matrix"],
    ("math", "inverse"): ["matrix"],
    ("math", "matrix_multiply"): ["A", "B"],
    ("math", "rank"): ["matrix"],
    ("math", "optimize"): ["expression", "x0"],
    ("math", "maximize"): ["expression", "x0"],
    ("math", "fit_curve"): ["x_data", "y_data", "model_str"],
    ("math", "minimize_interval"): ["expression", "a", "b"],
    ("math", "monte_carlo"): ["expression", "n_samples", "variable_ranges"],
    ("math", "descriptive_stats"): ["data"],
    ("math", "fit_distribution"): ["data"],
    ("math", "hypothesis_test"): ["data1", "data2"],
    ("chemistry", "molecule_analysis"): ["smiles"],
    ("chemistry", "reaction_kinetics"): ["k", "A0"],
    ("chemistry", "molecular_descriptors"): ["smiles"],
    ("chemistry", "parse_smiles"): ["smiles"],
    ("chemistry", "molecule_image"): ["smiles"],
    ("chemistry", "3d_coordinates"): ["smiles"],
    ("chemistry", "first_order"): ["k", "A0"],
    ("chemistry", "second_order"): ["k", "A0"],
    ("chemistry", "consecutive"): ["k1", "k2", "A0"],
    ("chemistry", "michaelis_menten"): ["Vmax", "Km", "S0"],
    ("chemistry", "equilibrium"): ["k_forward", "k_reverse", "A0", "B0"],
    ("materials", "create_lattice"): ["lattice_type", "a"],
    ("materials", "fcc_lattice"): ["a"],
    ("materials", "bcc_lattice"): ["a"],
    ("materials", "simple_cubic"): ["a"],
    ("materials", "stress_test"): ["force", "area"],
    ("materials", "youngs_modulus"): ["stress", "strain"],
    ("materials", "elastic_deformation"): ["E", "force", "area", "L0"],
    ("materials", "stress_strain_curve"): ["E", "yield_stress", "ultimate_stress"],
    ("materials", "elastic_plastic"): ["E", "yield_stress", "E_tangent", "force", "area", "L0"],
    ("materials", "ramberg_osgood"): ["E", "K", "n_exp", "stress_max"],
    ("materials", "element_properties"): ["symbol"],
    ("materials", "compare_elements"): ["symbols"],
    ("materials", "build_structure"): ["lattice_params", "species", "coords"],
    ("materials", "common_structure"): ["formula"],
    ("materials", "phase_diagram"): ["el1", "el2"],
    ("materials", "property_trends"): [],
    ("materials", "oxidation_states"): ["formula"],
    ("materials", "gpu_backend_status"): [],
    ("materials", "alloy_property_prediction"): ["composition"],
    ("materials", "degradation_prediction"): ["composition", "environment", "exposure_hours"],
    ("materials", "microstructure_diffusion"): [],
    ("materials", "surrogate_model_plan"): ["dataset_size", "target_properties"],
    # CALPHAD
    ("materials", "calphad_phase_equilibrium"): ["composition"],
    ("materials", "calphad_phase_scan"):         ["composition"],
    ("materials", "calphad_binary_diagram"):     ["element_a", "element_b"],
    ("materials", "tcp_phase_check"):            ["composition"],
    ("materials", "oxide_phase_equilibrium"):    ["composition"],
    # DFT
    ("materials", "dft_formation_energy"):       ["species", "positions_angstrom", "lattice_vectors_angstrom"],
    ("materials", "dft_relaxation"):             ["species", "positions_angstrom", "lattice_vectors_angstrom"],
    ("materials", "generate_sqs"):               ["composition"],
    # MLIP
    ("materials", "mlip_energy_forces"):         ["species", "positions_angstrom", "lattice_vectors_angstrom"],
    ("materials", "mlip_batch_screen"):          ["compositions"],
    ("materials", "mlip_relax"):                 ["species", "positions_angstrom", "lattice_vectors_angstrom"],
    # NVIDIA new routes
    ("nvidia", "warp_allen_cahn_field"):  ["grid_shape", "M", "kappa", "dt", "steps"],
    ("nvidia", "warp_diffusion_3d"):      ["grid_shape", "alpha", "steps"],
    ("nvidia", "gpu_diagnostics"):        [],
    ("materials", "allen_cahn_simulation"): [],
    ("materials", "diffusion_3d"):          [],
    ("materials", "gpu_diagnostics"):       [],
    # Diffusion
    ("materials", "diffusion_profile"):        ["D", "time_s"],
    ("materials", "steady_state_flux"):        ["D", "C_high", "C_low", "thickness_m"],
    ("materials", "arrhenius_diffusivity"):    ["D0", "activation_energy_J_mol"],
    ("materials", "darken_interdiffusion"):    ["D_A_m2_s", "D_B_m2_s", "x_B"],
    ("materials", "grain_boundary_diffusion"): ["D_bulk_m2_s", "D_gb_m2_s"],
    ("materials", "process_window_map"):       ["D0", "activation_energy_J_mol", "times_s", "temperatures_K"],
    # Phase transformation
    ("materials", "jmak_kinetics"):            ["k", "n", "t_max_s"],
    ("materials", "jmak_temperature_series"):  ["k0", "Q_J_mol", "n", "temperatures_K", "t_max_s"],
    ("materials", "ttt_diagram"):              ["T_high_K", "T_low_K", "t_nose_s", "T_nose_K"],
    ("materials", "grain_growth"):             ["d0_um", "temperature_K", "time_s"],
    ("materials", "precipitation_kinetics"):   ["incubation_time_s", "Q_nucleation_J_mol",
                                                "Q_growth_J_mol", "temperature_K", "T_ref_K", "t_max_s"],
    # Oxidation and chemical stability
    ("materials", "oxidation_kinetics"):       [],
    ("materials", "ellingham_diagram"):        [],
    ("materials", "pilling_bedworth_ratio"):   [],
    ("materials", "corrosion_risk"):           ["metal", "pH"],
    ("materials", "scc_risk_index"):           ["material", "environment"],
    # Forging
    ("materials", "flow_stress"):             [],
    ("materials", "hall_petch"):              [],
    ("materials", "dynamic_recrystallization"): [],
    ("materials", "forging_force"):           [],
    ("materials", "processing_map"):          [],
    ("materials", "strain_rate_sensitivity"): [],
    # Casting
    ("materials", "solidification_time"):     [],
    ("materials", "cooling_curve"):           [],
    ("materials", "dendrite_arm_spacing"):    [],
    ("materials", "scheil_microsegregation"): [],
    ("materials", "niyama_criterion"):        [],
    ("materials", "hot_tearing"):             [],
    ("materials", "mould_filling"):           [],
    ("materials", "open_die_pass_schedule"): [],
    ("materials", "closed_die_analysis"):    [],
    ("materials", "investment_casting"):     [],
    ("materials", "die_casting"):            [],
    ("materials", "application_case"):       [],
    ("atomic", "electron_config"): ["Z"],
    ("atomic", "compare_elements"): ["z_list"],
    ("atomic", "shell_diagram"): ["Z"],
    ("atomic", "orbital_diagram"): ["Z"],
    ("atomic", "radial_probability"): ["n", "l"],
    ("atomic", "radial_normalization"): ["n", "l"],
    ("atomic", "hydrogen_energy_levels"): [],
    ("atomic", "hydrogen_energy_diagram"): [],
    ("atomic", "orbital_2d"): ["n", "l"],
    ("atomic", "radial_comparison"): ["n"],
    ("atomic", "ionization_trend"): [],
    ("atomic", "create_element"): ["Z"],
    ("atomic", "fusion_to_element"): ["Z1", "Z2"],
    ("atomic", "compare_isotopes"): ["Z", "N_list"],
    ("atomic", "effective_nuclear_charge"): ["Z"],
    ("atomic", "slater_ionization"): ["Z"],
    ("atomic", "bulk_crystal"): ["symbol"],
    ("atomic", "molecule"): ["name"],
    ("atomic", "crystal_plot"): ["symbol"],
    ("atomic", "molecule_plot"): ["name"],
    ("atomic", "compare_crystals"): ["symbol"],
    ("atomic", "surface_slab"): ["symbol"],
    ("atomic", "ase_element_data"): ["symbol"],
    ("nuclear", "analyze_nucleus"): ["Z", "N"],
    ("nuclear", "binding_energy_curve"): [],
    ("nuclear", "nuclear_chart"): [],
    ("nuclear", "fusion_energy"): ["Z1", "N1", "Z2", "N2"],
    ("nuclear", "fission_energy"): ["Z", "N"],
    ("nuclear", "decay"): ["N0", "half_life_s"],
    ("nuclear", "decay_chain"): ["chain"],
    ("nuclear", "decay_plot"): ["N0", "half_life_s"],
    ("nuclear", "decay_chain_plot"): ["chain"],
    ("nuclear", "alpha_decay"): ["Z", "N"],
    ("nuclear", "beta_minus"): ["Z", "N"],
    ("nuclear", "beta_plus"): ["Z", "N"],
    ("nuclear", "separation_energy"): ["Z", "N"],
    ("nuclear", "alpha_halflife"): ["Z", "N"],
    ("data", "arxiv_search"): ["query"],
    ("data", "arxiv_paper"): ["arxiv_id"],
    ("data", "arxiv_references"): ["domain", "exp_type"],
    ("data", "pubchem_search"): ["name"],
    ("data", "pubchem_by_cid"): ["cid"],
    ("data", "pubchem_synonyms"): ["name"],
    ("data", "pubchem_safety"): ["cid"],
    ("data", "pubchem_similar"): ["cid"],
    ("data", "pubchem_substructure"): ["smiles"],
    ("control", "transfer_function"): ["numerator", "denominator"],
    ("control", "pid_controller"): ["Kp", "Ki", "Kd"],
    ("control", "step_response"): ["numerator", "denominator"],
    ("control", "bode"): ["numerator", "denominator"],
    ("control", "root_locus"): ["numerator", "denominator"],
    ("control", "nyquist"): ["numerator", "denominator"],
    ("control", "closed_loop"): [
        "plant_num", "plant_den", "controller_num", "controller_den"
    ],
    ("control", "design_pid"): ["plant_num", "plant_den", "Kp", "Ki", "Kd"],
    ("hybrid", "reaction_kinetics"): ["k", "A0"],
    ("hybrid", "consecutive_kinetics"): ["k1", "k2", "A0"],
    # NVIDIA GPU backends
    ("nvidia", "alchemi_relax"):        ["species", "positions_angstrom", "lattice_vectors_angstrom"],
    ("nvidia", "alchemi_md"):           ["species", "positions_angstrom", "lattice_vectors_angstrom"],
    ("nvidia", "alchemi_mlip"):         ["species", "positions_angstrom", "lattice_vectors_angstrom"],
    ("nvidia", "alchemi_batch_screen"): ["compositions"],
    ("nvidia", "warp_diffusion"):       ["grid_shape", "alpha", "steps"],
    ("nvidia", "warp_allen_cahn"):      ["phi", "M", "kappa", "dt", "steps"],
    ("nvidia", "warp_allen_cahn_field"):["grid_shape", "M", "kappa", "dt", "steps"],
    ("nvidia", "warp_diffusion_3d"):    ["grid_shape", "alpha", "steps"],
    ("nvidia", "nemo_train_fno"):       ["X_train", "y_train"],
    ("nvidia", "nemo_sampling_plan"):   ["composition_space", "target_properties"],
    ("nvidia", "cupy_diffusion_sweep"): ["D0", "Q_J_mol", "temperature_range_K", "time_range_s"],
    ("nvidia", "gpu_diagnostics"):      [],
    # GPU-backed materials experiments
    ("materials", "allen_cahn_simulation"): [],
    ("materials", "diffusion_3d"):          [],
    ("materials", "gpu_diagnostics"):       [],
    # QD geometry engine — MAP-Elites
    ("qdgeometry", "mapelites_alloy_search"):    ["elements", "bounds"],
    ("qdgeometry", "mapelites_geometry_search"): [],
    # QD geometry engine — primitives
    ("qdgeometry", "geometry_primitive"):        ["primitive"],
    ("qdgeometry", "constraint_loss"):           ["primitive"],
    # QD geometry engine — gradient descent (JAX autodiff)
    ("qdgeometry", "gradient_descent"):          ["primitive", "init_params"],
    # QD geometry engine — SAIL
    ("qdgeometry", "sail_geometry_search"):      [],
    ("qdgeometry", "sail_alloy_search"):         ["elements", "bounds"],
    # QD geometry engine — Grammar-Guided GP
    ("qdgeometry", "grammar_guided_search"):     [],
    # QD geometry engine — Neural Implicit
    ("qdgeometry", "fit_neural_implicit"):       ["primitive"],
    # QD geometry engine — GPU multi-physics blade simulation
    ("qdgeometry", "blade_simulation"):          [],
    # QD geometry engine — Warp kernel FD solver
    ("qdgeometry", "warp_blade_thermal"):        [],
    # QD geometry engine — FNO2d surrogate
    ("qdgeometry", "fno_generate_data"):         [],
    ("qdgeometry", "fno_train"):                 [],
    # QD geometry engine — CalculiX FEM validation
    ("qdgeometry", "calculix_validate"):         [],
    ("qdgeometry", "calculix_thermal"):          [],
}

# Chemical safety keyword blocklist (SMILES substrings / names)
_DANGEROUS_SMILES_KEYWORDS = [
    "VX", "GB", "GA", "GD",  # nerve agent codes
    "sarin", "novichok", "mustard",
]

_DANGEROUS_REACTION_TYPES = [
    "explosive_synthesis", "nerve_agent", "bioweapon",
    "radiological", "chemical_weapon",
]

# Extreme value thresholds that trigger stability warnings
_STABILITY_THRESHOLDS: dict[str, float] = {
    "v0": 1e6,          # m/s — far below c but already unphysical for macro
    "mass": 1e30,       # kg
    "n_samples": 1e8,   # Monte Carlo samples
    "t_span": 1e6,      # seconds
    "k": 1e10,          # rate constant
    "temperature": 1e9, # Kelvin
    "pressure": 1e12,   # Pascal
}


class SimLabValidator:
    """Validates experiment requests for safety and correctness."""

    def validate_experiment(self, req: ExperimentRequest) -> ValidationResult:
        """Full pre-flight validation of an experiment request.

        Returns a ValidationResult with valid=False and blocked=True
        if the request should be outright rejected.
        """
        errors: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []

        # 1. Domain check
        valid_domains = {
            "math",
            "physics",
            "chemistry",
            "materials",
            "atomic",
            "nuclear",
            "data",
            "control",
            "hybrid",
            "nvidia",
            "qdgeometry",
        }
        if req.domain not in valid_domains:
            errors.append(f"Unknown domain: {req.domain!r}")
        else:
            passed.append("domain_check")

        # 2. Required parameters
        key = (req.domain, req.type)
        if key not in _REQUIRED_PARAMS:
            available = sorted(
                exp_type for domain, exp_type in _REQUIRED_PARAMS if domain == req.domain
            )
            errors.append(
                f"Unknown experiment type for domain {req.domain!r}: {req.type!r}. "
                f"Available types: {available}"
            )
        else:
            passed.append("known_experiment")
            required = _REQUIRED_PARAMS[key]
            missing = [p for p in required if p not in req.parameters]
            missing.extend(self.check_type_specific_required_params(req))
            if missing:
                errors.append(
                    f"Missing required parameters for {req.domain}/{req.type}: {missing}"
                )
            else:
                passed.append("required_params")

        # 3. Physical constraints
        phys_warnings = self.check_physical_constraints(req.domain, req.type, req.parameters)
        warnings.extend(phys_warnings)
        if not phys_warnings:
            passed.append("physical_constraints")

        # 4. Chemical safety
        smiles = req.parameters.get("smiles", "")
        reaction_type = req.type
        chem_warnings = self.check_chemical_safety(smiles, reaction_type)
        warnings.extend(chem_warnings)
        if not chem_warnings:
            passed.append("chemical_safety")

        # 5. Solver stability
        stab_warnings = self.check_solver_stability(req.parameters)
        warnings.extend(stab_warnings)
        if not stab_warnings:
            passed.append("solver_stability")

        # 6. Hard block check
        blocked = self.flag_unsafe_request(req)
        if blocked:
            errors.append("Request flagged as unsafe and has been blocked.")

        valid = len(errors) == 0 and not blocked
        return ValidationResult(
            valid=valid,
            blocked=blocked,
            warnings=warnings,
            errors=errors,
            checks_passed=passed,
        )

    def check_physical_constraints(
        self, domain: str, sim_type: str, params: dict
    ) -> list[str]:
        """Check that parameter values obey physical laws."""
        warnings: list[str] = []

        # Velocity < c
        for key in ("v0", "velocity", "v1", "v2"):
            v = params.get(key)
            if v is not None and abs(float(v)) >= CONSTANTS.c:
                warnings.append(
                    f"Parameter {key}={v} >= speed of light ({CONSTANTS.c:.3e} m/s). "
                    "Relativistic effects not modelled."
                )

        # Temperature > 0 K
        for key in ("T", "T_hot", "T_cold", "temperature"):
            t = params.get(key)
            if t is not None and float(t) <= 0:
                warnings.append(
                    f"Temperature {key}={t} K is at or below absolute zero."
                )

        # Carnot: T_hot > T_cold
        if sim_type == "carnot_efficiency":
            T_hot = params.get("T_hot")
            T_cold = params.get("T_cold")
            if T_hot is not None and T_cold is not None:
                if float(T_hot) <= float(T_cold):
                    warnings.append(
                        "Carnot efficiency requires T_hot > T_cold. Check your temperatures."
                    )

        # Negative mass
        for key in ("mass", "m1", "m2"):
            m = params.get(key)
            if m is not None and float(m) <= 0:
                warnings.append(f"Mass {key}={m} must be positive.")

        # Pendulum length > 0
        if sim_type == "pendulum":
            length = params.get("length")
            if length is not None and float(length) <= 0:
                warnings.append("Pendulum length must be positive.")

        return warnings

    def check_chemical_safety(self, smiles: str, reaction_type: str) -> list[str]:
        """Flag obviously dangerous chemical requests."""
        warnings: list[str] = []
        smiles_upper = smiles.upper()

        for keyword in _DANGEROUS_SMILES_KEYWORDS:
            if keyword.upper() in smiles_upper:
                warnings.append(
                    f"SMILES contains potentially dangerous substance keyword: {keyword!r}. "
                    "Simulation blocked for safety."
                )

        for dangerous in _DANGEROUS_REACTION_TYPES:
            if dangerous.lower() in reaction_type.lower():
                warnings.append(
                    f"Reaction type {reaction_type!r} is on the blocked list."
                )

        return warnings

    def check_solver_stability(self, params: dict) -> list[str]:
        """Warn about extreme parameter values that may cause numerical instability."""
        warnings: list[str] = []
        for param_name, threshold in _STABILITY_THRESHOLDS.items():
            val = params.get(param_name)
            if val is not None:
                try:
                    fval = float(val)
                    if abs(fval) > threshold:
                        warnings.append(
                            f"Parameter {param_name}={fval:.3e} exceeds stability threshold "
                            f"({threshold:.3e}). Results may be unreliable."
                        )
                except (TypeError, ValueError):
                    pass
        return warnings

    def check_type_specific_required_params(self, req: ExperimentRequest) -> list[str]:
        """Validate conditional parameter requirements that depend on parameter values."""
        params = req.parameters
        missing: list[str] = []

        if req.domain == "physics" and req.type == "ideal_gas":
            gas_vars = ("P", "V", "n", "T")
            known = [name for name in gas_vars if params.get(name) is not None]
            if len(known) != 3:
                missing.append(
                    "ideal_gas requires exactly three known values among P, V, n, T"
                )

        if req.domain == "physics" and req.type == "capacitance":
            geometry = str(params.get("geometry", "")).lower()
            geometry_required = {
                "parallel_plate": ["area", "separation"],
                "cylindrical": ["length", "inner_r", "outer_r"],
                "spherical": ["inner_r", "outer_r"],
            }
            if geometry in geometry_required:
                missing.extend(
                    p for p in geometry_required[geometry] if p not in params
                )
            elif geometry:
                missing.append(
                    "geometry must be one of: parallel_plate, cylindrical, spherical"
                )

        return missing

    def flag_unsafe_request(self, req: ExperimentRequest) -> bool:
        """Return True if the request should be outright blocked."""
        # Block dangerous reaction types
        for dangerous in _DANGEROUS_REACTION_TYPES:
            if dangerous.lower() in req.type.lower():
                return True

        # Block if SMILES contains dangerous keywords
        smiles = req.parameters.get("smiles", "")
        for keyword in _DANGEROUS_SMILES_KEYWORDS:
            if keyword.upper() in smiles.upper():
                return True

        return False
