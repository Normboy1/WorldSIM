"""Regression tests for physics / math / chemistry corrections."""

from __future__ import annotations

import math
from pathlib import Path
import tomllib

import pytest

from simlab.core.engine.simlab_core import SimLabCore
from simlab.core.schemas.experiment import ExperimentRequest
from simlab.engines.chemistry.kinetics import KineticsEngine
from simlab.engines.math.symbolic import SymbolicEngine
from simlab.engines.physics.classical import ClassicalMechanicsEngine


def test_environment_g_merged_for_projectile():
    core = SimLabCore()
    r_earth = core.simulate_physics(
        "projectile_motion",
        {"v0": 25.0, "angle_deg": 45.0},
        environment={"g": 9.81},
    )
    r_moon = core.simulate_physics(
        "projectile_motion",
        {"v0": 25.0, "angle_deg": 45.0},
        environment={"g": 1.62},
    )
    assert r_earth.status == "success" and r_moon.status == "success"
    assert r_moon.results["range"] > r_earth.results["range"]


def test_projectile_matches_analytical_range():
    eng = ClassicalMechanicsEngine()
    v0, angle, g = 25.0, 45.0, 9.81
    res = eng.simulate_projectile_motion(v0, angle, g=g)
    v0y = v0 * math.sin(math.radians(angle))
    expected_range = v0**2 * math.sin(math.radians(2 * angle)) / g
    assert abs(res["range"] - expected_range) < 1e-6
    assert abs(res["max_height"] - v0y**2 / (2 * g)) < 1e-6


def test_gravity_requires_positive_r():
    eng = ClassicalMechanicsEngine()
    with pytest.raises(ValueError, match="positive"):
        eng.simulate_gravity(1.0, 1.0, r=-1.0)


def test_ideal_gas_rejects_nonpositive_knowns():
    from simlab.engines.physics.thermodynamics import ThermodynamicsEngine

    th = ThermodynamicsEngine()
    with pytest.raises(ValueError):
        th.ideal_gas_law(P=1e5, V=-1.0, n=1.0, T=None)


def test_symbolic_solve_list_normalisation():
    sym = SymbolicEngine()
    out = sym.solve_equation("x**2 - 4", variable="x")
    assert out["count"] == 2
    assert set(out["solutions"]) == {"-2", "2"}


def test_critical_points_no_bool_on_symbolic():
    sym = SymbolicEngine()
    out = sym.find_critical_points("x**2", variable="x")
    assert len(out["critical_points"]) == 1
    assert out["critical_points"][0]["type"] == "minimum"


def test_kinetics_first_order_negative_k():
    kin = KineticsEngine()
    with pytest.raises(ValueError):
        kin.simulate_first_order(k=-0.1, A0=1.0)


def test_linear_system_singular_raises_clear_error():
    from simlab.engines.math.linear_algebra import LinearAlgebraEngine

    la = LinearAlgebraEngine()
    with pytest.raises(ValueError, match="no unique solution"):
        la.solve_linear_system([[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0])


def test_experiment_request_environment_merge_via_dispatcher():
    core = SimLabCore()
    req = ExperimentRequest(
        domain="physics",
        type="pendulum",
        parameters={"length": 1.0, "theta0_deg": 30.0, "t_span": 1.0},
        environment={"g": 1.62},
    )
    res = core.run_experiment(req)
    assert res.status == "success"
    assert abs(res.results["natural_frequency"] - math.sqrt(1.62 / 1.0)) < 1e-9


def test_equation_split_skips_relational_operators():
    from simlab.engines.math._sympy_parse import try_split_binary_equals

    assert try_split_binary_equals("x**2 = 4") == ("x**2", "4")
    assert try_split_binary_equals("x <= 1") is None
    assert try_split_binary_equals("a == b") is None


def test_safe_sympify_invalid_expression():
    from simlab.engines.math._sympy_parse import safe_sympify, sympy_locals_with_var

    loc = sympy_locals_with_var("x")
    with pytest.raises(ValueError, match="Could not parse"):
        safe_sympify("x**", loc)


def test_ode_rejects_bad_t_span():
    from simlab.engines.math.ode_solver import ODESolver

    ode = ODESolver()
    with pytest.raises(ValueError, match="t_end"):
        ode.solve_ode("-y", 1.0, [10.0, 0.0], t_eval_points=20)
    with pytest.raises(ValueError, match="t_eval_points"):
        ode.solve_ode("-y", 1.0, [0.0, 1.0], t_eval_points=1)


def test_descriptive_stats_empty_raises():
    from simlab.engines.math.statistics import StatisticsEngine

    with pytest.raises(ValueError, match="non-empty"):
        StatisticsEngine().descriptive_stats([])


def test_monte_carlo_rejects_inverted_range():
    from simlab.engines.math.statistics import StatisticsEngine

    with pytest.raises(ValueError, match="max > min"):
        StatisticsEngine().run_monte_carlo(
            "x", 10, {"x": [1.0, 0.0]}, seed=0
        )


def test_anova_requires_second_group():
    from simlab.engines.math.statistics import StatisticsEngine

    with pytest.raises(ValueError, match="anova requires data2"):
        StatisticsEngine().hypothesis_test([1, 2, 3], None, test="anova")


def test_atomic_domain_dispatches_electron_config():
    core = SimLabCore()
    result = core.simulate_atomic("electron_config", {"Z": 6})

    assert result.status == "success"
    assert result.domain == "atomic"
    assert result.results["symbol"] == "C"
    assert result.results["config_noble"]


def test_nuclear_domain_dispatches_analyze_nucleus():
    core = SimLabCore()
    result = core.simulate_nuclear("analyze_nucleus", {"Z": 6, "N": 6})

    assert result.status == "success"
    assert result.domain == "nuclear"
    assert result.results["symbol"] == "C"
    assert result.results["A"] == 12
    assert result.results["binding_energy_MeV"] > 0
    assert result.results["empirical_comparison"]["isotope"] == "12C"
    assert result.results["empirical_comparison"]["absolute_error_MeV"] < 10.0


def test_new_domains_are_registered_without_optional_imports():
    core = SimLabCore()

    assert ("atomic", "create_element") in core.available_experiments("atomic")
    assert ("atomic", "radial_normalization") in core.available_experiments("atomic")
    assert ("nuclear", "fusion_energy") in core.available_experiments("nuclear")
    assert ("data", "arxiv_search") in core.available_experiments("data")
    assert ("control", "transfer_function") in core.available_experiments("control")
    assert ("materials", "alloy_property_prediction") in core.available_experiments("materials")
    assert ("materials", "degradation_prediction") in core.available_experiments("materials")
    assert ("materials", "microstructure_diffusion") in core.available_experiments("materials")


def test_packaging_metadata_exposes_new_entry_points_and_extras():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text())

    scripts = config["project"]["scripts"]
    assert scripts["simlab-knowledge-mcp"].endswith("simlab_knowledge_server.server:main")

    extras = config["project"]["optional-dependencies"]
    assert "ase" in extras["atomic"]
    assert "pymatgen" in extras["materials"]
    assert "control" in extras["control"]
    assert "nvidia-physicsnemo" in extras["nvidia"]
    assert "warp-lang" in extras["nvidia"]
    assert "nvidia-cupynumeric" in extras["nvidia"]

    pytest_config = config["tool"]["pytest"]["ini_options"]
    assert pytest_config["pythonpath"] == ["."]
    assert "asyncio_mode" not in pytest_config


def test_validator_required_params_cover_all_registered_routes():
    from simlab.core.router.dispatcher import _ROUTING_TABLE
    from simlab.core.validation.validator import _REQUIRED_PARAMS

    assert set(_REQUIRED_PARAMS) == set(_ROUTING_TABLE)


def test_engine_errors_do_not_expose_tracebacks():
    core = SimLabCore()
    result = core.simulate_physics(
        "gravity",
        {"m1": 1.0, "m2": 1.0, "r": -1.0},
    )

    assert result.status == "error"
    assert result.errors
    assert all("Traceback" not in error for error in result.errors)


def test_validator_rejects_unknown_experiment_type_before_dispatch():
    core = SimLabCore()
    result = core.simulate_physics("not_a_simulation", {})

    assert result.status == "error"
    assert "Unknown experiment type" in result.errors[0]


def test_validator_handles_conditional_required_parameters():
    core = SimLabCore()

    ideal_gas = core.simulate_physics("ideal_gas", {"P": 101325.0, "V": 1.0})
    assert ideal_gas.status == "error"
    assert "exactly three known values" in ideal_gas.errors[0]

    capacitance = core.simulate_physics(
        "capacitance",
        {"geometry": "parallel_plate", "area": 1.0},
    )
    assert capacitance.status == "error"
    assert "separation" in capacitance.errors[0]


def test_proof_engine_rejects_unsafe_path_components(tmp_path):
    from simlab.proof.proof_engine import ProofEngine

    proof = ProofEngine(proof_root=tmp_path)
    with pytest.raises(ValueError, match="safe path component"):
        proof.save_proof(
            exp_id="../outside",
            domain="math",
            exp_type="solve_equation",
            parameters={},
            results={},
            plots_b64={},
        )


def test_cors_defaults_are_not_wildcard_credentials(monkeypatch):
    monkeypatch.delenv("SIMLAB_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("SIMLAB_CORS_ALLOW_CREDENTIALS", raising=False)

    from simlab.api.fastapi_server.cors import cors_allow_credentials, cors_origins

    origins = cors_origins()
    assert "*" not in origins
    assert cors_allow_credentials(origins) is False

    monkeypatch.setenv("SIMLAB_CORS_ORIGINS", "*")
    monkeypatch.setenv("SIMLAB_CORS_ALLOW_CREDENTIALS", "true")
    origins = cors_origins()
    assert origins == ["*"]
    assert cors_allow_credentials(origins) is False


def test_hydrogen_radial_wavefunctions_are_normalized():
    from simlab.engines.atomic.hydrogen_orbitals import HydrogenOrbitalEngine

    engine = HydrogenOrbitalEngine()
    one_s = engine.radial_probability_integral(1, 0)
    two_p = engine.radial_probability_integral(2, 1)

    assert one_s["normalization_error"] < 1e-3
    assert two_p["normalization_error"] < 1e-3


def test_atomic_radial_normalization_dispatches():
    core = SimLabCore()
    result = core.simulate_atomic("radial_normalization", {"n": 2, "l": 0})

    assert result.status == "success"
    assert result.results["normalization_error"] < 1e-3


def test_cached_http_client_uses_fresh_and_stale_cache(tmp_path):
    import httpx

    from simlab.engines.data.http_cache import CachedHTTPClient

    calls = {"count": 0}

    def ok_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"ok": True})

    client = CachedHTTPClient(
        cache_dir=tmp_path,
        ttl_s=3600,
        transport=httpx.MockTransport(ok_handler),
    )
    first = client.get_text("https://example.test/data", params={"q": "water"})
    second = client.get_text("https://example.test/data", params={"q": "water"})

    assert first.source == "network"
    assert second.source == "cache"
    assert calls["count"] == 1

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    stale_client = CachedHTTPClient(
        cache_dir=tmp_path,
        ttl_s=-1,
        transport=httpx.MockTransport(failing_handler),
    )
    stale = stale_client.get_text("https://example.test/data", params={"q": "water"})
    assert stale.source == "stale_cache"
    assert stale.json() == {"ok": True}


def test_materials_gpu_backend_status_runs_without_optional_stack():
    core = SimLabCore()
    result = core.simulate_materials("gpu_backend_status", {})

    assert result.status == "success"
    assert "alchemi" in result.results["available"]
    assert "physicsnemo" in result.results["roles"]
    assert result.results["fallback_policy"].startswith("Use NumPy")


def test_alloy_property_prediction_routes_and_scores_superalloy():
    core = SimLabCore()
    result = core.simulate_materials(
        "alloy_property_prediction",
        {
            "composition": {"Ni": 0.62, "Cr": 0.18, "Co": 0.10, "Al": 0.06, "Ti": 0.04},
            "temperature_K": 1050.0,
            "heat_treatment": "solution + aging",
        },
    )

    assert result.status == "success"
    assert result.results["alloy_family"] == "nickel_superalloy_candidate"
    assert result.results["property_estimates"]["oxidation_resistance_score_raw"] > 0.6
    assert result.results["property_estimates"]["yield_strength_proxy_MPa"] > 100.0
    assert any(run["backend"] == "PhysicsNeMo" for run in result.results["recommended_next_runs"])


def test_degradation_prediction_reports_corrosion_and_hydrogen_risk():
    core = SimLabCore()
    result = core.simulate_materials(
        "degradation_prediction",
        {
            "composition": {"Fe": 0.82, "Cr": 0.12, "Ni": 0.06},
            "environment": {
                "temperature_K": 900.0,
                "oxygen_partial_pressure_atm": 0.21,
                "chloride_mol_L": 0.1,
                "pH": 6.5,
                "hydrogen_partial_pressure_atm": 0.02,
            },
            "exposure_hours": 100.0,
        },
    )

    assert result.status == "success"
    estimates = result.results["degradation_estimates"]
    assert estimates["oxide_thickness_um"] >= 0.0
    assert 0.0 <= estimates["corrosion_risk_score"] <= 1.0
    assert 0.0 <= estimates["hydrogen_embrittlement_risk_score"] <= 1.0


def test_microstructure_diffusion_cpu_fallback_summary():
    core = SimLabCore()
    result = core.simulate_materials(
        "microstructure_diffusion",
        {"grid_shape": [12, 10], "time_s": 2.0, "steps": 5},
    )

    assert result.status == "success"
    assert result.results["backend"] in {"numpy", "warp", "numpy_cpu", "warp_gpu", "warp_cuda"}
    assert result.results["field_summary"]["max"] <= 1.0
    assert len(result.results["field_summary"]["centerline"]) == 12


def test_surrogate_model_plan_targets_physicsnemo_workflow():
    core = SimLabCore()
    result = core.simulate_materials(
        "surrogate_model_plan",
        {
            "dataset_size": 100,
            "target_properties": ["phase_stability_score", "oxide_thickness_um"],
        },
    )

    assert result.status == "success"
    assert result.results["dataset_split"]["train"] == 70
    assert "physics residual for diffusion or reaction kinetics" in result.results["loss_terms"]


def test_materials_visual_outputs_return_png_payloads():
    import base64

    core = SimLabCore()
    requests = [
        (
            "alloy_property_prediction",
            {
                "composition": {"Ni": 0.62, "Cr": 0.18, "Co": 0.10, "Al": 0.06, "Ti": 0.04},
                "temperature_K": 1050.0,
            },
            2,
        ),
        (
            "degradation_prediction",
            {
                "composition": {"Fe": 0.82, "Cr": 0.12, "Ni": 0.06},
                "environment": {"temperature_K": 900.0, "chloride_mol_L": 0.1},
                "exposure_hours": 100.0,
            },
            2,
        ),
        (
            "microstructure_diffusion",
            {"grid_shape": [12, 10], "diffusivity_m2_s": 0.5, "time_s": 2.0, "steps": 10},
            2,
        ),
        (
            "surrogate_model_plan",
            {"dataset_size": 100, "target_properties": ["phase_stability_score"]},
            1,
        ),
    ]

    for sim_type, params, min_plots in requests:
        result = core.simulate_materials(sim_type, params, outputs=["plot"])

        assert result.status == "success"
        assert len(result.plots) >= min_plots
        first_png = base64.b64decode(result.plots[0])[:8]
        assert first_png == b"\x89PNG\r\n\x1a\n"
