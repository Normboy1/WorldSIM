"""SimLabCore: main orchestrator for the SIMLAB platform.

Combines validation, dispatch, and convenience wrappers.
"""

from __future__ import annotations

import statistics
from typing import Any

from simlab.core.router.dispatcher import ExperimentDispatcher
from simlab.core.schemas.experiment import ExperimentRequest, ExperimentResult
from simlab.core.validation.validator import SimLabValidator


class SimLabCore:
    """Top-level orchestrator.

    Usage::

        core = SimLabCore()
        result = core.run_experiment(ExperimentRequest(
            domain="physics",
            type="projectile_motion",
            parameters={"v0": 25.0, "angle_deg": 45.0},
            outputs=["plot"],
        ))
    """

    def __init__(self) -> None:
        self._validator = SimLabValidator()
        self._dispatcher = ExperimentDispatcher()

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run_experiment(self, req: ExperimentRequest) -> ExperimentResult:
        """Validate and run an experiment, returning a structured result.

        Validation failures are surfaced as ExperimentResult with status='error'.
        """
        # 1. Validate
        try:
            validation = self._validator.validate_experiment(req)
        except Exception as exc:
            return ExperimentResult(
                experiment_id=req.experiment_id,
                domain=req.domain,
                type=req.type,
                status="error",
                errors=[f"Validation error: {exc}"],
            )

        if not validation.valid:
            return ExperimentResult(
                experiment_id=req.experiment_id,
                domain=req.domain,
                type=req.type,
                status="error",
                errors=validation.errors,
                warnings=validation.warnings,
            )

        # 2. Dispatch
        try:
            result = self._dispatcher.dispatch(req)
        except Exception as exc:
            return ExperimentResult(
                experiment_id=req.experiment_id,
                domain=req.domain,
                type=req.type,
                status="error",
                errors=[f"Experiment failed: {exc}"],
            )

        # 3. Attach validation warnings to result
        if validation.warnings:
            result.warnings.extend(validation.warnings)

        return result

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def solve_math(
        self,
        math_type: str,
        parameters: dict[str, Any],
        outputs: list[str] | None = None,
    ) -> ExperimentResult:
        """Shortcut for math domain experiments.

        Parameters
        ----------
        math_type:  experiment type, e.g. 'solve_equation', 'differentiate', 'eigenvalues'
        parameters: domain-specific parameters dict
        outputs:    optional list of output formats ('plot', 'json', etc.)
        """
        req = ExperimentRequest(
            domain="math",
            type=math_type,
            parameters=parameters,
            outputs=outputs or [],
        )
        return self.run_experiment(req)

    def simulate_physics(
        self,
        sim_type: str,
        parameters: dict[str, Any],
        outputs: list[str] | None = None,
        environment: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        """Shortcut for physics domain simulations.

        Parameters
        ----------
        sim_type:    simulation type, e.g. 'projectile_motion', 'pendulum', 'electric_field'
        parameters:  physics parameters
        environment: environment overrides (e.g. {"g": 1.62} for moon gravity)
        """
        req = ExperimentRequest(
            domain="physics",
            type=sim_type,
            parameters=parameters,
            environment=environment or {},
            outputs=outputs or [],
        )
        return self.run_experiment(req)

    def simulate_chemistry(
        self,
        sim_type: str,
        parameters: dict[str, Any],
        outputs: list[str] | None = None,
    ) -> ExperimentResult:
        """Shortcut for chemistry domain simulations."""
        req = ExperimentRequest(
            domain="chemistry",
            type=sim_type,
            parameters=parameters,
            outputs=outputs or [],
        )
        return self.run_experiment(req)

    def simulate_materials(
        self,
        sim_type: str,
        parameters: dict[str, Any],
        outputs: list[str] | None = None,
    ) -> ExperimentResult:
        """Shortcut for materials domain simulations."""
        req = ExperimentRequest(
            domain="materials",
            type=sim_type,
            parameters=parameters,
            outputs=outputs or [],
        )
        return self.run_experiment(req)

    def simulate_atomic(
        self,
        sim_type: str,
        parameters: dict[str, Any],
        outputs: list[str] | None = None,
    ) -> ExperimentResult:
        """Shortcut for atomic-domain simulations."""
        req = ExperimentRequest(
            domain="atomic",
            type=sim_type,
            parameters=parameters,
            outputs=outputs or [],
        )
        return self.run_experiment(req)

    def simulate_nuclear(
        self,
        sim_type: str,
        parameters: dict[str, Any],
        outputs: list[str] | None = None,
    ) -> ExperimentResult:
        """Shortcut for nuclear-domain simulations."""
        req = ExperimentRequest(
            domain="nuclear",
            type=sim_type,
            parameters=parameters,
            outputs=outputs or [],
        )
        return self.run_experiment(req)

    def query_data(
        self,
        query_type: str,
        parameters: dict[str, Any],
    ) -> ExperimentResult:
        """Shortcut for data-source lookups such as arXiv and PubChem."""
        req = ExperimentRequest(
            domain="data",
            type=query_type,
            parameters=parameters,
        )
        return self.run_experiment(req)

    def analyze_control(
        self,
        analysis_type: str,
        parameters: dict[str, Any],
        outputs: list[str] | None = None,
    ) -> ExperimentResult:
        """Shortcut for control-systems analyses."""
        req = ExperimentRequest(
            domain="control",
            type=analysis_type,
            parameters=parameters,
            outputs=outputs or [],
        )
        return self.run_experiment(req)

    def simulate_nvidia(
        self,
        sim_type: str,
        parameters: dict[str, Any],
        outputs: list[str] | None = None,
    ) -> ExperimentResult:
        """Shortcut for NVIDIA GPU backend experiments (ALCHEMI, Warp, PhysicsNeMo, CuPy).

        sim_type options:
          alchemi_relax, alchemi_md, alchemi_mlip, alchemi_batch_screen
          warp_diffusion, warp_allen_cahn
          nemo_train_fno, nemo_sampling_plan
          cupy_diffusion_sweep
        """
        req = ExperimentRequest(
            domain="nvidia",
            type=sim_type,
            parameters=parameters,
            outputs=outputs or [],
        )
        return self.run_experiment(req)

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def analyze_results(self, result: ExperimentResult) -> dict[str, Any]:
        """Produce a lightweight summary of numeric results.

        Returns a dict with summary statistics for any list-valued result fields.
        """
        summary: dict[str, Any] = {
            "experiment_id": result.experiment_id,
            "domain": result.domain,
            "type": result.type,
            "status": result.status,
            "n_plots": len(result.plots),
            "n_warnings": len(result.warnings),
            "n_errors": len(result.errors),
            "numeric_summaries": {},
        }

        for key, val in result.results.items():
            if isinstance(val, list) and len(val) > 1:
                try:
                    floats = [float(v) for v in val]
                    summary["numeric_summaries"][key] = {
                        "min": min(floats),
                        "max": max(floats),
                        "mean": statistics.mean(floats),
                        "stdev": statistics.stdev(floats) if len(floats) > 1 else 0.0,
                        "n": len(floats),
                    }
                except (TypeError, ValueError):
                    pass
            elif isinstance(val, (int, float)):
                summary["numeric_summaries"][key] = {"value": float(val)}

        return summary

    def compare_experiments(self, results: list[ExperimentResult]) -> dict[str, Any]:
        """Compare a list of ExperimentResult objects.

        Returns a comparison dict with status breakdown and shared numeric key comparisons.
        """
        if not results:
            return {"error": "No results provided."}

        status_counts: dict[str, int] = {}
        for r in results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

        # Find scalar result keys common to all successful results
        successful = [r for r in results if r.status == "success"]
        shared_scalar_keys: set[str] = set()
        if successful:
            shared_scalar_keys = {
                k for k, v in successful[0].results.items()
                if isinstance(v, (int, float))
            }
            for r in successful[1:]:
                shared_scalar_keys &= {
                    k for k, v in r.results.items() if isinstance(v, (int, float))
                }

        comparison: dict[str, list[Any]] = {}
        for key in shared_scalar_keys:
            comparison[key] = [r.results[key] for r in successful]

        return {
            "n_experiments": len(results),
            "status_counts": status_counts,
            "experiment_ids": [r.experiment_id for r in results],
            "domains": list({r.domain for r in results}),
            "types": list({r.type for r in results}),
            "scalar_comparison": comparison,
        }

    def available_experiments(self, domain: str | None = None) -> list[tuple[str, str]]:
        """Return all registered (domain, type) experiment pairs."""
        return self._dispatcher.available_types(domain)
