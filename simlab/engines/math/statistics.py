"""Statistics engine using NumPy and SciPy.

Provides Monte Carlo simulation, descriptive stats, distribution fitting,
and hypothesis testing.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy import stats as sci_stats

from simlab.engines.math._sympy_parse import safe_sympify, sympy_locals_for_symbols, sympy_locals_with_var


def _build_callable(f_str: str, variable: str = "x"):
    """Compile an expression string into a numpy-compatible callable."""
    local_dict = sympy_locals_with_var(variable)
    expr = safe_sympify(f_str, local_dict)
    return sp.lambdify(local_dict[variable], expr, modules=["numpy"])


class StatisticsEngine:
    """Numerical statistics and Monte Carlo engine."""

    def run_monte_carlo(
        self,
        f_str: str,
        n_samples: int,
        variable_ranges: dict[str, list[float]],
        seed: int | None = None,
    ) -> dict:
        """Monte Carlo integration of f over the given variable ranges.

        Uses uniform random sampling over the hypercube defined by variable_ranges.

        Parameters
        ----------
        f_str:
            Expression string. Must reference variables matching keys in variable_ranges.
        n_samples:
            Number of random samples to draw.
        variable_ranges:
            Dict mapping variable name to [min, max].
        seed:
            Optional RNG seed for reproducibility.
        """
        if int(n_samples) < 1:
            raise ValueError("n_samples must be at least 1.")
        if not variable_ranges:
            raise ValueError("variable_ranges must not be empty.")

        rng = np.random.default_rng(seed)
        var_names = list(variable_ranges.keys())
        for var, (lo, hi) in variable_ranges.items():
            lo_f, hi_f = float(lo), float(hi)
            if not (np.isfinite(lo_f) and np.isfinite(hi_f)):
                raise ValueError(f"Bounds for {var!r} must be finite.")
            if hi_f <= lo_f:
                raise ValueError(f"variable_ranges[{var!r}] must have max > min.")

        local_dict = sympy_locals_for_symbols(var_names)
        expr = safe_sympify(f_str, local_dict)
        syms = [local_dict[v] for v in var_names]
        f_lam = sp.lambdify(syms, expr, modules=["numpy"])

        # Sample each variable uniformly over its range
        samples = {}
        volume = 1.0
        for var, (lo, hi) in variable_ranges.items():
            samples[var] = rng.uniform(lo, hi, n_samples)
            volume *= (hi - lo)

        f_vals = f_lam(*[samples[v] for v in var_names])
        f_vals = np.asarray(f_vals, dtype=float)

        integral = float(np.mean(f_vals)) * volume
        std_error = float(np.std(f_vals) / np.sqrt(n_samples)) * volume

        return {
            "integral": integral,
            "std_error": std_error,
            "n_samples": n_samples,
            "mean_f": float(np.mean(f_vals)),
            "volume": volume,
            "confidence_95": [integral - 1.96 * std_error, integral + 1.96 * std_error],
        }

    def descriptive_stats(self, data: list[float]) -> dict:
        """Compute descriptive statistics for a dataset."""
        if not data:
            raise ValueError("data must be a non-empty list of numbers.")
        arr = np.array(data, dtype=float)
        q1, q2, q3 = np.percentile(arr, [25, 50, 75])
        mode_result = sci_stats.mode(arr, keepdims=True)

        return {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "mode": float(mode_result.mode[0]),
            "std": float(np.std(arr, ddof=1)),
            "variance": float(np.var(arr, ddof=1)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "q1": float(q1),
            "q2": float(q2),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "skewness": float(sci_stats.skew(arr)),
            "kurtosis": float(sci_stats.kurtosis(arr)),
        }

    def fit_distribution(self, data: list[float], dist_name: str = "norm") -> dict:
        """Fit a SciPy distribution to data and return parameters + goodness-of-fit.

        Parameters
        ----------
        dist_name:
            Any distribution name from scipy.stats, e.g. 'norm', 'expon', 'lognorm'.
        """
        if not data:
            raise ValueError("data must be a non-empty list of numbers.")
        arr = np.array(data, dtype=float)
        dist = getattr(sci_stats, dist_name, None)
        if dist is None:
            raise ValueError(f"Unknown distribution: {dist_name!r}")

        params = dist.fit(arr)
        # KS test for goodness of fit
        ks_stat, ks_pval = sci_stats.kstest(arr, dist_name, args=params)

        param_names = dist.shapes.split(",") if dist.shapes else []
        param_names = [p.strip() for p in param_names] + ["loc", "scale"]

        return {
            "distribution": dist_name,
            "parameters": dict(zip(param_names[-len(params):], [float(p) for p in params])),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "good_fit": bool(ks_pval > 0.05),
            "log_likelihood": float(np.sum(dist.logpdf(arr, *params))),
        }

    def hypothesis_test(
        self,
        data1: list[float],
        data2: list[float] | None = None,
        test: str = "ttest",
        mu: float = 0.0,
    ) -> dict:
        """Run a hypothesis test.

        Supported tests: 'ttest' (Welch's t-test), 'ttest_1samp', 'mannwhitney',
        'wilcoxon', 'anova', 'ks_2samp', 'levene'.

        Parameters
        ----------
        data1, data2: input samples (data2 optional for one-sample tests)
        test:         name of the test
        mu:           null hypothesis mean for one-sample t-test
        """
        if not data1:
            raise ValueError("data1 must be a non-empty list of numbers.")
        arr1 = np.array(data1, dtype=float)
        arr2 = np.array(data2, dtype=float) if data2 is not None else None

        def _two_sample_check():
            if arr2 is None:
                raise ValueError(f"Test {test!r} requires data2.")
            if len(arr2) == 0:
                raise ValueError("data2 must be non-empty for two-sample tests.")

        if test == "ttest":
            _two_sample_check()
            stat, pval = sci_stats.ttest_ind(arr1, arr2, equal_var=False)
            test_name = "Welch's t-test (two-sample)"
        elif test == "ttest_1samp":
            stat, pval = sci_stats.ttest_1samp(arr1, mu)
            test_name = f"One-sample t-test (mu={mu})"
        elif test == "mannwhitney":
            _two_sample_check()
            stat, pval = sci_stats.mannwhitneyu(arr1, arr2, alternative="two-sided")
            test_name = "Mann-Whitney U test"
        elif test == "wilcoxon":
            _two_sample_check()
            stat, pval = sci_stats.wilcoxon(arr1, arr2)
            test_name = "Wilcoxon signed-rank test"
        elif test == "ks_2samp":
            _two_sample_check()
            stat, pval = sci_stats.ks_2samp(arr1, arr2)
            test_name = "Kolmogorov-Smirnov two-sample test"
        elif test == "levene":
            _two_sample_check()
            stat, pval = sci_stats.levene(arr1, arr2)
            test_name = "Levene's test for equal variances"
        elif test == "anova":
            groups = [arr1] + ([arr2] if arr2 is not None else [])
            if len(groups) < 2:
                raise ValueError("anova requires data2 (at least two groups).")
            stat, pval = sci_stats.f_oneway(*groups)
            test_name = "One-way ANOVA"
        else:
            raise ValueError(f"Unknown test: {test!r}")

        return {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(pval),
            "reject_null_alpha_05": bool(pval < 0.05),
            "reject_null_alpha_01": bool(pval < 0.01),
            "n1": len(arr1),
            "n2": len(arr2) if arr2 is not None else None,
        }
