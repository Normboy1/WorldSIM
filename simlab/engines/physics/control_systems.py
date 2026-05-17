"""Control systems engine using python-control.

Supports transfer functions, state-space models, PID design,
Bode/Nyquist/root-locus plots, and step/impulse response.
"""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np

try:
    import control
    from control import (
        TransferFunction, StateSpace,
        step_response, impulse_response, frequency_response,
        bode_plot, nyquist_plot, root_locus_plot,
        feedback, series, parallel,
        margin, poles, zeros,
    )
    _CTRL_OK = True
except ImportError:
    _CTRL_OK = False


def _pid(Kp: float, Ki: float, Kd: float) -> "TransferFunction":
    """Build a PID controller C(s) = Kp + Ki/s + Kd·s as a TransferFunction.

    python-control 0.10+ removed the ``pid`` helper, so construct it directly:
    C(s) = (Kd·s² + Kp·s + Ki) / s.
    """
    return TransferFunction([Kd, Kp, Ki], [1.0, 0.0])


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _require_ctrl() -> None:
    if not _CTRL_OK:
        raise ImportError("python-control not installed. Run: pip install control")


class ControlSystemsEngine:
    """Classical and modern control systems analysis."""

    # ── System creation ────────────────────────────────────────────────────

    def transfer_function(self, numerator: list[float], denominator: list[float]) -> dict:
        """Create and analyse a transfer function G(s) = num(s) / den(s).

        Example: G(s) = 1 / (s^2 + 2s + 1)  → num=[1], den=[1,2,1]
        """
        _require_ctrl()
        G = TransferFunction(numerator, denominator)
        ps = [complex(p) for p in poles(G)]
        zs = [complex(z) for z in zeros(G)]
        gm, pm, wg, wp = margin(G)

        return {
            "numerator": numerator,
            "denominator": denominator,
            "poles": [{"re": p.real, "im": p.imag} for p in ps],
            "zeros": [{"re": z.real, "im": z.imag} for z in zs],
            "gain_margin_dB": round(20 * np.log10(gm), 3) if gm and np.isfinite(gm) else None,
            "phase_margin_deg": round(float(pm), 3) if pm and np.isfinite(pm) else None,
            "gain_crossover_rad_s": round(float(wp), 4) if wp and np.isfinite(wp) else None,
            "phase_crossover_rad_s": round(float(wg), 4) if wg and np.isfinite(wg) else None,
            "is_stable": all(p.real < 0 for p in ps),
            "dc_gain": round(float(G.dcgain()), 4) if np.isfinite(float(G.dcgain())) else None,
        }

    def pid_controller(self, Kp: float, Ki: float, Kd: float) -> dict:
        """Build a PID controller C(s) = Kp + Ki/s + Kd*s."""
        _require_ctrl()
        C = _pid(Kp, Ki, Kd)
        ps = [complex(p) for p in poles(C)]
        return {
            "Kp": Kp, "Ki": Ki, "Kd": Kd,
            "poles": [{"re": p.real, "im": p.imag} for p in ps],
            "description": f"C(s) = {Kp} + {Ki}/s + {Kd}·s",
        }

    # ── Analysis plots ─────────────────────────────────────────────────────

    def plot_step_response(self, numerator: list[float], denominator: list[float],
                           t_end: float = 10.0, label: str = "G(s)") -> str:
        """Plot unit step response. Returns base64 PNG."""
        _require_ctrl()
        G = TransferFunction(numerator, denominator)
        t, y = step_response(G, T=np.linspace(0, t_end, 500))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, y, color="#2196F3", linewidth=2.2, label=label)
        ax.axhline(y[-1], color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(t[-1] * 0.02, float(y[-1]) * 1.02,
                f"Steady state ≈ {float(y[-1]):.3f}", fontsize=9, color="gray")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Output", fontsize=12)
        ax.set_title(f"Unit Step Response — {label}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_bode(self, numerator: list[float], denominator: list[float],
                  label: str = "G(s)") -> str:
        """Plot Bode magnitude and phase diagram. Returns base64 PNG."""
        _require_ctrl()
        G = TransferFunction(numerator, denominator)
        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        mag, phase, omega = bode_plot(G, plot=False)
        mag_db = 20 * np.log10(np.abs(mag) + 1e-12)
        phase_deg = np.degrees(phase)

        ax_mag.semilogx(omega, mag_db, color="#2196F3", linewidth=2)
        ax_mag.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_mag.set_ylabel("Magnitude (dB)", fontsize=11)
        ax_mag.set_title(f"Bode Plot — {label}", fontsize=13)
        ax_mag.grid(True, which="both", alpha=0.3)

        ax_ph.semilogx(omega, phase_deg, color="#E91E63", linewidth=2)
        ax_ph.axhline(-180, color="gray", linestyle="--", alpha=0.5)
        ax_ph.set_xlabel("Frequency (rad/s)", fontsize=11)
        ax_ph.set_ylabel("Phase (°)", fontsize=11)
        ax_ph.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_root_locus(self, numerator: list[float], denominator: list[float],
                        label: str = "G(s)") -> str:
        """Plot root locus. Returns base64 PNG."""
        _require_ctrl()
        G = TransferFunction(numerator, denominator)
        fig, ax = plt.subplots(figsize=(8, 7))
        root_locus_plot(G, plot=True, ax=ax)
        ax.set_title(f"Root Locus — {label}", fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_nyquist(self, numerator: list[float], denominator: list[float],
                     label: str = "G(s)") -> str:
        """Plot Nyquist diagram. Returns base64 PNG."""
        _require_ctrl()
        G = TransferFunction(numerator, denominator)
        fig, ax = plt.subplots(figsize=(8, 7))
        nyquist_plot(G, plot=True, ax=ax)
        ax.set_title(f"Nyquist Diagram — {label}", fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def closed_loop_analysis(self, plant_num: list[float], plant_den: list[float],
                              controller_num: list[float], controller_den: list[float]) -> dict:
        """Analyse a unity-feedback closed-loop system T = C*G / (1 + C*G)."""
        _require_ctrl()
        G = TransferFunction(plant_num, plant_den)
        C = TransferFunction(controller_num, controller_den)
        T = feedback(series(C, G))

        ps = [complex(p) for p in poles(T)]
        t, y = step_response(T, T=np.linspace(0, 20, 500))
        y_arr = np.asarray(y).ravel()

        # Rise time (10% → 90%)
        ss = float(y_arr[-1])
        try:
            t10 = float(t[np.where(y_arr >= 0.1 * ss)[0][0]])
            t90 = float(t[np.where(y_arr >= 0.9 * ss)[0][0]])
            rise_time = round(t90 - t10, 4)
        except IndexError:
            rise_time = None

        # Settling time (within 2%)
        try:
            settle_idx = np.where(np.abs(y_arr - ss) > 0.02 * abs(ss))[0]
            settling_time = round(float(t[settle_idx[-1]]), 4) if len(settle_idx) else 0.0
        except Exception:
            settling_time = None

        overshoot = round((float(y_arr.max()) - ss) / ss * 100, 2) if ss > 0 else None

        return {
            "closed_loop_poles": [{"re": p.real, "im": p.imag} for p in ps],
            "is_stable": all(p.real < 0 for p in ps),
            "steady_state_gain": round(ss, 4),
            "rise_time_s": rise_time,
            "settling_time_s": settling_time,
            "percent_overshoot": overshoot,
        }

    def design_pid_for_plant(self, plant_num: list[float], plant_den: list[float],
                              Kp: float, Ki: float, Kd: float) -> dict:
        """Design and analyse a PID-controlled plant."""
        _require_ctrl()
        G = TransferFunction(plant_num, plant_den)
        C = _pid(Kp, Ki, Kd)
        T = feedback(series(C, G))
        analysis = self.closed_loop_analysis(plant_num, plant_den, [Kp], [1])

        return {
            "pid_gains": {"Kp": Kp, "Ki": Ki, "Kd": Kd},
            "closed_loop": analysis,
        }
