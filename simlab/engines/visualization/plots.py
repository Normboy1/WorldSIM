"""Plot engine using Matplotlib.

All public methods return base64-encoded PNG strings so they can be
embedded in JSON responses, HTML reports, or MCP tool outputs.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from simlab.engines.math._sympy_parse import safe_sympify, sympy_locals_with_var


def _fig_to_b64(fig: plt.Figure) -> str:
    """Serialize a Matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _parse_expr(expr_str: str, x_sym: sp.Symbol) -> sp.Expr:
    local_dict = sympy_locals_with_var("x")
    return safe_sympify(expr_str, local_dict)


class PlotEngine:
    """Matplotlib-backed plotting engine, returning base64 PNG strings."""

    def plot_function(
        self,
        expr_str: str,
        x_range: list[float],
        title: str = "",
        n_points: int = 500,
        xlabel: str = "x",
        ylabel: str = "f(x)",
    ) -> str:
        """Plot a symbolic function f(x) over x_range=[x_min, x_max].

        Returns a base64-encoded PNG string.
        """
        x_sym = sp.Symbol("x")
        expr = _parse_expr(expr_str, x_sym)
        f = sp.lambdify(x_sym, expr, modules=["numpy"])

        x_arr = np.linspace(x_range[0], x_range[1], n_points)
        y_arr = np.array([float(np.real(f(xi))) for xi in x_arr])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_arr, y_arr, linewidth=2, color="#2196F3")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title or f"f(x) = {expr_str}", fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_data(
        self,
        x: list[float],
        y: list[float],
        title: str = "",
        xlabel: str = "x",
        ylabel: str = "y",
        plot_type: str = "line",
        color: str = "#2196F3",
    ) -> str:
        """Plot x/y data as a line, scatter, bar, or step chart.

        Returns a base64-encoded PNG string.
        """
        x_arr = np.array(x)
        y_arr = np.array(y)

        fig, ax = plt.subplots(figsize=(8, 5))

        if plot_type == "scatter":
            ax.scatter(x_arr, y_arr, color=color, s=20, alpha=0.7)
        elif plot_type == "bar":
            ax.bar(x_arr, y_arr, color=color, alpha=0.8)
        elif plot_type == "step":
            ax.step(x_arr, y_arr, color=color, linewidth=2, where="mid")
        elif plot_type == "area":
            ax.fill_between(x_arr, y_arr, alpha=0.4, color=color)
            ax.plot(x_arr, y_arr, color=color, linewidth=2)
        else:  # line
            ax.plot(x_arr, y_arr, color=color, linewidth=2)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_bar_labels(
        self,
        labels: list[str],
        values: list[float],
        title: str = "",
        ylabel: str = "Value",
        color: str = "#2196F3",
    ) -> str:
        """Plot categorical values as a bar chart. Returns base64 PNG."""
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        ax.bar(x, np.array(values, dtype=float), color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_heatmap(
        self,
        matrix: list[list[float]],
        title: str = "",
        xlabel: str = "x",
        ylabel: str = "y",
        colorbar_label: str = "Value",
        cmap: str = "viridis",
    ) -> str:
        """Plot a 2D scalar field as a heatmap. Returns base64 PNG."""
        arr = np.array(matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(arr, origin="lower", cmap=cmap, aspect="auto")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(colorbar_label)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_trajectory(
        self,
        x_arr: list[float],
        y_arr: list[float],
        title: str = "Trajectory",
        xlabel: str = "x [m]",
        ylabel: str = "y [m]",
        color: str = "#E91E63",
    ) -> str:
        """Plot a 2D trajectory (e.g. projectile path).

        Marks start and end points. Returns base64 PNG.
        """
        xa = np.array(x_arr)
        ya = np.array(y_arr)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(xa, ya, color=color, linewidth=2.5)
        ax.scatter([xa[0]], [ya[0]], color="green", s=80, zorder=5, label="Start")
        ax.scatter([xa[-1]], [ya[-1]], color="red", s=80, zorder=5, label="End")
        ax.fill_between(xa, ya, alpha=0.12, color=color)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_multiple(
        self,
        datasets: list[dict[str, Any]],
        title: str = "",
        xlabel: str = "x",
        ylabel: str = "y",
    ) -> str:
        """Plot multiple datasets on one axes.

        Each dataset dict should have 'x', 'y', and optionally 'label' and 'color'.
        Returns base64 PNG.
        """
        fig, ax = plt.subplots(figsize=(9, 5))

        for i, ds in enumerate(datasets):
            x = np.array(ds["x"])
            y = np.array(ds["y"])
            label = ds.get("label", f"Series {i+1}")
            color = ds.get("color", None)
            ax.plot(x, y, label=label, color=color, linewidth=2)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_phase_portrait(
        self,
        x_arr: list[float],
        y_arr: list[float],
        title: str = "Phase Portrait",
        xlabel: str = "x",
        ylabel: str = "dx/dt",
    ) -> str:
        """Plot a phase portrait (position vs velocity or state1 vs state2).

        Color-maps the trajectory by time. Returns base64 PNG.
        """
        xa = np.array(x_arr)
        ya = np.array(y_arr)
        t_norm = np.linspace(0, 1, len(xa))

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(xa, ya, c=t_norm, cmap="plasma", s=5, alpha=0.8)
        plt.colorbar(sc, ax=ax, label="Normalized time")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_time_series(
        self,
        t_arr: list[float],
        y_arrs: list[list[float]],
        labels: list[str],
        title: str = "",
        xlabel: str = "t",
        ylabel: str = "",
    ) -> str:
        """Plot one or more time series on the same axes.

        Parameters
        ----------
        t_arr:  shared time array
        y_arrs: list of y arrays, one per series
        labels: series labels (must match length of y_arrs)
        """
        t = np.array(t_arr)
        fig, ax = plt.subplots(figsize=(9, 5))

        colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
        for i, (y, lbl) in enumerate(zip(y_arrs, labels)):
            ax.plot(t, np.array(y), label=lbl, color=colors[i % len(colors)], linewidth=2)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_histogram(
        self,
        data: list[float],
        bins: int = 30,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Count",
    ) -> str:
        """Plot a histogram. Returns base64 PNG."""
        arr = np.array(data)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(arr, bins=bins, color="#2196F3", edgecolor="white", alpha=0.85)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return _fig_to_b64(fig)
