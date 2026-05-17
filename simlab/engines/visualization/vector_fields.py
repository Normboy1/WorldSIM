"""Vector field visualization engine using Matplotlib.

All methods return base64-encoded PNG strings.
"""

from __future__ import annotations

import base64
import io
import math

import matplotlib.pyplot as plt
import numpy as np

from simlab.core.constants.physical import CONSTANTS


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


class VectorFieldEngine:
    """Matplotlib-backed vector field visualizations."""

    def plot_vector_field(
        self,
        X: list[list[float]],
        U: list[list[float]],
        V: list[list[float]],
        Y: list[list[float]] | None = None,
        title: str = "Vector Field",
        xlabel: str = "x",
        ylabel: str = "y",
        use_streamplot: bool = True,
    ) -> str:
        """Plot a 2D vector field using streamlines or quiver arrows.

        Parameters
        ----------
        X, Y: 2D grid arrays (from np.meshgrid)
        U, V: x and y components of the field
        use_streamplot: True for streamplot, False for quiver
        """
        X_arr = np.array(X)
        U_arr = np.array(U)
        V_arr = np.array(V)
        Y_arr = np.array(Y) if Y is not None else X_arr.T

        magnitude = np.sqrt(U_arr**2 + V_arr**2)

        fig, ax = plt.subplots(figsize=(8, 7))

        if use_streamplot:
            # streamplot needs 1D x/y axes
            x_lin = X_arr[0, :]
            y_lin = Y_arr[:, 0]
            strm = ax.streamplot(
                x_lin, y_lin, U_arr, V_arr,
                color=magnitude, cmap="viridis", linewidth=1.5, density=1.5,
            )
            plt.colorbar(strm.lines, ax=ax, label="|Field|")
        else:
            q = ax.quiver(X_arr, Y_arr, U_arr, V_arr, magnitude, cmap="viridis", alpha=0.8)
            plt.colorbar(q, ax=ax, label="|Field|")

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_electric_field(
        self,
        charges: list[float],
        positions: list[list[float]],
        grid_extent: float = 5.0,
        grid_points: int = 25,
    ) -> str:
        """Compute and plot the electric field of point charges.

        Returns a base64 PNG with streamlines colored by field magnitude.
        Positive charges are plotted in red, negative in blue.
        """
        k_e = 1 / (4 * math.pi * CONSTANTS.epsilon_0)
        x_lin = np.linspace(-grid_extent, grid_extent, grid_points)
        y_lin = np.linspace(-grid_extent, grid_extent, grid_points)
        X, Y = np.meshgrid(x_lin, y_lin)

        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)

        for q, (px, py) in zip(charges, positions):
            dx = X - px
            dy = Y - py
            r2 = dx**2 + dy**2 + 1e-20
            r = np.sqrt(r2)
            E_mag = k_e * q / r2
            Ex += E_mag * dx / r
            Ey += E_mag * dy / r

        magnitude = np.sqrt(Ex**2 + Ey**2)
        # Log-scale for better visual range
        log_mag = np.log1p(magnitude)

        fig, ax = plt.subplots(figsize=(8, 7))

        strm = ax.streamplot(
            x_lin, y_lin, Ex, Ey,
            color=log_mag, cmap="RdYlBu_r", linewidth=1.2, density=1.5,
        )
        plt.colorbar(strm.lines, ax=ax, label="log(1 + |E|) [V/m]")

        # Mark charge positions
        for q, (px, py) in zip(charges, positions):
            color = "red" if q > 0 else "blue"
            marker = "+" if q > 0 else "_"
            ax.scatter(px, py, c=color, s=200, zorder=10, linewidths=3)
            ax.annotate(
                f"{'+'if q>0 else '-'}{abs(q):.1e}C",
                (px, py), textcoords="offset points", xytext=(8, 8), fontsize=9
            )

        ax.set_xlim(-grid_extent, grid_extent)
        ax.set_ylim(-grid_extent, grid_extent)
        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("y [m]", fontsize=12)
        ax.set_title("Electric Field", fontsize=13)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_magnetic_field(
        self,
        wire_data: list[dict],
        grid_extent: float = 5.0,
        grid_points: int = 25,
    ) -> str:
        """Plot the magnetic field of current-carrying wires.

        Parameters
        ----------
        wire_data:
            List of dicts: [{"x": float, "y": float, "current": float}, ...]
        """
        mu0 = CONSTANTS.mu_0
        x_lin = np.linspace(-grid_extent, grid_extent, grid_points)
        y_lin = np.linspace(-grid_extent, grid_extent, grid_points)
        X, Y = np.meshgrid(x_lin, y_lin)

        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)

        for wire in wire_data:
            wx, wy = wire["x"], wire["y"]
            I = wire.get("current", 1.0)
            dx = X - wx
            dy = Y - wy
            r2 = dx**2 + dy**2 + 1e-20
            factor = (mu0 * I) / (2 * math.pi * r2)
            Bx += -factor * dy
            By += factor * dx

        magnitude = np.sqrt(Bx**2 + By**2)
        log_mag = np.log1p(magnitude)

        fig, ax = plt.subplots(figsize=(8, 7))
        strm = ax.streamplot(
            x_lin, y_lin, Bx, By,
            color=log_mag, cmap="plasma", linewidth=1.2, density=1.5
        )
        plt.colorbar(strm.lines, ax=ax, label="log(1 + |B|) [T]")

        for wire in wire_data:
            ax.scatter(wire["x"], wire["y"], c="orange", s=100, zorder=10)

        ax.set_xlim(-grid_extent, grid_extent)
        ax.set_ylim(-grid_extent, grid_extent)
        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("y [m]", fontsize=12)
        ax.set_title("Magnetic Field", fontsize=13)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _fig_to_b64(fig)
