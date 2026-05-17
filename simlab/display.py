"""SIMLAB display utilities.

Decodes base64 PNG strings from engine outputs and opens them in the
system image viewer (xdg-open on Linux).  Also provides matplotlib
figure helpers for scripts that run with a live DISPLAY.
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Switch to an interactive backend when a display is available.
_DISPLAY = os.environ.get("DISPLAY", "")
if _DISPLAY and matplotlib.get_backend().lower() in ("agg", ""):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass


def show(b64_png: str, title: str = "SIMLAB Plot", block: bool = False) -> Path:
    """Decode a base64 PNG and open it in the system viewer.

    Returns the temp file path so callers can reference it later.
    Set block=True to wait for the viewer to close.
    """
    raw = base64.b64decode(b64_png)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".png", prefix="simlab_", delete=False
    )
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    path = Path(tmp.name)
    _open_image(path, block=block)
    return path


def show_all(b64_plots: list[str], titles: list[str] | None = None) -> list[Path]:
    """Open multiple base64 PNG plots. Returns list of temp file paths."""
    titles = titles or [f"SIMLAB Plot {i+1}" for i in range(len(b64_plots))]
    paths = []
    for b64, title in zip(b64_plots, titles):
        paths.append(show(b64, title=title, block=False))
    return paths


def show_result(result: dict) -> list[Path]:
    """Open all plots embedded in an ExperimentResult dict."""
    plots = result.get("plots", [])
    if not plots:
        print("[SIMLAB Display] No plots found in result.")
        return []
    return show_all(plots)


def save(b64_png: str, path: str | Path, fmt: str = "png") -> Path:
    """Save a base64 PNG to a file without opening a viewer."""
    raw = base64.b64decode(b64_png)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return p


def fig_to_b64(fig: plt.Figure) -> str:
    """Convert a live Matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def show_fig(fig: plt.Figure, block: bool = True) -> None:
    """Show a live Matplotlib figure — uses plt.show() when a display is
    available, otherwise saves to a temp file and opens with xdg-open."""
    if _DISPLAY:
        plt.figure(fig.number)
        plt.show(block=block)
    else:
        b64 = fig_to_b64(fig)
        show(b64)


# ── internal ────────────────────────────────────────────────────────────────

def _open_image(path: Path, block: bool = False) -> None:
    viewers = ["eog", "feh", "display", "xdg-open"]
    for viewer in viewers:
        try:
            if block:
                subprocess.run([viewer, str(path)], check=False)
            else:
                subprocess.Popen(
                    [viewer, str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return
        except FileNotFoundError:
            continue
    print(f"[SIMLAB Display] Could not find an image viewer. Plot saved to: {path}")
