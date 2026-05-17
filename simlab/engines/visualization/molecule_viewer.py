"""Molecule structure visualization.

Uses RDKit Draw for 2D structure images (graceful fallback if not installed).
Returns base64-encoded PNG strings.
"""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    _RDKIT_AVAILABLE = True
except ImportError:
    pass


def _fallback_image(smiles: str, size: tuple[int, int]) -> str:
    """Generate a placeholder image with the SMILES string when RDKit is unavailable."""
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax.text(
        0.5, 0.5,
        f"RDKit not available\n\nSMILES:\n{smiles}",
        ha="center", va="center", fontsize=10, wrap=True,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


class MoleculeViewer:
    """2D and grid molecule structure visualizer."""

    @property
    def available(self) -> bool:
        return _RDKIT_AVAILABLE

    def render_2d_molecule(self, smiles: str, size: tuple[int, int] = (400, 300)) -> str:
        """Render a 2D molecule structure image.

        Returns a base64-encoded PNG string.
        Falls back to a placeholder if RDKit is not installed.
        """
        if not _RDKIT_AVAILABLE:
            return _fallback_image(smiles, size)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _fallback_image(f"Invalid SMILES: {smiles}", size)

        AllChem.Compute2DCoords(mol)

        # Use the SVG drawer for better quality, then convert to PNG via PIL
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            drawer.drawOptions().addStereoAnnotation = True
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            png_data = drawer.GetDrawingText()
            return base64.b64encode(png_data).decode("ascii")
        except Exception:
            # Fallback to simpler Draw.MolToImage
            img = Draw.MolToImage(mol, size=size)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("ascii")

    def render_molecule_grid(
        self,
        smiles_list: list[str],
        legends: list[str] | None = None,
        mol_size: tuple[int, int] = (300, 200),
        mols_per_row: int = 3,
    ) -> str:
        """Render a grid of molecule structure images.

        Returns a base64-encoded PNG string of the grid.
        Falls back to a placeholder if RDKit is not installed.
        """
        if not _RDKIT_AVAILABLE:
            return _fallback_image("\n".join(smiles_list), (900, 400))

        mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)

        if not mols:
            return _fallback_image("No valid SMILES", (900, 400))

        if legends is None:
            legends = smiles_list[: len(mols)]

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=mol_size,
            legends=legends[: len(mols)],
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
