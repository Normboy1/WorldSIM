"""RDKit molecular chemistry engine.

All RDKit imports are guarded with try/except so the module can be imported
even if RDKit is not installed. Methods will raise a descriptive RuntimeError
in that case rather than a cryptic ImportError.
"""

from __future__ import annotations

import base64
import io

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
    _RDKIT_AVAILABLE = True
except ImportError:
    pass


def _require_rdkit():
    if not _RDKIT_AVAILABLE:
        raise RuntimeError(
            "RDKit is not installed. Install it with: pip install rdkit-pypi\n"
            "or: pip install simlab[chemistry]"
        )


class RDKitEngine:
    """RDKit-backed molecular analysis engine."""

    @property
    def available(self) -> bool:
        """True if RDKit is installed and usable."""
        return _RDKIT_AVAILABLE

    def parse_smiles(self, smiles: str) -> dict:
        """Check whether a SMILES string is valid and return basic info."""
        _require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "valid": False,
                "smiles": smiles,
                "canonical_smiles": None,
                "error": "RDKit could not parse this SMILES string.",
            }
        canonical = Chem.MolToSmiles(mol)
        return {
            "valid": True,
            "smiles": smiles,
            "canonical_smiles": canonical,
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
        }

    def analyze_molecule(self, smiles: str) -> dict:
        """Full molecular analysis: formula, MW, atom/bond counts, rings, H-bond donors/acceptors."""
        _require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        mol_h = Chem.AddHs(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = Descriptors.MolWt(mol)
        exact_mw = Descriptors.ExactMolWt(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        atom_counts: dict[str, int] = {}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            atom_counts[sym] = atom_counts.get(sym, 0) + 1

        return {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "molecular_formula": formula,
            "molecular_weight": float(mw),
            "exact_molecular_weight": float(exact_mw),
            "num_atoms": mol.GetNumAtoms(),
            "num_atoms_with_H": mol_h.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": int(num_rings),
            "num_aromatic_rings": int(num_aromatic_rings),
            "h_bond_donors": int(hbd),
            "h_bond_acceptors": int(hba),
            "rotatable_bonds": int(rot_bonds),
            "atom_counts": atom_counts,
        }

    def calculate_molecular_weight(self, smiles: str) -> float:
        """Return the average molecular weight."""
        _require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        return float(Descriptors.MolWt(mol))

    def calculate_descriptors(self, smiles: str) -> dict:
        """Compute a set of common molecular descriptors using RDKit."""
        _require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        logP = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        mw = Descriptors.MolWt(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        frac_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)

        # Lipinski's Rule of Five
        ro5_violations = sum([
            mw > 500,
            logP > 5,
            hbd > 5,
            hba > 10,
        ])

        return {
            "logP": float(logP),
            "TPSA": float(tpsa),
            "rotatable_bonds": int(rot_bonds),
            "h_bond_donors": int(hbd),
            "h_bond_acceptors": int(hba),
            "num_rings": int(rings),
            "num_aromatic_rings": int(aromatic_rings),
            "molecular_weight": float(mw),
            "num_heavy_atoms": int(heavy_atoms),
            "fraction_csp3": float(frac_sp3),
            "lipinski_ro5_violations": int(ro5_violations),
            "drug_like": bool(ro5_violations <= 1),
        }

    def generate_2d_image(self, smiles: str, size: tuple[int, int] = (400, 300)) -> str:
        """Render a 2D structure image and return as a base64-encoded PNG string."""
        _require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def generate_3d_coordinates(self, smiles: str) -> dict:
        """Generate 3D atomic coordinates via ETKDG embedding.

        Returns a dict with atom symbols and their (x, y, z) positions.
        """
        _require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        mol = Chem.AddHs(mol)
        result = -1
        for name in ("ETKDGv3", "ETKDGv2", "ETKDG"):
            factory = getattr(AllChem, name, None)
            if factory is not None:
                try:
                    result = AllChem.EmbedMolecule(mol, factory())
                except TypeError:
                    result = AllChem.EmbedMolecule(mol, factory)
                if result >= 0:
                    break
        if result == -1:
            # Fallback to basic embedding
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            raise RuntimeError("3D embedding failed for this molecule.")

        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()

        atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atoms.append({
                "index": i,
                "symbol": atom.GetSymbol(),
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z),
            })

        return {
            "atoms": atoms,
            "num_atoms": len(atoms),
            "smiles": smiles,
            "method": "ETKDG (v3/v2/v1 fallback) + MMFF94",
        }
