"""Example: Molecular analysis of caffeine (and aspirin for comparison).

Demonstrates:
- SMILES parsing and validation
- Molecular formula / MW calculation
- Lipinski descriptors
- 2D image generation (if RDKit is available)
- SimLabCore integration
"""

import sys
import os
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simlab.core.engine.simlab_core import SimLabCore
from simlab.engines.chemistry.rdkit_backend import RDKitEngine
from simlab.engines.visualization.molecule_viewer import MoleculeViewer


# Caffeine SMILES
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
# Aspirin SMILES
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
# Ethanol SMILES
ETHANOL = "CCO"


def print_analysis(name: str, smiles: str, engine: RDKitEngine):
    print(f"\n{'='*50}")
    print(f"Molecule: {name}")
    print(f"SMILES:   {smiles}")

    # Parse
    parse = engine.parse_smiles(smiles)
    if not parse["valid"]:
        print(f"ERROR: {parse['error']}")
        return

    print(f"Canonical SMILES: {parse['canonical_smiles']}")

    # Full analysis
    info = engine.analyze_molecule(smiles)
    print(f"\nMolecular Formula: {info['molecular_formula']}")
    print(f"Molecular Weight:  {info['molecular_weight']:.3f} g/mol")
    print(f"Atom counts:       {info['atom_counts']}")
    print(f"Num rings:         {info['num_rings']} ({info['num_aromatic_rings']} aromatic)")
    print(f"H-bond donors:     {info['h_bond_donors']}")
    print(f"H-bond acceptors:  {info['h_bond_acceptors']}")
    print(f"Rotatable bonds:   {info['rotatable_bonds']}")

    # Descriptors
    desc = engine.calculate_descriptors(smiles)
    print(f"\nLogP:              {desc['logP']:.3f}")
    print(f"TPSA:              {desc['TPSA']:.2f} Å²")
    print(f"Fraction Csp3:     {desc['fraction_csp3']:.3f}")
    print(f"Lipinski violations: {desc['lipinski_ro5_violations']} / 4")
    print(f"Drug-like:         {desc['drug_like']}")


def main():
    core = SimLabCore()
    engine = RDKitEngine()
    viewer = MoleculeViewer()

    print("SIMLAB Chemistry: Molecular Analysis")
    print("=" * 50)
    print(f"RDKit available: {engine.available}")

    if not engine.available:
        print("\nRDKit not installed. Install with: pip install rdkit-pypi")
        print("Running basic demo without molecular analysis...")
        print(f"\nCaffeine SMILES: {CAFFEINE}")
        print("(Install RDKit for full analysis)")

        # Still demo the kinetics part
        from simlab.engines.chemistry.kinetics import KineticsEngine
        kin = KineticsEngine()
        r = kin.simulate_first_order(k=0.1, A0=1.0, t_span=30)
        print(f"\nFirst-order decay demo (k=0.1, A0=1.0):")
        print(f"  t=0:   A = {r['A'][0]:.4f}")
        print(f"  t=t½:  A ≈ {r['A'][len(r['A'])//2]:.4f}")
        print(f"  t=end: A = {r['A'][-1]:.4f}")
        print(f"  Half-life: {r['half_life']:.3f} s")
        return

    # Analyze molecules
    for name, smiles in [("Caffeine", CAFFEINE), ("Aspirin", ASPIRIN), ("Ethanol", ETHANOL)]:
        print_analysis(name, smiles, engine)

    # Generate 2D image of caffeine
    print(f"\n{'='*50}")
    print("Generating 2D structure images...")
    for name, smiles in [("Caffeine", CAFFEINE), ("Aspirin", ASPIRIN)]:
        try:
            b64 = viewer.render_2d_molecule(smiles, size=(400, 300))
            fname = f"{name.lower()}_2d.png"
            out_path = os.path.join(os.path.dirname(__file__), fname)
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"  {name} structure saved to: {out_path}")
        except Exception as e:
            print(f"  {name}: Could not render image ({e})")

    # Grid of molecules
    try:
        b64_grid = viewer.render_molecule_grid(
            [CAFFEINE, ASPIRIN, ETHANOL],
            legends=["Caffeine", "Aspirin", "Ethanol"],
        )
        grid_path = os.path.join(os.path.dirname(__file__), "molecule_grid.png")
        with open(grid_path, "wb") as f:
            f.write(base64.b64decode(b64_grid))
        print(f"  Molecule grid saved to: {grid_path}")
    except Exception as e:
        print(f"  Grid render failed: {e}")

    # Via SimLabCore
    print(f"\n--- Via SimLabCore ---")
    result = core.simulate_chemistry("molecule_analysis", {"smiles": CAFFEINE})
    print(f"Status: {result.status}")
    if result.status == "success":
        r = result.results
        print(f"Caffeine: {r.get('molecular_formula')}, MW={r.get('molecular_weight'):.2f} g/mol")
    elif result.errors:
        print(f"Error: {result.errors[0][:200]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
