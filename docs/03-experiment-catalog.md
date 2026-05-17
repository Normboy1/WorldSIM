# Experiment catalog

Every runnable experiment is a pair **`(domain, type)`** registered in `simlab.core.router.dispatcher._ROUTING_TABLE`. Parameters are passed as a dict; some keys are renamed via `param_map` (for example `equation` → `equation_str` for `solve_equation`).

Below is a **reference list** aligned with the codebase (SIMLAB v0.1).

---

## `math`

| `type` | Engine area | Notes |
|--------|-------------|--------|
| `solve_equation` | Symbolic | Param `equation`; optional `variable`. |
| `simplify` | Symbolic | `expression` → `expr_str`. |
| `differentiate` | Symbolic | `expression`, `variable`, `order`. |
| `integrate` | Symbolic | `expression`, optional `limits`. |
| `expand` | Symbolic | |
| `factor` | Symbolic | |
| `critical_points` | Symbolic | |
| `taylor_series` | Symbolic | `point`, `order`. |
| `solve_ode` | ODE | `equation` → RHS string `f(t,y)`; `y0`, `t_span`. |
| `solve_system_ode` | ODE | List `equations`, `y0`, `t_span`; optional `var_names`. |
| `solve_2nd_order` | ODE | Linear 2nd order: `coeff_a`, `coeff_b`, `coeff_c`, `y0`, `yp0`, `t_span`. |
| `solve_linear` | Linear algebra | `A`, `b`. |
| `eigenvalues` | Linear algebra | `matrix` → `A`. |
| `svd` | Linear algebra | |
| `determinant` | Linear algebra | |
| `inverse` | Linear algebra | |
| `matrix_multiply` | Linear algebra | `A`, `B`. |
| `rank` | Linear algebra | |
| `optimize` | Optimization | `expression`, `x0`. |
| `maximize` | Optimization | |
| `fit_curve` | Optimization | `x_data`, `y_data`, `model_str`, optional `p0`. |
| `minimize_interval` | Optimization | `expression`, interval `a`, `b`. |
| `monte_carlo` | Statistics | `expression`, `n_samples`, `variable_ranges`, optional `seed`. |
| `descriptive_stats` | Statistics | `data`. |
| `fit_distribution` | Statistics | `data`, optional `dist_name`. |
| `hypothesis_test` | Statistics | `data1`, optional `data2`, `test`, optional `mu`. |

---

## `physics`

| `type` | Topic |
|--------|--------|
| `projectile_motion` | Classical |
| `pendulum` | Classical |
| `spring_mass` | Classical |
| `collision` | Classical |
| `gravity` | Classical |
| `circular_motion` | Classical |
| `electric_field` | EM |
| `coulomb_force` | EM |
| `magnetic_field` | EM |
| `capacitance` | EM |
| `heat_transfer` | Thermo |
| `ideal_gas` | Thermo |
| `entropy` | Thermo |
| `carnot_efficiency` | Thermo |

Use `environment` (e.g. `{"g": 1.62}`) for defaults merged into parameters before dispatch.

---

## `chemistry`

| `type` | Backend |
|--------|---------|
| `molecule_analysis` | RDKit |
| `molecular_descriptors` | RDKit |
| `parse_smiles` | RDKit |
| `molecule_image` | RDKit |
| `3d_coordinates` | RDKit |
| `first_order` | Kinetics |
| `second_order` | Kinetics |
| `consecutive` | Kinetics |
| `michaelis_menten` | Kinetics |
| `equilibrium` | Kinetics |
| `reaction_kinetics` | Kinetics (alias of first-order) |

---

## `materials`

| `type` | Backend |
|--------|---------|
| `create_lattice` | Lattice (dispatched by `lattice_type`) |
| `fcc_lattice` | Lattice |
| `bcc_lattice` | Lattice |
| `simple_cubic` | Lattice |
| `stress_test` | Stress–strain |
| `youngs_modulus` | Stress–strain |
| `elastic_deformation` | Stress–strain |
| `stress_strain_curve` | Stress–strain |
| `element_properties` | pymatgen (optional) |
| `compare_elements` | pymatgen |
| `build_structure` | pymatgen |
| `common_structure` | pymatgen |
| `phase_diagram` | pymatgen |
| `property_trends` | pymatgen |
| `oxidation_states` | pymatgen |

---

## `atomic`

| `type` | Backend |
|--------|---------|
| `electron_config` | Electron shell / Aufbau |
| `compare_elements` | Electron config |
| `shell_diagram` | Plots |
| `orbital_diagram` | Plots |
| `ionization_trend` | Plots |
| `hydrogen_energy_levels` | Hydrogen-like orbitals |
| `hydrogen_energy_diagram` | Plots |
| `radial_probability` | Plots |
| `radial_normalization` | Check / integral |
| `orbital_2d` | Plots |
| `radial_comparison` | Plots |
| `create_element` | Element builder |
| `fusion_to_element` | Element builder |
| `compare_isotopes` | Element builder |
| `bulk_crystal` | ASE (optional) |
| `molecule` | ASE |
| `crystal_plot` | ASE |
| `molecule_plot` | ASE |
| `compare_crystals` | ASE |
| `surface_slab` | ASE |
| `ase_element_data` | ASE |

---

## `nuclear`

| `type` | Backend |
|--------|---------|
| `analyze_nucleus` | Nuclear engine (SEMF-style analysis) |
| `binding_energy_curve` | Nuclear engine |
| `nuclear_chart` | Nuclear engine |
| `fusion_energy` | Nuclear engine |
| `fission_energy` | Nuclear engine |
| `decay` | Decay |
| `decay_chain` | Decay |
| `decay_plot` | Decay |
| `decay_chain_plot` | Decay |
| `alpha_decay` | Decay |
| `beta_minus` | Decay |
| `beta_plus` | Decay |

---

## `data`

| `type` | Source |
|--------|--------|
| `arxiv_search` | arXiv API |
| `arxiv_paper` | arXiv |
| `arxiv_references` | arXiv |
| `pubchem_search` | PubChem |
| `pubchem_by_cid` | PubChem |
| `pubchem_synonyms` | PubChem |
| `pubchem_safety` | PubChem |
| `pubchem_similar` | PubChem |
| `pubchem_substructure` | PubChem |

---

## `control`

| `type` | Topic |
|--------|--------|
| `transfer_function` | LTI |
| `pid_controller` | PID |
| `step_response` | Time domain plot |
| `bode` | Frequency domain |
| `root_locus` | Plot |
| `nyquist` | Plot |
| `closed_loop` | Analysis |
| `design_pid` | Synthesis |

Requires optional **control** dependency.

---

## `hybrid`

| `type` | Maps to |
|--------|---------|
| `reaction_kinetics` | Chemistry first-order kinetics |
| `consecutive_kinetics` | Consecutive reactions |

---

## Discovering types in code

```python
from simlab.core.engine.simlab_core import SimLabCore

core = SimLabCore()
core.available_experiments()           # all (domain, type) pairs
core.available_experiments("math")   # math only
```

The authoritative list is **`_ROUTING_TABLE`** in `simlab/core/router/dispatcher.py`.
