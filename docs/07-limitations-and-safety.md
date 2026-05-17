# Limitations, safety, and expectations

## Intended fidelity

SIMLAB prioritizes **a single consistent API** and **transparent routing** over replacing specialized research software. Models are often **idealized** (e.g. no air resistance on projectiles unless explicitly added, simplified nuclear parameterizations, network-backed data with caching).

Use results for **education**, **prototyping**, and **agent toolchains**; validate independently for **publication-grade** or **safety-critical** decisions.

## Optional dependencies

Routes may be **listed** even if the backing library is not installed. A dispatch error at runtime usually means you need the matching extra (`chemistry`, `atomic`, `materials`, `control`, etc.).

## Validation and blocking

`SimLabValidator` checks:

- Known **domains** and **required parameters** for common `(domain, type)` pairs.
- **Physical plausibility hints** (e.g. speeds vs. \(c\), positive masses, temperature ordering for Carnot).
- **Keyword-based** chemical safety warnings and blocks for certain SMILES substrings / disallowed “reaction type” strings.

Blocking is **heuristic**, not a substitute for institutional biosafety, export control, or chemistry lab protocols.

## Symbolic and string-driven math

Math engines accept **strings** that are parsed with SymPy. Malformed input should raise **`ValueError`** with a clear message (`safe_sympify`). Equation parsing avoids naive splits on `=` when relational operators (`<=`, `==`, …) appear.

## Data APIs

arXiv and PubChem access is **network-dependent**. Configure **`SIMLAB_DATA_CACHE_*`** for offline-friendly behavior when stale cache is acceptable.

## Environment merge

`ExperimentRequest.environment` is **merged into** the parameter dict passed to engines (`environment` first, then `parameters` overrides). This is convenient for defaults like **`g`**, but agents should not rely on undocumented keys; engines only consume what their methods expect.

## Versioning

This documentation matches the **0.1.x** line. Experiment catalogs and tools may expand; re-run `SimLabCore().available_experiments()` after upgrades.
