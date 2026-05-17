"""SIMLAB Proof Engine.

For every experiment run, this module:
  1. Saves all plot PNGs to proof/{domain}/{exp_id}/plots/
  2. Writes a LaTeX source file with results, tables, figures, and references
  3. Compiles it to PDF with pdflatex
  4. Returns the proof directory path and PDF path

The LaTeX template includes:
  - AMS math for equations
  - Proper scientific figure environment
  - BibTeX-style references section
  - Metadata: timestamp, solver, parameters, results
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROOF_ROOT = Path(__file__).resolve().parents[3] / "proof"
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Known scientific references keyed by domain/type
_REFERENCES: dict[str, list[dict]] = {
    "math": [
        {"key": "sympy", "title": "SymPy: symbolic computing in Python",
         "author": "Meurer et al.", "year": 2017,
         "journal": "PeerJ Computer Science", "doi": "10.7717/peerj-cs.103"},
        {"key": "scipy", "title": "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python",
         "author": "Virtanen et al.", "year": 2020,
         "journal": "Nature Methods", "doi": "10.1038/s41592-019-0686-2"},
    ],
    "physics": [
        {"key": "goldstein", "title": "Classical Mechanics (3rd ed.)",
         "author": "Goldstein, Poole, Safko", "year": 2002, "publisher": "Addison Wesley"},
        {"key": "griffiths_em", "title": "Introduction to Electrodynamics (4th ed.)",
         "author": "Griffiths, D.J.", "year": 2013, "publisher": "Pearson"},
    ],
    "chemistry": [
        {"key": "rdkit", "title": "RDKit: Open-source cheminformatics",
         "author": "Landrum, G.", "year": 2006, "url": "https://www.rdkit.org"},
        {"key": "atkins", "title": "Physical Chemistry (11th ed.)",
         "author": "Atkins, P.; de Paula, J.; Keeler, J.", "year": 2018, "publisher": "Oxford University Press"},
    ],
    "atomic": [
        {"key": "griffiths_qm", "title": "Introduction to Quantum Mechanics (3rd ed.)",
         "author": "Griffiths, D.J.; Schroeter, D.F.", "year": 2018, "publisher": "Cambridge University Press"},
        {"key": "atkins_pc", "title": "Physical Chemistry: Quanta, Matter, and Change",
         "author": "Atkins, P.; de Paula, J.; Friedman, R.", "year": 2014, "publisher": "Oxford University Press"},
        {"key": "nist_data", "title": "NIST Atomic Spectra Database",
         "author": "Kramida et al.", "year": 2023,
         "url": "https://physics.nist.gov/asd"},
    ],
    "nuclear": [
        {"key": "bethe1936", "title": "Nuclear Physics A: Stationary States of Nuclei",
         "author": "Bethe, H.A.; Bacher, R.F.", "year": 1936,
         "journal": "Reviews of Modern Physics", "doi": "10.1103/RevModPhys.8.82"},
        {"key": "weizsacker1935", "title": "Zur Theorie der Kernmassen",
         "author": "von Weizs\\\"acker, C.F.", "year": 1935,
         "journal": "Zeitschrift f\\\"ur Physik", "doi": "10.1007/BF01330559"},
        {"key": "krane", "title": "Introductory Nuclear Physics",
         "author": "Krane, K.S.", "year": 1988, "publisher": "Wiley"},
        {"key": "nubase2020", "title": "The NUBASE2020 evaluation of nuclear physics properties",
         "author": "Kondev et al.", "year": 2021,
         "journal": "Chinese Physics C", "doi": "10.1088/1674-1137/abddae"},
    ],
    "hybrid": [
        {"key": "scipy", "title": "SciPy 1.0",
         "author": "Virtanen et al.", "year": 2020,
         "journal": "Nature Methods", "doi": "10.1038/s41592-019-0686-2"},
    ],
    "materials": [
        {"key": "ashcroft", "title": "Solid State Physics",
         "author": "Ashcroft, N.W.; Mermin, N.D.", "year": 1976, "publisher": "Holt, Rinehart and Winston"},
    ],
}


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _safe_path_component(value: str, field_name: str) -> str:
    """Validate user-controlled path components before composing proof paths."""
    component = str(value)
    if (
        not _SAFE_PATH_COMPONENT.fullmatch(component)
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
    ):
        raise ValueError(
            f"{field_name} must be a safe path component containing only "
            "letters, numbers, '.', '_', or '-'."
        )
    return component


def _ref_to_bibtex(ref: dict) -> str:
    key = ref["key"]
    if "journal" in ref:
        btype = "article"
        lines = [f"@article{{{key},"]
    elif "publisher" in ref:
        btype = "book"
        lines = [f"@book{{{key},"]
    elif "url" in ref:
        btype = "misc"
        lines = [f"@misc{{{key},"]
    else:
        lines = [f"@misc{{{key},"]

    lines.append(f"  author  = {{{ref.get('author', 'Unknown')}}},")
    lines.append(f"  title   = {{{ref.get('title', '')}}},")
    lines.append(f"  year    = {{{ref.get('year', '')}}},")
    if "journal" in ref:
        lines.append(f"  journal = {{{ref['journal']}}},")
    if "doi" in ref:
        lines.append(f"  doi     = {{{ref['doi']}}},")
    if "publisher" in ref:
        lines.append(f"  publisher = {{{ref['publisher']}}},")
    if "url" in ref:
        lines.append(f"  url     = {{{ref['url']}}},")
    lines.append("}")
    return "\n".join(lines)


def _dict_to_latex_table(data: dict, caption: str = "") -> str:
    rows = []
    for k, v in data.items():
        if isinstance(v, dict):
            continue
        val_str = _escape_latex(str(v))
        key_str = _escape_latex(str(k).replace("_", " "))
        rows.append(f"    \\texttt{{{key_str}}} & {val_str} \\\\")

    if not rows:
        return ""

    cap_line = f"  \\caption{{{_escape_latex(caption)}}}" if caption else ""
    return (
        "\\begin{table}[H]\n"
        "  \\centering\n"
        "  \\begin{tabular}{ll}\n"
        "    \\hline\n"
        "    \\textbf{Parameter} & \\textbf{Value} \\\\\n"
        "    \\hline\n"
        + "\n".join(rows) + "\n"
        "    \\hline\n"
        "  \\end{tabular}\n"
        + (cap_line + "\n" if cap_line else "")
        + "\\end{table}\n"
    )


class ProofEngine:
    """Saves experiment artifacts and generates LaTeX/PDF proofs."""

    def __init__(self, proof_root: str | Path | None = None) -> None:
        self.proof_root = Path(proof_root) if proof_root else _PROOF_ROOT
        self.proof_root.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def save_proof(
        self,
        exp_id: str,
        domain: str,
        exp_type: str,
        parameters: dict,
        results: dict,
        plots_b64: dict[str, str],        # name → base64 PNG
        status: str = "success",
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
        title: str | None = None,
        extra_notes: str = "",
    ) -> dict:
        """Write all proof artifacts for an experiment.

        Returns dict with keys: proof_dir, tex_path, pdf_path, plot_paths.
        """
        domain_dir = _safe_path_component(domain, "domain")
        exp_dir = _safe_path_component(exp_id, "exp_id")
        proof_dir = self.proof_root / domain_dir / exp_dir
        plots_dir = proof_dir / "plots"
        proof_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save plots as PNG files
        plot_paths: dict[str, Path] = {}
        for name, b64 in plots_b64.items():
            if not isinstance(b64, str) or not b64:
                continue
            png_path = plots_dir / f"{name}.png"
            png_path.write_bytes(base64.b64decode(b64))
            plot_paths[name] = png_path

        # 2. Save raw data as JSON
        data_path = proof_dir / "data.json"
        data_path.write_text(json.dumps({
            "experiment_id": exp_id,
            "domain": domain,
            "type": exp_type,
            "parameters": parameters,
            "results": _strip_plots(results),
            "status": status,
            "warnings": warnings or [],
            "errors": errors or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

        # 3. Write BibTeX
        refs = _REFERENCES.get(domain, _REFERENCES["math"])
        bib_path = proof_dir / "refs.bib"
        bib_path.write_text("\n\n".join(_ref_to_bibtex(r) for r in refs))

        # 4. Build and write LaTeX
        tex_source = self._build_latex(
            exp_id=exp_id,
            domain=domain,
            exp_type=exp_type,
            parameters=parameters,
            results=results,
            plot_paths=plot_paths,
            status=status,
            warnings=warnings or [],
            errors=errors or [],
            refs=refs,
            title=title,
            extra_notes=extra_notes,
        )
        tex_path = proof_dir / "report.tex"
        tex_path.write_text(tex_source, encoding="utf-8")

        # 5. Compile PDF
        pdf_path = self._compile_pdf(tex_path, bib_path)

        return {
            "proof_dir": str(proof_dir),
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path) if pdf_path else None,
            "data_path": str(data_path),
            "plot_paths": {k: str(v) for k, v in plot_paths.items()},
            "compiled_ok": pdf_path is not None,
        }

    # ── LaTeX builder ──────────────────────────────────────────────────────

    def _build_latex(
        self,
        exp_id: str,
        domain: str,
        exp_type: str,
        parameters: dict,
        results: dict,
        plot_paths: dict[str, Path],
        status: str,
        warnings: list[str],
        errors: list[str],
        refs: list[dict],
        title: str | None,
        extra_notes: str,
    ) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        doc_title = title or f"SIMLAB Experiment: {exp_type.replace('_', ' ').title()}"
        status_mark = {"success": "[SUCCESS]", "error": "[ERROR]",
                       "partial": "[PARTIAL]"}.get(status, status.upper())

        # Parameters table
        param_table = _dict_to_latex_table(parameters, caption="Experiment Parameters")

        # Results — flatten one level
        flat_results = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)) and v is not None:
                flat_results[k] = v
            elif isinstance(v, list) and len(v) <= 20:
                flat_results[k] = str(v)
        result_table = _dict_to_latex_table(flat_results, caption="Key Results")

        # Figures — using absolute paths, h placement (no float package needed)
        figures_tex = ""
        for name, path in plot_paths.items():
            caption = _escape_latex(name.replace("_", " ").title())
            abs_path = path.as_posix()
            figures_tex += (
                "\\begin{figure}[h]\n"
                "  \\centering\n"
                f"  \\includegraphics[width=0.85\\textwidth]{{{abs_path}}}\n"
                f"  \\caption{{{caption}}}\n"
                f"  \\label{{fig:{name}}}\n"
                "\\end{figure}\n\n"
            )

        warnings_tex = ""
        if warnings:
            items = "\n".join(f"  \\item {_escape_latex(w)}" for w in warnings)
            warnings_tex = "\\subsection*{Warnings}\n\\begin{itemize}\n" + items + "\n\\end{itemize}\n"

        errors_tex = ""
        if errors:
            items = "\n".join(f"  \\item {_escape_latex(e)}" for e in errors)
            errors_tex = "\\subsection*{Errors}\n\\begin{itemize}\n" + items + "\n\\end{itemize}\n"

        notes_tex = ""
        if extra_notes:
            notes_tex = "\\subsection*{Notes}\n" + _escape_latex(extra_notes) + "\n"

        domain_intro = {
            "math":      "This experiment applies symbolic and numerical mathematical methods "
                         "using SymPy [1] and SciPy [2].",
            "physics":   "This simulation applies classical mechanics and numerical integration "
                         "following the framework of Goldstein et al.~[1].",
            "chemistry": "Molecular analysis performed using RDKit [1], "
                         "following methods from Atkins et al.~[2].",
            "atomic":    "Atomic structure computed via the Aufbau principle and hydrogen-like "
                         "wave functions, per Griffiths [1]. "
                         "Elemental data sourced from the NIST Atomic Spectra Database [3].",
            "nuclear":   "Binding energies computed using the Bethe--Weizsacker semi-empirical "
                         "mass formula [1, 2]. "
                         "Nuclear data cross-referenced with NUBASE2020 [4].",
            "materials": "Material properties computed following Ashcroft & Mermin [1].",
            "hybrid":    "Hybrid simulation combining multiple domain solvers using SciPy [1].",
        }.get(domain, "")

        # Build manual bibliography
        bib_lines = []
        for i, r in enumerate(refs, 1):
            parts = [r.get("author", ""), f"\\textit{{{_escape_latex(r.get('title', ''))}}}",
                     str(r.get("year", ""))]
            if "journal" in r:
                parts.append(_escape_latex(r["journal"]))
            if "publisher" in r:
                parts.append(_escape_latex(r["publisher"]))
            if "doi" in r:
                parts.append(f"DOI: {_escape_latex(r['doi'])}")
            if "url" in r:
                parts.append(f"URL: {_escape_latex(r['url'])}")
            bib_lines.append(f"  \\bibitem{{ref{i}}} {', '.join(p for p in parts if p)}.")
        bib_tex = "\n".join(bib_lines)

        # Build the full document using a plain string (no f-string for LaTeX body)
        # to avoid conflicts between LaTeX {} and Python format fields.
        # Dynamic values are inserted via tagged placeholders then replaced.
        TEMPLATE = (
            "% SIMLAB Proof Document -- Auto-generated\n"
            "% Experiment: __EXP_ID__\n"
            "% Generated:  __TS__\n\n"
            "\\documentclass[12pt,a4paper]{article}\n\n"
            "\\usepackage{amsmath, amssymb}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{geometry}\n"
            "\\usepackage{fancyhdr}\n\n"
            "\\geometry{margin=2.5cm}\n\n"
            "\\pagestyle{fancy}\n"
            "\\fancyhf{}\n"
            "\\lhead{\\textbf{SIMLAB} --- Scientific Simulation Platform}\n"
            "\\rhead{Experiment \\texttt{__EXP_ID__}}\n"
            "\\cfoot{\\thepage}\n\n"
            "\\begin{document}\n\n"
            "\\begin{center}\n"
            "{\\LARGE \\textbf{__TITLE__}}\\\\\n"
            "\\vspace{0.4cm}\n"
            "{\\large SIMLAB Scientific Simulation Platform}\\\\\n"
            "\\vspace{0.3cm}\n"
            "\\hrule\n"
            "\\vspace{0.3cm}\n"
            "\\begin{tabular}{rl}\n"
            "  \\textbf{Experiment ID:} & \\texttt{__EXP_ID__} \\\\\n"
            "  \\textbf{Domain:}        & __DOMAIN__ \\\\\n"
            "  \\textbf{Type:}          & \\texttt{__TYPE__} \\\\\n"
            "  \\textbf{Status:}        & \\textbf{__STATUS__} \\\\\n"
            "  \\textbf{Generated:}     & __TS__ \\\\\n"
            "\\end{tabular}\n"
            "\\vspace{0.3cm}\n"
            "\\hrule\n"
            "\\end{center}\n\n"
            "\\tableofcontents\n"
            "\\newpage\n\n"
            "\\section{Overview}\n"
            "__INTRO__\n\n"
            "This report documents experiment \\texttt{__EXP_ID__}, a\n"
            "\\textit{__TYPE_PRETTY__} simulation in the \\textit{__DOMAIN__} domain.\n\n"
            "\\section{Experiment Parameters}\n"
            "__PARAM_TABLE__\n\n"
            "\\section{Results}\n"
            "__RESULT_TABLE__\n\n"
            "__WARNINGS__\n"
            "__ERRORS__\n"
            "__NOTES__\n"
            "\\section{Visualizations}\n"
            "__FIGURES__\n\n"
            "\\section{Methodology}\n"
            "\\subsection*{Computational Framework}\n"
            "All simulations are executed within the SIMLAB engine pipeline:\n"
            "\\begin{enumerate}\n"
            "  \\item Parameter validation and unit checking\n"
            "  \\item Solver selection (domain-specific backend)\n"
            "  \\item Numerical integration / symbolic computation\n"
            "  \\item Result post-processing and visualization\n"
            "  \\item Proof generation and archiving to \\texttt{proof/} directory\n"
            "\\end{enumerate}\n\n"
            "\\subsection*{Solvers}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Symbolic math:} SymPy\n"
            "  \\item \\textbf{Numerical methods:} NumPy / SciPy (RK45 / LSODA)\n"
            "  \\item \\textbf{Nuclear SEMF:} Bethe--Weizsacker formula\n"
            "\\end{itemize}\n\n"
            "\\section{Raw Data Archive}\n"
            "Full numerical results are in \\texttt{data.json}. "
            "Plots are in the \\texttt{plots/} subdirectory.\n\n"
            "\\section{References}\n"
            "\\begin{thebibliography}{99}\n"
            "__BIBLIOGRAPHY__\n"
            "\\end{thebibliography}\n\n"
            "\\end{document}\n"
        )

        no_result_msg = "\\textit{Results stored in data.json (arrays omitted from table).}"
        no_fig_msg = "\\textit{No visualizations generated for this experiment.}"

        doc = TEMPLATE
        doc = doc.replace("__EXP_ID__", _escape_latex(exp_id))
        doc = doc.replace("__TS__", _escape_latex(ts))
        doc = doc.replace("__TITLE__", _escape_latex(doc_title))
        doc = doc.replace("__DOMAIN__", _escape_latex(domain.title()))
        doc = doc.replace("__TYPE__", _escape_latex(exp_type))
        doc = doc.replace("__TYPE_PRETTY__", _escape_latex(exp_type.replace("_", " ")))
        doc = doc.replace("__STATUS__", _escape_latex(status_mark))
        doc = doc.replace("__INTRO__", domain_intro)
        doc = doc.replace("__PARAM_TABLE__", param_table or "\\textit{No parameters recorded.}")
        doc = doc.replace("__RESULT_TABLE__", result_table or no_result_msg)
        doc = doc.replace("__WARNINGS__", warnings_tex)
        doc = doc.replace("__ERRORS__", errors_tex)
        doc = doc.replace("__NOTES__", notes_tex)
        doc = doc.replace("__FIGURES__", figures_tex or no_fig_msg)
        doc = doc.replace("__BIBLIOGRAPHY__", bib_tex)
        return doc

    # ── PDF compilation ────────────────────────────────────────────────────

    def _compile_pdf(self, tex_path: Path, bib_path: Path | None = None) -> Path | None:
        """Run pdflatex twice in the proof dir. Returns PDF path or None."""
        work_dir = tex_path.parent
        opts = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory", str(work_dir),
            str(tex_path),
        ]
        try:
            subprocess.run(opts, capture_output=True, check=False, cwd=work_dir)
            subprocess.run(opts, capture_output=True, check=False, cwd=work_dir)
            pdf_path = tex_path.with_suffix(".pdf")
            return pdf_path if pdf_path.exists() else None
        except FileNotFoundError:
            return None


def _strip_plots(d: dict) -> dict:
    """Remove base64 plot blobs from a result dict before JSON serialisation."""
    clean = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > 200 and _looks_like_b64(v):
            clean[k] = "<base64 PNG — see plots/ directory>"
        elif isinstance(v, dict):
            clean[k] = _strip_plots(v)
        else:
            clean[k] = v
    return clean


def _looks_like_b64(s: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9+/]+=*$", s[:80]))
