"""arXiv data engine.

Queries the arXiv API (free, no key) to search scientific papers
relevant to SIMLAB experiments: physics, chemistry, math, nuclear.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from simlab.engines.data.http_cache import CachedHTTPClient

_BASE = "http://export.arxiv.org/api/query"
_TIMEOUT = 20.0

_CATEGORY_MAP = {
    "physics":   "physics.class-ph OR physics.comp-ph",
    "nuclear":   "nucl-th OR nucl-ex",
    "atomic":    "physics.atom-ph OR quant-ph",
    "chemistry": "physics.chem-ph OR q-bio.BM",
    "math":      "math.NA OR math.MP OR math-ph",
    "materials": "cond-mat.mtrl-sci OR cond-mat.str-el",
    "fluid":     "physics.flu-dyn",
    "control":   "eess.SY OR cs.SY",
}


def _parse_feed(xml: str) -> list[dict]:
    """Minimal Atom-feed parser (no lxml dependency)."""
    papers = []
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
    for entry in entries:
        def get(tag: str) -> str:
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.DOTALL)
            return m.group(1).strip() if m else ""

        def get_all(tag: str) -> list[str]:
            return re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.DOTALL)

        arxiv_id_raw = get("id")
        arxiv_id = re.search(r"abs/(.+)$", arxiv_id_raw)
        arxiv_id = arxiv_id.group(1) if arxiv_id else arxiv_id_raw

        authors = [re.sub(r"<[^>]+>", "", a).strip() for a in get_all("name")]
        cats = [re.search(r'term="([^"]+)"', c).group(1)
                for c in re.findall(r'<category[^>]*/>', entry)
                if re.search(r'term="([^"]+)"', c)]
        published = get("published")[:10]  # YYYY-MM-DD

        papers.append({
            "arxiv_id": arxiv_id,
            "title": re.sub(r"\s+", " ", get("title")),
            "authors": authors[:5],  # first 5
            "published": published,
            "summary": re.sub(r"\s+", " ", get("summary"))[:300] + "…",
            "categories": cats,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return papers


class ArXivEngine:
    """arXiv paper search for SIMLAB scientific context."""

    def __init__(
        self,
        cache_dir: str | None = None,
        cache_ttl_s: float | None = None,
        use_cache: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = CachedHTTPClient(
            timeout=_TIMEOUT,
            cache_dir=cache_dir,
            ttl_s=cache_ttl_s,
            use_cache=use_cache,
            transport=transport,
        )

    def search(self, query: str, domain: str | None = None,
               max_results: int = 8, sort_by: str = "relevance") -> dict:
        """Search arXiv for papers.

        Parameters
        ----------
        query      : search terms
        domain     : optional SIMLAB domain to prepend category filter
                     ('physics','nuclear','atomic','chemistry','math','materials','fluid','control')
        max_results: max papers to return
        sort_by    : 'relevance' | 'lastUpdatedDate' | 'submittedDate'
        """
        cat_filter = _CATEGORY_MAP.get(domain or "", "")
        if cat_filter:
            search_query = f"({cat_filter}) AND all:{query}"
        else:
            search_query = f"all:{query}"

        params = {
            "search_query": search_query,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        try:
            response = self._client.get_text(_BASE, params=params)
            papers = _parse_feed(response.text)
        except Exception as exc:
            return {"error": str(exc), "query": query}

        return {
            "query": query,
            "domain_filter": domain,
            "count": len(papers),
            "papers": papers,
            "data_source": {
                "provider": "arXiv",
                "access": response.source,
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }

    def get_paper(self, arxiv_id: str) -> dict:
        """Fetch metadata for a specific arXiv paper by ID (e.g. '2301.12345')."""
        params = {"id_list": arxiv_id, "max_results": 1}
        try:
            response = self._client.get_text(_BASE, params=params)
            papers = _parse_feed(response.text)
            if not papers:
                return {"error": "Not found", "arxiv_id": arxiv_id}
            paper = papers[0]
            paper["data_source"] = {
                "provider": "arXiv",
                "access": response.source,
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            return paper
        except Exception as exc:
            return {"error": str(exc), "arxiv_id": arxiv_id}

    def find_references_for_experiment(self, domain: str, exp_type: str) -> dict:
        """Auto-search arXiv for papers relevant to a SIMLAB experiment type."""
        query_map = {
            "projectile_motion":     "classical mechanics projectile trajectory",
            "pendulum":              "nonlinear pendulum chaos bifurcation",
            "spring_mass":           "harmonic oscillator damping resonance",
            "electric_field":        "electrostatics Coulomb field simulation",
            "heat_transfer":         "heat conduction numerical simulation",
            "solve_ode":             "numerical ODE solver Runge-Kutta stiff systems",
            "molecule_analysis":     "cheminformatics molecular descriptors QSAR",
            "reaction_kinetics":     "chemical kinetics reaction rate simulation",
            "create_element":        "nuclear binding energy semi-empirical mass formula",
            "nuclear_fusion":        "nuclear fusion Q-value reaction cross section",
            "decay_chain":           "radioactive decay chain Bateman equations",
            "hydrogen_orbitals":     "hydrogen atom wave function quantum mechanics",
            "crystal_structure":     "crystal structure lattice DFT first principles",
            "phase_diagram":         "binary phase diagram thermodynamics CALPHAD",
            "control_systems":       "PID control transfer function stability",
            "molecular_dynamics":    "molecular dynamics force field simulation",
        }
        query = query_map.get(exp_type, f"{domain} {exp_type.replace('_', ' ')} simulation")
        return self.search(query, domain=domain, max_results=5, sort_by="relevance")
