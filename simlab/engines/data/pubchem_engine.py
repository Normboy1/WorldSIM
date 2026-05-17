"""PubChem data engine.

Queries the free PubChem REST API (no key required) for chemical compound
data: properties, synonyms, safety, bioassays, and similar compounds.
"""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote

import httpx

from simlab.engines.data.http_cache import CachedHTTPClient

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_TIMEOUT = 15.0


class PubChemEngine:
    """Direct PubChem API client for chemical data retrieval."""

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

    def _get(self, url: str, params: dict | None = None) -> tuple[dict, str]:
        data, source = self._client.get_json(url, params=params)
        return data, source

    @staticmethod
    def _source(provider: str, access: str | list[str]) -> dict:
        return {
            "provider": provider,
            "access": access,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def search_compound(self, name: str) -> dict:
        """Search PubChem by compound name. Returns CID and basic properties."""
        encoded_name = quote(name, safe="")
        url = f"{_BASE}/compound/name/{encoded_name}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,InChIKey,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,Charge/JSON"
        try:
            data, property_source = self._get(url)
            props = data["PropertyTable"]["Properties"][0]
        except Exception as exc:
            return {"error": str(exc), "query": name}

        # Get CID
        cid_url = f"{_BASE}/compound/name/{encoded_name}/cids/JSON"
        try:
            cid_data, cid_source = self._get(cid_url)
            cid = cid_data["IdentifierList"]["CID"][0]
        except Exception:
            cid_source = "unavailable"
            cid = props.get("CID")

        return {
            "query": name,
            "cid": cid,
            "molecular_formula": props.get("MolecularFormula"),
            "molecular_weight_g_mol": props.get("MolecularWeight"),
            "iupac_name": props.get("IUPACName"),
            "canonical_smiles": props.get("CanonicalSMILES"),
            "inchikey": props.get("InChIKey"),
            "logP": props.get("XLogP"),
            "tpsa_A2": props.get("TPSA"),
            "hbond_donors": props.get("HBondDonorCount"),
            "hbond_acceptors": props.get("HBondAcceptorCount"),
            "rotatable_bonds": props.get("RotatableBondCount"),
            "heavy_atoms": props.get("HeavyAtomCount"),
            "charge": props.get("Charge"),
            "data_source": self._source("PubChem", [property_source, cid_source]),
        }

    def get_by_cid(self, cid: int) -> dict:
        """Fetch compound properties by PubChem CID."""
        url = (f"{_BASE}/compound/cid/{cid}/property/"
               "MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,"
               "IsomericSMILES,InChI,InChIKey,XLogP,TPSA,HBondDonorCount,"
               "HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,Complexity/JSON")
        try:
            data, source = self._get(url)
            props = data["PropertyTable"]["Properties"][0]
            props["cid"] = cid
            props["data_source"] = self._source("PubChem", source)
            return props
        except Exception as exc:
            return {"error": str(exc), "cid": cid}

    def get_synonyms(self, name: str, limit: int = 20) -> dict:
        """Return synonyms for a compound name."""
        url = f"{_BASE}/compound/name/{quote(name, safe='')}/synonyms/JSON"
        try:
            data, source = self._get(url)
            syns = data["InformationList"]["Information"][0]["Synonym"]
            return {
                "query": name,
                "synonyms": syns[:limit],
                "total": len(syns),
                "data_source": self._source("PubChem", source),
            }
        except Exception as exc:
            return {"error": str(exc), "query": name}

    def get_safety(self, cid: int) -> dict:
        """Fetch GHS safety/hazard information for a compound by CID."""
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        try:
            data, source = self._get(url)
            sections = data.get("Record", {}).get("Section", [])
        except Exception as exc:
            return {"error": str(exc), "cid": cid}

        safety_info: dict = {"cid": cid, "ghs_hazards": [], "signal_word": None,
                             "ghs_pictograms": [],
                             "data_source": self._source("PubChem", source)}

        for sec in sections:
            if "Safety" in sec.get("TOCHeading", ""):
                for subsec in sec.get("Section", []):
                    heading = subsec.get("TOCHeading", "")
                    if "GHS" in heading or "Hazard" in heading:
                        for info in subsec.get("Information", []):
                            for val in info.get("Value", {}).get("StringWithMarkup", []):
                                text = val.get("String", "")
                                if text:
                                    safety_info["ghs_hazards"].append(text)
        return safety_info

    def similar_compounds(self, cid: int, threshold: int = 90, limit: int = 10) -> dict:
        """Find structurally similar compounds by Tanimoto similarity."""
        url = f"{_BASE}/compound/fastsimilarity_2d/cid/{cid}/cids/JSON"
        try:
            data, source = self._get(url, params={"Threshold": threshold})
            cids = data["IdentifierList"]["CID"][:limit]
            return {"source_cid": cid, "threshold": threshold,
                    "similar_cids": cids, "count": len(cids),
                    "data_source": self._source("PubChem", source)}
        except Exception as exc:
            return {"error": str(exc), "cid": cid}

    def substructure_search(self, smiles: str, limit: int = 10) -> dict:
        """Substructure search by SMILES."""
        url = f"{_BASE}/compound/fastsubstructure/smiles/{quote(smiles, safe='')}/cids/JSON"
        try:
            data, source = self._get(url)
            cids = data["IdentifierList"]["CID"][:limit]
            return {
                "smiles": smiles,
                "matching_cids": cids,
                "count": len(cids),
                "data_source": self._source("PubChem", source),
            }
        except Exception as exc:
            return {"error": str(exc), "smiles": smiles}
