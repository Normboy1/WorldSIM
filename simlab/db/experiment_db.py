"""SIMLAB experiment knowledge database.

Stores every experiment run — parameters, results, plots, proof paths,
pass/fail status, and free-form notes — in a local SQLite database so
the AI can query what has already been tried and what worked.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default DB lives in the proof/ folder next to the project root.
_DEFAULT_DB = Path(__file__).resolve().parents[3] / "proof" / "simlab.db"


_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS experiments (
    id             TEXT PRIMARY KEY,
    domain         TEXT NOT NULL,
    exp_type       TEXT NOT NULL,
    parameters     TEXT NOT NULL,        -- JSON
    results        TEXT,                 -- JSON
    status         TEXT NOT NULL DEFAULT 'pending',  -- success | error | partial
    warnings       TEXT,                -- JSON list
    errors         TEXT,                -- JSON list
    proof_dir      TEXT,                -- path to proof/ subfolder
    pdf_path       TEXT,                -- compiled PDF path
    created_at     TEXT NOT NULL,
    duration_ms    REAL,
    solver         TEXT
);

CREATE TABLE IF NOT EXISTS plots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  TEXT NOT NULL REFERENCES experiments(id),
    name           TEXT NOT NULL,
    file_path      TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  TEXT REFERENCES experiments(id),
    category       TEXT NOT NULL DEFAULT 'general',  -- worked | failed | insight | warning
    body           TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    domain         TEXT NOT NULL,
    exp_type       TEXT,
    key            TEXT NOT NULL,
    value          TEXT NOT NULL,
    confidence     REAL DEFAULT 1.0,    -- 0.0–1.0
    source_exp_id  TEXT REFERENCES experiments(id),
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exp_domain    ON experiments(domain);
CREATE INDEX IF NOT EXISTS idx_exp_status    ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_exp_type      ON experiments(exp_type);
CREATE INDEX IF NOT EXISTS idx_notes_cat     ON notes(category);
CREATE INDEX IF NOT EXISTS idx_know_domain   ON knowledge(domain);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentDB:
    """SQLite-backed store for SIMLAB experiment history and knowledge."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── Experiments ────────────────────────────────────────────────────────

    def save_experiment(
        self,
        exp_id: str,
        domain: str,
        exp_type: str,
        parameters: dict,
        results: dict | None = None,
        status: str = "success",
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
        proof_dir: str | None = None,
        pdf_path: str | None = None,
        duration_ms: float | None = None,
        solver: str | None = None,
    ) -> str:
        """Insert or replace an experiment record. Returns exp_id."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                  (id, domain, exp_type, parameters, results, status,
                   warnings, errors, proof_dir, pdf_path, created_at, duration_ms, solver)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    exp_id, domain, exp_type,
                    json.dumps(parameters),
                    json.dumps(results) if results else None,
                    status,
                    json.dumps(warnings or []),
                    json.dumps(errors or []),
                    proof_dir,
                    pdf_path,
                    _now(),
                    duration_ms,
                    solver,
                ),
            )
        return exp_id

    def get_experiment(self, exp_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (exp_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def list_experiments(
        self,
        domain: str | None = None,
        status: str | None = None,
        exp_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        clauses, params = [], []
        if domain:
            clauses.append("domain = ?"); params.append(domain)
        if status:
            clauses.append("status = ?"); params.append(status)
        if exp_type:
            clauses.append("exp_type = ?"); params.append(exp_type)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM experiments {where} ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def search_experiments(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across domain, type, parameters, results."""
        like = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM experiments
                   WHERE domain LIKE ? OR exp_type LIKE ?
                      OR parameters LIKE ? OR results LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (like, like, like, like, limit),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def delete_experiment(self, exp_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
        return cur.rowcount > 0

    # ── Plots ──────────────────────────────────────────────────────────────

    def save_plot(self, experiment_id: str, name: str, file_path: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO plots (experiment_id, name, file_path, created_at) VALUES (?,?,?,?)",
                (experiment_id, name, file_path, _now()),
            )
        return cur.lastrowid

    def get_plots(self, experiment_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM plots WHERE experiment_id = ?", (experiment_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Notes ──────────────────────────────────────────────────────────────

    def add_note(
        self,
        body: str,
        category: str = "general",
        experiment_id: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO notes (experiment_id, category, body, created_at) VALUES (?,?,?,?)",
                (experiment_id, category, body, _now()),
            )
        return cur.lastrowid

    def get_notes(
        self,
        experiment_id: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        clauses, params = [], []
        if experiment_id:
            clauses.append("experiment_id = ?"); params.append(experiment_id)
        if category:
            clauses.append("category = ?"); params.append(category)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM notes {where} ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Knowledge ──────────────────────────────────────────────────────────

    def store_knowledge(
        self,
        domain: str,
        key: str,
        value: str,
        exp_type: str | None = None,
        confidence: float = 1.0,
        source_exp_id: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO knowledge
                   (domain, exp_type, key, value, confidence, source_exp_id, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (domain, exp_type, key, value, confidence, source_exp_id, _now()),
            )
        return cur.lastrowid

    def get_knowledge(
        self,
        domain: str | None = None,
        exp_type: str | None = None,
        key: str | None = None,
    ) -> list[dict]:
        clauses, params = [], []
        if domain:
            clauses.append("domain = ?"); params.append(domain)
        if exp_type:
            clauses.append("exp_type = ?"); params.append(exp_type)
        if key:
            clauses.append("key LIKE ?"); params.append(f"%{key}%")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM knowledge {where} ORDER BY confidence DESC, created_at DESC",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Summary / reporting ────────────────────────────────────────────────

    def summary_stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            by_status = {
                r["status"]: r["cnt"]
                for r in conn.execute(
                    "SELECT status, COUNT(*) AS cnt FROM experiments GROUP BY status"
                ).fetchall()
            }
            by_domain = {
                r["domain"]: r["cnt"]
                for r in conn.execute(
                    "SELECT domain, COUNT(*) AS cnt FROM experiments GROUP BY domain"
                ).fetchall()
            }
            notes_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            knowledge_count = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]

        return {
            "total_experiments": total,
            "by_status": by_status,
            "by_domain": by_domain,
            "total_notes": notes_count,
            "total_knowledge_entries": knowledge_count,
        }

    def what_worked(self, domain: str | None = None) -> list[dict]:
        return self.list_experiments(domain=domain, status="success", limit=100)

    def what_failed(self, domain: str | None = None) -> list[dict]:
        return self.list_experiments(domain=domain, status="error", limit=100)

    def recent(self, n: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (n,)
            ).fetchall()
        return [_row_to_dict(r) for r in rows]


# ── helpers ────────────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for field in ("parameters", "results", "warnings", "errors"):
        if isinstance(d.get(field), str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d
