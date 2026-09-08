"""SQLite-backed persistence for rules, learned mappings, training data, history.

We use plain SQLite (stdlib only) so there are no extra services to run and the
whole knowledge base is a single portable ``.db`` file. Rules and mappings are
what give the tool "memory"; the ``training_data`` table accumulates labelled
examples (from human approvals) that the optional ML layer trains on.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from models.schemas import KeywordRule, LearnedMapping, RunSummary, TrainingExample


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    """Thin, thread-safe data-access layer over a SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit reruns across threads; serialise access with a lock and
        # allow cross-thread use of the single connection.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL + busy_timeout so Streamlit reruns don't lock each other on a
        # volume. Backward compatible for local / tests (WAL is fine locally).
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    # --------------------------------------------------------------- schema
    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rules (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword       TEXT NOT NULL,
                    account_code  TEXT NOT NULL,
                    fields        TEXT NOT NULL DEFAULT '[]',
                    match_type    TEXT NOT NULL DEFAULT 'contains',
                    case_sensitive INTEGER NOT NULL DEFAULT 0,
                    enabled       INTEGER NOT NULL DEFAULT 1,
                    notes         TEXT NOT NULL DEFAULT '',
                    priority      INTEGER NOT NULL DEFAULT 100,
                    client_id     TEXT,
                    created_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learned_mappings (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature    TEXT NOT NULL,
                    account_code TEXT NOT NULL,
                    hits         INTEGER NOT NULL DEFAULT 1,
                    last_seen    TEXT NOT NULL,
                    -- '' = shared/default client; a value scopes the mapping.
                    client_id    TEXT NOT NULL DEFAULT '',
                    UNIQUE(signature, client_id)
                );

                -- Pairings a reviewer explicitly rejected (signature ↛ code).
                -- The engine never auto-fills a blocked pair again.
                CREATE TABLE IF NOT EXISTS blocked_mappings (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature    TEXT NOT NULL,
                    account_code TEXT NOT NULL,
                    client_id    TEXT NOT NULL DEFAULT '',
                    created_at   TEXT NOT NULL,
                    UNIQUE(signature, account_code, client_id)
                );

                -- Per-transaction "Rule Notes" keyed by a *base* text signature
                -- (the transaction text only, without the notes themselves) so
                -- that notes re-attach to the same rows across future uploads.
                CREATE TABLE IF NOT EXISTS rule_notes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_sig     TEXT NOT NULL UNIQUE,
                    notes        TEXT NOT NULL DEFAULT '',
                    updated_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_data (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    text         TEXT NOT NULL,
                    label        TEXT NOT NULL,
                    confidence   REAL NOT NULL DEFAULT 1.0,
                    engine_used  TEXT NOT NULL DEFAULT 'manual',
                    timestamp    TEXT NOT NULL,
                    approved_by  TEXT NOT NULL DEFAULT 'user',
                    -- '' = shared/default client; a value scopes the example.
                    client_id    TEXT NOT NULL DEFAULT '',
                    UNIQUE(text, label, client_id)
                );

                CREATE TABLE IF NOT EXISTS run_history (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at              TEXT NOT NULL,
                    file_name           TEXT NOT NULL DEFAULT '',
                    total_rows          INTEGER NOT NULL DEFAULT 0,
                    seeds               INTEGER NOT NULL DEFAULT 0,
                    auto_filled         INTEGER NOT NULL DEFAULT 0,
                    filled_review       INTEGER NOT NULL DEFAULT 0,
                    needs_review        INTEGER NOT NULL DEFAULT 0,
                    no_match            INTEGER NOT NULL DEFAULT 0,
                    groups_found        INTEGER NOT NULL DEFAULT 0,
                    embedding_backend   TEXT NOT NULL DEFAULT '',
                    similarity_threshold REAL NOT NULL DEFAULT 0,
                    auto_apply_cutoff   REAL NOT NULL DEFAULT 0,
                    notes               TEXT NOT NULL DEFAULT ''
                );

                -- Account-Level Knowledge Base. Every rule/approval reinforces a
                -- profile for a canonical GL account (e.g. "6618S"): the keywords
                -- and sample transaction texts that map to it, how often it's been
                -- used, and human notes/nuances. This is the tool's long-term
                -- "memory" of what each account looks like — the more it learns,
                -- the better (and more explainable) future matching becomes.
                CREATE TABLE IF NOT EXISTS account_profiles (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_code  TEXT NOT NULL,          -- canonical, e.g. "6618S"
                    base_account  TEXT NOT NULL DEFAULT '',  -- e.g. "6618"
                    client_id     TEXT,                   -- NULL = general/shared
                    keywords      TEXT NOT NULL DEFAULT '[]',    -- JSON array
                    sample_texts  TEXT NOT NULL DEFAULT '[]',    -- JSON array
                    usage_count   INTEGER NOT NULL DEFAULT 0,
                    last_updated  TEXT NOT NULL DEFAULT '',
                    notes         TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_account_profiles_code
                    ON account_profiles(account_code);
                CREATE INDEX IF NOT EXISTS idx_account_profiles_code_client
                    ON account_profiles(account_code, client_id);
                """
            )
        self._ensure_rule_columns()
        self._ensure_client_isolation()

    def _ensure_rule_columns(self) -> None:
        """Add newer ``rules`` columns to a database created before they existed.

        SQLite ``ALTER TABLE ADD COLUMN`` is safe and idempotent here because we
        first inspect the live schema. Keeps upgrades seamless — no data loss.
        """
        with self._lock, self._conn:
            cols = {r["name"] for r in
                    self._conn.execute("PRAGMA table_info(rules)").fetchall()}
            if "priority" not in cols:
                self._conn.execute(
                    "ALTER TABLE rules ADD COLUMN priority INTEGER NOT NULL "
                    "DEFAULT 100")
            if "client_id" not in cols:
                self._conn.execute("ALTER TABLE rules ADD COLUMN client_id TEXT")

    def _table_columns(self, table: str) -> set:
        return {r["name"] for r in
                self._conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _ensure_client_isolation(self) -> None:
        """Add ``client_id`` to learned_mappings / training_data on old DBs.

        SQLite cannot alter a UNIQUE constraint in place, so when a pre-existing
        table lacks ``client_id`` we rebuild it under the new schema and copy
        every row across, assigning the shared/default client (``''``). No data
        is dropped — existing memory simply becomes the default client's memory.
        """
        with self._lock, self._conn:
            if "client_id" not in self._table_columns("learned_mappings"):
                self._conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS learned_mappings_new (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        signature    TEXT NOT NULL,
                        account_code TEXT NOT NULL,
                        hits         INTEGER NOT NULL DEFAULT 1,
                        last_seen    TEXT NOT NULL,
                        client_id    TEXT NOT NULL DEFAULT '',
                        UNIQUE(signature, client_id)
                    );
                    INSERT OR IGNORE INTO learned_mappings_new
                        (id, signature, account_code, hits, last_seen, client_id)
                        SELECT id, signature, account_code, hits, last_seen, ''
                        FROM learned_mappings;
                    DROP TABLE learned_mappings;
                    ALTER TABLE learned_mappings_new RENAME TO learned_mappings;
                    """
                )
            if "client_id" not in self._table_columns("training_data"):
                self._conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS training_data_new (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        text         TEXT NOT NULL,
                        label        TEXT NOT NULL,
                        confidence   REAL NOT NULL DEFAULT 1.0,
                        engine_used  TEXT NOT NULL DEFAULT 'manual',
                        timestamp    TEXT NOT NULL,
                        approved_by  TEXT NOT NULL DEFAULT 'user',
                        client_id    TEXT NOT NULL DEFAULT '',
                        UNIQUE(text, label, client_id)
                    );
                    INSERT OR IGNORE INTO training_data_new
                        (id, text, label, confidence, engine_used, timestamp,
                         approved_by, client_id)
                        SELECT id, text, label, confidence, engine_used,
                               timestamp, approved_by, ''
                        FROM training_data;
                    DROP TABLE training_data;
                    ALTER TABLE training_data_new RENAME TO training_data;
                    """
                )

    def reset_all(self) -> None:
        """Drop every data table and recreate the empty schema.

        Used by Fresh Demo Mode so a deployed Space looks brand new: all rules,
        learned mappings, rule notes, training examples and run history are
        removed, and AUTOINCREMENT id counters reset to 1 (dropping the tables
        clears their ``sqlite_sequence`` rows). The schema is preserved — the app
        keeps full functionality and users can create new clients/rules.
        """
        with self._lock, self._conn:
            self._conn.executescript(
                """
                DROP TABLE IF EXISTS rules;
                DROP TABLE IF EXISTS learned_mappings;
                DROP TABLE IF EXISTS blocked_mappings;
                DROP TABLE IF EXISTS rule_notes;
                DROP TABLE IF EXISTS training_data;
                DROP TABLE IF EXISTS run_history;
                DROP TABLE IF EXISTS account_profiles;
                """
            )
        self._init_schema()

    # ---------------------------------------------------------------- rules
    def add_rule(self, rule: KeywordRule) -> KeywordRule:
        created = (rule.created_at.isoformat() if rule.created_at
                   else _utcnow_iso())
        with self._lock, self._conn:
            cur = self._conn.execute(
                """INSERT INTO rules
                   (keyword, account_code, fields, match_type, case_sensitive,
                    enabled, notes, priority, client_id, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    rule.keyword,
                    rule.account_code,
                    json.dumps(rule.fields),
                    rule.match_type,
                    int(rule.case_sensitive),
                    int(rule.enabled),
                    rule.notes,
                    int(rule.priority),
                    rule.client_id,
                    created,
                ),
            )
            rule.id = cur.lastrowid
        return rule

    def update_rule(self, rule: KeywordRule) -> None:
        if rule.id is None:
            raise ValueError("Cannot update a rule without an id.")
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE rules SET keyword=?, account_code=?, fields=?,
                       match_type=?, case_sensitive=?, enabled=?, notes=?,
                       priority=?, client_id=?
                   WHERE id=?""",
                (
                    rule.keyword,
                    rule.account_code,
                    json.dumps(rule.fields),
                    rule.match_type,
                    int(rule.case_sensitive),
                    int(rule.enabled),
                    rule.notes,
                    int(rule.priority),
                    rule.client_id,
                    rule.id,
                ),
            )

    def delete_rule(self, rule_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))

    def delete_rules(self, rule_ids: List[int]) -> int:
        """Delete several rules at once. Returns the number removed."""
        ids = [int(i) for i in rule_ids]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"DELETE FROM rules WHERE id IN ({placeholders})", ids)
            return int(cur.rowcount or 0)

    def clear_rules(self) -> int:
        """Delete every keyword rule. Returns the number removed."""
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM rules")
            return int(cur.rowcount or 0)

    def list_rules(self, enabled_only: bool = False,
                   client_id: Optional[str] = None) -> List[KeywordRule]:
        """List rules ordered by priority then id.

        ``client_id=None`` returns *all* rules (default, backward compatible).
        Passing a client id returns that client's rules plus global ones
        (``client_id IS NULL``), so per-client scoping is additive.
        """
        query = "SELECT * FROM rules"
        conds: List[str] = []
        params: List[object] = []
        if enabled_only:
            conds.append("enabled=1")
        if client_id is not None:
            conds.append("(client_id IS NULL OR client_id=?)")
            params.append(client_id)
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY priority, id"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_rule(r) for r in rows]

    @staticmethod
    def _row_to_rule(row: sqlite3.Row) -> KeywordRule:
        keys = row.keys()
        return KeywordRule(
            id=row["id"],
            keyword=row["keyword"],
            account_code=row["account_code"],
            fields=json.loads(row["fields"]),
            match_type=row["match_type"],
            case_sensitive=bool(row["case_sensitive"]),
            enabled=bool(row["enabled"]),
            notes=row["notes"],
            priority=int(row["priority"]) if "priority" in keys else 100,
            client_id=row["client_id"] if "client_id" in keys else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ----------------------------------------------------- learned mappings
    def upsert_learned_mapping(self, signature: str, account_code: str,
                               client_id: Optional[str] = None) -> None:
        """Record (or reinforce) a confirmed text->code mapping.

        ``client_id=None`` writes to the shared/default client (``''``); a
        value scopes the mapping to that client. On a conflict the latest human
        decision wins (the code is updated) and the hit counter increments.
        """
        now = _utcnow_iso()
        cid = client_id or ""
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO learned_mappings
                       (signature, account_code, hits, last_seen, client_id)
                   VALUES (?,?,1,?,?)
                   ON CONFLICT(signature, client_id) DO UPDATE SET
                       hits = hits + 1,
                       account_code = excluded.account_code,
                       last_seen = excluded.last_seen""",
                (signature, account_code, now, cid),
            )

    def list_learned_mappings(
            self, client_id: Optional[str] = None) -> List[LearnedMapping]:
        """List mappings, most-hit first.

        ``client_id=None`` returns every mapping (backward compatible); a value
        narrows to that client's mappings plus the shared/default ones.
        """
        query = "SELECT * FROM learned_mappings"
        params: List[object] = []
        if client_id is not None:
            query += " WHERE client_id IN ('', ?)"
            params.append(client_id)
        query += " ORDER BY hits DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            LearnedMapping(
                id=r["id"],
                signature=r["signature"],
                account_code=r["account_code"],
                hits=r["hits"],
                last_seen=datetime.fromisoformat(r["last_seen"]),
                client_id=(r["client_id"] or None),
            )
            for r in rows
        ]

    def get_learned_lookup(self, client_id: Optional[str] = None) -> dict[str, str]:
        """Return a {signature: account_code} dict for fast lookups.

        With a ``client_id``, shared mappings are overlaid with that client's
        mappings (the client's win on conflict). Without one, every mapping is
        merged (highest-hit wins) — the historical behaviour.
        """
        mappings = self.list_learned_mappings(client_id=client_id)
        lookup: dict[str, str] = {}
        if client_id is None:
            # Ordered hits DESC — first occurrence (most-hit) wins.
            for m in mappings:
                lookup.setdefault(m.signature, m.account_code)
            return lookup
        # Shared ('' -> None) first, then the client's own mappings override.
        for m in sorted(mappings, key=lambda m: (m.client_id is not None,
                                                 -m.hits)):
            lookup[m.signature] = m.account_code
        return lookup

    def delete_learned_mapping(self, mapping_id: int) -> bool:
        """Delete one learned mapping by id. Returns True when a row was removed.

        Deleting stops the mapping from firing on the next run.
        """
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM learned_mappings WHERE id=?", (int(mapping_id),))
            return bool(cur.rowcount)

    def clear_learned_mappings(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM learned_mappings")

    # ----------------------------------------------------- blocked mappings
    def block_mapping(self, signature: str, account_code: str,
                      client_id: Optional[str] = None) -> None:
        """Record a rejected text->code pairing so it is never re-suggested."""
        signature = (signature or "").strip()
        account_code = (account_code or "").strip()
        if not signature or not account_code:
            return
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO blocked_mappings
                       (signature, account_code, client_id, created_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(signature, account_code, client_id) DO NOTHING""",
                (signature, account_code, client_id or "", _utcnow_iso()),
            )

    def get_blocked_lookup(self, client_id: Optional[str] = None) -> dict:
        """Return {signature: {blocked codes}} for the engine to honour."""
        query = "SELECT signature, account_code FROM blocked_mappings"
        params: List[object] = []
        if client_id is not None:
            query += " WHERE client_id IN ('', ?)"
            params.append(client_id)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        out: dict = {}
        for r in rows:
            out.setdefault(r["signature"], set()).add(r["account_code"])
        return out

    def count_blocked_mappings(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM blocked_mappings").fetchone()
        return int(row["c"]) if row else 0

    def clear_blocked_mappings(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM blocked_mappings")

    # ------------------------------------------------------------ rule notes
    def upsert_rule_note(self, base_sig: str, notes: str) -> None:
        """Persist a transaction's Rule Notes, keyed by its base signature.

        An empty/blank ``notes`` value deletes the stored note so cleared notes
        don't silently reappear on the next upload.
        """
        base_sig = (base_sig or "").strip()
        if not base_sig:
            return
        notes = (notes or "").strip()
        now = _utcnow_iso()
        with self._lock, self._conn:
            if not notes:
                self._conn.execute(
                    "DELETE FROM rule_notes WHERE base_sig=?", (base_sig,))
                return
            self._conn.execute(
                """INSERT INTO rule_notes (base_sig, notes, updated_at)
                   VALUES (?,?,?)
                   ON CONFLICT(base_sig) DO UPDATE SET
                       notes = excluded.notes,
                       updated_at = excluded.updated_at""",
                (base_sig, notes, now),
            )

    def get_rule_notes_lookup(self) -> dict[str, str]:
        """Return a {base_sig: notes} dict for re-attaching notes on upload."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT base_sig, notes FROM rule_notes").fetchall()
        return {r["base_sig"]: r["notes"] for r in rows}

    def count_rule_notes(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM rule_notes").fetchone()
        return int(row["c"]) if row else 0

    def clear_rule_notes(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM rule_notes")

    # --------------------------------------------------------- training data
    def add_training_example(
        self,
        text: str,
        label: str,
        confidence: float = 1.0,
        engine_used: str = "manual",
        approved_by: str = "user",
        client_id: Optional[str] = None,
    ) -> None:
        """Persist a labelled example (idempotent on (text, label, client))."""
        text = (text or "").strip()
        label = (label or "").strip()
        if not text or not label:
            return
        now = _utcnow_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO training_data
                   (text, label, confidence, engine_used, timestamp, approved_by,
                    client_id)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(text, label, client_id) DO UPDATE SET
                       confidence = excluded.confidence,
                       engine_used = excluded.engine_used,
                       timestamp = excluded.timestamp,
                       approved_by = excluded.approved_by""",
                (text, label, float(confidence), engine_used, now, approved_by,
                 client_id or ""),
            )

    def count_training_data(self, client_id: Optional[str] = None) -> int:
        """Number of training examples (a client id includes shared examples)."""
        query = "SELECT COUNT(*) AS c FROM training_data"
        params: List[object] = []
        if client_id is not None:
            query += " WHERE client_id IN ('', ?)"
            params.append(client_id)
        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return int(row["c"]) if row else 0

    def count_training_since(self, iso_timestamp: str,
                             client_id: Optional[str] = None) -> int:
        """Examples captured at/after an ISO timestamp (for the train banner)."""
        if not iso_timestamp:
            return self.count_training_data(client_id)
        query = "SELECT COUNT(*) AS c FROM training_data WHERE timestamp >= ?"
        params: List[object] = [iso_timestamp]
        if client_id is not None:
            query += " AND client_id IN ('', ?)"
            params.append(client_id)
        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return int(row["c"]) if row else 0

    def distinct_labels(self, client_id: Optional[str] = None) -> List[str]:
        query = "SELECT DISTINCT label FROM training_data"
        params: List[object] = []
        if client_id is not None:
            query += " WHERE client_id IN ('', ?)"
            params.append(client_id)
        query += " ORDER BY label"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [r["label"] for r in rows]

    def list_training_data(self, limit: int = 1000,
                           client_id: Optional[str] = None) -> List[TrainingExample]:
        query = "SELECT * FROM training_data"
        params: List[object] = []
        if client_id is not None:
            query += " WHERE client_id IN ('', ?)"
            params.append(client_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            TrainingExample(
                id=r["id"], text=r["text"], label=r["label"],
                confidence=r["confidence"], engine_used=r["engine_used"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                approved_by=r["approved_by"],
            )
            for r in rows
        ]

    def get_training_xy(self, client_id: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Return (texts, labels) for model training.

        ``client_id=None`` trains on everything (historical behaviour); a value
        trains on that client's examples plus the shared/default ones. When the
        same text carries different labels in the shared pool and the client
        scope, the **client's** label wins (last human write wins per scope) so
        the client model is never trained on contradictory examples.
        """
        query = "SELECT text, label FROM training_data"
        params: List[object] = []
        if client_id is not None:
            query += " WHERE client_id IN ('', ?)"
            params.append(client_id)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        if client_id is None:
            texts = [r["text"] for r in rows]
            labels = [r["label"] for r in rows]
            return texts, labels

        # Shared rows first, then the client's rows override per text.
        pairs: dict[str, str] = {}
        client_rows: List[Tuple[str, str]] = []
        for r in rows:
            pairs[r["text"]] = r["label"]
        with self._lock:
            client_rows = [
                (r["text"], r["label"]) for r in self._conn.execute(
                    "SELECT text, label FROM training_data WHERE client_id=?",
                    (client_id,)).fetchall()
            ]
        for text, label in client_rows:
            pairs[text] = label
        texts = list(pairs.keys())
        labels = [pairs[t] for t in texts]
        return texts, labels

    def clear_training_data(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM training_data")

    # -------------------------------------------------------------- history
    def add_run(self, summary: RunSummary) -> RunSummary:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """INSERT INTO run_history
                   (run_at, file_name, total_rows, seeds, auto_filled,
                    filled_review, needs_review, no_match, groups_found,
                    embedding_backend, similarity_threshold, auto_apply_cutoff, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    summary.run_at.isoformat(),
                    summary.file_name,
                    summary.total_rows,
                    summary.seeds,
                    summary.auto_filled,
                    summary.filled_review,
                    summary.needs_review,
                    summary.no_match,
                    summary.groups_found,
                    summary.embedding_backend,
                    summary.similarity_threshold,
                    summary.auto_apply_cutoff,
                    summary.notes,
                ),
            )
            summary.id = cur.lastrowid
        return summary

    def list_runs(self, limit: int = 100) -> List[RunSummary]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM run_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            RunSummary(
                id=r["id"],
                run_at=datetime.fromisoformat(r["run_at"]),
                file_name=r["file_name"],
                total_rows=r["total_rows"],
                seeds=r["seeds"],
                auto_filled=r["auto_filled"],
                filled_review=r["filled_review"],
                needs_review=r["needs_review"],
                no_match=r["no_match"],
                groups_found=r["groups_found"],
                embedding_backend=r["embedding_backend"],
                similarity_threshold=r["similarity_threshold"],
                auto_apply_cutoff=r["auto_apply_cutoff"],
                notes=r["notes"],
            )
            for r in rows
        ]

    # ------------------------------------------------- account knowledge base
    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "account_code": row["account_code"],
            "base_account": row["base_account"] or "",
            "client_id": row["client_id"],
            "keywords": json.loads(row["keywords"] or "[]"),
            "sample_texts": json.loads(row["sample_texts"] or "[]"),
            "usage_count": int(row["usage_count"] or 0),
            "last_updated": row["last_updated"] or "",
            "notes": row["notes"] or "",
        }

    def get_account_profile(self, account_code: str,
                            client_id: Optional[str] = None) -> Optional[dict]:
        """Fetch one profile for ``account_code`` scoped to ``client_id``.

        ``client_id=None`` matches the shared/general profile (``client_id IS
        NULL``); a value matches that client's profile exactly.
        """
        with self._lock:
            if client_id is None:
                row = self._conn.execute(
                    "SELECT * FROM account_profiles "
                    "WHERE account_code=? AND client_id IS NULL",
                    (account_code,)).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM account_profiles "
                    "WHERE account_code=? AND client_id=?",
                    (account_code, client_id)).fetchone()
        return self._row_to_profile(row) if row else None

    def save_account_profile(
        self,
        account_code: str,
        base_account: str,
        client_id: Optional[str],
        keywords: List[str],
        sample_texts: List[str],
        usage_count: int,
        notes: str = "",
    ) -> None:
        """Insert or update a profile (unique per ``account_code`` + ``client_id``)."""
        now = _utcnow_iso()
        kw = json.dumps(list(keywords))
        samples = json.dumps(list(sample_texts))
        with self._lock, self._conn:
            if client_id is None:
                existing = self._conn.execute(
                    "SELECT id FROM account_profiles "
                    "WHERE account_code=? AND client_id IS NULL",
                    (account_code,)).fetchone()
            else:
                existing = self._conn.execute(
                    "SELECT id FROM account_profiles "
                    "WHERE account_code=? AND client_id=?",
                    (account_code, client_id)).fetchone()
            if existing:
                self._conn.execute(
                    """UPDATE account_profiles SET base_account=?, keywords=?,
                           sample_texts=?, usage_count=?, last_updated=?, notes=?
                       WHERE id=?""",
                    (base_account, kw, samples, int(usage_count), now, notes,
                     existing["id"]))
            else:
                self._conn.execute(
                    """INSERT INTO account_profiles
                       (account_code, base_account, client_id, keywords,
                        sample_texts, usage_count, last_updated, notes)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (account_code, base_account, client_id, kw, samples,
                     int(usage_count), now, notes))

    def list_account_profiles(self,
                              client_id: Optional[str] = None) -> List[dict]:
        """List profiles ordered by usage (most-used first), then code.

        ``client_id=None`` returns *all* profiles; passing a client id narrows to
        that client's profiles plus the shared/general ones (``client_id IS NULL``).
        """
        query = "SELECT * FROM account_profiles"
        params: List[object] = []
        if client_id is not None:
            query += " WHERE client_id IS NULL OR client_id=?"
            params.append(client_id)
        query += " ORDER BY usage_count DESC, account_code"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_profile(r) for r in rows]

    def clear_account_profiles(self) -> int:
        """Delete every account profile. Returns the number removed."""
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM account_profiles")
            return int(cur.rowcount or 0)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
