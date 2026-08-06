from __future__ import annotations

"""Deterministic, fail-soft SQLite cache for expensive scanner stages.

The store only reads payloads it previously wrote into the scanner cache root.
Every compressed payload is checksum-verified before unpickling. Corrupt or
incompatible entries are deleted and treated as misses; scanner decisions are
always rebuilt from source inputs rather than inferred from a broken cache.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import dataclasses
import json
import os
import pickle
import sqlite3
import time
import zlib

import numpy as np
import pandas as pd

CACHE_SCHEMA_VERSION = "incremental-focus-cache-v1"


def _stable_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return "__NaN__"
        if np.isposinf(value):
            return "__Inf__"
        if np.isneginf(value):
            return "__-Inf__"
        return value
    if isinstance(value, np.generic):
        return _stable_scalar(value.item())
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _stable_scalar(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _stable_scalar(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_scalar(item) for item in value]
    if isinstance(value, set):
        return sorted((_stable_scalar(item) for item in value), key=str)
    if isinstance(value, pd.Series):
        return {
            "name": _stable_scalar(value.name),
            "index": [_stable_scalar(item) for item in value.index.tolist()],
            "values": [_stable_scalar(item) for item in value.tolist()],
        }
    if isinstance(value, pd.DataFrame):
        return {"dataframe_fingerprint": fingerprint_dataframe(value)}
    try:
        if pd.isna(value):
            return "__NaN__"
    except Exception:
        pass
    return str(value)


def stable_json(value: Any) -> str:
    return json.dumps(
        _stable_scalar(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def fingerprint_value(value: Any) -> str:
    return sha256(stable_json(value).encode("utf-8")).hexdigest()


def _hashable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    object_columns = [
        column for column in frame.columns
        if frame[column].dtype == "object"
    ]
    if not object_columns:
        return frame
    out = frame.copy(deep=False)
    for column in object_columns:
        out[column] = frame[column].map(
            lambda value: stable_json(value)
            if isinstance(value, (dict, list, tuple, set, pd.Series, pd.DataFrame))
            else _stable_scalar(value)
        )
    return out


def fingerprint_dataframe(frame: pd.DataFrame | None) -> str:
    if frame is None:
        return sha256(b"NONE").hexdigest()
    if not isinstance(frame, pd.DataFrame):
        return fingerprint_value(frame)
    digest = sha256()
    digest.update(CACHE_SCHEMA_VERSION.encode("utf-8"))
    digest.update(str(frame.shape).encode("utf-8"))
    digest.update(stable_json([str(column) for column in frame.columns]).encode("utf-8"))
    digest.update(stable_json([str(dtype) for dtype in frame.dtypes]).encode("utf-8"))
    try:
        hashed = pd.util.hash_pandas_object(
            _hashable_frame(frame), index=True, categorize=True,
        ).to_numpy(dtype="uint64", copy=False)
        digest.update(hashed.tobytes())
    except Exception:
        # Slow but deterministic fail-safe for unusual extension/object values.
        digest.update(
            stable_json({
                "index": [_stable_scalar(item) for item in frame.index.tolist()],
                "records": frame.to_dict("records"),
            }).encode("utf-8")
        )
    if frame.attrs:
        digest.update(stable_json(frame.attrs).encode("utf-8"))
    return digest.hexdigest()


def fingerprint_prepared(prepared: Mapping[str, pd.DataFrame] | None) -> str:
    digest = sha256()
    digest.update(CACHE_SCHEMA_VERSION.encode("utf-8"))
    for ticker in sorted((prepared or {}).keys(), key=lambda value: str(value).upper()):
        digest.update(str(ticker).upper().encode("utf-8"))
        digest.update(fingerprint_dataframe((prepared or {}).get(ticker)).encode("ascii"))
    return digest.hexdigest()


def combine_fingerprints(*parts: Any) -> str:
    digest = sha256()
    digest.update(CACHE_SCHEMA_VERSION.encode("utf-8"))
    for part in parts:
        if isinstance(part, str) and len(part) == 64:
            token = part
        elif isinstance(part, pd.DataFrame) or part is None:
            token = fingerprint_dataframe(part)
        else:
            token = fingerprint_value(part)
        digest.update(token.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def default_cache_path() -> Path:
    explicit = os.getenv("IDX_SCANNER_INCREMENTAL_DB", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
    else:
        base = os.getenv("IDX_SCANNER_CACHE_DIR", "").strip()
        root = Path(base).expanduser() if base else Path.home() / ".cache" / "idx_super_scanner"
        path = root / "incremental" / "focus_cache_v1.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class CacheLookup:
    state: str
    payload: Any = None
    elapsed_ms: float = 0.0
    payload_bytes: int = 0
    detail: str = ""


class IncrementalEvidenceStore:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_entries: int = 48,
        max_bytes: int = 1_073_741_824,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path).expanduser() if path else default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max(4, int(max_entries))
        self.max_bytes = max(16 * 1024 * 1024, int(max_bytes))
        self.read_only = bool(read_only)
        self._initialised = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=15.0, isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        if not self._initialised:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    stage TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    payload BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    payload_bytes INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (stage, cache_key)
                );
                CREATE INDEX IF NOT EXISTS idx_cache_entries_lru
                    ON cache_entries(last_accessed);
                CREATE INDEX IF NOT EXISTS idx_cache_entries_expiry
                    ON cache_entries(expires_at);
                """
            )
            self._initialised = True
        return connection

    def get(self, stage: str, cache_key: str) -> CacheLookup:
        started = time.perf_counter()
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT schema_version, expires_at, payload, payload_sha256,
                           payload_bytes
                    FROM cache_entries
                    WHERE stage = ? AND cache_key = ?
                    """,
                    (str(stage), str(cache_key)),
                ).fetchone()
                if row is None:
                    return CacheLookup(
                        "MISS", elapsed_ms=1000.0 * (time.perf_counter() - started),
                    )
                schema_version, expires_at, payload, expected_hash, payload_bytes = row
                now = time.time()
                if schema_version != CACHE_SCHEMA_VERSION or float(expires_at) < now:
                    if not self.read_only:
                        connection.execute(
                            "DELETE FROM cache_entries WHERE stage = ? AND cache_key = ?",
                            (str(stage), str(cache_key)),
                        )
                    return CacheLookup(
                        "EXPIRED", elapsed_ms=1000.0 * (time.perf_counter() - started),
                    )
                actual_hash = sha256(payload).hexdigest()
                if actual_hash != expected_hash:
                    if not self.read_only:
                        connection.execute(
                            "DELETE FROM cache_entries WHERE stage = ? AND cache_key = ?",
                            (str(stage), str(cache_key)),
                        )
                    return CacheLookup(
                        "CORRUPT", elapsed_ms=1000.0 * (time.perf_counter() - started),
                        payload_bytes=int(payload_bytes or 0),
                        detail="Checksum mismatch; entry removed.",
                    )
                restored = pickle.loads(zlib.decompress(payload))
                if not self.read_only:
                    connection.execute(
                        """
                        UPDATE cache_entries SET last_accessed = ?
                        WHERE stage = ? AND cache_key = ?
                        """,
                        (now, str(stage), str(cache_key)),
                    )
                return CacheLookup(
                    "HIT", payload=restored,
                    elapsed_ms=1000.0 * (time.perf_counter() - started),
                    payload_bytes=int(payload_bytes or 0),
                )
        except Exception as exc:
            return CacheLookup(
                "ERROR", elapsed_ms=1000.0 * (time.perf_counter() - started),
                detail=f"{type(exc).__name__}: {str(exc)[:180]}",
            )

    def put(
        self,
        stage: str,
        cache_key: str,
        payload: Any,
        *,
        ttl_seconds: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> CacheLookup:
        started = time.perf_counter()
        if self.read_only:
            return CacheLookup("READ_ONLY", elapsed_ms=0.0)
        try:
            raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(raw, level=3)
            checksum = sha256(compressed).hexdigest()
            now = time.time()
            expires_at = now + max(60.0, float(ttl_seconds))
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO cache_entries (
                        stage, cache_key, schema_version, created_at,
                        last_accessed, expires_at, payload, payload_sha256,
                        payload_bytes, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(stage, cache_key) DO UPDATE SET
                        schema_version=excluded.schema_version,
                        created_at=excluded.created_at,
                        last_accessed=excluded.last_accessed,
                        expires_at=excluded.expires_at,
                        payload=excluded.payload,
                        payload_sha256=excluded.payload_sha256,
                        payload_bytes=excluded.payload_bytes,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        str(stage), str(cache_key), CACHE_SCHEMA_VERSION,
                        now, now, expires_at, sqlite3.Binary(compressed), checksum,
                        len(compressed), stable_json(metadata or {}),
                    ),
                )
                self._prune(connection, now)
                connection.execute("COMMIT")
            return CacheLookup(
                "STORED", elapsed_ms=1000.0 * (time.perf_counter() - started),
                payload_bytes=len(compressed),
            )
        except Exception as exc:
            return CacheLookup(
                "ERROR", elapsed_ms=1000.0 * (time.perf_counter() - started),
                detail=f"{type(exc).__name__}: {str(exc)[:180]}",
            )

    def _prune(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute("DELETE FROM cache_entries WHERE expires_at < ?", (now,))
        count, total = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(payload_bytes), 0) FROM cache_entries"
        ).fetchone()
        count = int(count or 0)
        total = int(total or 0)
        while count > self.max_entries or total > self.max_bytes:
            victims = connection.execute(
                """
                SELECT stage, cache_key, payload_bytes
                FROM cache_entries
                ORDER BY last_accessed ASC
                LIMIT 8
                """
            ).fetchall()
            if not victims:
                break
            for stage, cache_key, payload_bytes in victims:
                connection.execute(
                    "DELETE FROM cache_entries WHERE stage = ? AND cache_key = ?",
                    (stage, cache_key),
                )
                count -= 1
                total -= int(payload_bytes or 0)
                if count <= self.max_entries and total <= self.max_bytes:
                    break

    def clear(self, stage: str | None = None) -> None:
        if self.read_only:
            return
        with self._connect() as connection:
            if stage:
                connection.execute("DELETE FROM cache_entries WHERE stage = ?", (stage,))
            else:
                connection.execute("DELETE FROM cache_entries")

    def inventory(self) -> pd.DataFrame:
        try:
            with self._connect() as connection:
                return pd.read_sql_query(
                    """
                    SELECT stage, cache_key, created_at, last_accessed,
                           expires_at, payload_bytes, metadata_json
                    FROM cache_entries
                    ORDER BY last_accessed DESC
                    """,
                    connection,
                )
        except Exception:
            return pd.DataFrame()


def cache_enabled(config: Any | None = None) -> bool:
    explicit = os.getenv("IDX_SCANNER_INCREMENTAL_CACHE_ENABLED")
    if explicit is not None:
        return explicit.strip().lower() not in {"0", "false", "no", "off"}
    # Prevent unrelated regression tests from sharing persistent state. Cache
    # tests can opt in explicitly through the environment variable above.
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return bool(getattr(config, "incremental_cache_enabled", True))


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheLookup",
    "IncrementalEvidenceStore",
    "cache_enabled",
    "combine_fingerprints",
    "default_cache_path",
    "fingerprint_dataframe",
    "fingerprint_prepared",
    "fingerprint_value",
    "stable_json",
]
