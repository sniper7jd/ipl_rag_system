from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class FileRecord:
    file_id: str
    file_path: str
    file_hash: str
    collection: str


class IngestionIndex:
    """Tiny SQLite index to avoid re-embedding the same file content."""

    def __init__(self, db_path: str = "data/ingestion.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_hash_collection ON files(file_hash, collection)"
            )

    def lookup_by_hash(self, file_hash: str, collection: str) -> Optional[FileRecord]:
        with self._connect() as con:
            row = con.execute(
                "SELECT file_id, file_path, file_hash, collection FROM files WHERE file_hash=? AND collection=?",
                (file_hash, collection),
            ).fetchone()
        if not row:
            return None
        return FileRecord(*row)

    def upsert_file(self, file_id: str, file_path: str, file_hash: str, collection: str) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO files(file_id, file_path, file_hash, collection)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    file_path=excluded.file_path,
                    file_hash=excluded.file_hash,
                    collection=excluded.collection
                """,
                (file_id, file_path, file_hash, collection),
            )

