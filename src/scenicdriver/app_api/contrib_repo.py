from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
import os
import secrets
import shutil
import sqlite3
import tempfile
import uuid

import pandas as pd


DEFAULT_PROFILE_COLUMNS = [
    "contributor_id", "display_name", "credits", "approved_labels",
    "pending_labels", "agreement_score", "agreement_count", "created_at", "updated_at",
    "token_sha256",
]
# Stored for QA/auth only; never part of a public profile or leaderboard row.
PRIVATE_PROFILE_COLUMNS = ["agreement_count", "token_sha256"]
DEFAULT_LABEL_COLUMNS = [
    "task_id", "contributor_id", "image_path", "region", "scenic_human",
    "confidence", "skip", "notes", "status", "timestamp", "agreement_score", "qa_reason",
]
EVENT_COLUMNS = ["timestamp", "contributor_id", "event", "credits_delta", "metadata"]
TASK_COLUMNS = ["task_id", "contributor_id", "image_path", "status"]
TEXT_COLUMNS = {
    column: str for column in [
        "contributor_id", "display_name", "task_id", "image_path", "region",
        "confidence", "notes", "status", "timestamp", "qa_reason", "created_at",
        "updated_at", "event", "metadata", "class_name", "token_sha256",
    ]
}


@dataclass
class ContribConfig:
    root_dir: Path = Path("data/processed/contrib")
    approved_seed_csv: Path = Path("data/raw/labels_human.csv")
    labels_source_csv: Path = Path("data/processed/heuristic_runs/masswhites_z14_learned_h4_v2/labels.csv")


class ContribRepo:
    """Transactional SQLite contributor store, with concurrent WAL readers.

    Each mutation commits all affected rows together; SQLite rolls back incomplete
    transactions on close or process interruption. Legacy contributor CSVs are
    imported only at initialization, then ignored. ``export_csvs`` creates explicit
    snapshots, not live storage. External source/QA CSVs are indexed in SQLite and
    refreshed only when their file identity changes.
    """

    def __init__(self, cfg: ContribConfig | None = None):
        self.cfg = cfg or ContribConfig()
        self.cfg.root_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cfg.root_dir / "contributors.sqlite3"
        self.profiles_path = self.cfg.root_dir / "contributor_profiles.csv"
        self.pending_path = self.cfg.root_dir / "annotations_pending.csv"
        self.approved_path = self.cfg.root_dir / "annotations_approved.csv"
        self.events_path = self.cfg.root_dir / "contrib_events.csv"
        self.tasks_path = self.cfg.root_dir / "issued_tasks.csv"
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("PRAGMA user_version").fetchone()[0] == 0:
                self._initialize(connection)
                self._import_legacy(connection)
                connection.execute("PRAGMA user_version=1")
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _transaction(self, *, write: bool = False):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            # Closing also discards the transaction if an explicit rollback fails.
            connection.close()

    @staticmethod
    def _initialize(connection):
        statements = [
            """CREATE TABLE profiles (
                contributor_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
                credits REAL NOT NULL DEFAULT 0, approved_labels INTEGER NOT NULL DEFAULT 0,
                pending_labels INTEGER NOT NULL DEFAULT 0, agreement_score REAL NOT NULL DEFAULT 0,
                agreement_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, token_sha256 TEXT NOT NULL DEFAULT '')""",
            "CREATE UNIQUE INDEX profile_tokens ON profiles(token_sha256) WHERE token_sha256 != ''",
            "CREATE INDEX profile_ranking ON profiles(credits DESC, approved_labels DESC)",
            """CREATE TABLE issued_tasks (
                task_id TEXT PRIMARY KEY, contributor_id TEXT NOT NULL, image_path TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('issued', 'submitted')),
                UNIQUE(contributor_id, image_path), UNIQUE(task_id, contributor_id, image_path))""",
            """CREATE TABLE annotations (
                task_id TEXT PRIMARY KEY, contributor_id TEXT NOT NULL, image_path TEXT NOT NULL,
                region TEXT NOT NULL, scenic_human REAL, confidence TEXT NOT NULL,
                skip INTEGER NOT NULL CHECK(skip IN (0, 1)), notes TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected', 'skipped')),
                timestamp TEXT NOT NULL, agreement_score REAL, qa_reason TEXT NOT NULL,
                UNIQUE(contributor_id, image_path),
                FOREIGN KEY(task_id, contributor_id, image_path)
                    REFERENCES issued_tasks(task_id, contributor_id, image_path))""",
            "CREATE INDEX annotation_status ON annotations(status)",
            """CREATE TABLE events (
                timestamp TEXT NOT NULL, contributor_id TEXT NOT NULL, event TEXT NOT NULL,
                credits_delta REAL NOT NULL, metadata TEXT NOT NULL)""",
            "CREATE INDEX event_contributor ON events(contributor_id)",
            """CREATE TABLE source_images (
                image_path TEXT PRIMARY KEY, region TEXT, class_id TEXT NOT NULL,
                class_name TEXT NOT NULL, scenic_score TEXT NOT NULL, position INTEGER NOT NULL)""",
            "CREATE INDEX source_region ON source_images(region COLLATE NOCASE, position)",
            "CREATE INDEX source_order ON source_images(position)",
            "CREATE TABLE seed_references (image_path TEXT PRIMARY KEY, mean REAL NOT NULL, count INTEGER NOT NULL)",
            "CREATE TABLE input_files (kind TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, has_region INTEGER NOT NULL)",
        ]
        for statement in statements:
            connection.execute(statement)

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, dtype=TEXT_COLUMNS, keep_default_na=False)

    @staticmethod
    def _insert(connection, table, columns, record):
        connection.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            [record[column] for column in columns],
        )

    def _import_legacy(self, connection):
        # Finish any interrupted old multi-CSV rollback before importing its snapshot.
        journal = self.cfg.root_dir / ".contrib-transaction.json"
        if journal.exists():
            entries = json.loads(journal.read_text(encoding="utf-8"))
            for entry in entries:
                backup = Path(entry["backup"])
                restore = backup.with_suffix(".restore")
                shutil.copyfile(backup, restore)
                os.replace(restore, entry["path"])
            journal.unlink()
            for entry in entries:
                Path(entry["backup"]).unlink(missing_ok=True)
                Path(entry["staged"]).unlink(missing_ok=True)
        for record in self._read_csv(self.profiles_path).to_dict(orient="records"):
            record["agreement_count"] = int(record.get("agreement_count") or 0)
            record["token_sha256"] = record.get("token_sha256", "")
            self._insert(connection, "profiles", DEFAULT_PROFILE_COLUMNS, record)
        for record in self._read_csv(self.tasks_path).to_dict(orient="records"):
            self._insert(connection, "issued_tasks", TASK_COLUMNS, record)
        for path in (self.pending_path, self.approved_path):
            for record in self._read_csv(path).to_dict(orient="records"):
                # CSV stores predating issued_tasks still retain their replay bindings.
                connection.execute(
                    "INSERT OR IGNORE INTO issued_tasks VALUES (?, ?, ?, 'submitted')",
                    (record["task_id"], record["contributor_id"], record["image_path"]),
                )
                connection.execute("UPDATE issued_tasks SET status='submitted' WHERE task_id=?",
                                   (record["task_id"],))
                record["skip"] = int(str(record["skip"]).lower() == "true")
                for column in ("scenic_human", "agreement_score"):
                    record[column] = None if record[column] == "" else float(record[column])
                self._insert(connection, "annotations", DEFAULT_LABEL_COLUMNS, record)
        for record in self._read_csv(self.events_path).to_dict(orient="records"):
            self._insert(connection, "events", EVENT_COLUMNS, record)

    def _refresh_input(self, connection, kind):
        path = self.cfg.labels_source_csv if kind == "source" else self.cfg.approved_seed_csv
        try:
            stat = path.stat()
            fingerprint = json.dumps([str(path.resolve()), stat.st_dev, stat.st_ino,
                                      stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
        except FileNotFoundError:
            fingerprint = json.dumps([str(path.resolve()), None])
        previous = connection.execute("SELECT * FROM input_files WHERE kind=?", (kind,)).fetchone()
        if previous is not None and previous["fingerprint"] == fingerprint:
            return bool(previous["has_region"])
        data = self._read_csv(path)
        has_region = "region" in data.columns
        if kind == "source":
            connection.execute("DELETE FROM source_images")
            if not data.empty:
                for position, record in enumerate(data.drop_duplicates("image_path").to_dict(orient="records")):
                    connection.execute("INSERT INTO source_images VALUES (?, ?, ?, ?, ?, ?)", (
                        record["image_path"], record.get("region"), str(record.get("class_id", "")),
                        record.get("class_name", ""), str(record.get("scenic_score", "")), position))
        else:
            connection.execute("DELETE FROM seed_references")
            if not data.empty:
                if "scenic_human" not in data.columns and "scenic_score" in data.columns:
                    data = data.rename(columns={"scenic_score": "scenic_human"})
                data["scenic_human"] = pd.to_numeric(data["scenic_human"], errors="coerce")
                references = data.loc[data["scenic_human"].map(math.isfinite)].groupby("image_path")["scenic_human"].agg(["mean", "count"])
                connection.executemany("INSERT INTO seed_references VALUES (?, ?, ?)",
                                       [(image, row["mean"], int(row["count"])) for image, row in references.iterrows()])
        connection.execute("INSERT OR REPLACE INTO input_files VALUES (?, ?, ?)",
                           (kind, fingerprint, int(has_region)))
        return has_region

    def export_csvs(self) -> dict[str, Path]:
        """Export all five legacy-shaped CSV snapshots from one read transaction."""
        exports = [
            ("profiles", self.profiles_path, DEFAULT_PROFILE_COLUMNS, "SELECT * FROM profiles ORDER BY rowid"),
            ("pending", self.pending_path, DEFAULT_LABEL_COLUMNS, "SELECT * FROM annotations WHERE status='pending' ORDER BY rowid"),
            ("approved", self.approved_path, DEFAULT_LABEL_COLUMNS, "SELECT * FROM annotations WHERE status='approved' ORDER BY rowid"),
            ("events", self.events_path, EVENT_COLUMNS, "SELECT * FROM events ORDER BY rowid"),
            ("tasks", self.tasks_path, TASK_COLUMNS, "SELECT * FROM issued_tasks ORDER BY rowid"),
        ]
        staged = []
        try:
            with self._transaction() as connection:
                for _, path, columns, query in exports:
                    records = [dict(row) for row in connection.execute(query)]
                    if "skip" in columns:
                        for record in records:
                            record["skip"] = bool(record["skip"])
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                                     dir=path.parent, prefix=".contrib-export-", delete=False) as stream:
                        staged.append((Path(stream.name), path))
                        pd.DataFrame(records, columns=columns).to_csv(stream, index=False)
                        stream.flush()
                        os.fsync(stream.fileno())
            for temporary, path in staged:
                os.replace(temporary, path)
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
        return {name: path for name, path, _, _ in exports}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _identifier(value: str, name: str) -> str:
        limit = 2048 if name == "image_path" else 128
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"{name} must not be blank or exceed {limit} characters")
        return value

    def _ensure_profile(self, connection, contributor_id):
        now = self._now()
        connection.execute(
            "INSERT OR IGNORE INTO profiles (contributor_id, display_name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (contributor_id, contributor_id, now, now),
        )

    @staticmethod
    def _public_profile(row):
        return {column: row[column] for column in DEFAULT_PROFILE_COLUMNS if column not in PRIVATE_PROFILE_COLUMNS}

    @staticmethod
    def _display_name(display_name: str) -> str:
        if not isinstance(display_name, str) or not display_name.strip() or len(display_name) > 128:
            raise ValueError("display_name must not be blank or exceed 128 characters")
        return display_name

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create_profile(self, display_name: str) -> tuple[dict, str]:
        """Return a public profile and one-time token; persist only its SHA-256."""
        self._display_name(display_name)
        token = secrets.token_urlsafe(32)
        with self._transaction(write=True) as connection:
            contributor_id = str(uuid.uuid4())
            now = self._now()
            try:
                connection.execute(
                    "INSERT INTO profiles (contributor_id, display_name, created_at, updated_at, token_sha256) VALUES (?, ?, ?, ?, ?)",
                    (contributor_id, display_name, now, now, self._token_hash(token)))
            except sqlite3.IntegrityError as exc:
                raise ValueError("contributor_id already exists") from exc
            row = connection.execute("SELECT * FROM profiles WHERE contributor_id=?", (contributor_id,)).fetchone()
            return self._public_profile(row), token

    def authenticate(self, token: str) -> str | None:
        if not isinstance(token, str) or not token:
            return None
        with self._transaction() as connection:
            row = connection.execute("SELECT contributor_id FROM profiles WHERE token_sha256 != '' AND token_sha256=?",
                                     (self._token_hash(token),)).fetchone()
            return None if row is None else row["contributor_id"]

    def update_display_name(self, contributor_id: str, display_name: str) -> dict:
        self._identifier(contributor_id, "contributor_id")
        self._display_name(display_name)
        with self._transaction(write=True) as connection:
            updated = connection.execute("UPDATE profiles SET display_name=?, updated_at=? WHERE contributor_id=?",
                                         (display_name, self._now(), contributor_id))
            if not updated.rowcount:
                raise ValueError("contributor does not exist")
            row = connection.execute("SELECT * FROM profiles WHERE contributor_id=?", (contributor_id,)).fetchone()
            return self._public_profile(row)

    def get_profile(self, contributor_id: str) -> dict | None:
        """Read-only public profile lookup; unknown contributors return None."""
        self._identifier(contributor_id, "contributor_id")
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM profiles WHERE contributor_id=?", (contributor_id,)).fetchone()
            return None if row is None else self._public_profile(row)

    @staticmethod
    def _image_region(row) -> str:
        if row.get("region"):
            return str(row["region"])
        parts = str(row["image_path"]).replace("\\", "/").split("/")
        generic = {"data", "raw", "images", "satellite", "terrain"}
        for index, part in enumerate(parts[:-1]):
            if index and part.startswith("z") and part[1:].isdigit():
                parent = parts[index - 1]
                return parent if parent not in generic else "unknown"
        return parts[0] if len(parts) > 1 and parts[0] not in generic else "unknown"

    @staticmethod
    def _optional_numeric(value, column: str) -> int | float | None:
        text = str(value).strip()
        if text.lower() in {"", "na", "n/a", "nan", "null", "none"}:
            return None
        try:
            number = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"{column} must be numeric") from exc
        if not number.is_finite():
            raise ValueError(f"{column} must be finite")
        if column == "class_id":
            if number != number.to_integral_value():
                raise ValueError("class_id must be an integer")
            return int(number)
        score = float(number)
        if not math.isfinite(score):
            raise ValueError(f"{column} must be finite")
        return score

    def next_tasks(self, contributor_id: str, region: str, count: int = 25) -> list[dict]:
        self._identifier(contributor_id, "contributor_id")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 100:
            raise ValueError("count must be between 1 and 100")
        with self._transaction(write=True) as connection:
            has_region = self._refresh_input(connection, "source")
            region = region.strip().lower()
            filter_sql = ""
            parameters = [contributor_id]
            if region:
                filter_sql = "AND s.region=? COLLATE NOCASE" if has_region else "AND instr(lower(s.image_path), ?) > 0"
                parameters.append(region)
            parameters.append(count)
            rows = connection.execute(f"""SELECT s.*, t.task_id FROM source_images s
                LEFT JOIN issued_tasks t ON t.contributor_id=? AND t.image_path=s.image_path
                WHERE (t.task_id IS NULL OR t.status='issued') {filter_sql}
                ORDER BY s.position LIMIT ?""", parameters).fetchall()
            out = []
            for row in rows:
                task_id = row["task_id"] or str(uuid.uuid4())
                out.append({"task_id": task_id, "image_path": row["image_path"],
                            "class_id": self._optional_numeric(row["class_id"], "class_id"),
                            "class_name": row["class_name"] or None,
                            "scenic_score": self._optional_numeric(row["scenic_score"], "scenic_score")})
                if row["task_id"] is None:
                    try:
                        connection.execute("INSERT INTO issued_tasks VALUES (?, ?, ?, 'issued')",
                                           (task_id, contributor_id, row["image_path"]))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("task_id already exists") from exc
            return out

    def submit_label(self, payload: dict) -> dict:
        contributor_id = self._identifier(payload["contributor_id"], "contributor_id")
        task_id = self._identifier(payload["task_id"], "task_id")
        self._identifier(payload["image_path"], "image_path")
        for field, limit in (("notes", 4096), ("confidence", 32)):
            value = payload.get(field, "")
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"{field} must not exceed {limit} characters")
        skip = bool(payload.get("skip", False))
        score = payload.get("scenic_human")
        if score is not None:
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 10:
                raise ValueError("scenic_human must be between 0 and 10")
        elif not skip:
            raise ValueError("scenic_human is required unless skip is true")
        with self._transaction(write=True) as connection:
            binding = connection.execute("SELECT * FROM issued_tasks WHERE task_id=? AND contributor_id=? AND image_path=?",
                                         (task_id, contributor_id, payload["image_path"])).fetchone()
            if binding is None:
                raise ValueError("task_id was not issued to this contributor for this image")
            if binding["status"] != "issued":
                raise ValueError("task_id has already been submitted")
            self._refresh_input(connection, "source")
            image = connection.execute("SELECT * FROM source_images WHERE image_path=?", (payload["image_path"],)).fetchone()
            if image is None:
                raise ValueError("assigned image is no longer available")
            record = {"task_id": task_id, "contributor_id": contributor_id,
                      "image_path": payload["image_path"], "region": self._image_region(dict(image)),
                      "scenic_human": score, "confidence": payload.get("confidence", "medium"),
                      "skip": skip, "notes": payload.get("notes", ""), "status": "pending",
                      "timestamp": self._now(), "agreement_score": None, "qa_reason": ""}
            try:
                self._insert(connection, "annotations", DEFAULT_LABEL_COLUMNS, record)
            except sqlite3.IntegrityError as exc:
                raise ValueError("image already submitted by this contributor") from exc
            self._ensure_profile(connection, contributor_id)
            connection.execute("UPDATE profiles SET pending_labels=pending_labels+1, updated_at=? WHERE contributor_id=?",
                               (self._now(), contributor_id))
            connection.execute("UPDATE issued_tasks SET status='submitted' WHERE task_id=?", (task_id,))
            return record

    def run_qa_promotion(self, min_overlap: int = 1, min_agreement: float = 0.65) -> dict:
        if not isinstance(min_overlap, int) or isinstance(min_overlap, bool) or min_overlap < 1:
            raise ValueError("min_overlap must be at least 1")
        if not math.isfinite(min_agreement) or not 0 <= min_agreement <= 1:
            raise ValueError("min_agreement must be finite and between 0 and 1")
        with self._transaction(write=True) as connection:
            pending = connection.execute("SELECT * FROM annotations WHERE status='pending' ORDER BY rowid").fetchall()
            promoted = rejected = skipped = 0
            remaining = len(pending)
            if pending:
                self._refresh_input(connection, "seed")
            for row in pending:
                contributor_id = self._identifier(row["contributor_id"], "contributor_id")
                self._identifier(row["task_id"], "task_id")
                rec = dict(row)
                rec["skip"] = bool(rec["skip"])
                credits = 0.0
                agreement = None
                if rec["skip"]:
                    rec.update(status="skipped", qa_reason="contributor_skipped")
                    skipped += 1
                else:
                    reference = connection.execute("SELECT * FROM seed_references WHERE image_path=?", (row["image_path"],)).fetchone()
                    if reference is None or reference["count"] < min_overlap:
                        continue
                    score = float(row["scenic_human"])
                    if not math.isfinite(score) or not 0 <= score <= 10:
                        raise ValueError("Invalid pending scenic_human score")
                    agreement = max(0.0, 1.0 - abs(score - float(reference["mean"])) / 10.0)
                    rec["agreement_score"] = round(agreement, 4)
                    if agreement >= min_agreement:
                        rec.update(status="approved", qa_reason="agreement_pass")
                        credits = 1.5 if agreement >= 0.8 else 1.0
                        promoted += 1
                    else:
                        rec.update(status="rejected", qa_reason="low_agreement")
                        rejected += 1
                self._ensure_profile(connection, contributor_id)
                connection.execute("UPDATE profiles SET pending_labels=max(0, pending_labels-1), updated_at=? WHERE contributor_id=?",
                                   (self._now(), contributor_id))
                if rec["status"] == "approved":
                    connection.execute("""UPDATE profiles SET credits=credits+?, approved_labels=approved_labels+1,
                        agreement_score=(CASE WHEN agreement_count=0 THEN 0 ELSE agreement_score*agreement_count END + ?) / (agreement_count+1),
                        agreement_count=agreement_count+1 WHERE contributor_id=?""", (credits, agreement, contributor_id))
                connection.execute("UPDATE annotations SET status=?, agreement_score=?, qa_reason=? WHERE task_id=?",
                                   (rec["status"], rec["agreement_score"], rec["qa_reason"], rec["task_id"]))
                connection.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                                   (self._now(), contributor_id, f"label_{rec['status']}", credits, json.dumps(rec)))
                remaining -= 1
            return {"pending_rows": len(pending), "promoted": promoted, "rejected": rejected,
                    "skipped": skipped, "remaining_pending": remaining}

    def leaderboard(self, limit: int = 25) -> list[dict]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._transaction() as connection:
            rows = connection.execute("SELECT * FROM profiles ORDER BY credits DESC, approved_labels DESC, rowid LIMIT ?", (limit,))
            return [self._public_profile(row) for row in rows]
