from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import math
from decimal import Decimal, InvalidOperation
import os
import shutil
import tempfile
import threading
import uuid

import pandas as pd


DEFAULT_PROFILE_COLUMNS = [
    "contributor_id", "display_name", "credits", "approved_labels",
    "pending_labels", "agreement_score", "created_at", "updated_at",
]
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
        "updated_at", "event", "metadata", "class_name",
    ]
}
_PROCESS_LOCK = threading.Lock()


@dataclass
class ContribConfig:
    root_dir: Path = Path("data/processed/contrib")
    approved_seed_csv: Path = Path("data/raw/labels_human.csv")
    labels_source_csv: Path = Path("data/processed/heuristic_runs/masswhites_z14_learned_h4_v2/labels.csv")


class ContribRepo:
    """CSV storage serialized across threads and processes.

    Multi-file commits stage every CSV before replacing any live file. A rollback
    journal restores the prior snapshot on failure, including recovery on the
    next locked access if the process was interrupted during replacement.
    Helpers never acquire the lock: public operations own the entire transaction.

    Optional task numbers normalize CSV missing markers without changing literal
    text or IDs. Class IDs must be integral, and scenic scores must be finite.
    """

    def __init__(self, cfg: ContribConfig | None = None):
        self.cfg = cfg or ContribConfig()
        self.cfg.root_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_path = self.cfg.root_dir / "contributor_profiles.csv"
        self.pending_path = self.cfg.root_dir / "annotations_pending.csv"
        self.approved_path = self.cfg.root_dir / "annotations_approved.csv"
        self.events_path = self.cfg.root_dir / "contrib_events.csv"
        self.tasks_path = self.cfg.root_dir / "issued_tasks.csv"
        self.journal_path = self.cfg.root_dir / ".contrib-transaction.json"
        with self._locked():
            for path, columns in [
                (self.profiles_path, DEFAULT_PROFILE_COLUMNS),
                (self.pending_path, DEFAULT_LABEL_COLUMNS),
                (self.approved_path, DEFAULT_LABEL_COLUMNS),
                (self.events_path, EVENT_COLUMNS), (self.tasks_path, TASK_COLUMNS),
            ]:
                if not path.exists():
                    self._write(path, pd.DataFrame(columns=columns))

    @contextmanager
    def _locked(self):
        with _PROCESS_LOCK:
            with (self.cfg.root_dir / ".contrib.lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    self._recover()
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _identifier(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must not be blank")
        return value

    def _read(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, dtype=TEXT_COLUMNS, keep_default_na=False)

    def _stage(self, path: Path, df: pd.DataFrame) -> Path:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=".contrib-", delete=False) as stream:
            staged = Path(stream.name)
            try:
                df.to_csv(stream, index=False)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                staged.unlink(missing_ok=True)
                raise
        return staged

    def _write(self, path: Path, df: pd.DataFrame) -> None:
        staged = self._stage(path, df)
        try:
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)

    def _recover(self) -> None:
        if not self.journal_path.exists():
            return
        entries = json.loads(self.journal_path.read_text(encoding="utf-8"))
        for entry in entries:
            backup = Path(entry["backup"])
            # Retain backups until recovery completes, so recovery is restartable.
            restore = backup.with_suffix(".restore")
            shutil.copyfile(backup, restore)
            os.replace(restore, entry["path"])
        self.journal_path.unlink()
        for entry in entries:
            Path(entry["backup"]).unlink(missing_ok=True)
            Path(entry["staged"]).unlink(missing_ok=True)

    def _persist(self, tables: dict[Path, pd.DataFrame]) -> None:
        entries = []
        journal_temp = None
        try:
            # All serialization and validation complete before the first live write.
            for path, table in tables.items():
                staged = self._stage(path, table)
                entry = {"path": str(path.resolve()), "staged": str(staged.resolve()),
                         "backup": str(staged.with_suffix(".backup").resolve())}
                entries.append(entry)
                shutil.copyfile(path, entry["backup"])
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cfg.root_dir,
                                             prefix=".contrib-journal-", delete=False) as stream:
                journal_temp = Path(stream.name)
                json.dump(entries, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(journal_temp, self.journal_path)
            for entry in entries:
                os.replace(entry["staged"], entry["path"])
            self.journal_path.unlink()
        except BaseException:
            self._recover()
            raise
        finally:
            if journal_temp is not None:
                journal_temp.unlink(missing_ok=True)
            # Preserve recovery material if rollback itself was interrupted/failed.
            if not self.journal_path.exists():
                for entry in entries:
                    Path(entry["backup"]).unlink(missing_ok=True)
                    Path(entry["staged"]).unlink(missing_ok=True)

    def _profile_row(self, profiles: pd.DataFrame, contributor_id: str,
                     display_name: str | None = None) -> tuple[pd.DataFrame, int]:
        mask = profiles["contributor_id"] == contributor_id
        if mask.any():
            idx = profiles.loc[mask].index[0]
            if display_name:
                profiles.at[idx, "display_name"] = display_name
            return profiles, idx
        now = self._now()
        record = {"contributor_id": contributor_id, "display_name": display_name or contributor_id,
                  "credits": 0.0, "approved_labels": 0, "pending_labels": 0,
                  "agreement_score": 0.0, "created_at": now, "updated_at": now}
        profiles = pd.concat([profiles, pd.DataFrame([record])], ignore_index=True)
        return profiles, profiles.index[-1]

    def upsert_profile(self, contributor_id: str, display_name: str | None = None) -> dict:
        self._identifier(contributor_id, "contributor_id")
        with self._locked():
            profiles, idx = self._profile_row(self._read(self.profiles_path), contributor_id, display_name)
            profiles.at[idx, "updated_at"] = self._now()
            self._write(self.profiles_path, profiles)
            return profiles.loc[idx].to_dict()

    def get_profile(self, contributor_id: str) -> dict:
        self._identifier(contributor_id, "contributor_id")
        with self._locked():
            profiles = self._read(self.profiles_path)
            rows = profiles.loc[profiles["contributor_id"] == contributor_id]
            if not rows.empty:
                return rows.iloc[0].to_dict()
            profiles, idx = self._profile_row(profiles, contributor_id)
            self._write(self.profiles_path, profiles)
            return profiles.loc[idx].to_dict()

    def _task_pool(self, region: str) -> pd.DataFrame:
        src = self._read(self.cfg.labels_source_csv)
        if src.empty:
            return pd.DataFrame(columns=["image_path"])
        region = region.strip().lower()
        if region:
            filtered = src.loc[src["image_path"].str.contains(region, case=False, regex=False)]
            if not filtered.empty:
                src = filtered
        cols = [c for c in ["image_path", "class_id", "class_name", "scenic_score"] if c in src.columns]
        return src[cols].drop_duplicates("image_path")

    @staticmethod
    def _optional_numeric(value, column: str) -> int | float | None:
        # Only numeric metadata is nullable; IDs and text retain literal NA values.
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
        with self._locked():
            pool = self._task_pool(region)
            if pool.empty:
                return []
            issued = self._read(self.tasks_path)
            done = set(issued.loc[(issued["contributor_id"] == contributor_id)
                                  & (issued["status"] == "submitted"), "image_path"])
            for path in (self.pending_path, self.approved_path):
                labels = self._read(path)
                done.update(labels.loc[labels["contributor_id"] == contributor_id, "image_path"])
            tasks = pool.loc[~pool["image_path"].isin(done)].head(max(1, int(count)))
            out, new_bindings = [], []
            for _, row in tasks.iterrows():
                binding = issued.loc[(issued["contributor_id"] == contributor_id)
                                     & (issued["image_path"] == row["image_path"])
                                     & (issued["status"] == "issued")]
                task_id = binding.iloc[0]["task_id"] if not binding.empty else str(uuid.uuid4())
                if binding.empty:
                    new_bindings.append({"task_id": task_id, "contributor_id": contributor_id,
                                         "image_path": row["image_path"], "status": "issued"})
                out.append({"task_id": task_id, "image_path": row["image_path"],
                            "class_id": self._optional_numeric(row.get("class_id", ""), "class_id"),
                            "class_name": row.get("class_name") or None,
                            "scenic_score": self._optional_numeric(row.get("scenic_score", ""), "scenic_score")})
            if new_bindings:
                self._write(self.tasks_path, pd.concat([issued, pd.DataFrame(new_bindings)], ignore_index=True))
            return out

    def submit_label(self, payload: dict) -> dict:
        contributor_id = self._identifier(payload["contributor_id"], "contributor_id")
        task_id = self._identifier(payload["task_id"], "task_id")
        skip = bool(payload.get("skip", False))
        score = payload.get("scenic_human")
        if score is not None:
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 10:
                raise ValueError("scenic_human must be between 0 and 10")
        elif not skip:
            raise ValueError("scenic_human is required unless skip is true")
        with self._locked():
            issued = self._read(self.tasks_path)
            binding = issued.loc[(issued["task_id"] == task_id)
                                 & (issued["contributor_id"] == contributor_id)
                                 & (issued["image_path"] == payload["image_path"])]
            if binding.empty:
                raise ValueError("task_id was not issued to this contributor for this image")
            if (binding["status"] != "issued").any():
                raise ValueError("task_id has already been submitted")
            pending = self._read(self.pending_path)
            approved = self._read(self.approved_path)
            for labels in (pending, approved):
                if ((labels["contributor_id"] == contributor_id)
                    & (labels["image_path"] == payload["image_path"])).any():
                    raise ValueError("image already submitted by this contributor")
            record = {"task_id": task_id, "contributor_id": contributor_id,
                      "image_path": payload["image_path"], "region": payload.get("region", "unknown"),
                      "scenic_human": score, "confidence": payload.get("confidence", "medium"),
                      "skip": skip, "notes": payload.get("notes", ""), "status": "pending",
                      "timestamp": self._now(), "agreement_score": None, "qa_reason": ""}
            pending = pd.concat([pending, pd.DataFrame([record])], ignore_index=True)
            profiles, idx = self._profile_row(self._read(self.profiles_path), contributor_id)
            profiles.at[idx, "pending_labels"] = int(profiles.at[idx, "pending_labels"]) + 1
            profiles.at[idx, "updated_at"] = self._now()
            issued.loc[binding.index, "status"] = "submitted"
            self._persist({self.pending_path: pending, self.profiles_path: profiles, self.tasks_path: issued})
            return record

    def run_qa_promotion(self, min_overlap: int = 1, min_agreement: float = 0.65) -> dict:
        with self._locked():
            pending = self._read(self.pending_path)
            if pending.empty:
                return {"pending_rows": 0, "promoted": 0, "rejected": 0, "skipped": 0, "remaining_pending": 0}
            approved = self._read(self.approved_path)
            profiles = self._read(self.profiles_path)
            events = self._read(self.events_path)
            seed = self._read(self.cfg.approved_seed_csv)
            if seed.empty:
                seed = pd.DataFrame(columns=["image_path", "scenic_human"])
            if "scenic_human" not in seed.columns and "scenic_score" in seed.columns:
                seed = seed.rename(columns={"scenic_score": "scenic_human"})
            seed["scenic_human"] = pd.to_numeric(seed["scenic_human"], errors="coerce")
            seed_ref = seed.dropna(subset=["scenic_human"]).groupby("image_path")["scenic_human"].mean()
            promoted = rejected = skipped = 0
            keep_pending, approved_records, event_records = [], [], []
            for _, row in pending.iterrows():
                contributor_id = self._identifier(row["contributor_id"], "contributor_id")
                self._identifier(row["task_id"], "task_id")
                rec = row.to_dict()
                credits = 0.0
                agreement = None
                if str(row["skip"]).lower() == "true":
                    rec.update(status="skipped", qa_reason="contributor_skipped")
                    skipped += 1
                else:
                    candidates = [seed_ref[row["image_path"]]] if row["image_path"] in seed_ref.index else []
                    if len(candidates) < min_overlap:
                        keep_pending.append(rec)
                        continue
                    score = float(row["scenic_human"])
                    if not math.isfinite(score) or not 0 <= score <= 10:
                        raise ValueError("Invalid pending scenic_human score")
                    agreement = max(0.0, 1.0 - abs(score - float(candidates[0])) / 10.0)
                    rec["agreement_score"] = round(agreement, 4)
                    if agreement >= min_agreement:
                        rec.update(status="approved", qa_reason="agreement_pass")
                        approved_records.append(rec)
                        credits = 1.5 if agreement >= 0.8 else 1.0
                        promoted += 1
                    else:
                        rec.update(status="rejected", qa_reason="low_agreement")
                        rejected += 1
                profiles, idx = self._profile_row(profiles, contributor_id)
                profiles.at[idx, "pending_labels"] = max(0, int(profiles.at[idx, "pending_labels"]) - 1)
                profiles.at[idx, "updated_at"] = self._now()
                if rec["status"] == "approved":
                    profiles.at[idx, "credits"] = float(profiles.at[idx, "credits"]) + credits
                    profiles.at[idx, "approved_labels"] = int(profiles.at[idx, "approved_labels"]) + 1
                    profiles.at[idx, "agreement_score"] = round((float(profiles.at[idx, "agreement_score"]) + agreement) / 2, 4)
                event_records.append({"timestamp": self._now(), "contributor_id": contributor_id,
                                      "event": f"label_{rec['status']}", "credits_delta": credits,
                                      "metadata": json.dumps(rec)})
            if approved_records:
                approved = pd.concat([approved, pd.DataFrame(approved_records)], ignore_index=True)
            if event_records:
                events = pd.concat([events, pd.DataFrame(event_records)], ignore_index=True)
            self._persist({self.profiles_path: profiles, self.events_path: events,
                           self.approved_path: approved,
                           self.pending_path: pd.DataFrame(keep_pending, columns=DEFAULT_LABEL_COLUMNS)})
            return {"pending_rows": len(pending), "promoted": promoted, "rejected": rejected,
                    "skipped": skipped, "remaining_pending": len(keep_pending)}

    def leaderboard(self, limit: int = 25) -> list[dict]:
        with self._locked():
            profiles = self._read(self.profiles_path)
            if profiles.empty:
                return []
            profiles = profiles.sort_values(["credits", "approved_labels"], ascending=[False, False]).head(max(1, int(limit)))
            return profiles.to_dict(orient="records")
