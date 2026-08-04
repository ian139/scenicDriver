from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import pandas as pd


DEFAULT_PROFILE_COLUMNS = [
    "contributor_id",
    "display_name",
    "credits",
    "approved_labels",
    "pending_labels",
    "agreement_score",
    "created_at",
    "updated_at",
]

DEFAULT_LABEL_COLUMNS = [
    "task_id",
    "contributor_id",
    "image_path",
    "region",
    "scenic_human",
    "confidence",
    "skip",
    "notes",
    "status",
    "timestamp",
    "agreement_score",
    "qa_reason",
]


@dataclass
class ContribConfig:
    root_dir: Path = Path("data/processed/contrib")
    approved_seed_csv: Path = Path("data/raw/labels_human.csv")
    labels_source_csv: Path = Path("data/processed/heuristic_runs/masswhites_z14_learned_h4_v2/labels.csv")


class ContribRepo:
    def __init__(self, cfg: ContribConfig | None = None):
        self.cfg = cfg or ContribConfig()
        self.cfg.root_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_path = self.cfg.root_dir / "contributor_profiles.csv"
        self.pending_path = self.cfg.root_dir / "annotations_pending.csv"
        self.approved_path = self.cfg.root_dir / "annotations_approved.csv"
        self.events_path = self.cfg.root_dir / "contrib_events.csv"
        self._ensure_files()

    def _ensure_files(self) -> None:
        self._ensure_csv(self.profiles_path, DEFAULT_PROFILE_COLUMNS)
        self._ensure_csv(self.pending_path, DEFAULT_LABEL_COLUMNS)
        self._ensure_csv(self.approved_path, DEFAULT_LABEL_COLUMNS)
        self._ensure_csv(self.events_path, ["timestamp", "contributor_id", "event", "credits_delta", "metadata"])

    @staticmethod
    def _ensure_csv(path: Path, columns: list[str]) -> None:
        if path.exists():
            return
        pd.DataFrame(columns=columns).to_csv(path, index=False)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _write(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def upsert_profile(self, contributor_id: str, display_name: str | None = None) -> dict:
        profiles = self._read(self.profiles_path)
        now = self._now()
        if profiles.empty:
            profiles = pd.DataFrame(columns=DEFAULT_PROFILE_COLUMNS)

        mask = profiles["contributor_id"].astype(str) == contributor_id
        if mask.any():
            idx = profiles.loc[mask].index[0]
            if display_name:
                profiles.at[idx, "display_name"] = display_name
            profiles.at[idx, "updated_at"] = now
        else:
            profiles = pd.concat(
                [
                    profiles,
                    pd.DataFrame(
                        [
                            {
                                "contributor_id": contributor_id,
                                "display_name": display_name or contributor_id,
                                "credits": 0.0,
                                "approved_labels": 0,
                                "pending_labels": 0,
                                "agreement_score": 0.0,
                                "created_at": now,
                                "updated_at": now,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        self._write(self.profiles_path, profiles)
        return self.get_profile(contributor_id)

    def get_profile(self, contributor_id: str) -> dict:
        profiles = self._read(self.profiles_path)
        row = profiles.loc[profiles["contributor_id"].astype(str) == contributor_id]
        if row.empty:
            return self.upsert_profile(contributor_id)
        return row.iloc[0].to_dict()

    def _task_pool(self, region: str) -> pd.DataFrame:
        src = self._read(self.cfg.labels_source_csv)
        if src.empty:
            return pd.DataFrame(columns=["image_path"])
        src["image_path"] = src["image_path"].astype(str)
        region = region.strip().lower()
        if region:
            src = src.loc[src["image_path"].str.contains(region, case=False, regex=False)]
        if src.empty:
            src = self._read(self.cfg.labels_source_csv)
        cols = [c for c in ["image_path", "class_id", "class_name", "scenic_score"] if c in src.columns]
        return src[cols].drop_duplicates("image_path")

    def next_tasks(self, contributor_id: str, region: str, count: int = 25) -> list[dict]:
        pool = self._task_pool(region)
        if pool.empty:
            return []
        pending = self._read(self.pending_path)
        approved = self._read(self.approved_path)
        done = set()
        for df in (pending, approved):
            if not df.empty:
                rows = df.loc[df["contributor_id"].astype(str) == contributor_id, "image_path"]
                done.update(rows.astype(str).tolist())
        tasks = pool.loc[~pool["image_path"].isin(done)].head(max(1, int(count)))
        out = []
        for _, row in tasks.iterrows():
            out.append(
                {
                    "task_id": str(uuid.uuid4()),
                    "image_path": str(row["image_path"]),
                    "class_id": int(row["class_id"]) if "class_id" in row and pd.notna(row["class_id"]) else None,
                    "class_name": str(row["class_name"]) if "class_name" in row and pd.notna(row["class_name"]) else None,
                    "scenic_score": float(row["scenic_score"]) if "scenic_score" in row and pd.notna(row["scenic_score"]) else None,
                }
            )
        return out

    def submit_label(self, payload: dict) -> dict:
        pending = self._read(self.pending_path)
        if pending.empty:
            pending = pd.DataFrame(columns=DEFAULT_LABEL_COLUMNS)
        record = {
            "task_id": payload["task_id"],
            "contributor_id": payload["contributor_id"],
            "image_path": payload["image_path"],
            "region": payload.get("region", "unknown"),
            "scenic_human": float(payload["scenic_human"]),
            "confidence": payload.get("confidence", "medium"),
            "skip": bool(payload.get("skip", False)),
            "notes": payload.get("notes", ""),
            "status": "pending",
            "timestamp": self._now(),
            "agreement_score": None,
            "qa_reason": "",
        }
        pending = pd.concat([pending, pd.DataFrame([record])], ignore_index=True)
        self._write(self.pending_path, pending)
        self._increment_pending(payload["contributor_id"], 1)
        return record

    def _increment_pending(self, contributor_id: str, delta: int) -> None:
        profiles = self._read(self.profiles_path)
        mask = profiles["contributor_id"].astype(str) == contributor_id
        if not mask.any():
            self.upsert_profile(contributor_id)
            profiles = self._read(self.profiles_path)
            mask = profiles["contributor_id"].astype(str) == contributor_id
        idx = profiles.loc[mask].index[0]
        profiles.at[idx, "pending_labels"] = max(0, int(profiles.at[idx, "pending_labels"]) + int(delta))
        profiles.at[idx, "updated_at"] = self._now()
        self._write(self.profiles_path, profiles)

    def _apply_credits(self, contributor_id: str, credits_delta: float, approved_delta: int, agreement_score: float) -> None:
        profiles = self._read(self.profiles_path)
        mask = profiles["contributor_id"].astype(str) == contributor_id
        if not mask.any():
            self.upsert_profile(contributor_id)
            profiles = self._read(self.profiles_path)
            mask = profiles["contributor_id"].astype(str) == contributor_id
        idx = profiles.loc[mask].index[0]
        profiles.at[idx, "credits"] = float(profiles.at[idx, "credits"]) + float(credits_delta)
        profiles.at[idx, "approved_labels"] = int(profiles.at[idx, "approved_labels"]) + int(approved_delta)
        profiles.at[idx, "pending_labels"] = max(0, int(profiles.at[idx, "pending_labels"]) - int(approved_delta))
        prev_agree = float(profiles.at[idx, "agreement_score"])
        profiles.at[idx, "agreement_score"] = round((prev_agree + agreement_score) / 2.0, 4)
        profiles.at[idx, "updated_at"] = self._now()
        self._write(self.profiles_path, profiles)

    def run_qa_promotion(self, min_overlap: int = 1, min_agreement: float = 0.65) -> dict:
        pending = self._read(self.pending_path)
        approved = self._read(self.approved_path)
        seed = self._read(self.cfg.approved_seed_csv)
        if pending.empty:
            return {"pending_rows": 0, "promoted": 0, "rejected": 0}

        if approved.empty:
            approved = pd.DataFrame(columns=DEFAULT_LABEL_COLUMNS)
        if seed.empty:
            seed = pd.DataFrame(columns=["image_path", "scenic_human"])
        if "scenic_human" not in seed.columns and "scenic_score" in seed.columns:
            seed = seed.rename(columns={"scenic_score": "scenic_human"})
        seed = seed[[c for c in ["image_path", "scenic_human"] if c in seed.columns]]
        seed["image_path"] = seed["image_path"].astype(str)
        seed["scenic_human"] = pd.to_numeric(seed["scenic_human"], errors="coerce")
        seed = seed.dropna(subset=["scenic_human"])
        seed_ref = seed.groupby("image_path", as_index=False)["scenic_human"].mean()

        promoted = 0
        rejected = 0
        keep_pending = []
        for _, row in pending.iterrows():
            image_path = str(row["image_path"])
            candidates = seed_ref.loc[seed_ref["image_path"] == image_path, "scenic_human"].tolist()
            if len(candidates) < min_overlap:
                keep_pending.append(row.to_dict())
                continue
            mean_ref = float(sum(candidates) / len(candidates))
            score = float(row["scenic_human"])
            diff = abs(score - mean_ref)
            agreement = max(0.0, 1.0 - (diff / 10.0))
            rec = row.to_dict()
            rec["agreement_score"] = round(agreement, 4)
            if agreement >= min_agreement:
                rec["status"] = "approved"
                rec["qa_reason"] = "agreement_pass"
                approved = pd.concat([approved, pd.DataFrame([rec])], ignore_index=True)
                credits = 1.0 * (1.5 if agreement >= 0.8 else 1.0)
                self._apply_credits(str(row["contributor_id"]), credits, 1, agreement)
                self._append_event(str(row["contributor_id"]), "label_approved", credits, rec)
                promoted += 1
            else:
                rec["status"] = "rejected"
                rec["qa_reason"] = "low_agreement"
                self._append_event(str(row["contributor_id"]), "label_rejected", 0.0, rec)
                rejected += 1

        self._write(self.approved_path, approved)
        self._write(self.pending_path, pd.DataFrame(keep_pending, columns=DEFAULT_LABEL_COLUMNS))
        return {
            "pending_rows": int(len(pending)),
            "promoted": int(promoted),
            "rejected": int(rejected),
            "remaining_pending": int(len(keep_pending)),
        }

    def _append_event(self, contributor_id: str, event: str, credits_delta: float, metadata: dict) -> None:
        events = self._read(self.events_path)
        if events.empty:
            events = pd.DataFrame(columns=["timestamp", "contributor_id", "event", "credits_delta", "metadata"])
        events = pd.concat(
            [
                events,
                pd.DataFrame(
                    [
                        {
                            "timestamp": self._now(),
                            "contributor_id": contributor_id,
                            "event": event,
                            "credits_delta": float(credits_delta),
                            "metadata": json.dumps(metadata),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        self._write(self.events_path, events)

    def leaderboard(self, limit: int = 25) -> list[dict]:
        profiles = self._read(self.profiles_path)
        if profiles.empty:
            return []
        profiles = profiles.sort_values(["credits", "approved_labels"], ascending=[False, False]).head(max(1, int(limit)))
        return profiles.to_dict(orient="records")

