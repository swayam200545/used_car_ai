"""
Prediction history storage.

Uses MongoDB when it is reachable (set MONGO_URI, default mongodb://localhost:27017).
If MongoDB is not running the app silently falls back to a local JSON file
(data/history.json) so the project still works on any laptop.
"""
import json
import os
import threading
import uuid
from datetime import datetime, timezone

try:
    from pymongo import MongoClient, DESCENDING
    from bson import ObjectId
except Exception:  # pymongo not installed
    MongoClient = None


class Store:
    def __init__(self, file_path=None):
        if file_path is None:
            if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
                file_path = "/tmp/history.json"
            else:
                file_path = os.path.join("data", "history.json")
        self.file_path = file_path
        self._lock = threading.Lock()
        self.col = None
        self.backend = "Local file"

        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        if MongoClient is not None:
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=1200)
                client.admin.command("ping")
                self.col = client[os.getenv("MONGO_DB", "used_car_ai")]["predictions"]
                self.backend = "MongoDB"
            except Exception:
                self.col = None

    # ---------------------------------------------------------------- helpers
    def _read(self):
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, rows):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(rows, f)

    # -------------------------------------------------------------------- API
    def add(self, record: dict) -> str:
        record = {**record, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if self.col is not None:
            return str(self.col.insert_one(record).inserted_id)
        with self._lock:
            rows = self._read()
            record["id"] = uuid.uuid4().hex[:12]
            rows.append(record)
            self._write(rows[-500:])
        return record["id"]

    def list(self, limit=50):
        if self.col is not None:
            out = []
            for d in self.col.find().sort("_id", DESCENDING).limit(limit):
                d["id"] = str(d.pop("_id"))
                out.append(d)
            return out
        with self._lock:
            return list(reversed(self._read()))[:limit]

    def count(self):
        if self.col is not None:
            return self.col.count_documents({})
        with self._lock:
            return len(self._read())

    def get(self, rec_id: str):
        if self.col is not None:
            try:
                d = self.col.find_one({"_id": ObjectId(rec_id)})
                if d:
                    d["id"] = str(d.pop("_id"))
                    return d
            except Exception:
                pass
            return None
        with self._lock:
            rows = self._read()
            for r in rows:
                if r.get("id") == rec_id:
                    return r
            return None

    def delete(self, rec_id: str) -> bool:
        if self.col is not None:
            try:
                return self.col.delete_one({"_id": ObjectId(rec_id)}).deleted_count > 0
            except Exception:
                return False
        with self._lock:
            rows = self._read()
            kept = [r for r in rows if r.get("id") != rec_id]
            self._write(kept)
            return len(kept) != len(rows)

    def clear(self):
        if self.col is not None:
            self.col.delete_many({})
            return
        with self._lock:
            self._write([])
