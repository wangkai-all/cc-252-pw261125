"""SQLite-backed persistence.

One connection guarded by a global write lock; all state changes that must be
atomic (segment state change + validation_version bump, index publish CAS)
happen inside a single transaction so a crash never leaves a half-written
state. Recovery after process restart is just: open the same db file.
"""
import json
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
CREATE TABLE IF NOT EXISTS upload_sessions (
  session_id TEXT PRIMARY KEY,
  segment_uid TEXT UNIQUE,
  device_id TEXT,
  meta_json TEXT,
  total_chunks INTEGER,
  chunk_size INTEGER,
  total_bytes INTEGER,
  created_at REAL,
  completed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
  session_id TEXT,
  chunk_index INTEGER,
  data BLOB,
  PRIMARY KEY (session_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS segments (
  segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
  segment_uid TEXT UNIQUE,
  device_id TEXT,
  run_id TEXT,
  counter_start INTEGER,
  nsamples INTEGER,
  sample_rate INTEGER,
  captured_at REAL,        -- device clock, seconds
  first_chunk_at REAL,     -- server time of first chunk
  completed_at REAL,       -- server time assembly finished
  backfill INTEGER,        -- 1 if uploaded long after capture (offline backlog)
  pcm_sha256 TEXT,
  wav_path TEXT,
  state TEXT,              -- pending | validated | rejected | retracted
  state_reason TEXT,
  validated_at REAL,
  validation_version INTEGER
);
CREATE TABLE IF NOT EXISTS index_versions (
  index_id INTEGER PRIMARY KEY AUTOINCREMENT,
  validation_version INTEGER,
  built_at REAL,
  report_json TEXT
);
CREATE TABLE IF NOT EXISTS device_state (
  device_id TEXT PRIMARY KEY,
  config_json TEXT,
  counter INTEGER,
  run_id TEXT,
  backlog_json TEXT,
  inflight_json TEXT
);
"""


class Storage:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.isolation_level = None  # autocommit; explicit BEGIN where atomic
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
            self._init_meta()

    def _init_meta(self):
        cur = self.conn.execute("SELECT value FROM meta WHERE key='validation_version'")
        if cur.fetchone() is None:
            self.conn.execute("INSERT INTO meta VALUES ('validation_version','0')")
            self.conn.execute("INSERT INTO meta VALUES ('published_index_id','')")
            self.conn.commit()

    # -- meta ---------------------------------------------------------------
    def get_meta(self, key):
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None

    def validation_version(self):
        return int(self.get_meta("validation_version") or 0)

    def published_index_id(self):
        v = self.get_meta("published_index_id")
        return int(v) if v else None

    # -- upload sessions ------------------------------------------------------
    def create_session(self, session_id, segment_uid, device_id, meta, total_chunks,
                       chunk_size, total_bytes):
        """Idempotent on segment_uid: returns (session_row, created?)."""
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM upload_sessions WHERE segment_uid=?", (segment_uid,)).fetchone()
            if row is not None:
                return dict(row), False
            try:
                self.conn.execute(
                    "INSERT INTO upload_sessions VALUES (?,?,?,?,?,?,?,?,0)",
                    (session_id, segment_uid, device_id, json.dumps(meta),
                     total_chunks, chunk_size, total_bytes, time.time()))
                self.conn.commit()
            except sqlite3.IntegrityError:
                row = self.conn.execute(
                    "SELECT * FROM upload_sessions WHERE segment_uid=?", (segment_uid,)).fetchone()
                return dict(row), False
            row = self.conn.execute(
                "SELECT * FROM upload_sessions WHERE session_id=?", (session_id,)).fetchone()
            return dict(row), True

    def get_session(self, session_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM upload_sessions WHERE session_id=?", (session_id,)).fetchone()
            return dict(row) if row else None

    def put_chunk(self, session_id, index, data):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?)",
                (session_id, index, sqlite3.Binary(data)))
            self.conn.commit()

    def chunk_bitmap(self, session_id):
        with self.lock:
            rows = self.conn.execute(
                "SELECT chunk_index FROM chunks WHERE session_id=? ORDER BY chunk_index",
                (session_id,)).fetchall()
            return [r["chunk_index"] for r in rows]

    def assemble_session(self, session_id):
        """Return full payload if all chunks present, else None."""
        with self.lock:
            sess = self.get_session(session_id)
            if not sess:
                return None
            rows = self.conn.execute(
                "SELECT chunk_index, data FROM chunks WHERE session_id=? ORDER BY chunk_index",
                (session_id,)).fetchall()
            if len(rows) != sess["total_chunks"]:
                return None
            if [r["chunk_index"] for r in rows] != list(range(sess["total_chunks"])):
                return None
            return b"".join(bytes(r["data"]) for r in rows)

    def complete_session(self, session_id):
        with self.lock:
            self.conn.execute("UPDATE upload_sessions SET completed=1 WHERE session_id=?",
                              (session_id,))
            self.conn.commit()

    # -- segments ---------------------------------------------------------------
    def insert_segment(self, seg):
        """Insert assembled segment as pending. Idempotent on segment_uid."""
        with self.lock:
            row = self.conn.execute(
                "SELECT segment_id, state FROM segments WHERE segment_uid=?",
                (seg["segment_uid"],)).fetchone()
            if row is not None:
                return row["segment_id"], False
            cur = self.conn.execute(
                """INSERT INTO segments
                   (segment_uid, device_id, run_id, counter_start, nsamples, sample_rate,
                    captured_at, first_chunk_at, completed_at, backfill, pcm_sha256,
                    wav_path, state, state_reason, validated_at, validation_version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (seg["segment_uid"], seg["device_id"], seg["run_id"], seg["counter_start"],
                 seg["nsamples"], seg["sample_rate"], seg["captured_at"],
                 seg["first_chunk_at"], seg["completed_at"], seg["backfill"],
                 seg["pcm_sha256"], seg["wav_path"], "pending", "", None,
                 self.validation_version()))
            self.conn.commit()
            return cur.lastrowid, True

    def set_segment_state(self, segment_id, state, reason):
        """State change + validation_version bump in ONE transaction.

        This is the linchpin of 'retraction must not leave stale indexes
        counting bad segments': any validation-state change immediately
        invalidates the published index's version basis.
        """
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                v = self.validation_version() + 1
                self.conn.execute(
                    "UPDATE segments SET state=?, state_reason=?, validated_at=?,"
                    " validation_version=? WHERE segment_id=?",
                    (state, reason, time.time(), v, segment_id))
                self.conn.execute("UPDATE meta SET value=? WHERE key='validation_version'",
                                  (str(v),))
                self.conn.commit()
                return v
            except Exception:
                self.conn.rollback()
                raise

    def get_segment(self, segment_id):
        with self.lock:
            row = self.conn.execute("SELECT * FROM segments WHERE segment_id=?",
                                    (segment_id,)).fetchone()
            return dict(row) if row else None

    def segment_by_uid(self, segment_uid):
        with self.lock:
            row = self.conn.execute(
                "SELECT segment_id, state FROM segments WHERE segment_uid=?",
                (segment_uid,)).fetchone()
            return dict(row) if row else None

    def list_segments(self, device_id=None, state=None):
        q = "SELECT * FROM segments"
        cond, args = [], []
        if device_id:
            cond.append("device_id=?"); args.append(device_id)
        if state:
            cond.append("state=?"); args.append(state)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY segment_id"
        with self.lock:
            return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def validated_segments(self):
        return self.list_segments(state="validated")

    # -- index versions ---------------------------------------------------------
    def insert_index(self, validation_version, report):
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO index_versions (validation_version, built_at, report_json)"
                " VALUES (?,?,?)",
                (validation_version, time.time(), json.dumps(report)))
            self.conn.commit()
            return cur.lastrowid

    def publish_index_cas(self, index_id, expect_validation_version):
        """Publish only if validation_version is unchanged since the build
        snapshot. Returns True if published, False if stale (caller rebuilds).
        """
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                v = self.validation_version()
                if v != expect_validation_version:
                    self.conn.rollback()
                    return False
                self.conn.execute("UPDATE meta SET value=? WHERE key='published_index_id'",
                                  (str(index_id),))
                self.conn.commit()
                return True
            except Exception:
                self.conn.rollback()
                raise

    def get_index(self, index_id):
        with self.lock:
            row = self.conn.execute("SELECT * FROM index_versions WHERE index_id=?",
                                    (index_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["report"] = json.loads(d.pop("report_json"))
            return d

    def list_indexes(self, limit=20):
        with self.lock:
            rows = self.conn.execute(
                "SELECT index_id, validation_version, built_at FROM index_versions"
                " ORDER BY index_id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    # -- device state -------------------------------------------------------------
    def save_device(self, device_id, config, counter, run_id, backlog, inflight):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO device_state VALUES (?,?,?,?,?,?)",
                (device_id, json.dumps(config), counter, run_id,
                 json.dumps(backlog), json.dumps(inflight)))
            self.conn.commit()

    def load_devices(self):
        with self.lock:
            rows = self.conn.execute("SELECT * FROM device_state").fetchall()
            out = {}
            for r in rows:
                out[r["device_id"]] = {
                    "config": json.loads(r["config_json"]),
                    "counter": r["counter"],
                    "run_id": r["run_id"],
                    "backlog": json.loads(r["backlog_json"]),
                    "inflight": json.loads(r["inflight_json"]),
                }
            return out

    def close(self):
        with self.lock:
            self.conn.close()
