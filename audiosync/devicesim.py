"""Device simulator.

Each device is a thread that captures small tone segments (deterministic
from its sample counter, so payloads can be regenerated after a restart)
and uploads them in chunks over HTTP. It simulates the messy reality:

  - dropped chunks (retried in a later pass)
  - corrupted chunks (checksum fails at validation -> rejected)
  - duplicate uploads of an already-sent range (indexer must dedupe)
  - conflicting re-captures of a range (indexer must flag, not count)
  - offline periods: segments are buffered and backfilled on reconnect
  - sample-rate switches mid-run (counter keeps counting)
  - reboots: new run_id, counter reset to 0

Device state (counter, run_id, backlog, in-flight segment) is persisted
through the server after every step, so killing and restarting the whole
process resumes capture and reception where they left off.
"""
import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid

from . import wavutil

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "segment_ms": 400,
    "chunk_size": 2048,
    "drop_prob": 0.05,
    "corrupt_prob": 0.02,
    "dup_prob": 0.03,
    "conflict_prob": 0.0,
    "offline_prob": 0.05,
    "offline_secs": 3.0,
    "running": False,
}


class Device(threading.Thread):
    def __init__(self, server, device_id, config=None, counter=0, run_id=None,
                 backlog=None, inflight=None):
        super().__init__(daemon=True, name="device-%s" % device_id)
        self.server = server  # base URL, e.g. http://127.0.0.1:8077
        self.device_id = device_id
        self.config = dict(DEFAULT_CONFIG, **(config or {}))
        self.counter = counter
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.backlog = backlog or []          # list of segment meta dicts
        self.inflight = inflight              # meta of segment being uploaded
        self.freq = 220.0 + (hash(device_id) % 8) * 110.0
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._next_capture = None
        self._offline_until = 0.0
        self.status = "init"
        self.last_error = ""

    # -- control (called from the web API) ------------------------------------
    def set_running(self, on):
        with self._lock:
            self.config["running"] = bool(on)
        self.persist()

    def update_config(self, **kw):
        with self._lock:
            for k, v in kw.items():
                if k in self.config and v is not None:
                    self.config[k] = v
        self.persist()

    def reboot(self):
        with self._lock:
            self.run_id = uuid.uuid4().hex[:8]
            self.counter = 0
            self.inflight = None
        self.persist()

    def stop(self):
        self._stop_flag.set()

    def snapshot(self):
        with self._lock:
            return {"device_id": self.device_id, "config": dict(self.config),
                    "counter": self.counter, "run_id": self.run_id,
                    "backlog": len(self.backlog), "status": self.status,
                    "last_error": self.last_error}

    # -- persistence via the server -------------------------------------------
    def persist(self):
        try:
            self._post("/api/device/state", {
                "device_id": self.device_id, "config": self.config,
                "counter": self.counter, "run_id": self.run_id,
                "backlog": self.backlog, "inflight": self.inflight})
        except Exception:
            pass  # server may be down; state is retried on next step

    # -- main loop --------------------------------------------------------------
    def run(self):
        """Queue model: capture on a wall-clock grid into a send queue; a
        throttled sender drains the queue in order. Going offline pauses
        only the sender — capture continues into the queue (offline
        replay). A segment uploaded long after its capture is what the
        server classifies as backfill."""
        self.status = "started"
        next_capture = None
        next_send = 0.0
        while not self._stop_flag.is_set():
            cfg = dict(self.config)
            if not cfg["running"]:
                self.status = "paused"
                next_capture = None
                time.sleep(0.2)
                continue
            try:
                now = time.time()
                period = cfg["segment_ms"] / 1000.0
                if next_capture is None:
                    next_capture = now + period
                # 1. capture on the grid
                if now >= next_capture:
                    meta = self._make_segment_meta(cfg, next_capture)
                    next_capture += period
                    with self._lock:
                        self.counter += meta["nsamples"]
                    self.backlog.append(meta)
                    self.persist()
                    continue
                # 2. maybe go offline (sender pauses, capture continues)
                if now >= self._offline_until and \
                        random.random() < cfg["offline_prob"] * period:
                    self._offline_until = now + cfg["offline_secs"]
                offline = now < self._offline_until
                # 3. throttled send, oldest first
                if self.backlog and not offline and now >= next_send:
                    meta = self.backlog[0]
                    fresh = now - meta["captured_at"] < 2 * period
                    self._upload(meta)
                    self.backlog.pop(0)
                    self.persist()
                    # throttle replay, but drain fast when deeply behind
                    next_send = time.time() + (0.02 if len(self.backlog) > 8
                                               else period * 0.4)
                    if fresh:
                        self._maybe_duplicate_or_conflict(meta, cfg)
                    continue
                self.status = "offline" if offline else (
                    "replaying(%d)" % len(self.backlog) if self.backlog
                    else "live")
                wake = next_capture
                if self.backlog and not offline:
                    wake = min(wake, next_send)
                time.sleep(min(0.1, max(0.005, wake - time.time())))
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                self.status = "server-unreachable"
                self.last_error = str(e)
                time.sleep(0.5)
            except Exception as e:  # keep the device alive; report and continue
                self.status = "error"
                self.last_error = repr(e)
                time.sleep(0.5)

    def _make_segment_meta(self, cfg, captured_at):
        rate = cfg["sample_rate"]
        nsamples = int(rate * cfg["segment_ms"] / 1000)
        start = self.counter
        pcm = wavutil.gen_tone(start, nsamples, rate, self.freq)
        return {
            "segment_uid": "%s:%s:%d" % (self.device_id, self.run_id, start),
            "run_id": self.run_id,
            "counter_start": start,
            "nsamples": nsamples,
            "sample_rate": rate,
            "captured_at": captured_at,
            "pcm_sha256": wavutil.pcm_sha256(pcm),
        }

    def _payload(self, meta):
        pcm = wavutil.gen_tone(meta["counter_start"], meta["nsamples"],
                               meta["sample_rate"], self.freq)
        return wavutil.make_wav(pcm, meta["sample_rate"])

    def _maybe_duplicate_or_conflict(self, meta, cfg):
        r = random.random()
        if r < cfg["conflict_prob"]:
            # buggy re-capture: same counter range, different content, but a
            # valid checksum of that content — passes validation, collides
            # at indexing
            evil = dict(meta)
            evil["segment_uid"] = meta["segment_uid"] + ":conflict"
            pcm = wavutil.gen_tone(meta["counter_start"], meta["nsamples"],
                                   meta["sample_rate"], self.freq * 1.5)
            evil["pcm_sha256"] = wavutil.pcm_sha256(pcm)
            self._upload(evil, payload=wavutil.make_wav(pcm, meta["sample_rate"]))
        elif r < cfg["conflict_prob"] + cfg["dup_prob"]:
            dup = dict(meta)
            dup["segment_uid"] = meta["segment_uid"] + ":dup%d" % random.randint(0, 1 << 30)
            dup["captured_at"] = time.time()
            self._upload(dup)

    # -- chunked upload ---------------------------------------------------------
    def _upload(self, meta, payload=None):
        cfg = dict(self.config)
        payload = payload if payload is not None else self._payload(meta)
        chunk_size = cfg["chunk_size"]
        chunks = [payload[i:i + chunk_size]
                  for i in range(0, len(payload), chunk_size)]
        resp = self._post("/api/upload/start", {
            "segment_uid": meta["segment_uid"], "device_id": self.device_id,
            "meta": meta, "total_chunks": len(chunks),
            "chunk_size": chunk_size, "total_bytes": len(payload)})
        session_id = resp["session_id"]
        if resp.get("already_completed"):
            return
        have = set(self._get("/api/upload/status?session_id=%s" % session_id)["have"])
        dropped = []
        for i, ch in enumerate(chunks):
            if i in have:
                continue
            if random.random() < cfg["drop_prob"]:
                dropped.append(i)
                continue
            data = ch
            if random.random() < cfg["corrupt_prob"] and len(ch) > 4:
                data = ch[:10] + bytes([ch[10] ^ 0xFF]) + ch[11:]
            self._post_chunk(session_id, i, data)
        for rnd in range(4):
            have = set(self._get("/api/upload/status?session_id=%s" % session_id)["have"])
            missing = [i for i in range(len(chunks)) if i not in have]
            if not missing:
                break
            for i in missing:
                self._post_chunk(session_id, i, chunks[i])
            time.sleep(0.1 * (rnd + 1))
        self._post("/api/upload/complete", {"session_id": session_id})

    # -- HTTP helpers -------------------------------------------------------------
    def _post(self, path, obj):
        req = urllib.request.Request(
            self.server + path, data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def _get(self, path):
        with urllib.request.urlopen(self.server + path, timeout=5) as r:
            return json.loads(r.read())

    def _post_chunk(self, session_id, index, data):
        req = urllib.request.Request(
            self.server + "/api/upload/chunk?session_id=%s&index=%d" % (session_id, index),
            data=data, headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())


def revive_devices(storage, server_url):
    """Recreate device threads from persisted state after a process restart."""
    devices = {}
    for device_id, st in storage.load_devices().items():
        d = Device(server_url, device_id, config=st["config"],
                   counter=st["counter"], run_id=st["run_id"],
                   backlog=st["backlog"], inflight=st["inflight"])
        devices[device_id] = d
        d.start()
    return devices
