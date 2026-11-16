"""HTTP server: chunked upload API, device control, validation/index views,
and the static web UI. Stdlib only (http.server).

Upload protocol (resumable, idempotent on segment_uid):
  POST /api/upload/start     {segment_uid, device_id, meta, total_chunks,
                              chunk_size, total_bytes}
                             -> {session_id, have:[indices], already_completed}
  GET  /api/upload/status?session_id=..   -> {have:[...], completed}
  POST /api/upload/chunk?session_id=..&index=N   (binary body)
  POST /api/upload/complete  {session_id}

Assembly, validation and indexing are separate stages: complete -> store
file + pending segment -> validator thread -> validated -> indexer thread
-> published index (CAS on validation_version).
"""
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import core
from .devicesim import Device, revive_devices
from .indexer import Indexer
from .storage import Storage
from .validator import Validator

# A segment is a usable alignment anchor only if its upload completed
# within a bounded transfer delay of capture. Longer lag means it sat in a
# device queue (offline replay / drain) and its receive time says nothing
# about capture time. 2s comfortably covers chunked transfer of a segment.
BACKFILL_DELAY_S = 2.0


class App:
    def __init__(self, db_path, data_dir, web_dir):
        self.storage = Storage(db_path)
        self.data_dir = data_dir
        self.web_dir = web_dir
        self.wav_dir = os.path.join(data_dir, "wav")
        os.makedirs(self.wav_dir, exist_ok=True)
        self.indexer = Indexer(self.storage)
        self.validator = Validator(self.storage, data_dir,
                                   on_change=self.indexer.request_rebuild)
        self.devices = {}
        self.devices_lock = threading.Lock()
        self.server_url = None  # set after bind

    def start_workers(self, server_url):
        self.server_url = server_url
        self.validator.start()
        self.indexer.start()
        with self.devices_lock:
            self.devices = revive_devices(self.storage, server_url)

    def shutdown(self):
        with self.devices_lock:
            devices = list(self.devices.values())
            for d in devices:
                d.stop()
        self.validator.stop()
        self.indexer.stop()
        for d in devices:
            d.join(timeout=3)
        self.validator.join(timeout=3)
        self.indexer.join(timeout=3)

    # -- upload ------------------------------------------------------------
    def upload_start(self, body):
        uid = body["segment_uid"]
        existing = self.storage.segment_by_uid(uid)
        if existing is not None:
            return {"already_completed": True, "segment_id": existing["segment_id"]}
        session_id = uuid.uuid4().hex
        sess, created = self.storage.create_session(
            session_id, uid, body["device_id"], body["meta"],
            int(body["total_chunks"]), int(body["chunk_size"]), int(body["total_bytes"]))
        return {"session_id": sess["session_id"],
                "have": self.storage.chunk_bitmap(sess["session_id"]),
                "already_completed": False}

    def upload_chunk(self, session_id, index, data):
        sess = self.storage.get_session(session_id)
        if not sess:
            return {"error": "unknown-session"}, 404
        if sess["completed"]:
            return {"error": "session-completed"}, 409
        if not (0 <= index < sess["total_chunks"]):
            return {"error": "bad-index"}, 400
        self.storage.put_chunk(session_id, index, data)
        return {"have": self.storage.chunk_bitmap(session_id)}, 200

    def upload_complete(self, body):
        session_id = body["session_id"]
        sess = self.storage.get_session(session_id)
        if not sess:
            return {"error": "unknown-session"}, 404
        if sess["completed"]:
            return {"ok": True, "note": "already-completed"}
        payload = self.storage.assemble_session(session_id)
        if payload is None:
            return {"error": "incomplete",
                    "have": self.storage.chunk_bitmap(session_id)}, 409
        if len(payload) != sess["total_bytes"]:
            return {"error": "size-mismatch"}, 400
        meta = json.loads(sess["meta_json"])
        now = time.time()
        captured = float(meta.get("captured_at", now))
        seg = {
            "segment_uid": sess["segment_uid"],
            "device_id": sess["device_id"],
            "run_id": meta.get("run_id", ""),
            "counter_start": int(meta["counter_start"]),
            "nsamples": int(meta["nsamples"]),
            "sample_rate": int(meta["sample_rate"]),
            "captured_at": captured,
            "first_chunk_at": sess["created_at"],
            "completed_at": now,
            "backfill": 1 if now - captured > BACKFILL_DELAY_S else 0,
            "pcm_sha256": meta.get("pcm_sha256", ""),
            "wav_path": os.path.join(self.wav_dir, sess["segment_uid"].replace(":", "_") + ".wav"),
        }
        with open(seg["wav_path"], "wb") as f:
            f.write(payload)
        segment_id, inserted = self.storage.insert_segment(seg)
        self.storage.complete_session(session_id)
        if inserted:
            self.validator.submit(segment_id)
        return {"ok": True, "segment_id": segment_id, "duplicate": not inserted}, 200

    # -- device control --------------------------------------------------------
    def device_create(self, body):
        device_id = body["device_id"]
        with self.devices_lock:
            if device_id in self.devices:
                return {"error": "exists"}, 409
            d = Device(self.server_url, device_id, config=body.get("config"))
            self.devices[device_id] = d
            d.start()
        d.persist()
        return {"ok": True}, 200

    def device_action(self, device_id, action, body):
        with self.devices_lock:
            d = self.devices.get(device_id)
        if not d:
            return {"error": "unknown-device"}, 404
        if action == "start":
            d.set_running(True)
        elif action == "stop":
            d.set_running(False)
        elif action == "config":
            d.update_config(**{k: v for k, v in body.items() if v is not None})
        elif action == "reboot":
            d.reboot()
        else:
            return {"error": "unknown-action"}, 400
        return {"ok": True}, 200

    def device_state_save(self, body):
        self.storage.save_device(body["device_id"], body["config"], body["counter"],
                                 body["run_id"], body.get("backlog", []),
                                 body.get("inflight"))
        return {"ok": True}, 200

    # -- views -------------------------------------------------------------------
    def state(self):
        with self.devices_lock:
            devices = [d.snapshot() for d in self.devices.values()]
        segs = self.storage.list_segments()
        counts = {}
        for s in segs:
            counts[s["state"]] = counts.get(s["state"], 0) + 1
        return {
            "devices": sorted(devices, key=lambda d: d["device_id"]),
            "validation_version": self.storage.validation_version(),
            "published_index_id": self.storage.published_index_id(),
            "segment_counts": counts,
            "indexes": self.storage.list_indexes(),
            "indexer": {"builds": self.indexer.builds,
                        "discarded_builds": self.indexer.discarded_builds,
                        "last_error": self.indexer.last_error},
        }

    def published_index(self):
        pid = self.storage.published_index_id()
        if pid is None:
            return None
        return self.storage.get_index(pid)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        # -- helpers ---------------------------------------------------------
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _json_body(self):
            raw = self._body()
            return json.loads(raw) if raw else {}

        # -- routing -----------------------------------------------------------
        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            p = u.path
            if p == "/" or p == "/index.html":
                return self._static("index.html", "text/html")
            if p in ("/app.js", "/style.css"):
                return self._static(p[1:], "application/javascript" if p.endswith(".js")
                                    else "text/css")
            if p == "/api/state":
                return self._json(app.state())
            if p == "/api/index":
                idx = app.published_index()
                return self._json(idx or {"error": "no-index-yet"},
                                  200 if idx else 404)
            if p.startswith("/api/index/"):
                idx = app.storage.get_index(int(p.rsplit("/", 1)[1]))
                return self._json(idx or {"error": "not-found"}, 200 if idx else 404)
            if p == "/api/segments":
                return self._json({"segments": app.storage.list_segments(
                    device_id=q.get("device", [None])[0],
                    state=q.get("state", [None])[0])})
            if p == "/api/upload/status":
                sid = q["session_id"][0]
                sess = app.storage.get_session(sid)
                if not sess:
                    return self._json({"error": "unknown-session"}, 404)
                return self._json({"have": app.storage.chunk_bitmap(sid),
                                   "completed": bool(sess["completed"]),
                                   "total_chunks": sess["total_chunks"]})
            if p == "/api/gaps":
                idx = app.published_index()
                if not idx:
                    return self._json({"gaps": []})
                gaps = []
                for dev, d in idx["report"]["devices"].items():
                    for ep in d["epochs"]:
                        for g in ep["time_gaps"]:
                            gaps.append({"device": dev, "run_id": ep["run_id"],
                                         "epoch_seq": ep["epoch_seq"],
                                         "time_range": g, "seconds": g[1] - g[0]})
                return self._json({"gaps": gaps})
            return self._json({"error": "not-found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            p = u.path
            try:
                if p == "/api/upload/start":
                    return self._json(app.upload_start(self._json_body()))
                if p == "/api/upload/chunk":
                    data = self._body()
                    res, code = app.upload_chunk(q["session_id"][0],
                                                 int(q["index"][0]), data)
                    return self._json(res, code)
                if p == "/api/upload/complete":
                    res, code = app.upload_complete(self._json_body())
                    return self._json(res, code)
                if p == "/api/device/create":
                    res, code = app.device_create(self._json_body())
                    return self._json(res, code)
                if p == "/api/device/state":
                    res, code = app.device_state_save(self._json_body())
                    return self._json(res, code)
                if p.startswith("/api/device/"):
                    parts = p.split("/")
                    device_id, action = parts[3], parts[4]
                    res, code = app.device_action(device_id, action, self._json_body())
                    return self._json(res, code)
                if p.startswith("/api/segments/"):
                    parts = p.split("/")
                    seg_id, action = int(parts[3]), parts[4]
                    if action == "retract":
                        v = app.validator.retract(seg_id, self._json_body().get("reason"))
                        return self._json({"ok": v is not None, "validation_version": v})
                    if action == "revalidate":
                        v = app.validator.revalidate(seg_id)
                        return self._json({"ok": v is not None, "validation_version": v})
                if p == "/api/index/rebuild":
                    app.indexer.request_rebuild()
                    return self._json({"ok": True})
            except (KeyError, ValueError) as e:
                return self._json({"error": "bad-request: %s" % e}, 400)
            return self._json({"error": "not-found"}, 404)

        def _static(self, name, ctype):
            path = os.path.join(app.web_dir, name)
            if not os.path.isfile(path):
                return self._json({"error": "not-found"}, 404)
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(db_path, data_dir, web_dir, host="127.0.0.1", port=8077):
    app = App(db_path, data_dir, web_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = "http://%s:%d" % (host, httpd.server_address[1])
    app.start_workers(url)
    return app, httpd, url
