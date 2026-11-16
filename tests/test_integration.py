"""Integration tests: real HTTP server + device simulator + restart recovery."""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync.app import serve  # noqa: E402
from audiosync import wavutil  # noqa: E402

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def post(url, path, obj=None, raw=None):
    data = raw if raw is not None else (json.dumps(obj).encode() if obj is not None else b"")
    req = urllib.request.Request(url + path, data=data, headers={
        "Content-Type": "application/octet-stream" if raw is not None
        else "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def get(url, path):
    try:
        with urllib.request.urlopen(url + path, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None


def wait_for(pred, timeout=15.0, interval=0.1, desc="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    raise AssertionError("timeout waiting for %s" % desc)


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="audiosync-test")
        self.db = os.path.join(self.dir, "audio.db")
        self.app, self.httpd, self.url = serve(self.db, self.dir, WEB,
                                               "127.0.0.1", 0)
        import threading
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.app.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app.storage.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def restart(self):
        """Simulate a process crash+restart on the same data directory."""
        self.app.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app.storage.close()
        self.app, self.httpd, self.url = serve(self.db, self.dir, WEB,
                                               "127.0.0.1", 0)
        import threading
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    # -- helpers ---------------------------------------------------------------
    def segments(self, state=None):
        q = "?state=%s" % state if state else ""
        r = get(self.url, "/api/segments" + q)
        return (r or {}).get("segments", [])

    def published(self):
        return get(self.url, "/api/index")

    def make_segment(self, uid, start=0, n=1600, rate=16000, corrupt=False):
        pcm = wavutil.gen_tone(start, n, rate, 440.0)
        wav = wavutil.make_wav(pcm, rate)
        sha = wavutil.pcm_sha256(pcm) if not corrupt else "0" * 64
        meta = {"run_id": "rtest", "counter_start": start, "nsamples": n,
                "sample_rate": rate, "captured_at": time.time(),
                "pcm_sha256": sha}
        return uid, meta, wav

    def upload(self, uid, meta, wav, chunk_size=1024, skip_chunks=()):
        chunks = [wav[i:i + chunk_size] for i in range(0, len(wav), chunk_size)]
        r = post(self.url, "/api/upload/start", {
            "segment_uid": uid, "device_id": "dtest", "meta": meta,
            "total_chunks": len(chunks), "chunk_size": chunk_size,
            "total_bytes": len(wav)})
        sid = r["session_id"]
        for i, ch in enumerate(chunks):
            if i in skip_chunks:
                continue
            post(self.url, "/api/upload/chunk?session_id=%s&index=%d" % (sid, i),
                 raw=ch)
        return sid, len(chunks)

    # -- tests -------------------------------------------------------------------
    def test_end_to_end_and_retract(self):
        post(self.url, "/api/device/create", {"device_id": "devA", "config": {
            "segment_ms": 100, "chunk_size": 1024, "drop_prob": 0.0,
            "corrupt_prob": 0.0, "dup_prob": 0.0, "offline_prob": 0.0}})
        post(self.url, "/api/device/devA/start", {})
        wait_for(lambda: len(self.segments("validated")) >= 4, desc="4 validated")

        def index_with(n):
            idx = self.published()
            if idx and idx.get("report", {}).get("stats", {}).get(
                    "segments_used", 0) >= n:
                return idx
            return None
        idx = wait_for(lambda: index_with(4), desc="published index with 4 segments")
        rep = idx["report"]
        self.assertGreater(rep["stats"]["coverage_seconds"], 0.2)
        dev = rep["devices"]["devA"]
        self.assertEqual(len(dev["epochs"]), 1)
        run = dev["epochs"][0]["rate_runs"][0]
        self.assertGreaterEqual(run["n_anchors"], 4)
        # free-fit drift over a sub-second window is noisy; sanity-bound it
        self.assertLess(abs(run["drift_ppm"] or 0), 30000)
        self.assertLess(run["resid_max"], 0.5)

        # retraction must bump validation_version and the published index
        # must stop counting the retracted segment
        victim = self.segments("validated")[0]
        before = idx["report"]["validation_version"]
        r = post(self.url, "/api/segments/%d/retract" % victim["segment_id"],
                 {"reason": "test"})
        self.assertTrue(r["ok"])
        self.assertGreater(r["validation_version"], before)

        def retracted_out():
            idx2 = self.published()
            if not idx2 or "report" not in idx2:
                return None
            rep2 = idx2["report"]
            if rep2["validation_version"] < r["validation_version"]:
                return None
            ids = [p["segment_id"] for p in rep2["provenance"]]
            return rep2 if victim["segment_id"] not in ids else None
        wait_for(retracted_out, desc="published index without retracted segment")

        post(self.url, "/api/device/devA/stop", {})

    def test_corrupt_segment_rejected(self):
        uid, meta, wav = self.make_segment("bad1", corrupt=True)
        sid, _ = self.upload(uid, meta, wav)
        post(self.url, "/api/upload/complete", {"session_id": sid})
        wait_for(lambda: self.segments("rejected"), desc="rejection")
        seg = self.segments("rejected")[0]
        self.assertIn("checksum", seg["state_reason"])

    def test_restart_resumes_upload_and_devices(self):
        # partial upload before "crash"
        uid, meta, wav = self.make_segment("resume1")
        sid, total = self.upload(uid, meta, wav, skip_chunks=(2, 3))
        # device running before crash
        post(self.url, "/api/device/create", {"device_id": "devB", "config": {
            "segment_ms": 100, "drop_prob": 0.0, "corrupt_prob": 0.0,
            "offline_prob": 0.0}})
        post(self.url, "/api/device/devB/start", {})
        wait_for(lambda: len(self.segments("validated")) >= 1,
                 desc="one validated before crash")
        v_before = get(self.url, "/api/state")["validation_version"]

        self.restart()

        # upload session survived: bitmap shows received chunks
        st = get(self.url, "/api/upload/status?session_id=%s" % sid)
        self.assertEqual(len(st["have"]), total - 2)
        # resume: re-open with same uid (idempotent), send missing, complete
        chunks = [wav[i:i + 1024] for i in range(0, len(wav), 1024)]
        r = post(self.url, "/api/upload/start", {
            "segment_uid": uid, "device_id": "dtest", "meta": meta,
            "total_chunks": len(chunks), "chunk_size": 1024,
            "total_bytes": len(wav)})
        sid2 = r["session_id"]
        for i in (2, 3):
            post(self.url, "/api/upload/chunk?session_id=%s&index=%d" % (sid2, i),
                 raw=chunks[i])
        post(self.url, "/api/upload/complete", {"session_id": sid2})
        wait_for(lambda: any(s["segment_uid"] == "resume1"
                             for s in self.segments("validated")),
                 desc="resumed segment validated")
        # device thread revived from persisted state and keeps sending
        wait_for(lambda: get(self.url, "/api/state")["validation_version"]
                 > v_before + 1, desc="device resumed after restart")
        devs = get(self.url, "/api/state")["devices"]
        self.assertEqual(devs[0]["device_id"], "devB")
        self.assertTrue(devs[0]["config"]["running"])

    def test_duplicate_and_conflict_via_api(self):
        uid, meta, wav = self.make_segment("orig", start=0)
        sid, _ = self.upload(uid, meta, wav)
        post(self.url, "/api/upload/complete", {"session_id": sid})
        # identical duplicate: deduped, not a conflict
        uid2, meta2, _ = self.make_segment("dup", start=0)
        sid2, _ = self.upload(uid2, meta2, wav)
        post(self.url, "/api/upload/complete", {"session_id": sid2})
        # conflicting content on the same counter range
        pcm2 = wavutil.gen_tone(0, 1600, 16000, 999.0)
        wav2 = wavutil.make_wav(pcm2, 16000)
        meta3 = dict(meta, pcm_sha256=wavutil.pcm_sha256(pcm2))
        sid3, _ = self.upload("evil", meta3, wav2)
        post(self.url, "/api/upload/complete", {"session_id": sid3})
        wait_for(lambda: len(self.segments("validated")) >= 3,
                 desc="3 validated")

        def has_conflict():
            idx = self.published()
            if not idx or "report" not in idx:
                return None
            rep = idx["report"]
            return rep if rep["conflicts"] else None
        rep = wait_for(has_conflict, desc="conflict in published index")
        self.assertEqual(rep["conflicts"][0]["counter_range"], [0, 1600])
        dev = rep["devices"]["dtest"]
        self.assertEqual(dev["duplicates_deduped"], 1)
        # conflicting range excluded from coverage
        self.assertEqual(dev["epochs"][0]["covered_counter"], [])


if __name__ == "__main__":
    unittest.main()
