#!/usr/bin/env python3
"""End-to-end demo scenario against a running server (default :8077).

Narrative:
  1. two devices start streaming (dev-1 flaky: drops/corrupts/dups/offline,
     dev-2 48kHz, occasionally sends conflicting re-captures)
  2. dev-1 switches sample rate mid-run        -> rate runs + chaining
  3. dev-2 reboots (sample counter resets)     -> new epoch, reset boundary
  4. one segment is retracted via the API      -> index republished without it
  5. report summary of the published index

Usage: python3 demo_scenario.py [base_url]
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"


def api(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def wait_v(vmin, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = api("/api/state")["validation_version"]
        if v >= vmin:
            return v
        time.sleep(2)
    raise SystemExit("timeout waiting for validation_version >= %d" % vmin)


def show(tag):
    idx = api("/api/index")
    if "report" not in idx:
        print("[%s] no published index yet" % tag)
        return
    r = idx["report"]
    print("\n[%s] published index #%d (validation v%d)" %
          (tag, idx["index_id"], r["validation_version"]))
    for dev, d in sorted(r["devices"].items()):
        print("  %s: coverage %.1fs, gaps %.1fs, deduped %d" %
              (dev, d["coverage_seconds"], d["gap_seconds"],
               d["duplicates_deduped"]))
        for ep in d["epochs"]:
            for rr in ep["rate_runs"]:
                drift = ("%.0fppm" % rr["drift_ppm"]) \
                    if rr["drift_ppm"] is not None else "-"
                print("    run %s#%d @%dHz anchors=%d drift=%s conf=%s" %
                      (ep["run_id"][:6], ep["epoch_seq"], rr["rate"],
                       rr["n_anchors"], drift, rr["confidence"]))
    print("  conflicts: %d, unconfirmable: %d" %
          (len(r["conflicts"]), len(r["unconfirmable"])))


def main():
    print("base:", BASE)
    api("/api/device/create", {"device_id": "dev-1", "config": {
        "segment_ms": 300, "drop_prob": 0.08, "corrupt_prob": 0.02,
        "dup_prob": 0.04, "offline_prob": 0.15}})
    api("/api/device/create", {"device_id": "dev-2", "config": {
        "segment_ms": 300, "sample_rate": 48000, "conflict_prob": 0.05}})
    api("/api/device/dev-1/start", {})
    api("/api/device/dev-2/start", {})
    print("devices started, streaming 25s ...")
    wait_v(150)
    show("t1: live streaming")

    print("\n>> dev-1 switches sample rate 16000 -> 8000")
    api("/api/device/dev-1/config", {"sample_rate": 8000})
    wait_v(api("/api/state")["validation_version"] + 60)
    show("t2: after rate switch")

    print("\n>> dev-2 reboots (sample counter resets)")
    api("/api/device/dev-2/reboot", {})
    wait_v(api("/api/state")["validation_version"] + 60)
    show("t3: after counter reset")

    print("\n>> retracting one validated segment of dev-1")
    segs = [s for s in api("/api/segments?device=dev-1&state=validated")
            ["segments"]]
    victim = segs[len(segs) // 2]
    r = api("/api/segments/%d/retract" % victim["segment_id"],
            {"reason": "demo retract"})
    print("   retracted segment %d -> validation v%d"
          % (victim["segment_id"], r["validation_version"]))
    time.sleep(2)
    idx = api("/api/index")
    ids = [p["segment_id"] for p in idx["report"]["provenance"]]
    assert victim["segment_id"] not in ids, "retracted segment still indexed!"
    assert idx["report"]["validation_version"] >= r["validation_version"]
    print("   published index #%d no longer counts it (checked %d segments)"
          % (idx["index_id"], len(ids)))
    show("t4: after retract")

    print("\nDemo done. Server state is in ./data — kill and restart"
          " run_server.py to watch recovery: devices resume, upload"
          " sessions continue, pending segments re-validate.")


if __name__ == "__main__":
    main()
