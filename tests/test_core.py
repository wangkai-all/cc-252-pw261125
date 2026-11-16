"""Unit tests for the time-index domain logic."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiosync import core  # noqa: E402

T0 = 1_700_000_000.0


def seg(sid, device="d1", run="r1", start=0, n=8000, rate=16000,
        captured=T0, completed=None, backfill=0, sha="a"):
    return {
        "segment_id": sid, "segment_uid": "u%d" % sid, "device_id": device,
        "run_id": run, "counter_start": start, "nsamples": n,
        "sample_rate": rate, "captured_at": captured,
        "completed_at": completed if completed is not None else captured + 0.1,
        "backfill": backfill, "pcm_sha256": sha, "state": "validated",
    }


def live_chain(n_seg, start=0, n=8000, rate=16000, t0=T0, device="d1",
               run="r1", sha_prefix="a", gap_every=None):
    """A clean live chain: segment k covers [start+k*n, ...); captured_at is
    the device-clock capture start, completed_at is capture end + 50ms
    transfer (what a real live upload looks like server-side)."""
    out = []
    sid = 1
    c = start
    k = 0
    while len(out) < n_seg:
        if gap_every and k and k % gap_every == 0:
            c += n  # a lost segment: counter jumps, time jumps too
        t_start = t0 + (c - start) / rate
        out.append(seg(sid, device=device, run=run, start=c, n=n, rate=rate,
                       captured=t_start, completed=t_start + n / rate + 0.05,
                       sha="%s%d" % (sha_prefix, sid)))
        sid += 1
        c += n
        k += 1
    return out


class TestEpochs(unittest.TestCase):
    def test_overlap_is_union_not_sum(self):
        # two 0.5s segments overlapping by half: union 0.75s, not 1.0s
        segs = [seg(1, start=0, n=8000), seg(2, start=4000, n=8000,
                                             captured=T0 + 0.25)]
        idx = core.build_index(segs, 1)
        dev = idx["devices"]["d1"]
        ep = dev["epochs"][0]
        self.assertEqual(ep["covered_counter"], [[0, 12000]])
        self.assertAlmostEqual(dev["coverage_seconds"], 12000 / 16000, places=3)

    def test_counter_reset_new_epoch(self):
        segs = live_chain(3, start=0)
        # device reboots: new run, counter back to 0
        segs += live_chain(2, start=0, run="r2", t0=T0 + 10, sha_prefix="b")
        for i, s in enumerate(segs):
            s["segment_id"] = i + 1
        idx = core.build_index(segs, 1)
        eps = idx["devices"]["d1"]["epochs"]
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[0]["run_id"], "r1")
        self.assertEqual(eps[1]["run_id"], "r2")
        reasons = [u["reason"] for u in idx["unconfirmable"]]
        self.assertIn("counter-reset-boundary", reasons)

    def test_reset_same_run_detected(self):
        # first life starts at counter 100000 (device had been running);
        # counter then wraps to 0 without a run change -> new epoch, because
        # the new life's range lies entirely before the old life's start.
        segs = live_chain(3, start=100000, run="r1")
        segs += live_chain(2, start=0, run="r1", t0=T0 + 10, sha_prefix="b")
        for i, s in enumerate(segs):
            s["segment_id"] = i + 1
        idx = core.build_index(segs, 1)
        eps = idx["devices"]["d1"]["epochs"]
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[1]["epoch_seq"], 1)

    def test_backfill_not_misread_as_reset(self):
        # early segments arrive late (offline replay): captured_at orders
        # them correctly, so no phantom epoch appears.
        segs = live_chain(2, start=16000, t0=T0 + 1.0)  # live, counter 16000+
        late = live_chain(2, start=0, t0=T0)            # the beginning, replayed
        for s in late:
            s["completed_at"] = T0 + 30.0   # arrived much later
            s["backfill"] = 1
        segs = segs + late
        for i, s in enumerate(segs):
            s["segment_id"] = i + 1
        idx = core.build_index(segs, 1)
        eps = idx["devices"]["d1"]["epochs"]
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["counter_range"], [0, 32000])


class TestAlignment(unittest.TestCase):
    def test_rate_locked_offset(self):
        segs = live_chain(5)
        idx = core.build_index(segs, 1)
        run = idx["devices"]["d1"]["epochs"][0]["rate_runs"][0]
        self.assertEqual(run["n_anchors"], 5)
        self.assertAlmostEqual(run["offset"], T0 + 0.05, places=2)
        self.assertLess(abs(run["drift_ppm"]), 1.0)
        self.assertEqual(run["confidence"], "high")

    def test_free_fit_detects_drift(self):
        # device clock runs 1% fast vs its nominal sample rate
        segs = []
        for k in range(6):
            c = k * 8000
            t_start = T0 + (c / 16000) * 1.01
            segs.append(seg(k + 1, start=c, captured=t_start,
                            completed=t_start + 0.5 * 1.01))
        idx = core.build_index(segs, 1)
        run = idx["devices"]["d1"]["epochs"][0]["rate_runs"][0]
        self.assertIsNotNone(run["drift_ppm"])
        self.assertGreater(run["drift_ppm"], 5000)
        self.assertEqual(run["confidence"], "low")
        reasons = [u["reason"] for u in idx["unconfirmable"]]
        self.assertIn("degraded-alignment", reasons)

    def test_backfill_not_used_as_anchor(self):
        segs = live_chain(4)
        bf = seg(99, start=32000, captured=T0 + 2.0, completed=T0 + 60.0,
                 backfill=1, sha="bf")
        segs.append(bf)
        idx = core.build_index(segs, 1)
        run = idx["devices"]["d1"]["epochs"][0]["rate_runs"][0]
        self.assertEqual(run["n_anchors"], 4)
        self.assertAlmostEqual(run["offset"], T0 + 0.05, places=2)

    def test_clock_disagreement_flagged(self):
        segs = live_chain(4)
        # backfilled segment whose device clock disagrees wildly with the fit
        bf = seg(99, start=32000, captured=T0 + 100.0, completed=T0 + 100.1,
                 backfill=1, sha="bf")
        segs.append(bf)
        idx = core.build_index(segs, 1)
        reasons = [u["reason"] for u in idx["unconfirmable"]]
        self.assertIn("clock-disagreement", reasons)

    def test_rate_switch_chained(self):
        segs = live_chain(3, rate=16000)
        # device switches to 48kHz mid-epoch; new run arrives only as
        # backfill, so it must chain from the previous run's fit
        c = 24000
        for k in range(2):
            segs.append(seg(50 + k, start=c + k * 19200, n=19200, rate=48000,
                            captured=T0 + 1.5 + k * 0.4,
                            completed=T0 + 90.0, backfill=1, sha="c%d" % k))
        idx = core.build_index(segs, 1)
        runs = idx["devices"]["d1"]["epochs"][0]["rate_runs"]
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[1]["rate"], 48000)
        self.assertEqual(runs[1]["confidence"], "chained")
        self.assertIsNotNone(runs[1]["offset"])

    def test_unanchored_run_unconfirmable(self):
        # epoch starts with backfill-only segments: nothing to anchor to
        segs = [seg(1, start=0, captured=T0, completed=T0 + 50, backfill=1),
                seg(2, start=8000, captured=T0 + 0.5, completed=T0 + 50,
                    backfill=1)]
        idx = core.build_index(segs, 1)
        run = idx["devices"]["d1"]["epochs"][0]["rate_runs"][0]
        self.assertEqual(run["confidence"], "unanchored")
        self.assertIsNone(run["offset"])
        reasons = [u["reason"] for u in idx["unconfirmable"]]
        self.assertIn("unanchored-run", reasons)


class TestCoverageAndConflicts(unittest.TestCase):
    def test_gap_detection(self):
        segs = live_chain(4, gap_every=2)  # one segment lost in the middle
        idx = core.build_index(segs, 1)
        ep = idx["devices"]["d1"]["epochs"][0]
        self.assertEqual(len(ep["counter_gaps"]), 1)
        self.assertEqual(ep["counter_gaps"][0], [16000, 24000])
        self.assertEqual(len(ep["time_gaps"]), 1)
        self.assertAlmostEqual(ep["time_gaps"][0][1] - ep["time_gaps"][0][0],
                               0.5, places=2)

    def test_duplicates_deduped(self):
        segs = live_chain(2)
        dup = dict(segs[0])
        dup["segment_id"] = 99
        dup["segment_uid"] = "u99"
        segs.append(dup)
        idx = core.build_index(segs, 1)
        dev = idx["devices"]["d1"]
        self.assertEqual(dev["duplicates_deduped"], 1)
        self.assertEqual(len(idx["conflicts"]), 0)
        self.assertAlmostEqual(dev["coverage_seconds"], 1.0, places=3)

    def test_conflict_excluded_and_reported(self):
        segs = live_chain(2)
        evil = dict(segs[0])
        evil["segment_id"] = 99
        evil["segment_uid"] = "u99"
        evil["pcm_sha256"] = "different"
        segs.append(evil)
        idx = core.build_index(segs, 1)
        self.assertEqual(len(idx["conflicts"]), 1)
        self.assertEqual(idx["conflicts"][0]["counter_range"], [0, 8000])
        # conflicting range excluded from coverage
        ep = idx["devices"]["d1"]["epochs"][0]
        self.assertEqual(ep["covered_counter"], [[8000, 16000]])
        reasons = [u["reason"] for u in idx["unconfirmable"]]
        self.assertIn("conflict", reasons)


if __name__ == "__main__":
    unittest.main()
