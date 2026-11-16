"""Time-index domain logic.

Terminology
-----------
counter      device-side monotonically increasing *sample* counter. Each
             segment declares the counter value of its first sample.
epoch        a maximal span of one device 'counter life'. A new epoch starts
             when the run_id changes (device rebooted) or when, walking
             segments in device-capture order, a segment's whole counter
             range falls before everything seen so far (counter reset
             without a run change, e.g. 32-bit wrap / firmware bug).
rate run     contiguous span of an epoch captured at one sample rate. A
             device may switch rates mid-epoch; the counter keeps counting
             samples, so each rate run needs its own time alignment.

Alignment (device sample counter -> server receive time)
--------------------------------------------------------
Only *live* segments (uploaded right after capture) are usable as anchors:
for them the server receive-complete time approximates the end of capture
plus a bounded transfer delay. Anchors therefore pair
x = counter_end (counter one past the last sample) with y = completed_at.
Backfilled (offline-replayed) segments arrive arbitrarily late, so their
receive time says nothing about capture time; they are mapped through the
epoch fit and cross-checked against the device clock instead.

Two fits are computed per rate run and compared:
  rate-locked  slope fixed at 1/rate, offset = median(y - x/rate). Robust.
  free fit     least-squares slope+offset. Reveals clock drift vs the
               nominal sample rate (drift_ppm) but is outlier-sensitive.
The published mapping uses the rate-locked fit; the free fit is reported
alongside so an operator can see the discrepancy. Residual statistics give
the error bound applied to every mapped interval.
"""
import statistics

# tolerances
RESET_OVERLAP_TOL = 0          # samples; strict "entirely before" reset test
BACKFILL_DELAY_S = 5.0         # uploaded this long after capture => backfill
MIN_ERR_S = 0.25               # floor for mapping error (network jitter etc.)
DEGRADED_RESID_S = 0.75        # residual max above this => degraded alignment
DEGRADED_DRIFT_PPM = 5000.0    # |drift| above this => degraded alignment
CLOCK_DISAGREE_S = 1.0         # device-clock vs fit disagreement => unconfirmable
CHAIN_TOL_S = 1.0              # continuity tolerance when chaining rate runs


def seg_counter_end(s):
    return s["counter_start"] + s["nsamples"]


def seg_mid_counter(s):
    return s["counter_start"] + s["nsamples"] / 2.0


def classify_backfill(seg):
    """A segment is backfill if it arrived well after its device capture time
    would suggest. We cannot compare clocks directly (unknown offset), so the
    device tells us capture and upload times on its own clock; the server
    also records arrival. Here the simulator supplies captured_at on the
    device clock and we treat (completed_at - captured_at) after offset
    normalisation per-device; simplest robust rule: device flags nothing, we
    use the server's first observation: if another segment of the same device
    with a *higher* counter completed earlier, this one arrived out of order
    => backfill. That and the explicit delay check are combined by the caller.
    """
    return bool(seg.get("backfill"))


def detect_epochs(segments):
    """Group a device's validated segments into epochs.

    Deterministic: sort by (captured_at, counter_start) — device capture
    order, so late-arriving backfill of early counters does not masquerade
    as a reset. A new epoch opens when run_id changes or the segment lies
    entirely before the current epoch's minimum counter (counter reset).
    Returns list of epochs, each a dict with run_id, epoch_seq, segments.
    """
    segs = sorted(segments, key=lambda s: (s["captured_at"], s["counter_start"]))
    epochs = []
    cur = None
    for s in segs:
        end = seg_counter_end(s)
        if cur is not None:
            reset = (s["run_id"] != cur["run_id"]) or \
                    (end <= cur["min_start"] - RESET_OVERLAP_TOL)
            if reset:
                epochs.append(cur)
                cur = None
        if cur is None:
            seq = 0 if not epochs or s["run_id"] != epochs[-1]["run_id"] \
                else epochs[-1]["epoch_seq"] + 1
            cur = _new_epoch(s, seq)
        cur["segments"].append(s)
        cur["min_start"] = min(cur["min_start"], s["counter_start"])
        cur["max_end"] = max(cur["max_end"], end)
    if cur is not None:
        epochs.append(cur)
    return epochs


def _new_epoch(seg, seq):
    return {"run_id": seg["run_id"], "epoch_seq": seq, "segments": [],
            "min_start": seg["counter_start"], "max_end": seg_counter_end(seg)}


def split_rate_runs(epoch_segments):
    """Contiguous (in counter order) runs of equal sample rate."""
    segs = sorted(epoch_segments, key=lambda s: s["counter_start"])
    runs = []
    for s in segs:
        if runs and runs[-1]["rate"] == s["sample_rate"]:
            runs[-1]["segments"].append(s)
        else:
            runs.append({"rate": s["sample_rate"], "segments": [s]})
    return runs


def _least_squares(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None, None
    slope = (n * sxy - sx * sy) / denom
    offset = (sy - slope * sx) / n
    return slope, offset


def align_rate_run(run, prev_run_fit):
    """Fit counter->server-time for one rate run.

    Returns a fit dict:
      offset, rate, n_anchors, resid_rms, resid_max, err,
      drift_ppm (free fit, None if <2 anchors),
      method_compare {rate_locked_offset, free:{slope,offset}},
      confidence: high | low | chained | unanchored
    or None if no mapping is possible at all.
    """
    rate = run["rate"]
    anchors = [s for s in run["segments"] if not classify_backfill(s)]
    xs = [seg_counter_end(s) for s in anchors]  # completed_at ~ end of capture
    ys = [s["completed_at"] for s in anchors]

    fit = {"rate": rate, "n_anchors": len(anchors),
           "counter_range": [min(s["counter_start"] for s in run["segments"]),
                             max(seg_counter_end(s) for s in run["segments"])],
           "drift_ppm": None, "method_compare": None,
           "resid_rms": None, "resid_max": None}

    if anchors:
        # one robust pass: median offset, drop anchors whose residual is an
        # outlier vs the MAD (late "live" uploads, queue drains), refit on
        # the inliers. Outliers are reported, not silently used.
        def fit_on(pts):
            off = statistics.median(y - x / rate for x, y in pts)
            res = [y - (x / rate + off) for x, y in pts]
            return off, res

        pts = list(zip(xs, ys))
        offset, resids = fit_on(pts)
        med = statistics.median(resids)
        mad = statistics.median(abs(r - med) for r in resids)
        thresh = max(3 * 1.4826 * mad, MIN_ERR_S)
        inliers = [(x, y) for (x, y), r in zip(pts, resids) if abs(r) <= thresh]
        n_outliers = len(pts) - len(inliers)
        if inliers and n_outliers:
            offset, resids = fit_on(inliers)
        fit["n_outliers"] = n_outliers
        xs_fit = [x for x, _ in (inliers or pts)]
        ys_fit = [y for _, y in (inliers or pts)]
        rms = statistics.pstdev(resids) if len(resids) > 1 else 0.0
        rmax = max(abs(r) for r in resids)
        fit.update(offset=offset, resid_rms=rms, resid_max=rmax,
                   err=max(rmax, MIN_ERR_S))
        if len(xs_fit) >= 3:
            # drift needs >=3 points: with two, the free fit is exact and
            # the "drift" is meaningless
            slope, foff = _least_squares(xs_fit, ys_fit)
            if slope:
                fit["drift_ppm"] = (slope * rate - 1.0) * 1e6
                fit["method_compare"] = {
                    "rate_locked_offset": offset,
                    "free": {"slope": slope, "offset": foff},
                }
                # standard error of the slope: drift must be both
                # statistically significant and practically large, else
                # short noisy windows would constantly false-alarm
                n = len(xs_fit)
                xbar = sum(xs_fit) / n
                sxx = sum((x - xbar) ** 2 for x in xs_fit)
                if n > 2 and sxx > 0:
                    rss = sum((y - (slope * x + foff)) ** 2
                              for x, y in zip(xs_fit, ys_fit))
                    se_slope = (rss / (n - 2) / sxx) ** 0.5
                    fit["drift_se_ppm"] = se_slope * rate * 1e6
        degraded = (rmax > DEGRADED_RESID_S or
                    (fit["drift_ppm"] is not None and
                     abs(fit["drift_ppm"]) > DEGRADED_DRIFT_PPM and
                     abs(fit["drift_ppm"]) > 3 * fit.get("drift_se_ppm",
                                                         float("inf"))))
        # few anchors: offset is pinned but residuals/drift say nothing
        fit["confidence"] = "low" if (degraded or len(inliers or pts) < 3) \
            else "high"
        return fit

    # no live anchors: try chaining from the previous rate run's fit so the
    # mapping stays continuous across a rate switch (counter is monotonic).
    if prev_run_fit and prev_run_fit.get("offset") is not None:
        boundary = fit["counter_range"][0]
        prev_rate = prev_run_fit["rate"]
        t_boundary = boundary / prev_rate + prev_run_fit["offset"]
        offset = t_boundary - boundary / rate
        fit.update(offset=offset, err=max(prev_run_fit.get("err", MIN_ERR_S),
                                          MIN_ERR_S) + CHAIN_TOL_S,
                   confidence="chained")
        return fit

    fit.update(offset=None, err=None, confidence="unanchored")
    return fit


def map_counter(fit, counter):
    return counter / fit["rate"] + fit["offset"]


def union_intervals(intervals):
    """Merge overlapping/adjacent [a, b] intervals."""
    if not intervals:
        return []
    iv = sorted(intervals)
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def gaps_within(covered, lo, hi):
    """Complement of covered inside [lo, hi]."""
    out = []
    cur = lo
    for a, b in covered:
        if a > cur:
            out.append([cur, a])
        cur = max(cur, b)
    if cur < hi:
        out.append([cur, hi])
    return [g for g in out if g[1] - g[0] > 1e-9]


def find_conflicts(segments):
    """Duplicates and conflicts among a device's segments.

    Segments sharing the exact same counter range [counter_start,
    counter_start+nsamples) are one group:
      - one distinct content hash in the group  -> redundant duplicates;
        dedupe to a single representative (counted in `duplicates`).
      - several distinct hashes                 -> conflict: the range is
        unresolvable, ALL segments of the group are excluded from coverage
        and anchors, and the group is reported.

    Partial overlaps between different ranges are normal (devices send
    overlapping windows); coverage is the union, and content agreement
    inside the overlap cannot be verified without decoding, so they are
    left alone.
    """
    groups = {}
    for s in segments:
        key = (s["counter_start"], s["nsamples"])
        groups.setdefault(key, []).append(s)

    conflicts, clean, duplicates = [], [], 0
    for (start, n), members in sorted(groups.items()):
        hashes = {m["pcm_sha256"] for m in members}
        if len(hashes) == 1:
            duplicates += len(members) - 1
            clean.append(members[0])
        else:
            by_hash = {}
            for m in members:
                by_hash.setdefault(m["pcm_sha256"], 0)
                by_hash[m["pcm_sha256"]] += 1
            duplicates += sum(c - 1 for c in by_hash.values())
            conflicts.append({
                "counter_range": [start, start + n],
                "segment_ids": [m["segment_id"] for m in members],
            })
    return conflicts, clean, duplicates


def build_device_index(device_id, segments):
    """Build the full index for one device from its validated segments."""
    epochs = detect_epochs(segments)
    out_epochs = []
    all_unconfirmable = []
    all_conflicts = []
    total_dup = 0

    for ep in epochs:
        conflicts, clean, ndup = find_conflicts(ep["segments"])
        total_dup += ndup
        for c in conflicts:
            c.update(device=device_id, run_id=ep["run_id"], epoch_seq=ep["epoch_seq"])
            all_unconfirmable.append({
                "device": device_id, "reason": "conflict",
                "detail": "overlapping segments %s carry different content; "
                          "range excluded from coverage until resolved"
                          % c["segment_ids"],
                "counter_range": c["counter_range"], "time_range": None,
            })
        all_conflicts.extend(conflicts)

        runs = split_rate_runs(clean)
        fits = []
        prev_fit = None
        for run in runs:
            fit = align_rate_run(run, prev_fit)
            fits.append((run, fit))
            if fit.get("offset") is not None:
                prev_fit = fit

        # coverage in counter domain (clean validated segments only)
        c_intervals = [[s["counter_start"], seg_counter_end(s)] for s in clean]
        covered_counter = union_intervals(c_intervals)
        c_lo, c_hi = ep["min_start"], ep["max_end"]
        counter_gaps = gaps_within(covered_counter, c_lo, c_hi)

        # map to server time where a fit exists
        covered_time, seg_mappings = [], []
        for run, fit in fits:
            if fit.get("offset") is None:
                all_unconfirmable.append({
                    "device": device_id, "reason": "unanchored-run",
                    "detail": "rate run has no live anchors and cannot be "
                              "chained from a previous run; counter range "
                              "known, server-time mapping unknown",
                    "counter_range": fit["counter_range"], "time_range": None,
                })
                continue
            for s in run["segments"]:
                t0 = map_counter(fit, s["counter_start"])
                t1 = map_counter(fit, seg_counter_end(s))
                covered_time.append([t0, t1])
                seg_mappings.append({
                    "segment_id": s["segment_id"], "counter_range":
                    [s["counter_start"], seg_counter_end(s)],
                    "time_range": [t0, t1], "err": fit["err"],
                    "anchor": not classify_backfill(s),
                    "rate": run["rate"],
                })
            if fit["confidence"] in ("low", "chained"):
                all_unconfirmable.append({
                    "device": device_id,
                    "reason": "degraded-alignment" if fit["confidence"] == "low"
                              else "chained-alignment",
                    "detail": "fit residuals/drift exceed tolerance; mapped "
                              "times carry widened error bounds" if
                              fit["confidence"] == "low" else
                              "no live anchors in this rate run; offset "
                              "chained from previous run assuming counter "
                              "continuity across the rate switch",
                    "counter_range": fit["counter_range"],
                    "time_range": [map_counter(fit, fit["counter_range"][0]),
                                   map_counter(fit, fit["counter_range"][1])],
                })

        # backfill cross-check against device clock
        for run, fit in fits:
            if fit.get("offset") is None:
                continue
            for s in run["segments"]:
                if not classify_backfill(s):
                    continue
                # device stamps captured_at at capture start; offset is
                # anchor-independent: t(counter) - counter/rate
                implied = s["captured_at"] - s["counter_start"] / run["rate"]
                if abs(implied - fit["offset"]) > CLOCK_DISAGREE_S:
                    all_unconfirmable.append({
                        "device": device_id, "reason": "clock-disagreement",
                        "detail": "backfilled segment's device clock implies "
                                  "an offset %.2fs away from the epoch fit" %
                                  (implied - fit["offset"]),
                        "counter_range": [s["counter_start"], seg_counter_end(s)],
                        "time_range": [map_counter(fit, s["counter_start"]),
                                       map_counter(fit, seg_counter_end(s))],
                        "segment_id": s["segment_id"],
                    })

        covered_time_u = union_intervals(covered_time)
        t_lo = covered_time_u[0][0] if covered_time_u else None
        t_hi = covered_time_u[-1][1] if covered_time_u else None
        time_gaps = gaps_within(covered_time_u, t_lo, t_hi) if t_lo is not None else []

        out_epochs.append({
            "run_id": ep["run_id"], "epoch_seq": ep["epoch_seq"],
            "counter_range": [c_lo, c_hi],
            "rate_runs": [dict(fit, n_segments=len(run["segments"])) for run, fit in fits],
            "covered_counter": covered_counter,
            "counter_gaps": counter_gaps,
            "covered_time": covered_time_u,
            "time_gaps": time_gaps,
            "segment_mappings": seg_mappings,
        })

    # epoch boundaries: continuity across a reset is unconfirmable by sample
    # count; only receive times bound it.
    for a, b in zip(out_epochs, out_epochs[1:]):
        all_unconfirmable.append({
            "device": device_id, "reason": "counter-reset-boundary",
            "detail": "device sample counter reset between epochs "
                      "(run %s#%d -> %s#%d); sample-count chaining across "
                      "the boundary is impossible, only receive times apply"
                      % (a["run_id"], a["epoch_seq"], b["run_id"], b["epoch_seq"]),
            "counter_range": None,
            "time_range": ([a["covered_time"][-1][1], b["covered_time"][0][0]]
                           if a["covered_time"] and b["covered_time"] else None),
        })

    cov_s = sum(b - a for ep in out_epochs for a, b in ep["covered_time"])
    gap_s = sum(b - a for ep in out_epochs for a, b in ep["time_gaps"])
    return {
        "device": device_id,
        "epochs": out_epochs,
        "coverage_seconds": cov_s,
        "gap_seconds": gap_s,
        "duplicates_deduped": total_dup,
        "conflicts": all_conflicts,
        "unconfirmable": all_unconfirmable,
    }


def build_index(segments, validation_version):
    """Top-level: build the publishable index report from validated segments."""
    by_device = {}
    for s in segments:
        by_device.setdefault(s["device_id"], []).append(s)
    devices = {d: build_device_index(d, segs) for d, segs in sorted(by_device.items())}
    provenance = []
    for d in devices.values():
        for ep in d["epochs"]:
            for m in ep["segment_mappings"]:
                provenance.append(dict(device=d["device"], **m))
    return {
        "validation_version": validation_version,
        "devices": devices,
        "conflicts": [c for d in devices.values() for c in d["conflicts"]],
        "unconfirmable": [u for d in devices.values() for u in d["unconfirmable"]],
        "provenance": provenance,
        "stats": {
            "devices": len(devices),
            "segments_used": len(provenance),
            "coverage_seconds": sum(d["coverage_seconds"] for d in devices.values()),
            "gap_seconds": sum(d["gap_seconds"] for d in devices.values()),
            "conflicts": sum(len(d["conflicts"]) for d in devices.values()),
            "unconfirmable": sum(len(d["unconfirmable"]) for d in devices.values()),
        },
    }
