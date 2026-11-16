"""Segment validation.

Runs as its own worker thread, strictly separate from indexing. A segment
moves pending -> validated | rejected here, and validated -> retracted via
the API. Every state transition goes through Storage.set_segment_state,
which bumps validation_version in the same transaction, so any in-flight
index build immediately becomes unpublishable.
"""
import json
import os
import queue
import threading

from . import wavutil

ALLOWED_RATES = {8000, 16000, 22050, 44100, 48000}
MAX_SEGMENT_SECONDS = 10.0


def validate_segment_record(seg, wav_bytes):
    """Pure checks on one assembled segment. Returns (ok, reason)."""
    meta_ok, reason = _check_metadata(seg)
    if not meta_ok:
        return False, reason
    try:
        parsed = wavutil.parse_wav(wav_bytes)
    except ValueError as e:
        return False, "wav-parse: %s" % e
    if parsed["rate"] != seg["sample_rate"]:
        return False, "sample-rate-mismatch: meta=%d wav=%d" % (
            seg["sample_rate"], parsed["rate"])
    if parsed["frames"] != seg["nsamples"]:
        return False, "nsamples-mismatch: meta=%d wav=%d" % (
            seg["nsamples"], parsed["frames"])
    if wavutil.pcm_sha256(parsed["pcm"]) != seg["pcm_sha256"]:
        return False, "checksum-mismatch"
    return True, ""


def _check_metadata(seg):
    if seg["sample_rate"] not in ALLOWED_RATES:
        return False, "sample-rate-not-allowed: %d" % seg["sample_rate"]
    if seg["counter_start"] < 0:
        return False, "negative-counter"
    if seg["nsamples"] <= 0:
        return False, "empty-segment"
    if seg["nsamples"] / seg["sample_rate"] > MAX_SEGMENT_SECONDS:
        return False, "segment-too-long"
    if not seg["run_id"]:
        return False, "missing-run-id"
    return True, ""


class Validator(threading.Thread):
    """Pulls pending segments, validates, records verdicts."""

    def __init__(self, storage, data_dir, on_change=None):
        super().__init__(daemon=True, name="validator")
        self.storage = storage
        self.data_dir = data_dir
        self.queue = queue.Queue()
        self.on_change = on_change or (lambda: None)
        self._stop_flag = threading.Event()

    def submit(self, segment_id):
        self.queue.put(segment_id)

    def sweep_pending(self):
        """Recovery: anything left 'pending' across a restart is re-queued."""
        for seg in self.storage.list_segments(state="pending"):
            self.queue.put(seg["segment_id"])

    def stop(self):
        self._stop_flag.set()
        self.queue.put(None)

    def run(self):
        self.sweep_pending()
        while not self._stop_flag.is_set():
            try:
                segment_id = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if segment_id is None:
                continue
            self.validate_one(segment_id)

    def validate_one(self, segment_id):
        seg = self.storage.get_segment(segment_id)
        if not seg or seg["state"] != "pending":
            return
        wav_path = seg["wav_path"]
        try:
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
        except OSError as e:
            self.storage.set_segment_state(segment_id, "rejected", "io: %s" % e)
            self.on_change()
            return
        ok, reason = validate_segment_record(seg, wav_bytes)
        self.storage.set_segment_state(segment_id,
                                       "validated" if ok else "rejected", reason)
        self.on_change()

    def retract(self, segment_id, reason):
        seg = self.storage.get_segment(segment_id)
        if not seg or seg["state"] != "validated":
            return None
        v = self.storage.set_segment_state(segment_id, "retracted",
                                           reason or "manual retract")
        self.on_change()
        return v

    def revalidate(self, segment_id):
        seg = self.storage.get_segment(segment_id)
        if not seg or seg["state"] != "retracted":
            return None
        v = self.storage.set_segment_state(segment_id, "pending", "revalidate requested")
        self.on_change()
        self.submit(segment_id)
        return v
