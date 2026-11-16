"""Index builder / publisher worker.

Build protocol:
  1. read validation_version V
  2. snapshot validated segments
  3. compute report (pure function of the snapshot)
  4. store as a new index_versions row
  5. compare-and-swap publish: only if validation_version is still V

If validation changed mid-build (retraction, freshly validated backfill,
...) the CAS fails and the build is discarded — a stale index is never
published. The worker loops until it publishes against a quiet version, so
rebuilds racing with backfill converge on a self-consistent index.
"""
import threading
import time

from . import core


class Indexer(threading.Thread):
    def __init__(self, storage, debounce_s=0.3):
        super().__init__(daemon=True, name="indexer")
        self.storage = storage
        self.debounce_s = debounce_s
        self._dirty = threading.Event()
        self._stop_flag = threading.Event()
        self.last_build_at = None
        self.last_error = None
        self.builds = 0
        self.discarded_builds = 0

    def request_rebuild(self):
        self._dirty.set()

    def stop(self):
        self._stop_flag.set()
        self._dirty.set()

    def run(self):
        self._dirty.set()  # initial build from recovered state
        while not self._stop_flag.is_set():
            if not self._dirty.wait(timeout=0.5):
                continue
            self._dirty.clear()
            time.sleep(self.debounce_s)  # coalesce bursts of validations
            self.build_until_consistent()

    def build_until_consistent(self, max_tries=25):
        for _ in range(max_tries):
            v = self.storage.validation_version()
            published = self.storage.published_index_id()
            if published is not None:
                cur = self.storage.get_index(published)
                if cur and cur["validation_version"] == v:
                    return cur  # already consistent, nothing to do
            snapshot = self.storage.validated_segments()
            if self.storage.validation_version() != v:
                continue  # moved while reading; restart snapshot
            report = core.build_index(snapshot, v)
            index_id = self.storage.insert_index(v, report)
            self.builds += 1
            if self.storage.publish_index_cas(index_id, v):
                self.last_build_at = time.time()
                self.last_error = None
                return self.storage.get_index(index_id)
            self.discarded_builds += 1  # lost the race; rebuild
        self.last_error = "no self-consistent publish after %d tries" % max_tries
        return None
