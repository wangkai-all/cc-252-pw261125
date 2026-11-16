#!/usr/bin/env python3
"""Start the audio-sync server (upload API + validator + indexer + device
simulators + web UI). State lives in ./data; restart this process and
reception, validation and device capture resume from the persisted state.

  python3 run_server.py [--port 8077] [--db data/audio.db] [--reset]
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audiosync.app import serve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", default="data/audio.db")
    ap.add_argument("--data", default="data")
    ap.add_argument("--reset", action="store_true", help="wipe data dir first")
    args = ap.parse_args()

    if args.reset and os.path.exists(args.data):
        shutil.rmtree(args.data)
    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
    app, httpd, url = serve(args.db, args.data, web_dir, args.host, args.port)
    print("audio-sync listening on %s  (db=%s)" % (url, args.db))
    print("open %s/ for the control page" % url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
