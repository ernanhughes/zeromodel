#  zeromodel/pipeline/data/ingest_stream.py
from __future__ import annotations
import os
import time
import json
import numpy as np

class FileIngestStream:
    def __init__(self, run_dir, poll_ms=200):
        self.run_dir = run_dir
        self.poll = poll_ms / 1000.0
        self.seen = set()
        self.closed = False

    def __iter__(self):
        meta_path = f"{self.run_dir}/meta.jsonl"
        run_json = f"{self.run_dir}/run.json"
        while True:
            if os.path.exists(run_json):
                status = json.load(open(run_json)).get("status","open")
                self.closed = (status == "closed")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    for line in f:
                        off = f.tell()  # not strictly needed
                        rec = json.loads(line)
                        if rec["path"] in self.seen: 
                            continue
                        self.seen.add(rec["path"])
                        frame = np.load(rec["path"])
                        yield {"step": rec["step"], "frame": frame, "tags": rec.get("tags",{})}
            if self.closed: break
            time.sleep(self.poll)

