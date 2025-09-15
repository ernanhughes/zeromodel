#  zeromodel/pipeline/data/zmq_ingest_stream.py
from __future__ import annotations
import zmq
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

class ZmqIngestStream:
    def __init__(self, url="tcp://127.0.0.1:5557"):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.SUB); self.sock.connect(url); self.sock.setsockopt(zmq.SUBSCRIBE, b"")
    def __iter__(self):
        while True:
            pkt = msgpack.loads(self.sock.recv())
            yield {"step": pkt["step"], "frame": pkt["frame"], "tags": pkt.get("tags",{})}
