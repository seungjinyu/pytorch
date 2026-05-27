import os
import tempfile
import zmq
import torch

from .payload import Payload


class ZMQClient:
    def __init__(self, address):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(address)

    def send_payload(self, payload: Payload, y, batch_size=None, extra=None):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name

        payload.save_jin1(tmp_path)

        with open(tmp_path, "rb") as f:
            payload_bytes = f.read()

        os.remove(tmp_path)

        if batch_size is None:
            batch_size = y.size(0)

        msg = {
            "type": "PAYLOAD",
            "payload": payload_bytes,
            "model_output": payload.tensors["model.output"],
            "batch_size": batch_size,
            "y": y.detach().cpu().tolist(),
        }

        if extra is not None:
            msg.update(extra)

        self.sock.send_pyobj(msg)
        return self.sock.recv_pyobj()

    def stop(self):
        self.sock.send_pyobj({"type": "STOP"})
        return self.sock.recv_pyobj()


class ZMQServer:
    def __init__(self, bind_address):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(bind_address)

    def recv_payload(self, payload_path="/tmp/jin_payload_recv.bin"):
        msg = self.sock.recv_pyobj()

        if msg["type"] == "STOP":
            self.sock.send_pyobj({"status": "stopped"})
            return None

        payload_bytes = msg["payload"]

        with open(payload_path, "wb") as f:
            f.write(payload_bytes)

        payload = Payload()
        payload.add_tensor("model.output", msg["model_output"])

        req = {
            "payload": payload,
            "payload_path": payload_path,
            "y": torch.tensor(msg["y"], dtype=torch.long),
            "batch_size": msg["batch_size"],
            "num_bytes": len(payload_bytes),
        }

        # extra fields 보존
        for k, v in msg.items():
            if k not in {
                "type",
                "payload",
                "model_output",
                "y",
                "batch_size",
            }:
                req[k] = v

        return req

    def send_reply(self, reply):
        self.sock.send_pyobj(reply)