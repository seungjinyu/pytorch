import os
import tempfile
import time
import zmq
import torch

from .payload import Payload


# Client Class for node A 
class ZMQClient:
    def __init__(self, address):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(address)

    def send_payload(self, payload: Payload, y, batch_size=None, extra=None):

        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name
        t_save0 = time.perf_counter()
        payload.save_jin1(tmp_path)
        t_save1 = time.perf_counter()

        t_read0 = time.perf_counter()
        with open(tmp_path, "rb") as f:
            payload_bytes = f.read()
        t_read1 = time.perf_counter()

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
        t_send0 = time.perf_counter()
        self.sock.send_pyobj(msg)
        t_send1 = time.perf_counter()

        t_recv0 = time.perf_counter()
        reply = self.sock.recv_pyobj()
        t_recv1 = time.perf_counter()

        t1 = time.perf_counter()
        print(
            f"[ZMQClient][PROFILE] "
            f"save_jin1_ms={(t_save1 - t_save0) * 1000:.3f} "
            f"read_payload_bytes_ms={(t_read1 - t_read0) * 1000:.3f} "
            f"send_pyobj_ms={(t_send1 - t_send0) * 1000:.3f} "
            f"recv_pyobj_ms={(t_recv1 - t_recv0) * 1000:.3f} "
            f"total_send_payload_ms={(t1 - t0) * 1000:.3f}",
            flush=True,
        )
        return reply

    def stop(self):
        self.sock.send_pyobj({"type": "STOP"})
        return self.sock.recv_pyobj()
    
    def request_template_plan(self):
        self.sock.send_pyobj({
            "kind": "get_template_plan",
        })

        reply = self.sock.recv_pyobj()

        if reply.get("status") != "ok":
            raise RuntimeError(f"[ZMQClient] template plan request failed: {reply}")

        if reply.get("kind") != "template_plan":
            raise RuntimeError(f"[ZMQClient] unexpected reply: {reply}")

        return reply["template_plan"]

# Server Class for node B 
class ZMQServer:
    def __init__(self, bind_address):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(bind_address)

    def recv_payload(self, payload_path="/tmp/jin_payload_recv.bin"):

        t_total0 = time.perf_counter()

        t_recv0 = time.perf_counter()
        msg = self.sock.recv_pyobj()
        t_recv1 = time.perf_counter()

        if isinstance(msg, dict) and msg.get("kind") == "get_template_plan":
            return msg

        if msg["type"] == "STOP":
            self.sock.send_pyobj({"status": "stopped"})
            return None

        t_write0 = time.perf_counter()
        payload_bytes = msg["payload"]

        # Saves the payload from the node A to use it when the server overwrites the value.
        with open(payload_path, "wb") as f:
            f.write(payload_bytes)
        t_write1 = time.perf_counter()

        t_req0 = time.perf_counter()
        # Python 쪽 payload는 model.output만 있으면 됨
        payload = Payload()
        payload.add_tensor("model.output", msg["model_output"])

        print("[RECV_PAYLOAD_BYTES]", len(payload_bytes))
        print("[RECV_KEYS]", list(payload.tensors.keys()))

        req = {
            "payload": payload,
            "payload_path": payload_path,
            "y": torch.tensor(msg["y"], dtype=torch.long),
            "batch_size": msg["batch_size"],
            "num_bytes": len(payload_bytes),
        }
        t_req1 = time.perf_counter()

        for k, v in msg.items():
            if k not in {
                "type",
                "payload",
                "model_output",
                "y",
                "batch_size",
            }:
                req[k] = v
        t_total1 = time.perf_counter()

        print(
            f"[ZMQServer][RECV_PROFILE] "
            f"recv_pyobj_ms={(t_recv1 - t_recv0) * 1000:.3f} "
            f"write_payload_file_ms={(t_write1 - t_write0) * 1000:.3f} "
            f"build_req_ms={(t_req1 - t_req0) * 1000:.3f} "
            f"total_recv_payload_ms={(t_total1 - t_total0) * 1000:.3f}",
            flush=True,
        )
        return req

    def send_reply(self, reply):
        t0 = time.perf_counter()

        num_state_tensors = 0
        state_bytes = 0
        if isinstance(reply, dict) and "updated_state_dict" in reply:
            state = reply["updated_state_dict"]
            num_state_tensors = len(state)
            state_bytes = sum(
                v.numel() * v.element_size()
                for v in state.values()
                if hasattr(v, "numel")
            )

        num_grad_tensors = 0
        grad_bytes = 0
        if isinstance(reply, dict) and "grads" in reply:
            grads = reply["grads"]
            num_grad_tensors = len(grads)
            grad_bytes = sum(
                v.numel() * v.element_size()
                for v in grads.values()
                if hasattr(v, "numel")
            )

        t_send0 = time.perf_counter()
        self.sock.send_pyobj(reply)
        t_send1 = time.perf_counter()

        print(
            f"[ZMQServer][SEND_REPLY_PROFILE] "
            f"send_pyobj_ms={(t_send1 - t_send0) * 1000:.3f} "
            f"state_mb={state_bytes / 1024 / 1024:.3f} "
            f"state_tensors={num_state_tensors} "
            f"grads_mb={grad_bytes / 1024 / 1024:.3f} "
            f"grad_tensors={num_grad_tensors} "
            f"total_send_reply_ms={(t_send1 - t0) * 1000:.3f}",
            flush=True,
        )