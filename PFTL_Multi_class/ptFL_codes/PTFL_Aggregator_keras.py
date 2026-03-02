#!/usr/bin/env python3.10
# ============================================================
# server_pftl_shared_aggregator_strict_barrier.py
# ============================================================
# STRICT ROUND + BARRIER SERVER (matches fedavg.proto)
#
# - Aggregates SHARED layer weights only (FedAvg weighted)
# - Strict round sync:
#     accept update ONLY if client_round == server_round
#     else Ack(ok=False)
# - ONLINE accumulation (no buffering weight sets):
#     wsum += w * num_samples
# - Barrier behavior:
#     server_round increments ONLY after MIN_CLIENTS_TO_AGG unique clients send for that round
#     clients can block on GetSharedWeights until round advances
#
# - IMPORTANT for your "initial weights":
#     server starts with global_shared = None
#     GetSharedWeights returns empty list [] until first update arrives
#     clients MUST NOT set_shared([]) (handled in client code)
# ============================================================

import os, time, csv, pickle, threading
from datetime import datetime
from concurrent import futures

import numpy as np
import grpc

import myproto_pb2
import myproto_pb2_grpc

# ----------------- Settings -----------------
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "0.0.0.0:50051")
MIN_CLIENTS_TO_AGG = int(os.environ.get("MIN_CLIENTS_TO_AGG", "6"))

OUT_DIR = os.environ.get("OUT_DIR", "./server_logs_pftl_shared")
os.makedirs(OUT_DIR, exist_ok=True)

AGG_LOG = os.path.join(OUT_DIR, "server_agg_log.csv")
UPDATE_LOG = os.path.join(OUT_DIR, "server_update_log.csv")
PULL_LOG = os.path.join(OUT_DIR, "server_pull_log.csv")
LATEST_PKL = os.path.join(OUT_DIR, "global_shared_latest.pkl")  # overwritten each agg

SEED = 123
np.random.seed(SEED)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def save_latest(round_id: int, global_shared):
    obj = {"round": int(round_id), "global_shared": global_shared, "timestamp": ts()}
    with open(LATEST_PKL, "wb") as f:
        pickle.dump(obj, f)


def save_round_snapshot(out_dir: str, round_id: int, global_shared):
    path = os.path.join(out_dir, f"global_shared_round_{round_id:03d}.pkl")
    obj = {"round": int(round_id), "global_shared": global_shared, "timestamp": ts()}
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


class PFTLSharedAggregator(myproto_pb2_grpc.AggregatorServicer):
    def __init__(self):
        self.lock = threading.Lock()

        self.current_round = 0
        self.global_shared = None  # list of arrays [kernel, bias] once initialized

        # per-round online state
        self.seen_clients = set()
        self.num_clients_round = 0
        self.total_samples_round = 0
        self.wsum = None  # list of float64 arrays
        self.bytes_received_round = 0
        self.last_bytes_sent = 0

        ensure_csv(AGG_LOG, [
            "server_round_after_agg",
            "num_clients_used",
            "total_samples_used",
            "bytes_received_round",
            "bytes_sent_last_pull",
            "saved_round_pkl",
            "timestamp"
        ])
        ensure_csv(UPDATE_LOG, [
            "server_round",
            "client_id",
            "client_round",
            "num_samples",
            "bytes_received",
            "accepted",
            "timestamp"
        ])
        ensure_csv(PULL_LOG, [
            "server_round",
            "client_id",
            "bytes_sent",
            "timestamp"
        ])

        print(f"[SERVER] Address={SERVER_ADDRESS}")
        print(f"[SERVER] OUT_DIR={OUT_DIR}")
        print(f"[SERVER] MIN_CLIENTS_TO_AGG={MIN_CLIENTS_TO_AGG}")
        print("[SERVER] Strict round sync + online FedAvg")

    def _reset_round_state(self):
        self.seen_clients.clear()
        self.num_clients_round = 0
        self.total_samples_round = 0
        self.wsum = None
        self.bytes_received_round = 0

    def _accumulate(self, shared_weights, num_samples: int):
        if self.wsum is None:
            self.wsum = [np.array(w, dtype=np.float64) * float(num_samples) for w in shared_weights]
        else:
            if len(shared_weights) != len(self.wsum):
                raise ValueError("weight length mismatch (shared layer shape mismatch)")
            for i in range(len(self.wsum)):
                self.wsum[i] += np.array(shared_weights[i], dtype=np.float64) * float(num_samples)

        self.total_samples_round += int(num_samples)
        self.num_clients_round += 1

    def _finalize(self):
        total = float(max(1, self.total_samples_round))
        return [(ws / total).astype(np.float32) for ws in self.wsum]

    def SendSharedUpdate(self, request, context):
        md = dict(context.invocation_metadata())
        client_id = md.get("client_id", "unknown")

        bytes_received = len(request.weights)
        client_round = int(request.round)
        num_samples = int(request.num_samples)

        try:
            shared_weights = pickle.loads(request.weights)
        except Exception:
            with open(UPDATE_LOG, "a", newline="") as f:
                csv.writer(f).writerow([self.current_round, client_id, client_round, num_samples, bytes_received, 0, ts()])
            return myproto_pb2.Ack(ok=False)

        # must be list of arrays with at least 2 (kernel,bias)
        if not isinstance(shared_weights, list) or len(shared_weights) < 2:
            with open(UPDATE_LOG, "a", newline="") as f:
                csv.writer(f).writerow([self.current_round, client_id, client_round, num_samples, bytes_received, 0, ts()])
            return myproto_pb2.Ack(ok=False)

        with self.lock:
            # strict round sync
            if client_round != self.current_round:
                with open(UPDATE_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([self.current_round, client_id, client_round, num_samples, bytes_received, 0, ts()])
                return myproto_pb2.Ack(ok=False)

            # ignore duplicates in same round
            if client_id in self.seen_clients:
                with open(UPDATE_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([self.current_round, client_id, client_round, num_samples, bytes_received, 1, ts()])
                return myproto_pb2.Ack(ok=True)

            # init global from first-ever accepted update
            if self.global_shared is None:
                self.global_shared = shared_weights
                save_latest(self.current_round, self.global_shared)
                save_round_snapshot(OUT_DIR, self.current_round, self.global_shared)
                print(f"[SERVER] Initialized global_shared from first client at round {self.current_round}")

            # accumulate
            try:
                self.seen_clients.add(client_id)
                self._accumulate(shared_weights, num_samples)
                self.bytes_received_round += bytes_received
            except Exception:
                # if mismatch in shapes etc
                self.seen_clients.discard(client_id)
                with open(UPDATE_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([self.current_round, client_id, client_round, num_samples, bytes_received, 0, ts()])
                return myproto_pb2.Ack(ok=False)

            with open(UPDATE_LOG, "a", newline="") as f:
                csv.writer(f).writerow([self.current_round, client_id, client_round, num_samples, bytes_received, 1, ts()])

            print(f"[SERVER] update {client_id} round={self.current_round} "
                  f"count={self.num_clients_round}/{MIN_CLIENTS_TO_AGG} samples={num_samples}")

            # aggregate when enough clients
            if self.num_clients_round >= MIN_CLIENTS_TO_AGG:
                self.global_shared = self._finalize()
                self.current_round += 1

                snap = save_round_snapshot(OUT_DIR, self.current_round, self.global_shared)
                save_latest(self.current_round, self.global_shared)

                with open(AGG_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([
                        self.current_round,
                        self.num_clients_round,
                        self.total_samples_round,
                        self.bytes_received_round,
                        self.last_bytes_sent,
                        os.path.basename(snap),
                        ts()
                    ])

                print(f"[SERVER] AGG DONE -> server_round={self.current_round} "
                      f"(clients={self.num_clients_round}, total_samples={self.total_samples_round})")

                self._reset_round_state()

            return myproto_pb2.Ack(ok=True)

    def GetSharedWeights(self, request, context):
        md = dict(context.invocation_metadata())
        client_id = md.get("client_id", "unknown")

        with self.lock:
            payload = pickle.dumps([] if self.global_shared is None else self.global_shared)
            server_round = int(self.current_round)
            bytes_sent = len(payload)
            self.last_bytes_sent = bytes_sent

        with open(PULL_LOG, "a", newline="") as f:
            csv.writer(f).writerow([server_round, client_id, bytes_sent, ts()])

        return myproto_pb2.SharedWeights(weights=payload, round=server_round)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    myproto_pb2_grpc.add_AggregatorServicer_to_server(PFTLSharedAggregator(), server)
    server.add_insecure_port(SERVER_ADDRESS)
    server.start()
    print(f"[SERVER] Running on {SERVER_ADDRESS}")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("[SERVER] Shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
