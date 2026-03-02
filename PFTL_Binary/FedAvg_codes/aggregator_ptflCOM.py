#!/usr/bin/env python3.10
# ============================================================
# aggregator_fedavg_full.py
#

# Expected pickled payload:
#   {"input_adapter":[...], "feat1":[...], "shared_dense":[...], "clf":[...]}
# ============================================================

import os, time, csv, pickle, threading
from concurrent import futures
import numpy as np
import grpc
import myproto_pb2
import myproto_pb2_grpc

SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

# FedAvg over ALL trainable layers
SHARED_LAYER_NAMES = ["input_adapter", "feat1", "shared_dense", "clf"]

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

class AggregatorService(myproto_pb2_grpc.AggregatorServicer):
    def __init__(self, num_clients=6):
        self.num_clients = int(num_clients)
        self.current_round = 0
        self.client_updates = {}  # client_id -> (shared_dict, num_samples)
        self.lock = threading.Lock()
        self.round_start_time = None

        self.comm_log_path = "aggregator_comm_log.csv"
        if not os.path.exists(self.comm_log_path):
            with open(self.comm_log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "client_id", "bytes_received", "round_duration_sec", "timestamp"]
                )

        self.global_shared = None      # dict: {layer_name: [arrays...]}
        self.expected_shapes = None    # dict: {layer_name: [shape0, shape1, ...]}

        print("[Aggregator] FedAvg FULL init. Shared layers:", SHARED_LAYER_NAMES)

    # -------------------
    # Validation helpers
    # -------------------
    def _validate_payload(self, shared_dict):
        if not isinstance(shared_dict, dict):
            return False, "Payload is not a dict"

        # Must contain all required layers
        for lname in SHARED_LAYER_NAMES:
            if lname not in shared_dict:
                return False, f"Missing layer '{lname}'"

        # validate arrays + NaN
        for lname in SHARED_LAYER_NAMES:
            wlist = shared_dict[lname]
            if not isinstance(wlist, (list, tuple)) or len(wlist) == 0:
                return False, f"Bad weights list for '{lname}'"

            for i, arr in enumerate(wlist):
                if not isinstance(arr, np.ndarray):
                    try:
                        arr = np.asarray(arr)
                        wlist[i] = arr
                    except Exception:
                        return False, f"Weight '{lname}[{i}]' is not ndarray/array-like"
                if np.isnan(arr).any():
                    return False, f"NaN in '{lname}[{i}]'"

        # learn shapes first time
        if self.expected_shapes is None:
            self.expected_shapes = {}
            for lname in SHARED_LAYER_NAMES:
                self.expected_shapes[lname] = [tuple(np.asarray(x).shape) for x in shared_dict[lname]]
            print("[Aggregator] Learned expected shapes from first update:")
            for lname in SHARED_LAYER_NAMES:
                print(f"  - {lname}: {self.expected_shapes[lname]}")
            return True, "OK"

        # shape check
        for lname in SHARED_LAYER_NAMES:
            exp_shapes = self.expected_shapes.get(lname, None)
            got_list = shared_dict[lname]
            if exp_shapes is None:
                return False, f"Expected shapes missing for '{lname}'"
            if len(got_list) != len(exp_shapes):
                return False, (
                    f"Bad weights length for '{lname}': got {len(got_list)} expected {len(exp_shapes)}"
                )
            for i, arr in enumerate(got_list):
                if tuple(np.asarray(arr).shape) != tuple(exp_shapes[i]):
                    return False, (
                        f"Shape mismatch '{lname}'[{i}]: got {np.asarray(arr).shape} expected {exp_shapes[i]}"
                    )

        return True, "OK"

    def _weighted_average(self, updates):
        # updates: list of (shared_dict, num_samples)
        total = sum(int(n) for _, n in updates)
        if total <= 0:
            print("[Aggregator] WARNING: total_samples<=0; keeping previous global.")
            return self.global_shared

        out = {}
        for lname in SHARED_LAYER_NAMES:
            wref = updates[0][0][lname]
            sums = [np.zeros_like(np.asarray(x), dtype=np.float64) for x in wref]

            for (sdict, n) in updates:
                n = int(n)
                wlist = sdict[lname]
                for i, arr in enumerate(wlist):
                    sums[i] += n * np.nan_to_num(np.asarray(arr), nan=0.0)

            out[lname] = [s / float(total) for s in sums]

        return out

    # ---------------- RPCs ----------------
    def SendSharedUpdate(self, request, context):
        md = dict(context.invocation_metadata())
        client_id = md.get("client_id", "")
        if not client_id:
            return myproto_pb2.Ack(status="ERROR: Missing client_id", current_round=self.current_round)

        # strict round guard
        if int(request.round) != int(self.current_round):
            return myproto_pb2.Ack(
                status=f"ERROR: Round mismatch server={self.current_round} client={request.round}",
                current_round=self.current_round,
            )

        try:
            shared_dict = pickle.loads(request.weights)
        except Exception as e:
            print(f"[Aggregator] Unpickle error from {client_id}: {e}")
            return myproto_pb2.Ack(status="ERROR: Bad weights (unpickle)", current_round=self.current_round)

        ok, msg = self._validate_payload(shared_dict)
        if not ok:
            print(f"[Aggregator] Reject {client_id}: {msg}")
            return myproto_pb2.Ack(status=f"ERROR: {msg}", current_round=self.current_round)

        num_samples = int(request.num_samples)

        with self.lock:
            if len(self.client_updates) == 0:
                self.round_start_time = time.time()

            self.client_updates[client_id] = (shared_dict, num_samples)

            bytes_received = len(request.weights)
            with open(self.comm_log_path, "a", newline="") as f:
                csv.writer(f).writerow([self.current_round, client_id, bytes_received, "", now_ts()])

            print(f"[Aggregator] Round {self.current_round}: got {client_id} ({len(self.client_updates)}/{self.num_clients})")

            if len(self.client_updates) < self.num_clients:
                return myproto_pb2.Ack(status="WAITING", current_round=self.current_round)

            print("[Aggregator] Aggregating FULL model layers:", SHARED_LAYER_NAMES)
            updates = list(self.client_updates.values())
            self.global_shared = self._weighted_average(updates)

            finished_round = self.current_round
            self.current_round += 1
            self.client_updates.clear()

            with open("aggregated_shared.pkl", "wb") as f:
                pickle.dump(self.global_shared, f, protocol=pickle.HIGHEST_PROTOCOL)

            dur = time.time() - self.round_start_time

            # backfill duration in comm log for this round
            try:
                with open(self.comm_log_path, "r") as f:
                    rows = list(csv.reader(f))
                with open(self.comm_log_path, "w", newline="") as f:
                    w = csv.writer(f)
                    for row in rows:
                        if row and row[0] == str(finished_round) and row[3] == "":
                            row[3] = f"{dur:.3f}"
                        w.writerow(row)
            except Exception as e:
                print(f"[Aggregator] WARNING: could not backfill duration: {e}")

            print(f"[Aggregator] Round {finished_round} complete in {dur:.2f}s -> current_round={self.current_round}")
            return myproto_pb2.Ack(status="OK", current_round=self.current_round)

    def GetSharedWeights(self, request, context):
        if self.global_shared is None:
            return myproto_pb2.SharedResponse(weights=b"", round=self.current_round)

        return myproto_pb2.SharedResponse(
            weights=pickle.dumps(self.global_shared, protocol=pickle.HIGHEST_PROTOCOL),
            round=self.current_round,
        )

if __name__ == "__main__":
    NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "6"))
    PORT = os.environ.get("PORT", "50051")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=20),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )

    myproto_pb2_grpc.add_AggregatorServicer_to_server(
        AggregatorService(num_clients=NUM_CLIENTS), server
    )
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"[Aggregator] Listening on :{PORT} | FedAvg FULL (all layers)")
    server.wait_for_termination()