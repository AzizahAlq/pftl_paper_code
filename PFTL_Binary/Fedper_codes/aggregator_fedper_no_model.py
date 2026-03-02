#!/usr/bin/env python3.10
# ============================================================
# aggregator_fedper_no_model.py
#
# FedPer: share backbone weights; keep head personalized
# - PRIVATE on clients: clf
# - SHARED global: input_adapter, feat1, shared_dense
#
# NO TensorFlow model is created here.
# Shapes are learned from the first accepted update, then enforced.
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

# -------------------
# Share these layers (FedPer backbone)
# -------------------
SHARED_LAYER_NAMES = ["input_adapter", "feat1", "shared_dense"]


def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class AggregatorService(myproto_pb2_grpc.AggregatorServicer):
    def __init__(self, num_clients=6):
        self.num_clients = int(num_clients)
        self.current_round = 0
        self.client_updates = {}
        self.lock = threading.Lock()
        self.round_start_time = None

        self.comm_log_path = "aggregator_comm_log.csv"
        if not os.path.exists(self.comm_log_path):
            with open(self.comm_log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "client_id", "bytes_received", "round_duration_sec", "timestamp"]
                )

        self.global_shared = None
        self.expected_shapes = None

        print("[Aggregator] FedPer NO-MODEL init. Shared layers:", SHARED_LAYER_NAMES)

    # -------------------
    # Validation
    # -------------------
    def _validate_payload(self, shared_dict):

        if not isinstance(shared_dict, dict):
            return False, "Payload is not a dict"

        for lname in SHARED_LAYER_NAMES:
            if lname not in shared_dict:
                return False, f"Missing layer '{lname}'"

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
                        return False, f"Weight '{lname}[{i}]' not ndarray"
                if np.isnan(arr).any():
                    return False, f"NaN in '{lname}[{i}]'"

        # Learn shapes first time
        if self.expected_shapes is None:
            exp = {}
            for lname in SHARED_LAYER_NAMES:
                exp[lname] = [tuple(np.asarray(x).shape) for x in shared_dict[lname]]
            self.expected_shapes = exp
            print("[Aggregator] Learned expected shapes:")
            for lname in SHARED_LAYER_NAMES:
                print(f"  - {lname}: {self.expected_shapes[lname]}")
            return True, "OK"

        # Shape check
        for lname in SHARED_LAYER_NAMES:
            exp_shapes = self.expected_shapes[lname]
            got_list = shared_dict[lname]

            if len(got_list) != len(exp_shapes):
                return False, f"Bad weights length for '{lname}'"

            for i, arr in enumerate(got_list):
                if tuple(np.asarray(arr).shape) != tuple(exp_shapes[i]):
                    return False, f"Shape mismatch '{lname}'[{i}]"

        return True, "OK"

    def _weighted_average_shared(self, updates):

        total = sum(int(n) for _, n in updates)
        if total <= 0:
            return self.global_shared

        out = {}

        for lname in SHARED_LAYER_NAMES:
            wref = updates[0][0][lname]
            sums = [np.zeros_like(np.asarray(x), dtype=np.float64) for x in wref]

            for (sdict, n) in updates:
                wlist = sdict[lname]
                for i, arr in enumerate(wlist):
                    sums[i] += int(n) * np.asarray(arr)

            out[lname] = [s / float(total) for s in sums]

        return out

    # ---------------- RPC ----------------
    def SendSharedUpdate(self, request, context):

        md = dict(context.invocation_metadata())
        client_id = md.get("client_id", "")

        if int(request.round) != int(self.current_round):
            return myproto_pb2.Ack(
                status="ERROR: Round mismatch",
                current_round=self.current_round
            )

        try:
            shared_dict = pickle.loads(request.weights)
        except Exception:
            return myproto_pb2.Ack(
                status="ERROR: Bad weights",
                current_round=self.current_round
            )

        ok, msg = self._validate_payload(shared_dict)
        if not ok:
            return myproto_pb2.Ack(
                status=f"ERROR: {msg}",
                current_round=self.current_round
            )

        num_samples = int(request.num_samples)

        with self.lock:

            if len(self.client_updates) == 0:
                self.round_start_time = time.time()

            self.client_updates[client_id] = (shared_dict, num_samples)

            if len(self.client_updates) < self.num_clients:
                return myproto_pb2.Ack(status="WAITING", current_round=self.current_round)

            updates = list(self.client_updates.values())
            self.global_shared = self._weighted_average_shared(updates)

            self.current_round += 1
            self.client_updates.clear()

            return myproto_pb2.Ack(status="OK", current_round=self.current_round)

    def GetSharedWeights(self, request, context):

        if self.global_shared is None:
            return myproto_pb2.SharedResponse(weights=b"", round=self.current_round)

        return myproto_pb2.SharedResponse(
            weights=pickle.dumps(self.global_shared, protocol=pickle.HIGHEST_PROTOCOL),
            round=self.current_round
        )


if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    myproto_pb2_grpc.add_AggregatorServicer_to_server(
        AggregatorService(num_clients=6), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("[Aggregator] Listening on :50051 | FedPer backbone (input_adapter + feat1 + shared_dense)")
    server.wait_for_termination()