#!/usr/bin/env python3.10
# ============================================================
# aggregator_fedper_no_input.py
# FedPer: share backbone, keep head personal
# Exclude input_adapter from sharing
#
# PRIVATE (clients): input_adapter, clf
# SHARED (global): feat1, shared_dense
# Proto matches your usage:
# - SendSharedUpdate(SharedUpdate{weights, round, num_samples})
# - GetSharedWeights(EmptyRequest) -> SharedResponse{weights, round}
# ============================================================

import os, random, time, csv, pickle, threading
from concurrent import futures

SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
random.seed(SEED); np.random.seed(SEED)

import grpc
import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras import models, layers, initializers

import myproto_pb2
import myproto_pb2_grpc

# -------------------
# Share these layers only
# -------------------
SHARED_LAYER_NAMES = ["feat1", "shared_dense"]

# Template dims (only to build weights shapes)
INPUT_STEPS_AGG = 83
CHANNELS = 1
PRIVATE_DIM = 4
SHARED_DIM  = 4
CONV_FILTERS = 16

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
                csv.writer(f).writerow(["round","client_id","bytes_received","round_duration_sec","timestamp"])

        self.global_shared = self._init_shared_weights()
        print("[Aggregator] FedPer global shared initialized:", SHARED_LAYER_NAMES)

    def _build_template_model(self, input_steps=INPUT_STEPS_AGG, channels=CHANNELS):
        ki = initializers.GlorotUniform(seed=SEED)
        bi = initializers.Zeros()

        m = models.Sequential([
            layers.Input(shape=(input_steps, channels)),

            # PRIVATE on clients (not shared)
            layers.Conv1D(CONV_FILTERS, 5, activation="relu", padding="same", name="input_adapter",
                          kernel_initializer=ki, bias_initializer=bi),
            layers.MaxPooling1D(2),

            # CRITICAL: fixed-size output so sharing works even if input_steps differ
            layers.GlobalAveragePooling1D(),

            # SHARED backbone
            layers.Dense(PRIVATE_DIM, activation="relu", name="feat1",
                         kernel_initializer=ki, bias_initializer=bi),
            layers.Dense(SHARED_DIM, activation="relu", name="shared_dense",
                         kernel_initializer=ki, bias_initializer=bi),

            # PRIVATE head (FedPer)
            layers.Dense(1, activation="sigmoid", name="clf",
                         kernel_initializer=ki, bias_initializer=bi),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return m

    def _init_shared_weights(self):
        m = self._build_template_model()
        shared = {}
        for lname in SHARED_LAYER_NAMES:
            shared[lname] = m.get_layer(lname).get_weights()
        return shared

    def _shape_check(self, shared_dict):
        for lname in SHARED_LAYER_NAMES:
            if lname not in shared_dict:
                return False, f"Missing layer {lname}"
            exp = self.global_shared[lname]
            got = shared_dict[lname]
            if len(got) != len(exp):
                return False, f"Bad weights length for {lname}"
            for i in range(len(exp)):
                if got[i].shape != exp[i].shape:
                    return False, f"Shape mismatch {lname}[{i}]: got {got[i].shape}, expected {exp[i].shape}"
                if np.isnan(got[i]).any():
                    return False, f"NaN in {lname}[{i}]"
        return True, "OK"

    def _weighted_average_shared(self, updates):
        total = sum(n for _, n in updates)
        if total <= 0:
            print("[Aggregator] WARNING: total_samples<=0; keeping previous.")
            return self.global_shared

        out = {}
        for lname in SHARED_LAYER_NAMES:
            wref = updates[0][0][lname]
            sums = [np.zeros_like(x, dtype=np.float64) for x in wref]
            for (sdict, n) in updates:
                wlist = sdict[lname]
                for i, arr in enumerate(wlist):
                    sums[i] += n * np.nan_to_num(arr)
            out[lname] = [s/total for s in sums]
        return out

    # ---------------- RPCs ----------------
    def SendSharedUpdate(self, request, context):
        md = dict(context.invocation_metadata())
        client_id = md.get("client_id", "")
        if not client_id:
            return myproto_pb2.Ack(status="ERROR: Missing client_id", current_round=self.current_round)

        # Basic round guard: ignore stale/future
        if int(request.round) != int(self.current_round):
            return myproto_pb2.Ack(status=f"ERROR: Round mismatch server={self.current_round} client={request.round}",
                                  current_round=self.current_round)

        try:
            shared_dict = pickle.loads(request.weights)  # {layer: [W,b], ...}
        except Exception as e:
            print(f"[Aggregator] Unpickle error from {client_id}: {e}")
            return myproto_pb2.Ack(status="ERROR: Bad weights", current_round=self.current_round)

        ok, msg = self._shape_check(shared_dict)
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
                csv.writer(f).writerow([
                    self.current_round, client_id, bytes_received, "",
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])

            print(f"[Aggregator] Round {self.current_round}: got {client_id} ({len(self.client_updates)}/{self.num_clients})")

            if len(self.client_updates) < self.num_clients:
                return myproto_pb2.Ack(status="WAITING", current_round=self.current_round)

            print("[Aggregator] Aggregating shared backbone:", SHARED_LAYER_NAMES)
            self.global_shared = self._weighted_average_shared(list(self.client_updates.values()))

            finished_round = self.current_round
            self.current_round += 1
            self.client_updates.clear()

            # persist global shared
            with open("aggregated_shared.pkl", "wb") as f:
                pickle.dump(self.global_shared, f)

            # fill duration
            dur = time.time() - self.round_start_time
            with open(self.comm_log_path, "r") as f:
                rows = list(csv.reader(f))
            with open(self.comm_log_path, "w", newline="") as f:
                w = csv.writer(f)
                for row in rows:
                    if row and row[0] == str(finished_round) and row[3] == "":
                        row[3] = f"{dur:.3f}"
                    w.writerow(row)

            print(f"[Aggregator] Round {finished_round} complete in {dur:.2f}s -> current_round={self.current_round}")
            return myproto_pb2.Ack(status="OK", current_round=self.current_round)

    def GetSharedWeights(self, request, context):
        return myproto_pb2.SharedResponse(
            weights=pickle.dumps(self.global_shared),
            round=self.current_round
        )

if __name__ == "__main__":
    NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "6"))
    PORT = os.environ.get("PORT", "50051")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    myproto_pb2_grpc.add_AggregatorServicer_to_server(AggregatorService(num_clients=NUM_CLIENTS), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"[Aggregator] Listening on :{PORT} | FedPer backbone sharing (exclude input_adapter & clf)")
    server.wait_for_termination()
