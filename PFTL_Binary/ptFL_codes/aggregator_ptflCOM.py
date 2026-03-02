#!/usr/bin/env python3.10
# ============================================================
# aggregator_PFTL_shared_dense_only_no_model.py
# PFTL Aggregator: aggregate ONLY shared_dense layer (FedAvg weighted by num_samples)
#
# NO TensorFlow / NO model template:
# - Shapes are learned from the FIRST received update.
# - Then all updates must match the learned shapes.
#
# Contract:
# - Clients send pickled shared_dense weights: [W, b]
# - Aggregator returns pickled global shared_dense weights: [W, b]
# - Round barrier: aggregator advances when ALL num_clients updates arrive for current_round
# ============================================================

import os, random, time, csv, pickle, threading
from concurrent import futures

# ---- Reproducible seeding (for any randomness we might use) ----
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
import numpy as np
random.seed(SEED); np.random.seed(SEED)

import grpc
import myproto_pb2
import myproto_pb2_grpc

SHARED_LAYER_NAME = "shared_dense"

class AggregatorService(myproto_pb2_grpc.AggregatorServicer):
    def __init__(self, num_clients=6, save_dir="agg_pftl_shared_dense_no_model"):
        self.num_clients = int(num_clients)
        self.current_round = 0

        # per-round buffer: client_id -> (weights, num_samples)
        self.client_updates = {}
        self.lock = threading.Lock()
        self.round_start_time = None

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.comm_log_path = os.path.join(self.save_dir, "aggregator_comm_log.csv")
        if not os.path.exists(self.comm_log_path):
            with open(self.comm_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "round", "client_id", "bytes_received", "num_samples",
                    "round_duration_sec", "timestamp"
                ])

        # global shared weights (unknown until first update arrives)
        self.global_shared = None      # will be [W,b]
        self.shared_shapes = None      # will be (W_shape, b_shape)

        print(f"[Aggregator] NO-MODEL mode | num_clients={self.num_clients} | save_dir={self.save_dir}")

    # -------------------------
    # Weighted average for [W,b]
    # -------------------------
    def _weighted_average(self, updates):
        """
        updates: list of (weights=[W,b], num_samples)
        returns: [W_avg, b_avg]
        """
        total = sum(int(n) for _, n in updates)
        if total <= 0:
            print("[Aggregator] WARNING: total_samples <= 0; keeping previous shared_dense.")
            return self.global_shared

        W_sum = None
        b_sum = None

        for (w, n) in updates:
            n = int(n)
            W = np.nan_to_num(w[0])
            b = np.nan_to_num(w[1])

            if W_sum is None:
                W_sum = n * W
                b_sum = n * b
            else:
                W_sum += n * W
                b_sum += n * b

        return [W_sum / total, b_sum / total]

    def _persist_global_shared(self, round_number_finished):
        """
        Save the global shared_dense weights as .pkl for the FINISHED round.
        """
        if self.global_shared is None:
            return
        path = os.path.join(
            self.save_dir,
            f"global_{SHARED_LAYER_NAME}_round_{round_number_finished:06d}.pkl"
        )
        with open(path, "wb") as f:
            pickle.dump(self.global_shared, f)

    # =========================
    # RPC: SendSharedUpdate
    # =========================
    def SendSharedUpdate(self, request, context):
        md = dict(context.invocation_metadata())
        client_id = md.get("client_id", "")
        if not client_id:
            return myproto_pb2.Ack(status="ERROR: Missing client_id", current_round=self.current_round)

        req_round = int(request.round)
        if req_round != int(self.current_round):
            return myproto_pb2.Ack(
                status=f"ERROR: Round mismatch (got {req_round}, expected {self.current_round})",
                current_round=self.current_round
            )

        # unpickle weights
        try:
            weights = pickle.loads(request.weights)  # expected: [W,b]
        except Exception as e:
            print(f"[Aggregator] Unpickle error from {client_id}: {e}")
            return myproto_pb2.Ack(status="ERROR: Bad weights", current_round=self.current_round)

        # validate structure
        if (not isinstance(weights, list)) or len(weights) != 2:
            print(f"[Aggregator] Bad format from {client_id}: expected [W,b]")
            return myproto_pb2.Ack(status="ERROR: Bad format", current_round=self.current_round)

        # validate NaNs
        try:
            if np.isnan(weights[0]).any() or np.isnan(weights[1]).any():
                print(f"[Aggregator] NaN weights from {client_id}, rejecting.")
                return myproto_pb2.Ack(status="ERROR: NaN in weights", current_round=self.current_round)
        except Exception:
            # in case weights are not numpy arrays
            return myproto_pb2.Ack(status="ERROR: Non-numpy weights", current_round=self.current_round)

        num_samples = int(request.num_samples)

        with self.lock:
            if len(self.client_updates) == 0:
                self.round_start_time = time.time()

            # If first-ever update: set global + shapes
            if self.global_shared is None:
                self.global_shared = [np.array(weights[0]), np.array(weights[1])]
                self.shared_shapes = (self.global_shared[0].shape, self.global_shared[1].shape)
                print(f"[Aggregator] Initialized global {SHARED_LAYER_NAME} from first update: shapes={self.shared_shapes}")

            # shape check
            exp_W, exp_b = self.shared_shapes
            if weights[0].shape != exp_W or weights[1].shape != exp_b:
                print(
                    f"[Aggregator] Shape mismatch from {client_id}: "
                    f"got {weights[0].shape}/{weights[1].shape}, expected {exp_W}/{exp_b}"
                )
                return myproto_pb2.Ack(status="ERROR: Shape mismatch", current_round=self.current_round)

            # store (overwrite if resend)
            self.client_updates[client_id] = ([np.array(weights[0]), np.array(weights[1])], num_samples)

            bytes_received = len(request.weights)
            with open(self.comm_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.current_round, client_id, bytes_received, num_samples, "",
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])

            print(f"[Aggregator] Round {self.current_round}: received {client_id} ({len(self.client_updates)}/{self.num_clients})")

            if len(self.client_updates) < self.num_clients:
                return myproto_pb2.Ack(status="WAITING", current_round=self.current_round)

            # aggregate
            finished_round = self.current_round
            print(f"[Aggregator] Aggregating {SHARED_LAYER_NAME} for round {finished_round} ...")
            self.global_shared = self._weighted_average(list(self.client_updates.values()))

            # advance round
            self.current_round += 1
            self.client_updates.clear()

            # persist finished round weights
            self._persist_global_shared(finished_round)

            # compute duration & backfill duration for finished_round rows
            dur = time.time() - self.round_start_time
            try:
                with open(self.comm_log_path, "r") as f:
                    rows = list(csv.reader(f))
                with open(self.comm_log_path, "w", newline="") as f:
                    w = csv.writer(f)
                    for row in rows:
                        if row and row[0] == str(finished_round) and row[4] == "":
                            row[4] = f"{dur:.3f}"
                        w.writerow(row)
            except Exception as e:
                print(f"[Aggregator] NOTE: could not backfill duration: {e}")

            print(f"[Aggregator] Round {finished_round} complete in {dur:.2f}s. Now current_round={self.current_round}")
            return myproto_pb2.Ack(status="OK", current_round=self.current_round)

    # =========================
    # RPC: GetSharedWeights
    # =========================
    def GetSharedWeights(self, request, context):
        if self.global_shared is None:
            # return empty until first update arrives
            return myproto_pb2.SharedResponse(weights=b"", round=self.current_round)

        return myproto_pb2.SharedResponse(
            weights=pickle.dumps(self.global_shared),
            round=self.current_round
        )

# =========================
# Main
# =========================
if __name__ == "__main__":
    NUM_CLIENTS = 6  # <<< set your actual client count >>>

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    myproto_pb2_grpc.add_AggregatorServicer_to_server(
        AggregatorService(num_clients=NUM_CLIENTS, save_dir="agg_pftl_shared_dense"),
        server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("[Aggregator] Listening on :50051 ... (PFTL shared_dense only )")
    server.wait_for_termination()