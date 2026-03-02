#!/usr/bin/env python3.10
# ============================================================
# client1_FEDAVG_full_model_LOCAL_GLOBAL_SAME_CSV.py
#
# FedAvg (FULL MODEL):
# - Share and aggregate ALL trainable layers:
#     input_adapter, feat1, shared_dense, clf
#
# Logs LOCAL (before sync) and GLOBAL (after sync) into SAME metrics CSV
# phases:
#   - local_before_sync
#   - global_after_sync
#   - final
#
# Also writes SUMMARY_CSV (one row per server round after sync):
#   local_macro_f1 vs global_macro_f1
#
# Strict barrier: wait until server_round >= sent_round + 1
#
# Server expectation:
#   - GetSharedWeights returns pickled dict: {layer_name: weights_list}
#   - SendSharedUpdate receives pickled dict payload similarly
# ============================================================

# ------- Reproducible seeding (TOP) -------
import os, random
SEED = 123
SPLIT_SEED = SEED

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
random.seed(SEED); np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# ----------------------------------------

import time, csv, pickle, grpc
from datetime import datetime
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from tensorflow.keras import models, layers, initializers, optimizers

import myproto_pb2, myproto_pb2_grpc

# -------------------------
# CONFIG
# -------------------------
SERVER_ADDRESS = "localhost:50051"
CLIENT_ID      = "client1"
DATASET_NAME   = "CIC-IoT-2022"

NUM_ROUNDS   = 8
BATCH_SIZE   = 256
LOCAL_EPOCHS = 1
BARRIER_POLL_SEC = 1.0

# IMPORTANT: for FULL FedAvg, shapes MUST MATCH across clients for ALL shared layers
CONV_FILTERS = 4
FEAT1_DIM    = 4
SHARED_DIM   = 4
LR = 1e-3

METHOD = "fedavg_full_model"
LAYERS_TO_SHARE = ["input_adapter", "feat1", "shared_dense", "clf"]  # FULL MODEL FedAvg

DATA_PATH = "/Users/azizahalq/Desktop/PFTL_Binary/Datasets_processed/D1_CIC_IOT_2022/CIC-IoT-V2_Balanced_16k+16k.csv"
LABEL_COL = "binary_label"

TEST_SIZE = 0.20
VAL_SIZE  = 0.10

# -------------------------
# LOGGING
# -------------------------
LOG_DIR = os.path.join("logs", METHOD)
os.makedirs(LOG_DIR, exist_ok=True)

METRICS_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_metrics.csv")
COMM_CSV    = os.path.join(LOG_DIR, f"{CLIENT_ID}_comm.csv")
SUMMARY_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_local_global_macro_f1_by_round.csv")

SUMMARY_HEADER = ["timestamp","method","dataset","seed","client","round","local_macro_f1","global_macro_f1"]

METRICS_HEADER = [
    "timestamp","method","dataset","seed","client",
    "round","phase",
    "acc",
    "prec_macro","rec_macro","f1_macro",
    "prec_weighted","rec_weighted","f1_weighted",
    "prec0","rec0","f10",
    "prec1","rec1","f11",
    "tn","fp","fn","tp"
]

COMM_HEADER = [
    "timestamp","method","dataset","seed","client",
    "round",
    "bytes_sent","bytes_recv","rtt_sec","train_time_sec",
    "server_round_after"
]

def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def append_csv(path, row, header):
    ensure_csv(path, header)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------------
# Data
# -------------------------
def load_dataset():
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1", low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    if LABEL_COL not in df.columns:
        raise ValueError(f"[{CLIENT_ID}] '{LABEL_COL}' not found. Example cols: {df.columns.tolist()[:25]}")

    y = df[LABEL_COL].astype(int)

    X = df.drop(columns=[LABEL_COL], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )

    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.125
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=SPLIT_SEED, stratify=y_trainval
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val   = sc.transform(X_val)
    X_test  = sc.transform(X_test)

    nf = X_train.shape[1]
    X_train = X_train.reshape((-1, nf, 1)).astype("float32")
    X_val   = X_val.reshape((-1, nf, 1)).astype("float32")
    X_test  = X_test.reshape((-1, nf, 1)).astype("float32")

    return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values, nf

# -------------------------
# Model
# -------------------------
def build_model(input_steps):
    ki = initializers.GlorotUniform(seed=SEED)
    bi = initializers.Zeros()

    model = models.Sequential([
        layers.Input(shape=(input_steps, 1)),

        # SHARED (FULL)
        layers.Conv1D(CONV_FILTERS, 5, activation="relu", padding="same",
                      name="input_adapter", kernel_initializer=ki, bias_initializer=bi),

        layers.MaxPooling1D(2, name="pool"),
        layers.GlobalAveragePooling1D(name="gap"),

        layers.Dense(FEAT1_DIM, activation="relu", name="feat1",
                     kernel_initializer=ki, bias_initializer=bi),

        layers.Dense(SHARED_DIM, activation="relu", name="shared_dense",
                     kernel_initializer=ki, bias_initializer=bi),

        layers.Dense(1, activation="sigmoid", name="clf",
                     kernel_initializer=ki, bias_initializer=bi),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    )
    return model

# -------------------------
# Shared payload helpers (dict of layers)
# -------------------------
def model_layer_names(model):
    return [l.name for l in model.layers]

def get_layers_payload(model, layer_names):
    names = set(model_layer_names(model))
    payload = {}
    for ln in layer_names:
        if ln not in names:
            raise ValueError(f"[{CLIENT_ID}] layer '{ln}' not found. Available: {sorted(list(names))}")
        w = model.get_layer(ln).get_weights()
        if (not isinstance(w, (list, tuple))) or len(w) == 0:
            raise ValueError(f"[{CLIENT_ID}] layer '{ln}' weights invalid/empty.")
        if any(np.isnan(np.asarray(x)).any() for x in w):
            raise ValueError(f"[{CLIENT_ID}] NaN detected in layer '{ln}'.")
        payload[ln] = w
    return payload

def set_layers_payload(model, payload):
    if not isinstance(payload, dict) or not payload:
        print(f"[{CLIENT_ID}] WARNING: global payload empty/invalid; skipping set.")
        return False

    names = set(model_layer_names(model))

    # Validate required layers exist in both payload and model
    for ln in LAYERS_TO_SHARE:
        if ln not in payload:
            print(f"[{CLIENT_ID}] WARNING: missing '{ln}' in payload; skipping set.")
            return False
        if ln not in names:
            print(f"[{CLIENT_ID}] WARNING: '{ln}' not in local model; skipping set.")
            return False
        w = payload.get(ln)
        if (not isinstance(w, (list, tuple))) or len(w) == 0:
            print(f"[{CLIENT_ID}] WARNING: bad format for '{ln}'; skipping set.")
            return False
        if any(np.isnan(np.asarray(x)).any() for x in w):
            print(f"[{CLIENT_ID}] WARNING: NaN in '{ln}'; skipping set.")
            return False

    for ln in LAYERS_TO_SHARE:
        try:
            model.get_layer(ln).set_weights(payload[ln])
        except Exception as e:
            print(f"[{CLIENT_ID}] WARNING: could not set '{ln}': {e}")
            return False

    return True

# -------------------------
# Metrics + print
# -------------------------
def compute_metrics(model, X, y_true):
    probs = model.predict(X, verbose=0).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.5)
    y_pred = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    mp, mr, mf1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    prec0, rec0, f10 = float(p_cls[0]), float(r_cls[0]), float(f_cls[0])
    prec1, rec1, f11 = float(p_cls[1]), float(r_cls[1]), float(f_cls[1])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    return {
        "y_pred": y_pred,
        "cm": cm,
        "acc": acc,
        "mp": float(mp), "mr": float(mr), "mf1": float(mf1),
        "wp": float(wp), "wr": float(wr), "wf1": float(wf1),
        "prec0": prec0, "rec0": rec0, "f10": f10,
        "prec1": prec1, "rec1": rec1, "f11": f11,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

def log_metrics(phase, rnd, m):
    append_csv(METRICS_CSV, [
        now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
        int(rnd), phase,
        round(m["acc"], 6),
        round(m["mp"], 6), round(m["mr"], 6), round(m["mf1"], 6),
        round(m["wp"], 6), round(m["wr"], 6), round(m["wf1"], 6),
        round(m["prec0"], 6), round(m["rec0"], 6), round(m["f10"], 6),
        round(m["prec1"], 6), round(m["rec1"], 6), round(m["f11"], 6),
        int(m["tn"]), int(m["fp"]), int(m["fn"]), int(m["tp"])
    ], METRICS_HEADER)

def print_client(title, m, y_true):
    print(f"\n[{CLIENT_ID}] ===== {title} =====")
    print("\n===== RESULTS =====")
    print(f"Accuracy        : {m['acc']:.6f}")
    print(f"Macro-Precision : {m['mp']:.6f}")
    print(f"Macro-Recall    : {m['mr']:.6f}")
    print(f"Macro-F1        : {m['mf1']:.6f}")

    print("\n===== WEIGHTED (support-weighted) METRICS =====")
    print(f"Weighted-Precision : {m['wp']:.6f}")
    print(f"Weighted-Recall    : {m['wr']:.6f}")
    print(f"Weighted-F1        : {m['wf1']:.6f}")

    print("\n==== Classification Report ====")
    print(classification_report(y_true, m["y_pred"], zero_division=0))

    print("\n==== Confusion Matrix ====")
    print(m["cm"])

# -------------------------
# Client
# -------------------------
class FedAvgFullClient:
    def __init__(self):
        self.Xtr, self.Xva, self.Xte, self.ytr, self.yva, self.yte, self.nf = load_dataset()

        classes = np.unique(self.ytr)
        cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=self.ytr)
        self.class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
        print(f"[{CLIENT_ID}] class_weight = {self.class_weights}")

        self.model = build_model(self.nf)

        self.channel = grpc.insecure_channel(SERVER_ADDRESS)
        self.stub = myproto_pb2_grpc.AggregatorStub(self.channel)

        self.current_round = 0
        ensure_csv(METRICS_CSV, METRICS_HEADER)
        ensure_csv(COMM_CSV, COMM_HEADER)
        ensure_csv(SUMMARY_CSV, SUMMARY_HEADER)

        # Bootstrap (safe)
        try:
            resp = self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])
            self.current_round = int(getattr(resp, "round", 0))

            glob = None
            if getattr(resp, "weights", None):
                try:
                    glob = pickle.loads(resp.weights)
                except Exception:
                    glob = None

            if isinstance(glob, dict) and glob:
                set_layers_payload(self.model, glob)

            print(f"[{CLIENT_ID}] Aggregator reachable (server_round={self.current_round})")
        except Exception as e:
            print(f"[{CLIENT_ID}] Bootstrap failed: {e}")

    def _pull_global(self):
        resp = self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])
        bytes_recv = len(resp.weights) if getattr(resp, "weights", None) else 0

        w = None
        if getattr(resp, "weights", None) and bytes_recv > 0:
            try:
                w = pickle.loads(resp.weights)
            except Exception:
                w = None

        return w, int(getattr(resp, "round", 0)), int(bytes_recv)

    def _ack_ok(self, ack) -> bool:
        if hasattr(ack, "ok"):
            return bool(ack.ok)
        if hasattr(ack, "status"):
            s = str(ack.status).upper()
            return s.startswith("OK") or s.startswith("WAIT")
        return True

    def run(self):
        for i in range(NUM_ROUNDS):
            print(f"\n[{CLIENT_ID}] ===== Loop {i+1}/{NUM_ROUNDS} | server_round={self.current_round} =====")

            # pull latest before training (safe)
            try:
                glob_w, r, _ = self._pull_global()
                if int(r) >= int(self.current_round) and isinstance(glob_w, dict) and glob_w:
                    set_layers_payload(self.model, glob_w)
                    self.current_round = int(r)
                    print(f"[{CLIENT_ID}] Pulled shared layers | server_round={self.current_round}")
            except Exception as e:
                print(f"[{CLIENT_ID}] Pull before train failed: {e}")

            # local training
            t_train0 = time.perf_counter()
            self.model.fit(
                self.Xtr, self.ytr,
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=True,
                class_weight=self.class_weights,
                validation_data=(self.Xva, self.yva)
            )
            train_time = time.perf_counter() - t_train0

            # LOCAL before sync
            sent_round = int(self.current_round)
            m_local = compute_metrics(self.model, self.Xte, self.yte)
            log_metrics("local_before_sync", sent_round, m_local)
            print_client("TEST EVALUATION (local_before_sync)", m_local, self.yte)
            local_mf1 = float(m_local["mf1"])

            # send update (ALL layers)
            try:
                payload_dict = get_layers_payload(self.model, LAYERS_TO_SHARE)
                payload = pickle.dumps(payload_dict, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"[{CLIENT_ID}] ERROR preparing payload: {e}")
                continue

            bytes_sent = len(payload)
            t0 = time.perf_counter()

            ack = self.stub.SendSharedUpdate(
                myproto_pb2.SharedUpdate(
                    weights=payload,
                    round=sent_round,
                    num_samples=int(self.Xtr.shape[0])
                ),
                metadata=[("client_id", CLIENT_ID)]
            )

            if not self._ack_ok(ack):
                print(f"[{CLIENT_ID}] Server rejected update (round mismatch/other). Resyncing...")
                try:
                    glob_w, r, _ = self._pull_global()
                    if isinstance(glob_w, dict) and glob_w:
                        set_layers_payload(self.model, glob_w)
                    self.current_round = int(r)
                except Exception as e:
                    print(f"[{CLIENT_ID}] Resync failed: {e}")
                continue

            # barrier: wait until server_round >= sent_round + 1
            target = sent_round + 1
            while True:
                time.sleep(BARRIER_POLL_SEC)
                glob_w, r_check, bytes_recv = self._pull_global()

                if int(r_check) >= target:
                    rtt = time.perf_counter() - t0

                    if isinstance(glob_w, dict) and glob_w:
                        ok = set_layers_payload(self.model, glob_w)
                        if not ok:
                            print(f"[{CLIENT_ID}] WARNING: invalid global payload at r={r_check}; keeping local.")
                    else:
                        print(f"[{CLIENT_ID}] WARNING: global payload empty/invalid at r={r_check}; keeping local.")

                    self.current_round = int(r_check)

                    # GLOBAL after sync
                    m_global = compute_metrics(self.model, self.Xte, self.yte)
                    log_metrics("global_after_sync", self.current_round, m_global)
                    print_client("TEST EVALUATION (global_after_sync)", m_global, self.yte)
                    global_mf1 = float(m_global["mf1"])

                    # summary (server round after sync)
                    append_csv(SUMMARY_CSV, [
                        now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
                        int(self.current_round),
                        round(float(local_mf1), 6),
                        round(float(global_mf1), 6),
                    ], SUMMARY_HEADER)

                    # comm
                    append_csv(COMM_CSV, [
                        now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
                        int(sent_round),
                        int(bytes_sent), int(bytes_recv),
                        round(rtt, 6), round(train_time, 6),
                        int(self.current_round)
                    ], COMM_HEADER)

                    print(f"[{CLIENT_ID}] Barrier passed -> server_round={self.current_round}")
                    break
                else:
                    print(f"[{CLIENT_ID}] Waiting server... (server_round={r_check}, target={target})")

        # FINAL
        m_final = compute_metrics(self.model, self.Xte, self.yte)
        log_metrics("final", self.current_round, m_final)
        print_client("FINAL TEST EVALUATION", m_final, self.yte)

        print("\n===== SAVED (CSV ONLY) =====")
        print("Metrics CSV :", METRICS_CSV)
        print("Comm CSV    :", COMM_CSV)
        print("Summary CSV :", SUMMARY_CSV)

if __name__ == "__main__":
    FedAvgFullClient().run()