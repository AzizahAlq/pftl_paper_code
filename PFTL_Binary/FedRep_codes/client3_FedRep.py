#!/usr/bin/env python3.10
# ============================================================
# client3_FEDREP_share_repr_head_private_CLIENT6_PRINT_NO_PLOTS.py
#
# FedRep:
#   - Shared (global): representation/backbone layers only
#       e.g., input_adapter, feat1, shared_dense
#   - Private (local): clf head (never sent / never overwritten)
#
# Training per round (FedRep schedule):
#   A) Train REPRESENTATION (backbone) with head FROZEN
#      -> send backbone update to server (FedAvg over shared layers)
#      -> pull aggregated backbone
#   B) Train HEAD (clf) with backbone FROZEN (local only)
#
# SAME split: Test=20%, Val=10% total (val_frac=0.125)
# Correct class_weight mapping
# STRICT barrier: wait server_round >= sent_round + 1 (timeout)
# Client6-style prints, CSV logs, NO plots
#
# NEW: save local vs global macro-F1 by round to:
#   logs/<METHOD>/<CLIENT_ID>_local_global_macro_f1_by_round.csv
# ============================================================

import os, random
SEED = 190
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

import myproto_pb2, myproto_pb2_grpc
from tensorflow.keras import models, layers, optimizers, initializers

# =========================
# CONFIG
# =========================
SERVER_ADDRESS = "localhost:50051"
CLIENT_ID      = "client3"
DATASET_NAME   = "CIC-IoT-2023"

NUM_ROUNDS   = 8
BATCH_SIZE   = 256

# FedRep schedule epochs (you can tune)
EPOCHS_REP  = 1   # representation/backbone training
EPOCHS_HEAD = 1   # head-only fine-tune

SLEEP_POLL        = 1.0
SYNC_TIMEOUT_SEC  = 600.0

METHOD = "fedrep_repr_shared_head_private"
HEAD_NAME = "clf"

# Shared representation layers (FedAvg over these only)
# If you want NO input_adapter sharing, remove it from this list.
REP_SHARED_LAYERS = ["input_adapter", "feat1", "shared_dense"]

PRIVATE_DIM  = 4
SHARED_DIM   = 4
CONV_FILTERS = 4
LR = 1e-3

DATA_PATH = "/Users/azizahalq/Desktop/PFTL_Binary/Datasets_processed/D3_CICIOT2023/cic_iot-2023_merged_32000.csv"
LABEL_COL = "binary_label"

TEST_SIZE = 0.20
VAL_SIZE  = 0.10

# =========================
# LOGGING
# =========================
LOG_DIR = os.path.join("logs", METHOD)
os.makedirs(LOG_DIR, exist_ok=True)

METRICS_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_metrics.csv")
COMM_CSV    = os.path.join(LOG_DIR, f"{CLIENT_ID}_comm.csv")

# NEW: local vs global macro-F1 by round
LOCAL_GLOBAL_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_local_global_macro_f1_by_round.csv")
LOCAL_GLOBAL_HEADER = ["round", "local_macro_f1", "global_macro_f1"]

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

def save_local_global_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOCAL_GLOBAL_HEADER)
        w.writerows(rows)

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =========================
# DATA (same split)
# =========================
def load_dataset():
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1", low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    if LABEL_COL not in df.columns:
        raise ValueError(f"[{CLIENT_ID}] '{LABEL_COL}' not found. Example cols: {df.columns.tolist()[:25]}")

    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL], errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )

    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.125
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_frac, random_state=SPLIT_SEED, stratify=y_train_full
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    nf = X_train.shape[1]
    X_train = X_train.reshape((-1, nf, 1)).astype("float32")
    X_val   = X_val.reshape((-1, nf, 1)).astype("float32")
    X_test  = X_test.reshape((-1, nf, 1)).astype("float32")

    return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values, nf

# =========================
# MODEL (GAP-stable)
# =========================
def build_model(input_steps):
    ki = initializers.GlorotUniform(seed=SEED)
    bi = initializers.Zeros()

    model = models.Sequential([
        layers.Input(shape=(input_steps, 1)),

        layers.Conv1D(CONV_FILTERS, 5, activation="relu", padding="same",
                      name="input_adapter", kernel_initializer=ki, bias_initializer=bi),
        layers.MaxPooling1D(pool_size=2, name="pool"),
        layers.GlobalAveragePooling1D(name="gap"),

        layers.Dense(PRIVATE_DIM, activation="relu", name="feat1",
                     kernel_initializer=ki, bias_initializer=bi),
        layers.Dense(SHARED_DIM, activation="relu", name="shared_dense",
                     kernel_initializer=ki, bias_initializer=bi),

        layers.Dense(1, activation="sigmoid", name=HEAD_NAME,
                     kernel_initializer=ki, bias_initializer=bi),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    )
    return model

# =========================
# SHARE HELPERS (repr only)
# =========================
def _layer_names(model):
    return [l.name for l in model.layers]

def get_repr_payload(model):
    names = set(_layer_names(model))
    payload = {}
    for ln in REP_SHARED_LAYERS:
        if ln not in names:
            raise ValueError(f"[{CLIENT_ID}] layer '{ln}' not found. Available: {sorted(list(names))}")
        w = model.get_layer(ln).get_weights()
        if not w:
            raise ValueError(f"[{CLIENT_ID}] layer '{ln}' has no weights (unexpected).")
        if any(np.isnan(np.asarray(x)).any() for x in w):
            raise ValueError(f"[{CLIENT_ID}] NaN detected in layer '{ln}'.")
        payload[ln] = w
    return payload

def set_repr_payload(model, payload):
    if not isinstance(payload, dict) or not payload:
        print(f"[{CLIENT_ID}] WARNING: global payload empty/invalid; skipping set.")
        return
    names = set(_layer_names(model))
    for ln, w in payload.items():
        if ln not in names:
            continue
        if ln == HEAD_NAME:
            continue
        if not isinstance(w, (list, tuple)) or len(w) == 0:
            continue
        if any(np.isnan(np.asarray(x)).any() for x in w):
            continue
        try:
            model.get_layer(ln).set_weights(w)
        except Exception as e:
            print(f"[{CLIENT_ID}] WARNING: could not set layer '{ln}': {e}")

# =========================
# FREEZE/UNFREEZE (FedRep schedule)
# =========================
def set_trainable_for_rep(model):
    # Train only representation layers, freeze head
    for lyr in model.layers:
        lyr.trainable = (lyr.name != HEAD_NAME)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    )

def set_trainable_for_head(model):
    # Train only head, freeze everything else
    for lyr in model.layers:
        lyr.trainable = (lyr.name == HEAD_NAME)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    )

# =========================
# METRICS + PRINT (Client6)
# =========================
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

def print_client6_style(title, m, y_true):
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

# =========================
# CLIENT (FedRep)
# =========================
class FedRepClient3:
    def __init__(self):
        self.Xtr, self.Xva, self.Xte, self.ytr, self.yva, self.yte, self.nf = load_dataset()

        classes = np.unique(self.ytr)
        cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=self.ytr)
        self.class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
        print(f"[{CLIENT_ID}] class_weight = {self.class_weights}")

        self.model = build_model(self.nf)

        self.stub = myproto_pb2_grpc.AggregatorStub(
            grpc.insecure_channel(
                SERVER_ADDRESS,
                options=[
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ],
            )
        )
        self.current_round = 0

        ensure_csv(METRICS_CSV, METRICS_HEADER)
        ensure_csv(COMM_CSV, COMM_HEADER)
        ensure_csv(LOCAL_GLOBAL_CSV, LOCAL_GLOBAL_HEADER)

        # Bootstrap representation from server (head stays local)
        try:
            resp = self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])
            if resp.weights:
                glob = pickle.loads(resp.weights)
                if isinstance(glob, dict):
                    set_repr_payload(self.model, glob)
            self.current_round = int(resp.round)
            print(f"[{CLIENT_ID}] Bootstrapped repr | server_round={self.current_round}")
        except Exception as e:
            print(f"[{CLIENT_ID}] Bootstrap failed: {e}")

    def _pull_global(self):
        resp = self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])
        w = pickle.loads(resp.weights) if resp.weights else None
        return w, int(resp.round), int(len(resp.weights) if resp.weights else 0)

    def run(self):
        local_global_rows = []

        for i in range(NUM_ROUNDS):
            print(f"\n[{CLIENT_ID}] ===== Round {i+1}/{NUM_ROUNDS} (server_round={self.current_round}) =====")

            # ---- (optional) pull latest repr BEFORE training to align round ----
            try:
                glob_w, r, _ = self._pull_global()
                if isinstance(glob_w, dict):
                    set_repr_payload(self.model, glob_w)
                if r >= int(self.current_round):
                    self.current_round = int(r)
            except Exception as e:
                print(f"[{CLIENT_ID}] Pull before train failed: {e}")

            # =========================================================
            # A) REPRESENTATION TRAIN (head frozen) + SEND to server
            # =========================================================
            set_trainable_for_rep(self.model)

            t_train0 = time.perf_counter()
            self.model.fit(
                self.Xtr, self.ytr,
                epochs=EPOCHS_REP,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=True,
                class_weight=self.class_weights,
                validation_data=(self.Xva, self.yva)
            )
            train_time = time.perf_counter() - t_train0

            m_local = compute_metrics(self.model, self.Xte, self.yte)
            log_metrics("local_before_sync", self.current_round, m_local)
            print_client6_style("TEST (after REP train) local_before_sync", m_local, self.yte)
            local_mf1 = float(m_local["mf1"])

            # Send ONLY representation layers
            try:
                payload_dict = get_repr_payload(self.model)
                payload = pickle.dumps(payload_dict, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"[{CLIENT_ID}] ERROR preparing payload: {e}")
                continue

            bytes_sent = int(len(payload))
            sent_round = int(self.current_round)
            t0 = time.perf_counter()

            ack = self.stub.SendSharedUpdate(
                myproto_pb2.SharedUpdate(
                    weights=payload,
                    round=sent_round,
                    num_samples=int(self.Xtr.shape[0])
                ),
                metadata=[("client_id", CLIENT_ID)]
            )

            if hasattr(ack, "status") and str(ack.status).startswith("ERROR"):
                print(f"[{CLIENT_ID}] Server rejected update: {ack.status} (server_round={ack.current_round})")
                self.current_round = int(getattr(ack, "current_round", sent_round))
                continue

            # Barrier: wait for global to advance
            target = sent_round + 1
            wait_start = time.perf_counter()

            global_mf1 = None

            while True:
                time.sleep(SLEEP_POLL)
                glob_w, r_check, bytes_recv = self._pull_global()

                if int(r_check) >= target:
                    rtt = time.perf_counter() - t0

                    if isinstance(glob_w, dict):
                        set_repr_payload(self.model, glob_w)
                    else:
                        print(f"[{CLIENT_ID}] WARNING: global payload invalid; keeping local repr.")

                    self.current_round = int(r_check)

                    m_global = compute_metrics(self.model, self.Xte, self.yte)
                    log_metrics("global_after_sync", self.current_round, m_global)
                    print_client6_style("TEST (after REPR sync) global_after_sync", m_global, self.yte)
                    global_mf1 = float(m_global["mf1"])

                    append_csv(COMM_CSV, [
                        now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
                        int(sent_round),
                        int(bytes_sent), int(bytes_recv),
                        round(float(rtt), 6), round(float(train_time), 6),
                        int(self.current_round)
                    ], COMM_HEADER)

                    print(f"[{CLIENT_ID}] Barrier passed -> server_round={self.current_round}")
                    break

                if (time.perf_counter() - wait_start) > SYNC_TIMEOUT_SEC:
                    print(f"[{CLIENT_ID}] WARNING: timeout after {SYNC_TIMEOUT_SEC}s waiting for global. Continue.")
                    break

                print(f"[{CLIENT_ID}] Waiting server... (server_round={r_check}, target={target})")

            # Save local/global MF1 row ONLY if we got the global
            if global_mf1 is not None:
                local_global_rows.append([int(self.current_round), float(local_mf1), float(global_mf1)])

            # =========================================================
            # B) HEAD TRAIN (backbone frozen) local-only
            # =========================================================
            set_trainable_for_head(self.model)
            self.model.fit(
                self.Xtr, self.ytr,
                epochs=EPOCHS_HEAD,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=True,
                class_weight=self.class_weights,
                validation_data=(self.Xva, self.yva)
            )

            # (optional) log after head tune each round (kept from your original)
            m_head = compute_metrics(self.model, self.Xte, self.yte)
            log_metrics("after_head_tune", self.current_round, m_head)
            print_client6_style("TEST (after HEAD tune) local", m_head, self.yte)

        # Save local/global summary at end
        save_local_global_csv(local_global_rows, LOCAL_GLOBAL_CSV)
        print(f"\n[{CLIENT_ID}] Saved local/global summary: {LOCAL_GLOBAL_CSV}")

        # FINAL
        m = compute_metrics(self.model, self.Xte, self.yte)
        log_metrics("final", self.current_round, m)
        print_client6_style("FINAL TEST EVALUATION", m, self.yte)

        print("\n===== SAVED (CSV ONLY) =====")
        print("Metrics CSV      :", METRICS_CSV)
        print("Comm CSV         :", COMM_CSV)
        print("Local/Global CSV :", LOCAL_GLOBAL_CSV)

if __name__ == "__main__":
    FedRepClient3().run()