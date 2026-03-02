#!/usr/bin/env python3.10
# ============================================================
# client1_FEDPER_share_all_except_clf_FULL_METRICS_CLASSWEIGHT.py
#
# FedPer (Backbone shared, Head private):
# - Share ONLY: input_adapter, feat1, shared_dense
# - Do NOT share: clf  (private personalized head)
#
# Uses GAP (GlobalAveragePooling1D) so shapes are stable.
# FULL metrics + confusion + CSV
# class_weight enabled
#
# NEW: saves Local vs Global Macro-F1 by round (SUMMARY_CSV):
#   client1_local_global_macro_f1_by_round.csv
# ============================================================

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

METHOD       = "fedper_share_all_except_clf"


NUM_ROUNDS   = 8
BATCH_SIZE   = 256
LOCAL_EPOCHS = 1
BARRIER_POLL_SEC = 1.0

# ---- FedPer: share backbone only (NO clf) ----
SHARED_LAYER_NAMES = ["input_adapter", "feat1", "shared_dense"]

CONV_FILTERS = 4
PRIVATE_DIM  = 4
SHARED_DIM   = 4
LR = 1e-3

CLIENT_ID      = "client5"
DATASET_NAME   = "UNSW-NB15"
DATA_PATH = "/Users/azizahalq/Desktop/PFTL_Binary/Datasets_processed/D3_CICIOT2023/final_balanced_nusw_nb15_minmax.csv"
LABEL_COL = "binary_label"

TEST_SIZE = 0.20
VAL_SIZE  = 0.10  # of total

# -------------------------
# LOGGING
# -------------------------
LOG_DIR = os.path.join("logs", METHOD)
os.makedirs(LOG_DIR, exist_ok=True)

METRICS_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_metrics.csv")
COMM_CSV    = os.path.join(LOG_DIR, f"{CLIENT_ID}_comm.csv")

# NEW summary CSV (one row per round after sync)
SUMMARY_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_local_global_macro_f1_by_round.csv")
SUMMARY_HEADER = ["timestamp","method","dataset","seed","client","round","local_macro_f1","global_macro_f1"]

METRICS_HEADER = [
    "timestamp","method","dataset","seed","client",
    "round","phase",
    "acc",
    "prec_bin","rec_bin","f1_bin",
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
    df.columns = df.columns.str.strip()

    if LABEL_COL not in df.columns:
        raise ValueError(f"[{CLIENT_ID}] '{LABEL_COL}' not found. Example cols: {df.columns.tolist()[:25]}")

    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL], errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Test = 20%
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )
    # Val = 10% of total => 0.125 of trainval
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=SPLIT_SEED, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    nf = X_train.shape[1]
    X_train = X_train.astype("float32").reshape((-1, nf, 1))
    X_val   = X_val.astype("float32").reshape((-1, nf, 1))
    X_test  = X_test.astype("float32").reshape((-1, nf, 1))

    return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values, nf

# -------------------------
# Model (GAP-stable)
# -------------------------
def build_model(input_steps):
    ki = initializers.GlorotUniform(seed=SEED)
    bi = initializers.Zeros()

    return models.Sequential([
        layers.Input(shape=(input_steps, 1)),

        # Shared backbone
        layers.Conv1D(CONV_FILTERS, 5, activation="relu", padding="same",
                      name="input_adapter", kernel_initializer=ki, bias_initializer=bi),
        layers.MaxPooling1D(2),
        layers.GlobalAveragePooling1D(),

        layers.Dense(PRIVATE_DIM, activation="relu", name="feat1",
                     kernel_initializer=ki, bias_initializer=bi),
        layers.Dense(SHARED_DIM, activation="relu", name="shared_dense",
                     kernel_initializer=ki, bias_initializer=bi),

        # Private head
        layers.Dense(1, activation="sigmoid", name="clf",
                     kernel_initializer=ki, bias_initializer=bi),
    ])

# -------------------------
# Shared payload helpers
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
        if not isinstance(w, (list, tuple)) or len(w) < 1:
            raise ValueError(f"[{CLIENT_ID}] layer '{ln}' weights invalid.")
        if any(np.isnan(np.asarray(x)).any() for x in w):
            raise ValueError(f"[{CLIENT_ID}] NaN detected in layer '{ln}'.")
        payload[ln] = w
    return payload

def set_layers_payload(model, payload):
    if not isinstance(payload, dict) or not payload:
        print(f"[{CLIENT_ID}] WARNING: global payload empty/invalid; skipping set.")
        return

    names = set(model_layer_names(model))
    for ln, w in payload.items():
        if ln not in names:
            continue
        if not isinstance(w, (list, tuple)) or len(w) < 1:
            continue
        if any(np.isnan(np.asarray(x)).any() for x in w):
            continue
        try:
            model.get_layer(ln).set_weights(w)
        except Exception as e:
            print(f"[{CLIENT_ID}] WARNING: could not set layer '{ln}': {e}")

# -------------------------
# Metrics + logging
# -------------------------
def eval_and_log(model, X, y, phase, rnd, print_report=False):
    probs = model.predict(X, verbose=0).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.5)
    y_pred = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y, y_pred))

    mp, mr, mf1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)
    bp, br, bf1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)

    p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    prec0, rec0, f10 = float(p_cls[0]), float(r_cls[0]), float(f_cls[0])
    prec1, rec1, f11 = float(p_cls[1]), float(r_cls[1]), float(f_cls[1])

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    print(f"\n[{CLIENT_ID}] ===== {phase.upper()} | round={rnd} =====")
    print(f"Accuracy        : {acc:.6f}")
    print(f"Macro-Precision : {mp:.6f} | Macro-Recall : {mr:.6f} | Macro-F1 : {mf1:.6f}")
    print(f"Weighted-Prec   : {wp:.6f} | Weighted-Rec : {wr:.6f} | Weighted-F1 : {wf1:.6f}")
    print(f"Binary(POS=1)   : P={bp:.6f} | R={br:.6f} | F1={bf1:.6f}")
    print("Confusion Matrix:\n", cm)

    if print_report:
        print("\n==== Classification Report ====")
        print(classification_report(y, y_pred, zero_division=0))

    append_csv(METRICS_CSV, [
        now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
        int(rnd), phase,
        round(acc, 6),
        round(float(bp), 6), round(float(br), 6), round(float(bf1), 6),
        round(float(mp), 6), round(float(mr), 6), round(float(mf1), 6),
        round(float(wp), 6), round(float(wr), 6), round(float(wf1), 6),
        round(prec0, 6), round(rec0, 6), round(f10, 6),
        round(prec1, 6), round(rec1, 6), round(f11, 6),
        tn, fp, fn, tp
    ], METRICS_HEADER)

    # return macro-f1 so we can store local vs global
    return float(mf1)

# -------------------------
# Client
# -------------------------
class FedPerClient:
    def __init__(self):
        self.Xtr, self.Xva, self.Xte, self.ytr, self.yva, self.yte, self.nf = load_dataset()

        classes = np.unique(self.ytr)
        cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=self.ytr)
        self.class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
        print(f"[{CLIENT_ID}] class_weight = {self.class_weights}")

        self.model = build_model(self.nf)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
        )

        self.channel = grpc.insecure_channel(SERVER_ADDRESS)
        self.stub = myproto_pb2_grpc.AggregatorStub(self.channel)

        self.current_round = 0
        ensure_csv(METRICS_CSV, METRICS_HEADER)
        ensure_csv(COMM_CSV, COMM_HEADER)
        ensure_csv(SUMMARY_CSV, SUMMARY_HEADER)  # NEW

        # Bootstrap: pull global backbone and set it (clf remains private)
        try:
            resp = self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])
            if resp.weights:
                global_payload = pickle.loads(resp.weights)
                if isinstance(global_payload, dict):
                    set_layers_payload(self.model, global_payload)
            self.current_round = int(resp.round)
            print(f"[{CLIENT_ID}] Bootstrapped backbone from server round={resp.round} (clf private).")
        except Exception as e:
            print(f"[{CLIENT_ID}] Bootstrap failed: {e}")

    def _pull_global(self):
        resp = self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])
        w = pickle.loads(resp.weights) if resp.weights else None
        bytes_recv = len(resp.weights) if resp.weights else 0
        return w, int(resp.round), int(bytes_recv)

    def run(self):
        for i in range(NUM_ROUNDS):
            print(f"\n[{CLIENT_ID}] ==== Loop {i+1}/{NUM_ROUNDS} | current_round={self.current_round} ====")

            # local train (clf updated locally too)
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
            local_mf1 = eval_and_log(self.model, self.Xte, self.yte, "local_before_sync", sent_round)

            # send ONLY backbone layers (NO clf)
            payload_dict = get_layers_payload(self.model, SHARED_LAYER_NAMES)
            payload = pickle.dumps(payload_dict, protocol=pickle.HIGHEST_PROTOCOL)
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

            if hasattr(ack, "status") and str(ack.status).startswith("ERROR"):
                print(f"[{CLIENT_ID}] Server rejected update: {ack.status} (server_round={ack.current_round})")
                try:
                    glob_w, r, _ = self._pull_global()
                    if isinstance(glob_w, dict):
                        set_layers_payload(self.model, glob_w)
                    self.current_round = int(r)
                except Exception as e:
                    print(f"[{CLIENT_ID}] Resync failed: {e}")
                continue

            target = sent_round + 1

            # barrier wait
            while True:
                time.sleep(BARRIER_POLL_SEC)
                glob_w, r_check, bytes_recv = self._pull_global()

                if int(r_check) >= target:
                    rtt = time.perf_counter() - t0

                    # apply global backbone only (clf stays as-is)
                    if isinstance(glob_w, dict):
                        set_layers_payload(self.model, glob_w)
                    else:
                        print(f"[{CLIENT_ID}] WARNING: global payload invalid; keep local.")

                    self.current_round = int(r_check)

                    # GLOBAL after sync
                    global_mf1 = eval_and_log(self.model, self.Xte, self.yte, "global_after_sync", self.current_round)

                    # NEW: summary row (after sync)
                    append_csv(SUMMARY_CSV, [
                        now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
                        int(self.current_round),
                        round(float(local_mf1), 6),
                        round(float(global_mf1), 6),
                    ], SUMMARY_HEADER)

                    # comm row
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
                    print(f"[{CLIENT_ID}] Waiting server... (server_round={r_check}, my_round={self.current_round})")

        eval_and_log(self.model, self.Xte, self.yte, "final", self.current_round, print_report=True)
        print(f"\n[{CLIENT_ID}] Saved metrics : {METRICS_CSV}")
        print(f"[{CLIENT_ID}] Saved comm    : {COMM_CSV}")
        print(f"[{CLIENT_ID}] Saved summary : {SUMMARY_CSV}")

if __name__ == "__main__":
    FedPerClient().run()