#!/usr/bin/env python3.10
# ============================================================
# client2_PFTL_share_shared_dense_gamma_PRINT_SAME_AS_CLIENT6_FIXED.py
# ============================================================

# ------- Reproducible seeding -------
import os, random
SEED = 190
SPLIT_SEED = SEED  #  ONE source-of-truth for splits

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
# ------------------------------------

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
CLIENT_ID      = "client2"
DATASET_NAME   = "CIC-BCCC-NRC-2024"

NUM_ROUNDS   = 8
BATCH_SIZE   = 256
LOCAL_EPOCHS = 1
SLEEP_POLL   = 1.0

PRIVATE_DIM  = 4
SHARED_DIM   = 4
CONV_FILTERS = 4
LR = 1e-3

# ---- PFTL gamma blending ----
GAMMA_GLOBAL = 0.3
GAMMA_LOCAL  = 1.0 - GAMMA_GLOBAL

SHARED_LAYER_NAME = "shared_dense"

DATA_PATH = "/Users/azizahalq/Desktop/PFTL_Binary/Datasets_processed/D2_CIC-BCCC-NRC_2024/CIC_TabularIoT_Balanced_16k+16k_no_labels.csv"
LABEL_COL = "binary_label"

TEST_SIZE = 0.20          # 20% test (of total)
VAL_SIZE  = 0.10          # 10% val (of total)

# -------------------------
# LOGGING (folder)
# -------------------------
METHOD = f"pftl_gamma_{GAMMA_GLOBAL}"
LOG_DIR = os.path.join("logs", METHOD)
os.makedirs(LOG_DIR, exist_ok=True)

COMM_CSV    = os.path.join(LOG_DIR, f"{CLIENT_ID}_comm.csv")
METRICS_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_metrics.csv")

#  NEW: one row per round (local vs global macro-f1)
SUMMARY_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_local_global_macro_f1_by_round.csv")
SUMMARY_HEADER = ["timestamp","method","dataset","seed","client","round","local_macro_f1","global_macro_f1"]

COMM_HEADER = [
    "timestamp","method","dataset","seed","client",
    "round",
    "bytes_sent","bytes_recv","rtt_sec","server_round_after"
]

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
# Data (SAME SPLIT)
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

    # 1) Test = 20% of total
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )

    # 2) Val = 10% of total => 0.125 of trainval
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
    return models.Sequential([
        layers.Input(shape=(input_steps, 1)),
        layers.Conv1D(CONV_FILTERS, 5, activation="relu", padding="same",
                      kernel_initializer=ki, bias_initializer=bi),
        layers.MaxPooling1D(2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(PRIVATE_DIM, activation="relu", kernel_initializer=ki, bias_initializer=bi),
        layers.Dense(SHARED_DIM, activation="relu", name=SHARED_LAYER_NAME, kernel_initializer=ki, bias_initializer=bi),
        layers.Dense(1, activation="sigmoid", name="clf", kernel_initializer=ki, bias_initializer=bi),
    ])

# -------------------------
# Shared helpers
# -------------------------
def is_valid_shared(w) -> bool:
    return isinstance(w, (list, tuple)) and len(w) == 2

def get_shared_layer(model):
    return model.get_layer(SHARED_LAYER_NAME).get_weights()

def set_shared_layer(model, w):
    if not is_valid_shared(w):
        print(f"[{CLIENT_ID}] WARNING: shared weights invalid format, skipping.")
        return
    if any(np.isnan(x).any() for x in w):
        print(f"[{CLIENT_ID}] WARNING: NaN in shared weights, skipping.")
        return
    model.get_layer(SHARED_LAYER_NAME).set_weights(list(w))

def blend_shared(local_w, global_w, gamma_local, gamma_global):
    if len(local_w) != len(global_w):
        raise ValueError("local/global shared weights length mismatch")
    return [gamma_local * l + gamma_global * g for l, g in zip(local_w, global_w)]

# -------------------------
# Metrics
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

# -------------------------
# Client
# -------------------------
class PFTLClient:
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

        ensure_csv(COMM_CSV, COMM_HEADER)
        ensure_csv(METRICS_CSV, METRICS_HEADER)
        ensure_csv(SUMMARY_CSV, SUMMARY_HEADER)

        # Bootstrap: pull server global (if valid) and blend into local
        try:
            resp = self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])
            srv_round = int(getattr(resp, "round", 0))
            self.current_round = srv_round

            global_shared = None
            if getattr(resp, "weights", None):
                try:
                    global_shared = pickle.loads(resp.weights)
                except Exception:
                    global_shared = None

            if is_valid_shared(global_shared):
                local_shared = get_shared_layer(self.model)
                blended = blend_shared(local_shared, global_shared, GAMMA_LOCAL, GAMMA_GLOBAL)
                set_shared_layer(self.model, blended)
                print(f"[{CLIENT_ID}] Aggregator reachable (server_round={self.current_round})")
            else:
                print(f"[{CLIENT_ID}] Aggregator reachable (server_round={self.current_round}) but no valid global yet.")
        except Exception as e:
            print(f"[{CLIENT_ID}] Bootstrap failed: {e}")

    def _ack_ok(self, ack) -> bool:
        # supports both proto styles:
        # 1) Ack(ok=bool)
        if hasattr(ack, "ok"):
            return bool(ack.ok)
        # 2) Ack(status="OK"/"WAITING"/"ERROR:..")
        if hasattr(ack, "status"):
            s = str(ack.status).upper()
            return s.startswith("OK") or s.startswith("WAIT")
        return True

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

    def run(self):
        for loop_i in range(NUM_ROUNDS):
            print(f"\n[{CLIENT_ID}] ===== Loop {loop_i+1}/{NUM_ROUNDS} (server_round={self.current_round}) =====")

            server_round_before = int(self.current_round)

            # Local train (round r)
            self.model.fit(
                self.Xtr, self.ytr,
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                shuffle=True,
                validation_data=(self.Xva, self.yva),
                class_weight=self.class_weights
            )

            # local after training
            m_local = compute_metrics(self.model, self.Xte, self.yte)
            log_metrics("local_after_training", server_round_before, m_local)

            payload = pickle.dumps(get_shared_layer(self.model))
            bytes_sent = len(payload)
            t0 = time.perf_counter()

            # send update for round r
            ack = self.stub.SendSharedUpdate(
                myproto_pb2.SharedUpdate(
                    weights=payload,
                    round=int(server_round_before),
                    num_samples=int(self.Xtr.shape[0])
                ),
                metadata=[("client_id", CLIENT_ID)]
            )

            if not self._ack_ok(ack):
                print(f"[{CLIENT_ID}] Update rejected. Resyncing round...")
                try:
                    glob_w, r, _ = self._pull_global()
                    self.current_round = int(r)
                    if is_valid_shared(glob_w):
                        local_shared = get_shared_layer(self.model)
                        blended = blend_shared(local_shared, glob_w, GAMMA_LOCAL, GAMMA_GLOBAL)
                        set_shared_layer(self.model, blended)
                except Exception as e:
                    print(f"[{CLIENT_ID}] Resync failed: {e}")
                continue

            # STRICT barrier: wait until server_round >= r+1
            target = server_round_before + 1

            while True:
                time.sleep(SLEEP_POLL)
                glob_w, r_check, bytes_recv = self._pull_global()

                if int(r_check) >= target:
                    rtt = time.perf_counter() - t0

                    # apply TRUE global after sync (optionally blend with local)
                    if is_valid_shared(glob_w):
                        local_shared = get_shared_layer(self.model)
                        blended = blend_shared(local_shared, glob_w, GAMMA_LOCAL, GAMMA_GLOBAL)
                        set_shared_layer(self.model, blended)
                    else:
                        print(f"[{CLIENT_ID}] WARNING: invalid server global at r={r_check}; keeping local shared.")

                    self.current_round = int(r_check)

                    # global after sync
                    m_global = compute_metrics(self.model, self.Xte, self.yte)
                    log_metrics("global_after_sync", self.current_round, m_global)

                    #  one-row summary CSV
                    append_csv(SUMMARY_CSV, [
                        now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
                        int(self.current_round),
                        round(float(m_local["mf1"]), 6),
                        round(float(m_global["mf1"]), 6),
                    ], SUMMARY_HEADER)

                    # comm log
                    append_csv(COMM_CSV, [
                        now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
                        int(server_round_before),
                        int(bytes_sent),
                        int(bytes_recv),
                        round(rtt, 6),
                        int(self.current_round)
                    ], COMM_HEADER)

                    print(f"[{CLIENT_ID}] Barrier passed -> server_round={self.current_round}")
                    break
                else:
                    print(f"[{CLIENT_ID}] Waiting server... (server_round={r_check}, target={target})")

        m = compute_metrics(self.model, self.Xte, self.yte)
        log_metrics("final", self.current_round, m)
        print_client6_style("FINAL TEST EVALUATION", m, self.yte)

        print("\n===== SAVED (CSV ONLY) =====")
        print("Metrics CSV:", METRICS_CSV)
        print("Comm CSV   :", COMM_CSV)
        print("Summary CSV:", SUMMARY_CSV)

if __name__ == "__main__":
    PFTLClient().run()