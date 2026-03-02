#!/usr/bin/env python3.10
# ============================================================
# client1_STANDALONE_CLIENT6_PRINT_FULL_METRICS_SAME_STYLE.py
#
# Standalone (NO federated sync, NO server usage).
#
# Changes to reflect your fixed PFTL style:
# 1) Deterministic seeding + SPLIT_SEED = SEED
# 2) SAME split protocol: Test=20% of total, Val=10% of total (val_frac=0.125)
# 3) SAME model backbone as your PFTL (Conv1D + MaxPool + GAP; no Flatten)
# 4) Correct class_weight mapping {class_value: weight}
# 5) Client6-style prints + confusion matrix + classification_report
# 6) CSV logs:
#    - METRICS_CSV: detailed metrics (phase=standalone_after_training per round, final)
#    - SUMMARY_CSV: one row per round with macro-F1 (so you can compare trends)
#    - COMM_CSV: exists (zeros) to keep file structure consistent (optional but handy)
#
# No plotting, no seaborn, no matplotlib.
# ============================================================

# ------- Reproducible seeding (TOP) -------
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
# ----------------------------------------

import time, csv
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

# -------------------------
# CONFIG
# -------------------------
CLIENT_ID      = "client1"
DATASET_NAME   = "CIC-IoT-2022"
METHOD         = "standalone"

NUM_ROUNDS   = 8
BATCH_SIZE   = 256
LOCAL_EPOCHS = 1

CONV_FILTERS = 4
PRIVATE_DIM  = 4
SHARED_DIM   = 4
LR = 1e-3

DATA_PATH = "/Users/azizahalq/Desktop/PFTL_Binary/Datasets_processed/D1_CIC_IOT_2022/CIC-IoT-V2_Balanced_16k+16k.csv"
LABEL_COL = "binary_label"

TEST_SIZE = 0.20
VAL_SIZE  = 0.10  # of total

# -------------------------
# LOGGING
# -------------------------
LOG_DIR = os.path.join("logs", METHOD)
os.makedirs(LOG_DIR, exist_ok=True)

METRICS_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_metrics.csv")
SUMMARY_CSV = os.path.join(LOG_DIR, f"{CLIENT_ID}_macro_f1_by_round.csv")
COMM_CSV    = os.path.join(LOG_DIR, f"{CLIENT_ID}_comm.csv")  # will be zeros for standalone

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

SUMMARY_HEADER = [
    "timestamp","method","dataset","seed","client",
    "round","macro_f1"
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
# Data (same split protocol)
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
    X_train = X_train.astype("float32").reshape((-1, nf, 1))
    X_val   = X_val.astype("float32").reshape((-1, nf, 1))
    X_test  = X_test.astype("float32").reshape((-1, nf, 1))

    return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values, nf

# -------------------------
# Model (match your PFTL backbone)
# -------------------------
def build_model(input_steps):
    ki = initializers.GlorotUniform(seed=SEED)
    bi = initializers.Zeros()

    return models.Sequential([
        layers.Input(shape=(input_steps, 1)),
        layers.Conv1D(CONV_FILTERS, 5, activation="relu", padding="same", name="input_adapter",
                      kernel_initializer=ki, bias_initializer=bi),
        layers.MaxPooling1D(2, name="pool"),
        layers.GlobalAveragePooling1D(name="gap"),
        layers.Dense(PRIVATE_DIM, activation="relu", name="feat1",
                     kernel_initializer=ki, bias_initializer=bi),
        layers.Dense(SHARED_DIM, activation="relu", name="shared_dense",
                     kernel_initializer=ki, bias_initializer=bi),
        layers.Dense(1, activation="sigmoid", name="clf",
                     kernel_initializer=ki, bias_initializer=bi),
    ])

# -------------------------
# Metrics + logging (same style as your PFTL eval_and_log)
# -------------------------
def eval_and_log(model, X, y, phase, rnd, print_report=False):
    probs = model.predict(X, verbose=0).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.5)
    y_pred = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y, y_pred))

    mp, mr, mf1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)
    bp, br, bf1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)

    p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(y, y_pred, labels=[0, 1], average=None, zero_division=0)
    prec0, rec0, f10 = float(p_cls[0]), float(r_cls[0]), float(f_cls[0])
    prec1, rec1, f11 = float(p_cls[1]), float(r_cls[1]), float(f_cls[1])

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    print(f"\n[{CLIENT_ID}] ===== {phase.upper()} | round={rnd} =====")
    print(f"Accuracy        : {acc:.6f}")
    print(f"Macro-Precision : {mp:.6f} | Macro-Recall : {mr:.6f} | Macro-F1 : {mf1:.6f}")
    print(f"Weighted-Prec   : {wp:.6f} | Weighted-Rec : {wr:.6f} | Weighted-F1 : {wf1:.6f}")
    print(f"Binary(POS=1)   : P={bp:.6f} | R={br:.6f} | F1={bf1:.6f}")
    print(f"Class0          : P0={prec0:.6f} | R0={rec0:.6f} | F1_0={f10:.6f}")
    print(f"Class1          : P1={prec1:.6f} | R1={rec1:.6f} | F1_1={f11:.6f}")
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

    return float(mf1)

# -------------------------
# Standalone runner
# -------------------------
class StandaloneClient:
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

        ensure_csv(METRICS_CSV, METRICS_HEADER)
        ensure_csv(SUMMARY_CSV, SUMMARY_HEADER)
        ensure_csv(COMM_CSV, COMM_HEADER)

    def run(self):
        for i in range(NUM_ROUNDS):
            print(f"\n[{CLIENT_ID}] ==== Loop {i+1}/{NUM_ROUNDS} ====")

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

            mf1 = eval_and_log(self.model, self.Xte, self.yte, "standalone_after_training", i)

            append_csv(SUMMARY_CSV, [
                now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
                int(i),
                round(float(mf1), 6),
            ], SUMMARY_HEADER)

            # keep comm file structure consistent (zeros)
            append_csv(COMM_CSV, [
                now_ts(), METHOD, DATASET_NAME, SEED, CLIENT_ID,
                int(i),
                0, 0, 0.0, round(float(train_time), 6),
                int(i)
            ], COMM_HEADER)

        eval_and_log(self.model, self.Xte, self.yte, "final", NUM_ROUNDS, print_report=True)

        print("\n===== SAVED (CSV ONLY) =====")
        print("Metrics CSV :", METRICS_CSV)
        print("Summary CSV :", SUMMARY_CSV)
        print("Comm CSV    :", COMM_CSV)

if __name__ == "__main__":
    StandaloneClient().run()