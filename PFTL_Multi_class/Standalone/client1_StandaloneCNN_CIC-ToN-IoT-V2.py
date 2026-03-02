#!/usr/bin/env python3.10
# ============================================================
# Standalone_CNN_CLIENT1_TON_IOT_MATCH_PTFL_FIXED.py
# ============================================================
# FIXED to truly match your PTFL client style:
#  Same cleaning logic (fixes the broken .str.replace("", "-", ...))
#  Same split logic (70/15/15 via TEST=0.15, VAL_FROM_TRAIN=0.15/0.85)
#  Same model architecture + SAME layer names: private_dense / shared_dense
#  Same optimizer style (Adam + clipnorm=1.0)
#  Correct class_weight mapping (was WRONG before: enumerate(cw))
#  Robust metric key handling in logger (binary vs sparse accuracy keys)
#  Epoch logger computes VAL macro/weighted metrics exactly like PTFL style
# ============================================================

import os
import time
import csv
import random
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------
# Reproducibility (Matches PTFL)
# -------------------------
SEED = int(os.environ.get("SEED", "190"))
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from tensorflow.keras import layers, models, optimizers, callbacks

# -------------------------
# Settings (Matches PTFL Client1)
# -------------------------
CSV_PATH  = os.environ.get(
    "CSV_PATH",
    "/Users/azizahalq/Desktop/PFTL_project3/Datasets_processed/D1_D2_CIC_IOT_and_CIC_TON_Dataset2023/CIC-ToN-IoT_new.csv"
)
LABEL_COL = os.environ.get("LABEL_COL", "Attack")

OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/Users/azizahalq/Desktop/PFTL_project3/standalone_outputs/client1_ton_iot_logs"
)
os.makedirs(OUT_DIR, exist_ok=True)

EPOCH_LOG_CSV   = os.path.join(OUT_DIR, "standalone_client1_epoch_log.csv")
RUN_SUMMARY_CSV = os.path.join(OUT_DIR, "standalone_client1_run_summary.csv")

# PTFL Matching Splits (70/15/15)
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.15"))
VAL_SIZE_FROM_TRAIN = float(os.environ.get("VAL_SIZE_FROM_TRAIN", str(0.15 / (1.0 - TEST_SIZE))))  # 0.176470588...

# Hyperparams
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1024"))
LR = float(os.environ.get("LR", "1e-3"))
PRIVATE_DIM = int(os.environ.get("PRIVATE_DIM", "16"))
SHARED_DIM  = int(os.environ.get("SHARED_DIM", "8"))

# -------------------------
# Helpers
# -------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_csv_headers():
    if not os.path.exists(EPOCH_LOG_CSV):
        with open(EPOCH_LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "seed",
                "epoch",
                "train_loss", "train_acc",
                "val_loss", "val_acc",
                "val_macro_precision", "val_macro_recall", "val_macro_f1",
                "val_weighted_precision", "val_weighted_recall", "val_weighted_f1",
                "epoch_time_sec", "total_time_sec",
                "timestamp"
            ])

    if not os.path.exists(RUN_SUMMARY_CSV):
        with open(RUN_SUMMARY_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "seed",
                "samples_total", "features", "classes",
                "train_n", "val_n", "test_n",
                "test_accuracy",
                "test_macro_precision", "test_macro_recall", "test_macro_f1",
                "test_weighted_precision", "test_weighted_recall", "test_weighted_f1",
                "total_train_time_sec"
            ])


def _clean_label_series(s: pd.Series) -> np.ndarray:
    # EXACTLY the same spirit as your PTFL clients
    return (
        s.astype(str)
         .str.replace("\ufeff", "", regex=False)
         .str.replace("�", "-", regex=False)   #  FIXED (was "" which is wrong)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
         .values
    )


def predict_labels(model, X_batch, num_classes: int):
    probs = model.predict(X_batch, batch_size=1024, verbose=0)
    if int(num_classes) == 2:
        # probs could be (N,1) or (N,) depending on TF
        probs = probs.reshape(-1)
        return (probs >= 0.5).astype(int)
    return np.argmax(probs, axis=1)


class EpochCSVLogger(callbacks.Callback):
    def __init__(self, X_val, y_val, num_classes: int):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = int(num_classes)
        self.t0_all = None
        self.t0_epoch = None

    def on_train_begin(self, logs=None):
        self.t0_all = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.t0_epoch = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = float(time.perf_counter() - self.t0_epoch)
        total_time = float(time.perf_counter() - self.t0_all)

        y_pred = predict_labels(self.model, self.X_val, self.num_classes)

        mp, mr, mf1, _ = precision_recall_fscore_support(
            self.y_val, y_pred, average="macro", zero_division=0
        )
        wp, wr, wf1, _ = precision_recall_fscore_support(
            self.y_val, y_pred, average="weighted", zero_division=0
        )

        # Robust metric key handling (BinaryAccuracy vs SparseCategoricalAccuracy)
        train_acc = logs.get("accuracy", logs.get("binary_accuracy", np.nan))
        val_acc   = logs.get("val_accuracy", logs.get("val_binary_accuracy", np.nan))

        with open(EPOCH_LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                int(SEED),
                int(epoch + 1),
                f"{float(logs.get('loss', np.nan)):.6f}",
                f"{float(train_acc) if train_acc == train_acc else np.nan:.6f}",
                f"{float(logs.get('val_loss', np.nan)):.6f}",
                f"{float(val_acc) if val_acc == val_acc else np.nan:.6f}",
                f"{float(mp):.6f}", f"{float(mr):.6f}", f"{float(mf1):.6f}",
                f"{float(wp):.6f}", f"{float(wr):.6f}", f"{float(wf1):.6f}",
                f"{epoch_time:.4f}",
                f"{total_time:.4f}",
                now_str(),
            ])


# -------------------------
# Model (MATCH PTFL layer names + optimizer style)
# -------------------------
def build_cnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 3, padding="valid", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, 3, padding="valid", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, 3, padding="valid", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)

    #  MUST match PTFL naming
    x = layers.Dense(PRIVATE_DIM, activation="relu", name="private_dense")(x)
    x = layers.Dense(SHARED_DIM,  activation="relu", name="shared_dense")(x)

    if int(num_classes) == 2:
        out = layers.Dense(1, activation="sigmoid", name="y_out")(x)
        loss = "binary_crossentropy"
        metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    else:
        out = layers.Dense(int(num_classes), activation="softmax", name="y_out")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model = models.Model(inp, out, name="Standalone_Client1_CNN_Match_PTFL")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),  # ✅ match PTFL stability
        loss=loss,
        metrics=metrics
    )
    return model


def main():
    ensure_csv_headers()

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found. Available columns:\n{list(df.columns)}")

    #  PTFL matching cleaning
    y_raw = _clean_label_series(df[LABEL_COL])

    X_df = df.drop(columns=[LABEL_COL], errors="ignore").select_dtypes(include=[np.number]).copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X_df.values.astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    num_classes = int(len(le.classes_))

    # Splits: 70/15/15
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE_FROM_TRAIN, random_state=SEED, stratify=y_train
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)[..., np.newaxis]
    X_val   = scaler.transform(X_val).astype(np.float32)[..., np.newaxis]
    X_test  = scaler.transform(X_test).astype(np.float32)[..., np.newaxis]

    #  Correct class_weight mapping (your old code was WRONG)
    classes_idx = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes_idx, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes_idx, cw)}

    model = build_cnn((X_train.shape[1], 1), num_classes)

    print("\n==== Model Summary ====")
    model.summary()

    print("\n===== DATA SUMMARY =====")
    print("CSV:", CSV_PATH)
    print("Samples:", len(X))
    print("Features:", X.shape[1])
    print("Classes:", num_classes)
    print("Train/Val/Test:", len(X_train), len(X_val), len(X_test))
    print("Label names:", list(le.classes_))

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[EpochCSVLogger(X_val, y_val, num_classes)],
        verbose=1
    )
    total_train_time = float(time.perf_counter() - t0)

    # Test Evaluation
    y_pred = predict_labels(model, X_test, num_classes)

    acc = float(accuracy_score(y_test, y_pred))
    mp, mr, mf1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print("\n===== STANDALONE RESULTS =====")
    print(f"Accuracy        : {acc:.6f}")
    print(f"Macro-Precision : {mp:.6f}")
    print(f"Macro-Recall    : {mr:.6f}")
    print(f"Macro-F1        : {mf1:.6f}")

    print("\n===== WEIGHTED (support-weighted) METRICS =====")
    print(f"Weighted-Precision : {wp:.6f}")
    print(f"Weighted-Recall    : {wr:.6f}")
    print(f"Weighted-F1        : {wf1:.6f}")

    print("\n==== Classification Report ====")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("\n==== Confusion Matrix ====")
    print(confusion_matrix(y_test, y_pred))

    with open(RUN_SUMMARY_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            now_str(), int(SEED),
            int(len(X)), int(X.shape[1]), int(num_classes),
            int(len(X_train)), int(len(X_val)), int(len(X_test)),
            f"{acc:.6f}",
            f"{float(mp):.6f}", f"{float(mr):.6f}", f"{float(mf1):.6f}",
            f"{float(wp):.6f}", f"{float(wr):.6f}", f"{float(wf1):.6f}",
            f"{total_train_time:.4f}",
        ])

    print("\n===== SAVED (CSV ONLY) =====")
    print("Epoch log :", EPOCH_LOG_CSV)
    print("Summary   :", RUN_SUMMARY_CSV)


if __name__ == "__main__":
    main()
