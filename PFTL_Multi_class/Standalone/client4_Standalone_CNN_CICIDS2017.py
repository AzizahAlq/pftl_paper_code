#!/usr/bin/env python3.10
# ============================================================
# Standalone_CNN_CLIENT4_CICIDS2017_SPLIT70_15_15_LOGS.py
# ============================================================

import os, time, csv, random
import numpy as np
import pandas as pd
from datetime import datetime

SEED = 190
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tensorflow.keras import layers, models, optimizers, callbacks

# -------------------------
# Data / Paths
# -------------------------
CSV_PATH  = "/Users/azizahalq/Desktop/PFTL_project3/Datasets_processed/D4_CICIDS2017/Merged_ALL_CIC_IDS_2017_TOTAL_200K.csv"
LABEL_COL = "Label"

OUT_DIR = "/Users/azizahalq/Desktop/PFTL_project3/standalone_outputs/client4_like_logs"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCH_LOG_CSV   = os.path.join(OUT_DIR, "standalone_client4_epoch_log.csv")
RUN_SUMMARY_CSV = os.path.join(OUT_DIR, "standalone_client4_run_summary.csv")

# ✅ Unified split rule (70/15/15)
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)  # 0.17647058823529413

# Train config
EPOCHS = 20
BATCH_SIZE = 2048
LR = 1e-3

PRIVATE_DIM = 16
SHARED_DIM  = 8

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_csv_headers():
    if not os.path.exists(EPOCH_LOG_CSV):
        with open(EPOCH_LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch","train_loss","train_acc","val_loss","val_acc",
                "val_macro_precision","val_macro_recall","val_macro_f1",
                "val_weighted_f1","epoch_time_sec","total_time_sec","timestamp"
            ])

    if not os.path.exists(RUN_SUMMARY_CSV):
        with open(RUN_SUMMARY_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp","seed","samples_total","features","classes",
                "train_n","val_n","test_n",
                "test_accuracy","test_macro_precision","test_macro_recall","test_macro_f1",
                "test_weighted_precision","test_weighted_recall","test_weighted_f1",
                "total_train_time_sec"
            ])

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
        epoch_time = time.perf_counter() - self.t0_epoch
        total_time = time.perf_counter() - self.t0_all

        probs = self.model.predict(self.X_val, batch_size=2048, verbose=0)
        if self.num_classes == 2:
            y_pred = (probs.reshape(-1) >= 0.5).astype(int)
        else:
            y_pred = np.argmax(probs, axis=1)

        mp, mr, mf1, _ = precision_recall_fscore_support(self.y_val, y_pred, average="macro", zero_division=0)
        _, _, wf1, _ = precision_recall_fscore_support(self.y_val, y_pred, average="weighted", zero_division=0)

        with open(EPOCH_LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                int(epoch+1),
                f"{float(logs.get('loss', np.nan)):.6f}",
                f"{float(logs.get('accuracy', np.nan)):.6f}",
                f"{float(logs.get('val_loss', np.nan)):.6f}",
                f"{float(logs.get('val_accuracy', np.nan)):.6f}",
                f"{mp:.6f}", f"{mr:.6f}", f"{mf1:.6f}", f"{wf1:.6f}",
                f"{epoch_time:.4f}", f"{total_time:.4f}", now_str()
            ])

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

    x = layers.Dense(PRIVATE_DIM, activation="relu", name="private_dense")(x)
    x = layers.Dense(SHARED_DIM,  activation="relu", name="shared_dense")(x)

    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid", name="y_out")(x)
        loss = "binary_crossentropy"
        metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    else:
        out = layers.Dense(num_classes, activation="softmax", name="y_out")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model = models.Model(inp, out, name="Standalone_Client4_CNN")
    model.compile(optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0), loss=loss, metrics=metrics)
    return model

def main():
    ensure_csv_headers()

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found. Available columns: {list(df.columns)}")

    y_raw = (
        df[LABEL_COL].astype(str)
          .str.replace("\ufeff", "", regex=False)
          .str.replace("�", "-", regex=False)
          .str.replace(r"\s+", " ", regex=True)
          .str.strip()
          .values
    )

    X_df = df.drop(columns=[LABEL_COL], errors="ignore").select_dtypes(include=[np.number]).copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X_df.values.astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    num_classes = int(len(le.classes_))

    #  unified split rule
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE_FROM_TRAIN, random_state=SEED, stratify=y_train
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)[..., np.newaxis]
    X_val   = scaler.transform(X_val)[..., np.newaxis]
    X_test  = scaler.transform(X_test)[..., np.newaxis]

    #  correct class_weight mapping
    classes_idx = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes_idx, y=y_train)
    class_weight = {int(i): float(w) for i, w in zip(classes_idx, cw)}

    model = build_cnn((X_train.shape[1], 1), num_classes)

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
    total_train_time = time.perf_counter() - t0

    probs = model.predict(X_test, batch_size=2048, verbose=0)
    y_pred = (probs.reshape(-1) >= 0.5).astype(int) if num_classes == 2 else np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    mp, mr, mf1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print("\n===== STANDALONE RESULTS (NO SAVING artifacts) =====")
    print(f"Accuracy        : {acc:.6f}")
    print(f"Macro-Precision : {mp:.6f}")
    print(f"Macro-Recall    : {mr:.6f}")
    print(f"Macro-F1        : {mf1:.6f}")

    print("\n===== WEIGHTED (support-weighted) METRICS =====")
    print(f"Weighted-Precision : {wp:.6f}")
    print(f"Weighted-Recall    : {wr:.6f}")
    print(f"Weighted-F1        : {wf1:.6f}")

    print("\n==== Classification Report ====")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0))

    print("\n==== Confusion Matrix ====")
    print(confusion_matrix(y_test, y_pred))

    with open(RUN_SUMMARY_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            now_str(), SEED, len(X), X.shape[1], num_classes,
            len(X_train), len(X_val), len(X_test),
            f"{acc:.6f}", f"{mp:.6f}", f"{mr:.6f}", f"{mf1:.6f}",
            f"{wp:.6f}", f"{wr:.6f}", f"{wf1:.6f}", f"{total_train_time:.4f}"
        ])

    print(f"\nDONE. Logs saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
