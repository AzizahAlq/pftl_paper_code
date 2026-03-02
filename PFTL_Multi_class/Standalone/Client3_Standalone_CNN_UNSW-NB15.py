#!/usr/bin/env python3.10
import os, time, csv, random
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------
# Reproducibility
# -------------------------
SEED = 190
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tensorflow.keras import layers, models, optimizers

# -------------------------
# Config
# -------------------------
CSV_PATH = "/Users/azizahalq/Desktop/PFTL_project3/Datasets_processed/D3_ UNSW_NB15/UNSW_NB15/UNSW_NB15_ALL_MERGED_final.csv"
TARGET_COL = "attack_cat"

OUT_DIR = "/Users/azizahalq/Desktop/PFTL_project3/standalone_logs/client3_unsw"
os.makedirs(OUT_DIR, exist_ok=True)
METRICS_LOG = os.path.join(OUT_DIR, "standalone_metrics_log.csv")

EPOCHS = 15
BATCH_SIZE = 512
LR = 1e-3

# ---- MATCH Client1/2 split rule ----
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)  # 0.17647058823529413

def build_cnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding="valid", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="valid", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu", name="private_dense")(x)
    x = layers.Dense(8, activation="relu", name="shared_dense")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    # keep same numeric handling style
    df = df.replace([np.inf, -np.inf], np.nan)

    y_raw = df[TARGET_COL].astype(str).str.strip().values
    X_df = df.drop(columns=[TARGET_COL, "id", "label"], errors="ignore") \
             .select_dtypes(include=[np.number]).fillna(0.0)

    X = X_df.values.astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    num_classes = int(len(le.classes_))

    # ---- MATCH Client1/2 split rule ----
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

    model = build_cnn((X_train.shape[1], 1), num_classes)

    if not os.path.exists(METRICS_LOG):
        with open(METRICS_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "val_macro_f1", "timestamp"])

    print("\n===== TRAINING STANDALONE (CLIENT 3) =====")
    for epoch in range(1, EPOCHS + 1):
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=1, batch_size=BATCH_SIZE, verbose=1)

        y_v_hat = np.argmax(model.predict(X_val, verbose=0), axis=1)
        _, _, mf1, _ = precision_recall_fscore_support(y_val, y_v_hat, average="macro", zero_division=0)

        with open(METRICS_LOG, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{mf1:.6f}", datetime.now()])

    # Final Evaluation
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
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

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("\n==== Confusion Matrix ====")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
