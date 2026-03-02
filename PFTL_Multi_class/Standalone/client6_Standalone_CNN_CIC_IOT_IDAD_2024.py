#!/usr/bin/env python3.10
# ============================================================
# Standalone_CNN_CLIENT6_IDAD_SPLIT70_15_15_MATCH_PTFL.py
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tensorflow.keras import layers, models, optimizers

# --- Paths ---
CSV_PATH  = "/Users/azizahalq/Desktop/PFTL_project3/Datasets_processed/D6_CIC_IOT_IDAD_2024/Client6_Balanced_Final.csv"
LABEL_COL = "Label"

OUT_DIR = "/Users/azizahalq/Desktop/PFTL_project3/standalone_outputs/client6_idad_logs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Split rule (70/15/15) ---
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)  # 0.17647058823529413

# --- Train config ---
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3

PRIVATE_DIM = 16
SHARED_DIM  = 8

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

    # must match PTFL: Conv1D(64) in the 3rd conv (as your client6 design)
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

    model = models.Model(inp, out, name="Standalone_Client6_CNN")
    model.compile(optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0), loss=loss, metrics=metrics)
    return model

def main():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    y_raw = (
        df[LABEL_COL].astype(str)
          .str.replace("\ufeff", "", regex=False)
          .str.replace("�", "-", regex=False)
          .str.replace(r"\s+", " ", regex=True)
          .str.strip()
          .values
    )

    X_df = df.drop(columns=[LABEL_COL], errors="ignore").select_dtypes(include=[np.number]).copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    num_classes = int(len(le.classes_))

    X = X_df.values.astype(np.float32)

    #  unified split rule
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE_FROM_TRAIN, random_state=SEED, stratify=y_train
    )

    # preprocessing (keep like other clients)
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_val   = imp.transform(X_val)
    X_test  = imp.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)[..., np.newaxis]
    X_val   = scaler.transform(X_val)[..., np.newaxis]
    X_test  = scaler.transform(X_test)[..., np.newaxis]

    model = build_cnn((X_train.shape[1], 1), num_classes)

    print("\n===== TRAINING STANDALONE (CLIENT 6) =====")
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    probs = model.predict(X_test, verbose=0)
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
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("\n==== Confusion Matrix ====")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
