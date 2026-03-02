#!/usr/bin/env python3.10
# ============================================================
# STANDALONE (CORRECT TEMPLATE)
# - ONE SEED only (SEED controls EVERYTHING)
# - SAME split: Test=20%, Val=10% of total (val_frac=0.125)
# - Correct class_weight mapping
# - Client6-style printing (NO plots)
# ============================================================

# ===================== Reproducibility (TOP) =====================
import os, random
SEED = 123
SPLIT_SEED = SEED  #  keep split seed tied to SEED (single-source-of-truth)

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# ================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from tensorflow.keras import models, layers, optimizers, initializers

# -------------------- Config (EDIT THESE) --------------------
CSV_PATH  = "/Users/azizahalq/Desktop/PFTL_Binary/Datasets_processed/D3_CICIOT2023/cic_iot-2023_merged_32000.csv"
LABEL_COL = "binary_label"

CONV_FILTERS = 4
PRIVATE_DIM  = 4
SHARED_DIM   = 4

BATCH_SIZE = 256
EPOCHS     = 8
LR         = 1e-3

TEST_SIZE = 0.20
VAL_SIZE  = 0.10  # of total
# -----------------------------------------------------------

# -------------------- Load --------------------
df = pd.read_csv(CSV_PATH, low_memory=False, encoding="ISO-8859-1")
df.columns = df.columns.astype(str).str.strip()

if LABEL_COL not in df.columns:
    raise ValueError(f"LABEL_COL='{LABEL_COL}' not found. Sample cols: {df.columns.tolist()[:25]}")

y = df[LABEL_COL].astype(int).to_numpy()

X = df.drop(columns=[LABEL_COL], errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

# -------------------- SAME split (stratified) --------------------
# 1) Test = 20% of total
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SPLIT_SEED,
    stratify=y
)

# 2) Val = 10% of total => val_frac of train_full (train_full is 80%)
val_frac = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.125
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=val_frac,
    random_state=SPLIT_SEED,
    stratify=y_train_full
)

print("Shapes -> Train:", X_train.shape, " Val:", X_val.shape, " Test:", X_test.shape)
print("Label mean -> Train:", float(np.mean(y_train)), " Val:", float(np.mean(y_val)), " Test:", float(np.mean(y_test)))

# -------------------- Class Weight (✅ correct mapping) --------------------
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
CLASS_WEIGHTS = {int(c): float(w) for c, w in zip(classes, cw)}
print("class_weight =", CLASS_WEIGHTS)

# -------------------- Scale (fit on train only) --------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Reshape for Conv1D
n_features = X_train.shape[1]
X_train = X_train.astype("float32").reshape((-1, n_features, 1))
X_val   = X_val.astype("float32").reshape((-1, n_features, 1))
X_test  = X_test.astype("float32").reshape((-1, n_features, 1))

# -------------------- Model --------------------
def build_model(input_steps: int):
    ki = initializers.GlorotUniform(seed=SEED)
    bi = initializers.Zeros()

    return models.Sequential([
        layers.Input(shape=(input_steps, 1)),

        layers.Conv1D(
            CONV_FILTERS, 5, activation="relu", padding="same",
            name="input_adapter", kernel_initializer=ki, bias_initializer=bi
        ),
        layers.MaxPooling1D(2),
        layers.GlobalAveragePooling1D(),

        layers.Dense(
            PRIVATE_DIM, activation="relu", name="feat1",
            kernel_initializer=ki, bias_initializer=bi
        ),

        layers.Dense(
            SHARED_DIM, activation="relu", name="shared_dense",
            kernel_initializer=ki, bias_initializer=bi
        ),

        layers.Dense(
            1, activation="sigmoid", name="clf",
            kernel_initializer=ki, bias_initializer=bi
        ),
    ])

model = build_model(n_features)
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
)

model.summary()

# -------------------- Train --------------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    shuffle=True,
    class_weight=CLASS_WEIGHTS
)

# -------------------- Client6-style printing --------------------
def client6_print(y_true, probs, title: str):
    probs = np.nan_to_num(np.asarray(probs).reshape(-1), nan=0.5)
    y_pred = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    mp, mr, mf1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n===== {title} =====")
    print(f"Accuracy        : {acc:.6f}")
    print(f"Macro-Precision : {float(mp):.6f}")
    print(f"Macro-Recall    : {float(mr):.6f}")
    print(f"Macro-F1        : {float(mf1):.6f}")

    print("\n===== WEIGHTED (support-weighted) METRICS =====")
    print(f"Weighted-Precision : {float(wp):.6f}")
    print(f"Weighted-Recall    : {float(wr):.6f}")
    print(f"Weighted-F1        : {float(wf1):.6f}")

    print("\n==== Classification Report ====")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\n==== Confusion Matrix ====")
    print(cm)

# -------------------- Evaluate --------------------
pred_train = model.predict(X_train, verbose=0)
pred_val   = model.predict(X_val,   verbose=0)
pred_test  = model.predict(X_test,  verbose=0)

client6_print(y_train, pred_train, "TRAIN EVALUATION (standalone)")
client6_print(y_val,   pred_val,   "VAL EVALUATION (standalone)")
client6_print(y_test,  pred_test,  "FINAL TEST EVALUATION (standalone)")