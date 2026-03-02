#!/usr/bin/env python3.10
# ============================================================
# ONE_SHOT_TL_UNSW_NB15_MULTICLASS_FROM_SERVER_PKL.py
# ============================================================
# One-shot transfer learning (MULTICLASS) using server PKL snapshot:
#  - LabelEncoder fit once (full dataset) -> split
#  - Load global_shared from server PKL (global_shared_round_XXX.pkl)
#  - Transfer ONLY "shared_dense"
#  - Blend local+global via gamma=(local, global)
#  - Train locally and save:
#      - metrics CSV
#      - clean confusion matrix PNG (outer border only, no inner borders)
#      - model .keras
#
# Fixes:
#  - classification_report mismatch -> uses labels=np.arange(num_classes)
#  - round_id auto-detected from PKL filename if possible
# ============================================================

import os, time, random, pickle, re
from datetime import datetime

# -------------------------
# Reproducible seeding
# -------------------------
SEED = 32
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

# -------------------------
# Imports
# -------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

from tensorflow.keras import layers, models, optimizers


# ============================================================
# PATHS + SETTINGS
# ============================================================
CSV_PATH  = "/Users/azizahalq/Desktop/PFTL_project3/Datasets_processed/D3_ UNSW_NB15/UNSW_NB15/UNSW_NB15_testing-set.csv"
LABEL_COL = "attack_cat"

# Server PKL snapshot (use EXACT existing file)
SERVER_PKL = "/Users/azizahalq/Desktop/PFTL_project3/server_logs_pftl_shared_transfer/global_shared_latest.pkl"

OUT_DIR = "/Users/azizahalq/Desktop/PFTL_project3/one_shot_unsw_nb15_multiclass"
os.makedirs(OUT_DIR, exist_ok=True)

# One-shot settings
gamma = (1, 0)     # (local, global)
EPOCHS = 20
BATCH  = 1024
LR     = 1e-4          # ✅ safer for transfer

# PTFL matching splits (70/15/15)
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)

# model dims (must match server shared_dense shape)
PRIVATE_DIM = 16
SHARED_DIM  = 8


# ============================================================
# Helpers
# ============================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text_series(s: pd.Series) -> np.ndarray:
    return (
        s.astype(str)
         .str.replace("\ufeff", "", regex=False)
         .str.replace("�", "-", regex=False)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
         .values
    )


def infer_round_from_pkl(pkl_path: str, fallback: int = 0) -> int:
    """
    Try to infer round id from filename global_shared_round_XXX.pkl
    """
    m = re.search(r"global_shared_round_(\d+)\.pkl$", os.path.basename(pkl_path))
    if m:
        return int(m.group(1))
    return int(fallback)


def blend(local_w, global_w, gamma=(0.5, 0.5)):
    # local_w and global_w are [kernel, bias]
    return [gamma[0] * l + gamma[1] * g for l, g in zip(local_w, global_w)]


def predict_labels_multiclass(model, X, batch_size=1024):
    probs = model.predict(X, batch_size=batch_size, verbose=0)
    return np.argmax(probs, axis=1)


# ============================================================
# Clean CM plotting (outer border only)
# ============================================================
def plot_confusion_matrix_clean_multiclass(cm, labels, outpath):
    cm = np.asarray(cm, dtype=int)
    n = len(labels)

    fig_w = max(4, min(14, 0.35 * n + 3))
    fig_h = max(3, min(12, 0.35 * n + 2.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    sns.heatmap(
        cm,
        ax=ax,
        cmap="Blues",
        square=True,
        annot=False,         # multiclass: keep clean
        linewidths=0,        # ✅ no inner borders
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        cbar_kws={"pad": 0.08, "shrink": 1.0}
    )

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)

    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    # ✅ outer black border only
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)

    plt.tight_layout(pad=0.25)
    plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================
# Model (MATCH PTFL layer names: private_dense / shared_dense)
# ============================================================
def build_cnn_multiclass(input_shape, num_classes: int):
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

    out = layers.Dense(int(num_classes), activation="softmax", name="y_out")(x)

    model = models.Model(inp, out, name="OneShot_UNSW_NB15_Multiclass")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model


# ============================================================
# Load server global_shared PKL
# ============================================================
def load_server_shared_pkl(pkl_path: str):
    """
    Reads server snapshot PKL:
      .../global_shared_round_XXX.pkl
    Returns:
      global_shared = [kernel, bias]
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing server shared file: {pkl_path}")

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if not (isinstance(obj, dict) and "global_shared" in obj):
        raise ValueError("PKL format unexpected. Expected dict with key 'global_shared'.")

    global_shared = obj["global_shared"]
    if not isinstance(global_shared, list) or len(global_shared) < 2:
        raise ValueError("global_shared in PKL must be a list like [kernel, bias].")

    return global_shared


# ============================================================
# MAIN
# ============================================================
def main():
    round_id = infer_round_from_pkl(SERVER_PKL, fallback=0)

    print("\n===== ONE-SHOT TL (UNSW-NB15) MULTICLASS =====")
    print("Seed :", SEED)
    print("PKL  :", SERVER_PKL)
    print("Round:", round_id)
    print("Gamma:", gamma)

    # ---------- load ----------
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found. Columns: {list(df.columns)}")

    y_raw = clean_text_series(df[LABEL_COL])

    # numeric features only
    X_df = df.drop(columns=[LABEL_COL], errors="ignore").select_dtypes(include=[np.number]).copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X_df.values.astype(np.float32)

    # ---------- encode labels ONCE ----------
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    class_names = list(le.classes_)
    num_classes = int(len(class_names))
    labels_all = np.arange(num_classes)  # ✅ stable labels for report
    print("Num classes:", num_classes)

    # ---------- splits 70/15/15 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE_FROM_TRAIN, random_state=SEED, stratify=y_train
    )

    # ---------- scale + reshape ----------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)[..., np.newaxis]
    X_val   = scaler.transform(X_val).astype(np.float32)[..., np.newaxis]
    X_test  = scaler.transform(X_test).astype(np.float32)[..., np.newaxis]

    # ---------- class weights (correct mapping) ----------
    classes_unique = np.unique(y_train)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes_unique,
        y=y_train
    )
    cw = {int(c): float(wi) for c, wi in zip(classes_unique, weights)}

    # ---------- load GLOBAL shared_dense from server PKL ----------
    global_shared = load_server_shared_pkl(SERVER_PKL)
    print("Loaded global_shared from:", SERVER_PKL)

    # ---------- build local model + blend shared_dense ----------
    model = build_cnn_multiclass((X_train.shape[1], 1), num_classes=num_classes)
    local_shared = model.get_layer("shared_dense").get_weights()

    # sanity check: shapes must match
    if len(local_shared) != len(global_shared):
        raise ValueError(f"shared_dense weight length mismatch: local={len(local_shared)} global={len(global_shared)}")

    for i, (lw, gw) in enumerate(zip(local_shared, global_shared)):
        if lw.shape != gw.shape:
            raise ValueError(
                f"shared_dense shape mismatch at idx {i}: local={lw.shape} global={gw.shape}. "
                f"Server shared_dense must match SHARED_DIM={SHARED_DIM}"
            )

    model.get_layer("shared_dense").set_weights(blend(local_shared, global_shared, gamma))

    # ---------- train ----------
    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=cw,
        verbose=1
    )
    train_time = time.perf_counter() - t0

    # ---------- test eval ----------
    y_hat = predict_labels_multiclass(model, X_test, batch_size=BATCH)
    cm = confusion_matrix(y_test, y_hat, labels=labels_all)

    acc = accuracy_score(y_test, y_hat)
    mp, mr, mf1, _ = precision_recall_fscore_support(y_test, y_hat, average="macro", zero_division=0, labels=labels_all)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_test, y_hat, average="weighted", zero_division=0, labels=labels_all)

    print("\n===== RESULTS (TEST) =====")
    print(f"Accuracy        : {acc:.6f}")
    print(f"Macro-Precision : {mp:.6f}")
    print(f"Macro-Recall    : {mr:.6f}")
    print(f"Macro-F1        : {mf1:.6f}")
    print(f"Weighted-Prec   : {wp:.6f}")
    print(f"Weighted-Recall : {wr:.6f}")
    print(f"Weighted-F1     : {wf1:.6f}")
    print(f"Train time (sec): {train_time:.2f}")

    print("\n==== Classification Report ====")
    # ✅ FIX: force same #labels as target_names
    print(classification_report(
        y_test, y_hat,
        labels=labels_all,
        target_names=class_names,
        zero_division=0
    ))

    print("\n==== Confusion Matrix ====")
    print(cm)

    # ---------- save artifacts ----------
    cm_png = os.path.join(OUT_DIR, f"CM_one_shot_round{round_id:03d}_seed{SEED}_g{gamma[0]}_{gamma[1]}.png")
    plot_confusion_matrix_clean_multiclass(cm, class_names, cm_png)

    csv_out = os.path.join(OUT_DIR, f"metrics_one_shot_round{round_id:03d}_seed{SEED}_g{gamma[0]}_{gamma[1]}.csv")
    pd.DataFrame([{
        "timestamp": now_str(),
        "seed": SEED,
        "round_id": round_id,
        "gamma_local": gamma[0],
        "gamma_global": gamma[1],
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "num_classes": num_classes,
        "acc": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf1,
        "weighted_precision": wp,
        "weighted_recall": wr,
        "weighted_f1": wf1,
        "train_time_sec": float(train_time),
        "server_pkl_path": SERVER_PKL
    }]).to_csv(csv_out, index=False)

    model_out = os.path.join(OUT_DIR, f"one_shot_unsw_nb15_multiclass_round{round_id:03d}_seed{SEED}.keras")
    model.save(model_out)

    print("\n===== SAVED =====")
    print("CSV  :", csv_out)
    print("CM   :", cm_png)
    print("Model:", model_out)


if __name__ == "__main__":
    main()
