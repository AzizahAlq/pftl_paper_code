#!/usr/bin/env python3.10
# ============================================================
# Unseen Client — One-shot Transfer (shared_dense only)
# Standalone vs One-shot transfer using saved server shared layer:
#   /Users/azizahalq/Desktop/PFTL_Binary/agg_pftl_shared_dense/global_shared_dense_round_000000.pkl
#
# Preprocess inside the script:
# - Raw label column: label
# - binary_label: 0 if BenignTraffic else 1
# - Balance: 16000 class0 + 16000 class1
# - Split: Test=20%, Val=10% of total (val_frac=0.125)
# - StandardScaler
# - class_weight mapping {class_value: weight}
# - Client6-style prints + CSV logs
# - BLUE/WHITE confusion matrices saved:
#     Before_TL.png , After_TL.png
# ============================================================

import os, random, time, csv, pickle
from datetime import datetime

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

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, initializers


# -------------------------
# CONFIG
# -------------------------
CLIENT_ID     = "unseen_one_shot"
DATASET_NAME  = "UNSEEN"

DATA_PATH      = "/Users/azizahalq/Desktop/PFTL_Binary/Datasets_processed/D3_CICIOT2023/Unseen_UNSW_NB15.csv"
RAW_LABEL_COL  = "attack_cat"
BENIGN_NAME    = "Normal"
N_PER_CLASS    = 20000

# one-shot transfer file (server shared_dense)
GLOBAL_SHARED_PKL = "/Users/azizahalq/Desktop/PFTL_Binary/agg_pftl_shared_dense/global_shared_dense_round_000000.pkl"

# gamma blend: W <- gamma_local*W_local + gamma_global*W_global
GAMMA_LOCAL  = 0.9
GAMMA_GLOBAL = 0.1

# training
TEST_SIZE = 0.20
VAL_SIZE  = 0.10
VAL_FRAC  = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.125

EPOCHS_STANDALONE = 8
EPOCHS_AFTER_TL   = 7   # you mentioned 7 local epochs after one-shot transfer
BATCH_SIZE        = 256
LR                = 1e-3

# model
CONV_FILTERS = 4
PRIVATE_DIM  = 4
SHARED_DIM   = 4
SHARED_LAYER_NAME = "shared_dense"

# outputs
OUT_DIR = "/Users/azizahalq/Desktop/PFTL_Binary/unseen2_one_shot_logs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

METRICS_CSV = os.path.join(OUT_DIR, f"{CLIENT_ID}_metrics.csv")
BEFORE_PNG  = os.path.join(FIG_DIR, "Before_TL.png")
AFTER_PNG   = os.path.join(FIG_DIR, "After_TL.png")

METHOD_STANDALONE = "standalone"
METHOD_ONESHOT    = f"one_shot_gamma_{GAMMA_LOCAL}_{GAMMA_GLOBAL}"

METRICS_HEADER = [
    "timestamp","method","dataset","seed","client",
    "phase",
    "acc",
    "prec_bin","rec_bin","f1_bin",
    "prec_macro","rec_macro","f1_macro",
    "prec_weighted","rec_weighted","f1_weighted",
    "prec0","rec0","f10",
    "prec1","rec1","f11",
    "tn","fp","fn","tp"
]


# -------------------------
# Utils
# -------------------------
def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def append_csv(path, row, header):
    ensure_csv(path, header)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def safe_to_numeric_df(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

# -------------------------
# Data preprocess + balance
# -------------------------
def load_and_preprocess_balanced():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    if RAW_LABEL_COL not in df.columns:
        raise ValueError(
            f"[{CLIENT_ID}] RAW_LABEL_COL='{RAW_LABEL_COL}' not found. "
            f"Available cols (last 15): {df.columns.tolist()[-15:]}"
        )

    raw = df[RAW_LABEL_COL].astype(str).str.strip()
    df["binary_label"] = (raw != BENIGN_NAME).astype(int)

    df0 = df[df["binary_label"] == 0]
    df1 = df[df["binary_label"] == 1]

    if len(df0) < N_PER_CLASS or len(df1) < N_PER_CLASS:
        raise ValueError(
            f"[{CLIENT_ID}] Not enough samples to balance.\n"
            f"class0(Benign)={len(df0)} class1(Attack)={len(df1)} required={N_PER_CLASS} each.\n"
            f"Tip: reduce N_PER_CLASS or regenerate data with more benign."
        )

    df0 = df0.sample(n=N_PER_CLASS, random_state=SEED, replace=False)
    df1 = df1.sample(n=N_PER_CLASS, random_state=SEED, replace=False)
    dfb = pd.concat([df0, df1], axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    y = dfb["binary_label"].astype(int)

    X = dfb.drop(columns=["binary_label"], errors="ignore")
    X = X.drop(columns=[RAW_LABEL_COL], errors="ignore")
    X = safe_to_numeric_df(X)

    return X, y

def split_scale_reshape(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_FRAC, random_state=SPLIT_SEED, stratify=y_trainval
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
# Model
# -------------------------
def build_model(input_steps):
    ki = initializers.GlorotUniform(seed=SEED)
    bi = initializers.Zeros()

    model = models.Sequential([
        layers.Input(shape=(input_steps, 1)),
        layers.Conv1D(CONV_FILTERS, 5, activation="relu", padding="same",
                      kernel_initializer=ki, bias_initializer=bi),
        layers.MaxPooling1D(2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(PRIVATE_DIM, activation="relu", name="feat1",
                     kernel_initializer=ki, bias_initializer=bi),
        layers.Dense(SHARED_DIM, activation="relu", name=SHARED_LAYER_NAME,
                     kernel_initializer=ki, bias_initializer=bi),
        layers.Dense(1, activation="sigmoid", name="clf",
                     kernel_initializer=ki, bias_initializer=bi),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    )
    return model

# -------------------------
# Shared layer load/blend
# -------------------------
def load_global_shared_weights(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"[{CLIENT_ID}] Global shared PKL not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    # Common possibilities:
    # 1) list/tuple: [W, b]
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        return obj

    # 2) dict with shared layer key
    if isinstance(obj, dict):
        for k in ["shared_dense", "shared", SHARED_LAYER_NAME, "weights"]:
            if k in obj:
                w = obj[k]
                if isinstance(w, (list, tuple)) and len(w) == 2:
                    return w
        # maybe dict is already {layer_name: [W,b]}
        # take first value that looks like [W,b]
        for v in obj.values():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return v

    raise ValueError(
        f"[{CLIENT_ID}] Unrecognized PKL content type: {type(obj)}. "
        "Expected [W,b] or dict containing [W,b]."
    )

def get_shared_layer(model):
    return model.get_layer(SHARED_LAYER_NAME).get_weights()

def set_shared_layer(model, w):
    if (not isinstance(w, (list, tuple))) or len(w) != 2:
        raise ValueError(f"[{CLIENT_ID}] shared_dense weights invalid format (need [W,b]).")
    if any(np.isnan(x).any() for x in w):
        raise ValueError(f"[{CLIENT_ID}] shared_dense weights contain NaN.")
    model.get_layer(SHARED_LAYER_NAME).set_weights(w)

def blend_shared(local_w, global_w, gamma_local, gamma_global):
    if len(local_w) != len(global_w):
        raise ValueError("local/global shared_dense weights length mismatch")
    return [gamma_local * l + gamma_global * g for l, g in zip(local_w, global_w)]

# -------------------------
# Metrics + plot
# -------------------------
def compute_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))

    mp, mr, mf1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    bp, br, bf1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    prec0, rec0, f10 = float(p_cls[0]), float(r_cls[0]), float(f_cls[0])
    prec1, rec1, f11 = float(p_cls[1]), float(r_cls[1]), float(f_cls[1])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    return {
        "acc": acc,
        "bp": float(bp), "br": float(br), "bf1": float(bf1),
        "mp": float(mp), "mr": float(mr), "mf1": float(mf1),
        "wp": float(wp), "wr": float(wr), "wf1": float(wf1),
        "prec0": prec0, "rec0": rec0, "f10": f10,
        "prec1": prec1, "rec1": rec1, "f11": f11,
        "cm": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

def eval_and_log(model, X, y, method, phase, print_report=False):
    probs = model.predict(X, verbose=0).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.5)
    y_pred = (probs >= 0.5).astype(int)

    m = compute_metrics(y, y_pred)

    print(f"\n[{CLIENT_ID}] ===== {phase.upper()} ({method}) =====")
    print("\n===== RESULTS =====")
    print(f"Accuracy        : {m['acc']:.6f}")
    print(f"Macro-Precision : {m['mp']:.6f}")
    print(f"Macro-Recall    : {m['mr']:.6f}")
    print(f"Macro-F1        : {m['mf1']:.6f}")

    print("\n===== WEIGHTED (support-weighted) METRICS =====")
    print(f"Weighted-Precision : {m['wp']:.6f}")
    print(f"Weighted-Recall    : {m['wr']:.6f}")
    print(f"Weighted-F1        : {m['wf1']:.6f}")

    print("\n==== Confusion Matrix ====")
    print(m["cm"])

    if print_report:
        print("\n==== Classification Report ====")
        print(classification_report(y, y_pred, zero_division=0))

    append_csv(METRICS_CSV, [
        now_ts(), method, DATASET_NAME, SEED, CLIENT_ID,
        phase,
        round(m["acc"], 6),
        round(m["bp"], 6), round(m["br"], 6), round(m["bf1"], 6),
        round(m["mp"], 6), round(m["mr"], 6), round(m["mf1"], 6),
        round(m["wp"], 6), round(m["wr"], 6), round(m["wf1"], 6),
        round(m["prec0"], 6), round(m["rec0"], 6), round(m["f10"], 6),
        round(m["prec1"], 6), round(m["rec1"], 6), round(m["f11"], 6),
        int(m["tn"]), int(m["fp"]), int(m["fn"]), int(m["tp"])
    ], METRICS_HEADER)

    return m

def save_confusion_matrix_png(cm, path):
    import numpy as np
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=int)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(cm, cmap="Blues", interpolation="nearest", vmin=0)

    # Axis labels (larger + padded)
    ax.set_xlabel("Predicted", fontsize=18, labelpad=12)
    ax.set_ylabel("Actual", fontsize=18, labelpad=12)

    # Tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Normal", "Attack"], fontsize=16)
    ax.set_yticklabels(["Normal", "Attack"], fontsize=16)

    # Annotate numbers
    thresh = cm.max() * 0.5 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color=color,
                fontsize=20,        # bigger numbers
                fontweight="bold"
            )

    # Bigger colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()

    # Important: avoid clipping labels
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[{CLIENT_ID}] Saved: {path}")

# -------------------------
# Main
# -------------------------
def main():
    ensure_csv(METRICS_CSV, METRICS_HEADER)

    # data
    X_raw, y_raw = load_and_preprocess_balanced()
    print(f"[{CLIENT_ID}] After balancing: total={len(y_raw)} | 0={(y_raw==0).sum()} 1={(y_raw==1).sum()}")

    Xtr, Xva, Xte, ytr, yva, yte, nf = split_scale_reshape(X_raw, y_raw)
    print(f"[{CLIENT_ID}] Split: train={len(ytr)} val={len(yva)} test={len(yte)} nf={nf}")

    # class_weight
    classes = np.unique(ytr)
    cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"[{CLIENT_ID}] class_weight = {class_weights}")

    # ============================================================
    # (A) Standalone baseline (no transfer)
    # ============================================================
    model_st = build_model(nf)

    # evaluate before training (optional)
    eval_and_log(model_st, Xte, yte, METHOD_STANDALONE, "before_training", print_report=False)

    t0 = time.perf_counter()
    model_st.fit(
        Xtr, ytr,
        epochs=EPOCHS_STANDALONE,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=True,
        class_weight=class_weights,
        validation_data=(Xva, yva)
    )
    print(f"\n[{CLIENT_ID}] Standalone training time: {time.perf_counter()-t0:.2f}s")

    m_before = eval_and_log(model_st, Xte, yte, METHOD_STANDALONE, "final_test", print_report=True)
    save_confusion_matrix_png(m_before["cm"], BEFORE_PNG)

    # ============================================================
    # (B) One-shot transfer (shared_dense only) + local training
    # ============================================================
    model_tl = build_model(nf)

    # Load global shared weights
    global_shared = load_global_shared_weights(GLOBAL_SHARED_PKL)
    local_shared  = get_shared_layer(model_tl)

    blended = blend_shared(local_shared, global_shared, GAMMA_LOCAL, GAMMA_GLOBAL)
    set_shared_layer(model_tl, blended)

    print(f"\n[{CLIENT_ID}] One-shot: loaded global shared_dense from:\n  {GLOBAL_SHARED_PKL}")
    print(f"[{CLIENT_ID}] One-shot: applied gamma blend (local={GAMMA_LOCAL}, global={GAMMA_GLOBAL})")

    # evaluate right after transfer (optional)
    eval_and_log(model_tl, Xte, yte, METHOD_ONESHOT, "after_transfer_before_training", print_report=False)

    t1 = time.perf_counter()
    model_tl.fit(
        Xtr, ytr,
        epochs=EPOCHS_AFTER_TL,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=True,
        class_weight=class_weights,
        validation_data=(Xva, yva)
    )
    print(f"\n[{CLIENT_ID}] After-TL training time: {time.perf_counter()-t1:.2f}s")

    m_after = eval_and_log(model_tl, Xte, yte, METHOD_ONESHOT, "final_test", print_report=True)
    save_confusion_matrix_png(m_after["cm"], AFTER_PNG)

    print("\n===== SAVED =====")
    print("Metrics CSV:", METRICS_CSV)
    print("Before PNG :", BEFORE_PNG)
    print("After PNG  :", AFTER_PNG)

if __name__ == "__main__":
    main()