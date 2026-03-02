#!/usr/bin/env python3.10
# ============================================================
# PTFL Client (Design-B + STRICT BARRIER) 
# ============================================================
# What you asked (fixed):
# 1) Client does NOT send anything at init. It starts LOCAL training first.
# 2) After finishing local training for server_round=r, it sends shared weights for round=r
# 3) STRICT BARRIER: after sending round=r, client blocks until server_round >= r+1
# 4) No "set_weights length 0" crash: never set_shared unless weights has (kernel,bias) len==2
# 5) Uses YOUR generated modules: myproto_pb2 / myproto_pb2_grpc
# 6) Logs include SEED and CURVE_CSV stores ONLY macro-f1 (local vs global), as in your v2 style
# ============================================================

import os, time, csv, pickle, random
from datetime import datetime

# -------------------------
# Reproducibility
# -------------------------
SEED = 45
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

random.seed(SEED)
import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import pandas as pd
import grpc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import layers, models, optimizers

# IMPORTANT: use YOUR proto-generated modules
import myproto_pb2
import myproto_pb2_grpc


# -------------------------
# PTFL Settings
# -------------------------
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:50051")
CLIENT_ID      = os.environ.get("CLIENT_ID", "client1")
NUM_ROUNDS     = int(os.environ.get("NUM_ROUNDS", "20"))

PRIVATE_DIM = 16
SHARED_DIM  = 8

# -------------------------
# Data / Paths
# -------------------------
CSV_PATH  = os.environ.get(
    "CSV_PATH",
    "/Users/azizahalq/Desktop/PFTL_project3/Datasets_processed/D1_D2_CIC_IOT_and_CIC_TON_Dataset2023/CIC-ToN-IoT_new.csv"
)
LABEL_COL = os.environ.get("LABEL_COL", "Attack")

OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/Users/azizahalq/Desktop/PFTL_project3/ptfl_logs/client1_ton_iot_v2"
)
os.makedirs(OUT_DIR, exist_ok=True)

COMM_LOG    = os.path.join(OUT_DIR, f"{CLIENT_ID}_comm_log.csv")
METRICS_LOG = os.path.join(OUT_DIR, f"{CLIENT_ID}_metrics_log.csv")  # two rows per round: LOCAL + GLOBAL
CURVE_CSV   = os.path.join(OUT_DIR, f"{CLIENT_ID}_local_vs_global_curve.csv")  # macro-f1 only

# -------------------------
# Split style (70/15/15)
# -------------------------
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.17647058823529413  # 0.15 / (1-0.15)

# Train config
EPOCHS_PER_ROUND = 1
BATCH_SIZE       = 1024
LR               = 1e-3

# -------------------------
# Safety Switch + Adaptive Gamma
# -------------------------
ADAPT_GAMMA = True
EPS = 0.001  # MACRO-F1 margin

GAMMA_GLOBAL_INIT = 0.50
ETA = 0.05
TAU = 0.05
GAMMA_MIN = 0.10
GAMMA_MAX = 0.90

# -------------------------
# Networking + waiting
# -------------------------
MAX_GRPC_MSG = 50 * 1024 * 1024

WAIT_FOR_SERVER_MAX_SEC = 600
WAIT_FOR_SERVER_SLEEP_SEC = 2.0

# Barrier: after sending server_round=r, wait until server_round >= r+1
BARRIER_TIMEOUT_SEC = 900
BARRIER_POLL_SEC = 1.0


# ============================================================
# Utils
# ============================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clip(x, lo, hi):
    return float(max(lo, min(hi, x)))


def safe_unpickle_weights(b: bytes):
    try:
        return pickle.loads(b)
    except Exception:
        return None


def is_valid_shared(weights_obj) -> bool:
    # shared_dense expects [kernel, bias] (len==2)
    return isinstance(weights_obj, (list, tuple)) and len(weights_obj) == 2


def predict_labels(model, X_batch, num_classes):
    probs = model.predict(X_batch, batch_size=1024, verbose=0)
    if num_classes == 2:
        return (probs.reshape(-1) >= 0.5).astype(int)
    return np.argmax(probs, axis=1)


def eval_all_metrics(model, X_eval, y_eval, num_classes):
    y_hat = predict_labels(model, X_eval, num_classes)
    acc = float(accuracy_score(y_eval, y_hat))

    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_eval, y_hat, average="macro", zero_division=0
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_eval, y_hat, average="weighted", zero_division=0
    )
    return acc, float(mp), float(mr), float(mf1), float(wp), float(wr), float(wf1)


def update_gamma_global(gamma_global, f1_local, f1_mixed, eta=ETA, tau=TAU):
    delta = float(f1_mixed - f1_local)
    step = float(eta) * float(np.tanh(delta / max(float(tau), 1e-8)))
    gamma_global = clip(float(gamma_global) + step, GAMMA_MIN, GAMMA_MAX)
    gamma_local = 1.0 - gamma_global
    return gamma_local, gamma_global, delta


def ensure_csv_headers():
    if not os.path.exists(COMM_LOG):
        with open(COMM_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["seed", "round", "bytes_sent", "rtt_sec", "barrier_wait_sec", "timestamp"])

    if not os.path.exists(METRICS_LOG):
        with open(METRICS_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "seed",
                "round",
                "stage",  # LOCAL / GLOBAL

                "train_loss", "train_acc",
                "val_loss", "val_acc",

                "acc",
                "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1",

                "switch_on",
                "eps_margin",
                "gamma_local",
                "gamma_global",
                "delta_macro_f1_mixed_minus_local",
                "sent_weights_type",

                "local_train_time_sec",
                "mix_time_sec",
                "round_total_time_sec",

                "timestamp"
            ])

    if not os.path.exists(CURVE_CSV):
        with open(CURVE_CSV, "w", newline="") as f:
            # macro-f1 only, like your v2 clients
            csv.writer(f).writerow(["seed", "round", "local_macro_f1", "global_macro_f1", "switch_on", "gamma_global"])


def log_stage_row(
    seed, round_id, stage,
    train_loss, train_acc, val_loss, val_acc,
    acc, mp, mr, mf1, wp, wr, wf1,
    switch_on, eps_margin,
    gamma_local, gamma_global, delta_f1,
    sent_type,
    local_train_time, mix_time, round_total_time,
):
    with open(METRICS_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            int(seed),
            int(round_id),
            str(stage),
            f"{float(train_loss):.6f}", f"{float(train_acc):.6f}",
            f"{float(val_loss):.6f}",   f"{float(val_acc):.6f}",

            f"{float(acc):.6f}",
            f"{float(mp):.6f}", f"{float(mr):.6f}", f"{float(mf1):.6f}",
            f"{float(wp):.6f}", f"{float(wr):.6f}", f"{float(wf1):.6f}",

            int(switch_on),
            f"{float(eps_margin):.6f}",
            f"{float(gamma_local):.6f}",
            f"{float(gamma_global):.6f}",
            f"{float(delta_f1):+.6f}",
            str(sent_type),

            f"{float(local_train_time):.4f}",
            f"{float(mix_time):.4f}",
            f"{float(round_total_time):.4f}",

            now_str()
        ])


# ============================================================
# Model
# ============================================================
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

    model = models.Model(inp, out, name=f"PTFL_{CLIENT_ID}_CNN")
    opt = optimizers.Adam(learning_rate=LR, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


# ============================================================
# Client
# ============================================================
class PTFLClientDesignB_StrictBarrier:
    def __init__(self):
        ensure_csv_headers()

        self.current_round = 0  # must follow server round
        self.gamma_global = float(GAMMA_GLOBAL_INIT)
        self.gamma_local  = 1.0 - self.gamma_global

        self.channel = grpc.insecure_channel(
            SERVER_ADDRESS,
            options=[
                ("grpc.max_send_message_length", MAX_GRPC_MSG),
                ("grpc.max_receive_message_length", MAX_GRPC_MSG),
            ],
        )
        self.stub = myproto_pb2_grpc.AggregatorStub(self.channel)

        self._load_data()
        self.model = build_cnn(self.input_shape, self.num_classes)

        print("\n==== Model Summary ====")
        self.model.summary()

        self.wait_for_aggregator_reachable()

        # IMPORTANT: Do NOT send anything here.
        # Client must start LOCAL training first, then send at end of round.
        w, r = self.pull_global_shared()
        self.current_round = int(r)
        if is_valid_shared(w):
            self.set_shared(w)
            print(f"[{CLIENT_ID}] Initial sync: loaded server global (server_round={r}).")
        else:
            print(f"[{CLIENT_ID}] Initial sync: server has no global yet (server_round={r}). "
                  f"I will start LOCAL training and send my shared weights after Round 1.")

    # -------------------------
    # gRPC helpers
    # -------------------------
    def wait_for_aggregator_reachable(self):
        t0 = time.perf_counter()
        print(f"[{CLIENT_ID}] Waiting for aggregator at {SERVER_ADDRESS} ...")
        while True:
            try:
                _w, r = self.pull_global_shared()
                print(f"[{CLIENT_ID}] Aggregator reachable (server_round={r}).")
                return
            except Exception as e:
                if time.perf_counter() - t0 > WAIT_FOR_SERVER_MAX_SEC:
                    raise RuntimeError(f"[{CLIENT_ID}] Aggregator not reachable after {WAIT_FOR_SERVER_MAX_SEC}s: {e}")
                time.sleep(WAIT_FOR_SERVER_SLEEP_SEC)

    def pull_global_shared(self):
        resp = self.stub.GetSharedWeights(
            myproto_pb2.EmptyRequest(),
            metadata=[("client_id", CLIENT_ID)]
        )
        w = safe_unpickle_weights(resp.weights)
        return w, int(resp.round)

    def send_shared_update(self, shared_weights, num_samples: int):
        payload = pickle.dumps(shared_weights)
        req = myproto_pb2.SharedUpdate(
            weights=payload,
            round=int(self.current_round),
            num_samples=int(num_samples)
        )
        t0 = time.perf_counter()
        ack = self.stub.SendSharedUpdate(req, metadata=[("client_id", CLIENT_ID)])
        rtt = time.perf_counter() - t0
        # your proto returns Ack{bool ok}
        ok = bool(getattr(ack, "ok", False))
        return ok, len(payload), rtt

    def send_update_retry_once(self, shared_weights, num_samples: int):
        ok, bytes_sent, rtt = self.send_shared_update(shared_weights, num_samples)
        if ok:
            return ok, bytes_sent, rtt

        # resync round and retry once
        _w, srv_round = self.pull_global_shared()
        if int(srv_round) != int(self.current_round):
            print(f"[{CLIENT_ID}] Resync after reject: {self.current_round} -> {srv_round}")
            self.current_round = int(srv_round)

        ok2, bytes_sent2, rtt2 = self.send_shared_update(shared_weights, num_samples)
        return ok2, bytes_sent2, rtt2

    def wait_for_server_round(self, target_round: int, timeout_sec: float):
        t0 = time.perf_counter()
        while True:
            w, r = self.pull_global_shared()
            if int(r) >= int(target_round):
                return w, int(r), (time.perf_counter() - t0)
            if (time.perf_counter() - t0) > timeout_sec:
                return w, int(r), (time.perf_counter() - t0)
            time.sleep(BARRIER_POLL_SEC)

    # -------------------------
    # Data
    # -------------------------
    def _load_data(self):
        df = pd.read_csv(CSV_PATH, low_memory=False)
        df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

        if LABEL_COL not in df.columns:
            raise ValueError(f"Label column '{LABEL_COL}' not found. Available columns:\n{list(df.columns)}")

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

        self.le = LabelEncoder()
        y = self.le.fit_transform(y_raw).astype(int)
        self.num_classes = int(len(self.le.classes_))

        # 70/15/15 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=VAL_SIZE_FROM_TRAIN, random_state=SEED, stratify=y_train
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_test  = self.scaler.transform(X_test)

        X_train = X_train[..., np.newaxis]
        X_val   = X_val[..., np.newaxis]
        X_test  = X_test[..., np.newaxis]

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.input_shape = (X_train.shape[1], 1)

        classes_idx = np.unique(self.y_train)
        cw = compute_class_weight(class_weight="balanced", classes=classes_idx, y=self.y_train)
        self.class_weight = {int(i): float(w) for i, w in zip(classes_idx, cw)}

        print("\n=== PTFL Client Readiness Summary ===")
        print(f"Client : {CLIENT_ID}")
        print(f"CSV    : {CSV_PATH}")
        print(f"Samples: {X.shape[0]}")
        print(f"Features: {X.shape[1]}")
        print(f"Classes : {self.num_classes}")
        print("Splits:", len(self.X_train), len(self.X_val), len(self.X_test))

    # -------------------------
    # Shared layer helpers
    # -------------------------
    def get_shared(self):
        return self.model.get_layer("shared_dense").get_weights()

    def set_shared(self, w):
        if not is_valid_shared(w):
            return
        if any(np.isnan(x).any() for x in w):
            return
        self.model.get_layer("shared_dense").set_weights(list(w))

    def blend(self, local_w, global_w):
        gl = float(self.gamma_local)
        gg = float(self.gamma_global)
        return [gl * l + gg * g for l, g in zip(local_w, global_w)]

    # -------------------------
    # Train
    # -------------------------
    def train_one_epoch(self):
        hist = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=EPOCHS_PER_ROUND,
            batch_size=BATCH_SIZE,
            class_weight=self.class_weight,
            verbose=1
        )
        train_loss = float(hist.history["loss"][0])
        val_loss   = float(hist.history["val_loss"][0])
        train_acc  = float(hist.history.get("accuracy", [np.nan])[0])
        val_acc    = float(hist.history.get("val_accuracy", [np.nan])[0])
        return train_loss, train_acc, val_loss, val_acc

    # -------------------------
    # Main loop (STRICT BARRIER)
    # -------------------------
    def run(self):
        for local_iter in range(NUM_ROUNDS):
            round_id = local_iter + 1
            round_start = time.perf_counter()

            # align to server round before training
            w_srv, srv_round = self.pull_global_shared()
            if int(srv_round) != int(self.current_round):
                print(f"[{CLIENT_ID}] Align current_round {self.current_round} -> {srv_round}")
                self.current_round = int(srv_round)
            if is_valid_shared(w_srv):
                self.set_shared(w_srv)

            print(f"\n[{CLIENT_ID}] ===== Round {round_id}/{NUM_ROUNDS} (server_round={self.current_round}) =====")

            # local train
            t0 = time.perf_counter()
            train_loss, train_acc, val_loss, val_acc = self.train_one_epoch()
            local_train_time = time.perf_counter() - t0

            local_w = self.get_shared()

            # eval local (VAL)
            local_acc, local_mp, local_mr, local_mf1, local_wp, local_wr, local_wf1 = eval_all_metrics(
                self.model, self.X_val, self.y_val, self.num_classes
            )

            # pull global and mix (if no global -> fallback local)
            t1 = time.perf_counter()
            global_w, _ = self.pull_global_shared()
            if not is_valid_shared(global_w):
                global_w = local_w

            mixed_w = self.blend(local_w, global_w)
            self.set_shared(mixed_w)
            mix_time = time.perf_counter() - t1

            # eval mixed (VAL)
            mixed_acc, mixed_mp, mixed_mr, mixed_mf1, mixed_wp, mixed_wr, mixed_wf1 = eval_all_metrics(
                self.model, self.X_val, self.y_val, self.num_classes
            )

            # safety switch (macro-f1)
            switch_on = int(mixed_mf1 >= (local_mf1 + EPS))
            if switch_on:
                sent_type = "mixed"
                weights_to_send = mixed_w
            else:
                self.set_shared(local_w)
                sent_type = "local"
                weights_to_send = local_w

            # adaptive gamma (macro-f1)
            if ADAPT_GAMMA:
                self.gamma_local, self.gamma_global, delta_f1 = update_gamma_global(
                    self.gamma_global, local_mf1, mixed_mf1
                )
            else:
                delta_f1 = float(mixed_mf1 - local_mf1)

            print(
                f"[{CLIENT_ID}] MacroF1 Local={local_mf1:.6f} | Mixed={mixed_mf1:.6f} | "
                f"Switch={'ON' if switch_on else 'OFF'} (eps={EPS}) | "
                f"Send={sent_type} | gamma_global={self.gamma_global:.3f} (Δ={delta_f1:+.6f})"
            )

            # curve csv (macro-f1 only)
            with open(CURVE_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    int(SEED),
                    int(round_id),
                    f"{local_mf1:.16f}",
                    f"{mixed_mf1:.16f}",
                    int(switch_on),
                    f"{self.gamma_global:.16f}",
                ])

            # send update (retry once if rejected)
            send_t0 = time.perf_counter()
            ok, bytes_sent, rtt = self.send_update_retry_once(weights_to_send, self.X_train.shape[0])
            if not ok:
                print(f"[{CLIENT_ID}] SendSharedUpdate rejected (ok=False) even after retry. Will resync next loop.")

            # STRICT BARRIER: wait until server advances to current_round + 1
            target_round = int(self.current_round) + 1
            w_new, new_round, waited = self.wait_for_server_round(target_round, timeout_sec=BARRIER_TIMEOUT_SEC)

            barrier_wait = waited
            if int(new_round) >= target_round:
                self.current_round = int(new_round)
                if is_valid_shared(w_new):
                    self.set_shared(w_new)
                print(f"[{CLIENT_ID}] Barrier passed -> server_round={self.current_round} (waited {barrier_wait:.1f}s)")
            else:
                print(f"[{CLIENT_ID}] Barrier TIMEOUT: server_round={new_round}, expected>={target_round}")

            # comm log
            with open(COMM_LOG, "a", newline="") as f:
                csv.writer(f).writerow([
                    int(SEED),
                    int(round_id),
                    int(bytes_sent),
                    f"{float(rtt):.3f}",
                    f"{float(barrier_wait):.3f}",
                    now_str()
                ])

            # metrics log (two rows)
            round_total_time = time.perf_counter() - round_start

            log_stage_row(
                SEED, round_id, "LOCAL",
                train_loss, train_acc, val_loss, val_acc,
                local_acc, local_mp, local_mr, local_mf1, local_wp, local_wr, local_wf1,
                switch_on, EPS,
                self.gamma_local, self.gamma_global, delta_f1,
                sent_type,
                local_train_time, mix_time, round_total_time,
            )
            log_stage_row(
                SEED, round_id, "GLOBAL",
                train_loss, train_acc, val_loss, val_acc,
                mixed_acc, mixed_mp, mixed_mr, mixed_mf1, mixed_wp, mixed_wr, mixed_wf1,
                switch_on, EPS,
                self.gamma_local, self.gamma_global, delta_f1,
                sent_type,
                local_train_time, mix_time, round_total_time,
            )

        # FINAL TEST
        print(f"\n[{CLIENT_ID}] ===== FINAL TEST EVALUATION =====")
        y_pred = predict_labels(self.model, self.X_test, self.num_classes)

        acc = float(accuracy_score(self.y_test, y_pred))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average="macro", zero_division=0
        )
        w_p, w_r, w_f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average="weighted", zero_division=0
        )

        print("\n===== STANDALONE RESULTS (NO SAVING artifacts) =====")
        print(f"Accuracy        : {acc:.6f}")
        print(f"Macro-Precision : {macro_p:.6f}")
        print(f"Macro-Recall    : {macro_r:.6f}")
        print(f"Macro-F1        : {macro_f1:.6f}")

        print("\n===== WEIGHTED (support-weighted) METRICS =====")
        print(f"Weighted-Precision : {w_p:.6f}")
        print(f"Weighted-Recall    : {w_r:.6f}")
        print(f"Weighted-F1        : {w_f1:.6f}")

        print("\n==== Classification Report ====")
        print(classification_report(self.y_test, y_pred, target_names=self.le.classes_, zero_division=0))

        print("\n==== Confusion Matrix ====")
        print(confusion_matrix(self.y_test, y_pred))

        print("\n===== SAVED (CSV ONLY) =====")
        print("Curve CSV  :", CURVE_CSV)
        print("Metrics CSV:", METRICS_LOG)
        print("Comm CSV   :", COMM_LOG)


if __name__ == "__main__":
    PTFLClientDesignB_StrictBarrier().run()
