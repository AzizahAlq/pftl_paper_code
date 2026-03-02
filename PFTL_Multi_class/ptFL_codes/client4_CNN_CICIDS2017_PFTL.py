#!/usr/bin/env python3.10
# ============================================================
# client4_CICIDS2017_PTFL_DESIGN_B_LOCAL_FIRST_STRICT_BARRIER_MATCH_STANDALONE.py
# ============================================================

import os, time, csv, pickle, random
from datetime import datetime

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tensorflow.keras import layers, models, optimizers

import myproto_pb2
import myproto_pb2_grpc

# -------------------------
# Config
# -------------------------
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:50051")
CLIENT_ID      = os.environ.get("CLIENT_ID", "client4")
NUM_ROUNDS     = int(os.environ.get("NUM_ROUNDS", "20"))

CSV_PATH  = os.environ.get(
    "CSV_PATH",
    "/Users/azizahalq/Desktop/PFTL_project3/Datasets_processed/D4_CICIDS2017/Merged_ALL_CIC_IDS_2017_TOTAL_200K.csv"
)
LABEL_COL = os.environ.get("LABEL_COL", "Label")

OUT_DIR = os.environ.get(
    "OUT_DIR",
    f"/Users/azizahalq/Desktop/PFTL_project3/ptfl_logs/{CLIENT_ID}_v2_localfirst_strict"
)
os.makedirs(OUT_DIR, exist_ok=True)

COMM_LOG    = os.path.join(OUT_DIR, f"{CLIENT_ID}_comm_log.csv")
METRICS_LOG = os.path.join(OUT_DIR, f"{CLIENT_ID}_metrics_log.csv")
CURVE_CSV   = os.path.join(OUT_DIR, f"{CLIENT_ID}_local_vs_global_curve.csv")

#  unified split rule
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)  # 0.17647058823529413

EPOCHS_PER_ROUND = 1
BATCH_SIZE       = 2048
LR               = 1e-3

# Safety Switch + Adaptive Gamma
ADAPT_GAMMA = True
EPS = 0.001
GAMMA_GLOBAL_INIT = 0.50
ETA = 0.05
TAU = 0.05
GAMMA_MIN = 0.10
GAMMA_MAX = 0.90

MAX_GRPC_MSG = 50 * 1024 * 1024

WAIT_SERVER_MAX_SEC     = 600
WAIT_SERVER_POLL_SEC    = 2.0
WAIT_ROUND_ADV_MAX_SEC  = 1800
WAIT_ROUND_ADV_POLL_SEC = 1.0

PRIVATE_DIM = 16
SHARED_DIM  = 8

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clip(x, lo, hi):
    return float(max(lo, min(hi, x)))

def update_gamma_global(gamma_global, f1_local, f1_mixed):
    delta = float(f1_mixed - f1_local)
    step = float(ETA) * float(np.tanh(delta / max(float(TAU), 1e-8)))
    gamma_global = clip(float(gamma_global) + step, GAMMA_MIN, GAMMA_MAX)
    gamma_local = 1.0 - gamma_global
    return gamma_local, gamma_global, delta

def is_valid_shared(w) -> bool:
    return isinstance(w, (list, tuple)) and len(w) == 2

def safe_unpickle(b: bytes):
    if not b:
        return None
    try:
        return pickle.loads(b)
    except Exception:
        return None

def predict_labels(model, X_batch, num_classes):
    probs = model.predict(X_batch, batch_size=1024, verbose=0)
    if num_classes == 2:
        return (probs.reshape(-1) >= 0.5).astype(int)
    return np.argmax(probs, axis=1)

def eval_all_metrics(model, X_eval, y_eval, num_classes):
    y_hat = predict_labels(model, X_eval, num_classes)
    acc = float(accuracy_score(y_eval, y_hat))
    mp, mr, mf1, _ = precision_recall_fscore_support(y_eval, y_hat, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_eval, y_hat, average="weighted", zero_division=0)
    return acc, float(mp), float(mr), float(mf1), float(wp), float(wr), float(wf1)

def ensure_csv_headers():
    if not os.path.exists(COMM_LOG):
        with open(COMM_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "seed","round","bytes_sent",
                "server_round_before_send","server_round_after_barrier",
                "rtt_sec","barrier_wait_sec","timestamp"
            ])
    if not os.path.exists(METRICS_LOG):
        with open(METRICS_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "seed","round","stage",
                "train_loss","train_acc",
                "val_loss","val_acc",
                "acc",
                "macro_precision","macro_recall","macro_f1",
                "weighted_precision","weighted_recall","weighted_f1",
                "switch_on","eps_margin",
                "gamma_local","gamma_global",
                "delta_macro_f1_mixed_minus_local",
                "sent_weights_type",
                "timestamp"
            ])
    if not os.path.exists(CURVE_CSV):
        with open(CURVE_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "seed","round","local_macro_f1","global_macro_f1","switch_on","gamma_global"
            ])

def log_stage_row(seed, round_id, stage,
                  train_loss, train_acc, val_loss, val_acc,
                  acc, mp, mr, mf1, wp, wr, wf1,
                  switch_on, gamma_local, gamma_global, delta_f1, sent_type):
    with open(METRICS_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            int(seed), int(round_id), str(stage),
            f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}",   f"{val_acc:.6f}",
            f"{acc:.6f}",
            f"{mp:.6f}", f"{mr:.6f}", f"{mf1:.6f}",
            f"{wp:.6f}", f"{wr:.6f}", f"{wf1:.6f}",
            int(switch_on), f"{EPS:.6f}",
            f"{gamma_local:.6f}", f"{gamma_global:.6f}",
            f"{delta_f1:+.6f}",
            str(sent_type),
            now_str()
        ])

# ✅ MUST MATCH STANDALONE CLIENT4 MODEL EXACTLY
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

    model = models.Model(inp, out, name=f"PTFL_{CLIENT_ID}_CNN_MATCH_STANDALONE")
    model.compile(optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0), loss=loss, metrics=metrics)
    return model

class PTFLClient4_LocalFirst_StrictBarrier:
    def __init__(self):
        ensure_csv_headers()

        self.current_round = 0
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

        self.wait_for_server()

        # LOCAL-FIRST initial sync
        w, r = self.pull_global()
        self.current_round = int(r)
        if is_valid_shared(w):
            ok = self.set_shared_if_valid(w)
            print(f"[{CLIENT_ID}] Initial sync: loaded global (server_round={self.current_round}, set_shared={ok}).")
        else:
            print(f"[{CLIENT_ID}] Initial sync: server returned empty global (server_round={self.current_round}). "
                  f"Starting LOCAL training first.")

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

        print("\n=== Client4 PTFL Readiness Summary ===")
        print(f"CSV: {CSV_PATH}")
        print(f"Samples: {X.shape[0]}")
        print(f"Features: {X.shape[1]}")
        print(f"Classes: {self.num_classes}")
        print("Top 15 label counts:")
        print(pd.Series(y_raw).value_counts().head(15))

        # ✅ unified split rule
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

        self.X_train = X_train[..., np.newaxis]
        self.X_val   = X_val[..., np.newaxis]
        self.X_test  = X_test[..., np.newaxis]
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.input_shape = (self.X_train.shape[1], 1)

        classes_idx = np.unique(self.y_train)
        cw = compute_class_weight(class_weight="balanced", classes=classes_idx, y=self.y_train)
        self.class_weight = {int(i): float(w) for i, w in zip(classes_idx, cw)}

        print("\n===== SPLIT SIZES (70/15/15) =====")
        n = len(y)
        print("Train:", len(self.X_train), f"({len(self.X_train)/n:.3f})")
        print("Val  :", len(self.X_val),   f"({len(self.X_val)/n:.3f})")
        print("Test :", len(self.X_test),  f"({len(self.X_test)/n:.3f})")
        print("Class weights enabled.")

        print("\nLogs:")
        print("  METRICS_LOG:", METRICS_LOG)
        print("  CURVE_CSV  :", CURVE_CSV)
        print("  COMM_LOG   :", COMM_LOG)

    def get_shared(self):
        return self.model.get_layer("shared_dense").get_weights()

    def set_shared_if_valid(self, w):
        if not is_valid_shared(w):
            return False
        try:
            if any(np.isnan(np.array(x)).any() for x in w):
                return False
            self.model.get_layer("shared_dense").set_weights(list(w))
            return True
        except Exception:
            return False

    def blend(self, local_w, global_w):
        gl = float(self.gamma_local)
        gg = float(self.gamma_global)
        return [gl * l + gg * g for l, g in zip(local_w, global_w)]

    # RPC
    def rpc_get(self):
        return self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])

    def rpc_send(self, payload, num_samples):
        return self.stub.SendSharedUpdate(
            myproto_pb2.SharedUpdate(weights=payload, round=int(self.current_round), num_samples=int(num_samples)),
            metadata=[("client_id", CLIENT_ID)]
        )

    def pull_global(self):
        resp = self.rpc_get()
        return safe_unpickle(resp.weights), int(resp.round)

    # waits
    def wait_for_server(self):
        print(f"[{CLIENT_ID}] Waiting for aggregator at {SERVER_ADDRESS} ...")
        t0 = time.perf_counter()
        while True:
            try:
                resp = self.rpc_get()
                _ = int(resp.round)
                print(f"[{CLIENT_ID}] Aggregator reachable (server_round={resp.round}).")
                return
            except Exception as e:
                if time.perf_counter() - t0 > WAIT_SERVER_MAX_SEC:
                    raise RuntimeError(f"[{CLIENT_ID}] Aggregator not reachable: {e}")
                time.sleep(WAIT_SERVER_POLL_SEC)

    def align_to_server(self):
        w, r = self.pull_global()
        self.current_round = int(r)
        if is_valid_shared(w):
            self.set_shared_if_valid(w)

    def wait_barrier_advance(self, before_round: int):
        t0 = time.perf_counter()
        target = int(before_round) + 1
        while True:
            w, r = self.pull_global()
            if int(r) >= target:
                if is_valid_shared(w):
                    self.set_shared_if_valid(w)
                return int(r), float(time.perf_counter() - t0)

            if (time.perf_counter() - t0) > WAIT_ROUND_ADV_MAX_SEC:
                raise RuntimeError(
                    f"[{CLIENT_ID}] Barrier timeout: server_round still {r}, expected >= {target}. "
                    f"Check MIN_CLIENTS_TO_AGG and that all clients are running."
                )
            time.sleep(WAIT_ROUND_ADV_POLL_SEC)

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

    def run(self):
        for local_iter in range(NUM_ROUNDS):
            round_id = local_iter + 1

            # LOCAL-FIRST
            self.align_to_server()

            print(f"\n[{CLIENT_ID}] ===== Round {round_id}/{NUM_ROUNDS} | server_round={self.current_round} =====")

            train_loss, train_acc, val_loss, val_acc = self.train_one_epoch()
            local_w = self.get_shared()

            local_acc, local_mp, local_mr, local_mf1, local_wp, local_wr, local_wf1 = eval_all_metrics(
                self.model, self.X_val, self.y_val, self.num_classes
            )

            global_w, _ = self.pull_global()
            if not is_valid_shared(global_w):
                global_w = local_w

            mixed_w = self.blend(local_w, global_w)
            self.set_shared_if_valid(mixed_w)

            mixed_acc, mixed_mp, mixed_mr, mixed_mf1, mixed_wp, mixed_wr, mixed_wf1 = eval_all_metrics(
                self.model, self.X_val, self.y_val, self.num_classes
            )

            switch_on = int(mixed_mf1 >= (local_mf1 + EPS))
            if switch_on:
                sent_type = "mixed"
                weights_to_send = mixed_w
            else:
                sent_type = "local"
                weights_to_send = local_w
                self.set_shared_if_valid(local_w)

            if ADAPT_GAMMA:
                self.gamma_local, self.gamma_global, delta_f1 = update_gamma_global(
                    self.gamma_global, local_mf1, mixed_mf1
                )
            else:
                delta_f1 = float(mixed_mf1 - local_mf1)

            print(
                f"[{CLIENT_ID}] MacroF1 Local={local_mf1:.6f} | Global={mixed_mf1:.6f} | "
                f"Switch={'ON' if switch_on else 'OFF'} | Send={sent_type} | "
                f"gamma_global={self.gamma_global:.3f} (Δ={delta_f1:+.6f})"
            )

            with open(CURVE_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    int(SEED), int(round_id),
                    f"{local_mf1:.16f}", f"{mixed_mf1:.16f}",
                    int(switch_on), f"{self.gamma_global:.16f}",
                ])

            # send (retry until ok)
            payload = pickle.dumps(weights_to_send)
            bytes_sent = int(len(payload))
            server_round_before = int(self.current_round)

            send_t0 = time.perf_counter()
            while True:
                ack = self.rpc_send(payload, num_samples=self.X_train.shape[0])
                if bool(getattr(ack, "ok", False)):
                    break
                self.align_to_server()
                server_round_before = int(self.current_round)
            rtt_sec = float(time.perf_counter() - send_t0)

            # strict barrier
            new_srv_round, waited_sec = self.wait_barrier_advance(server_round_before)
            self.current_round = int(new_srv_round)

            with open(COMM_LOG, "a", newline="") as f:
                csv.writer(f).writerow([
                    int(SEED), int(round_id), int(bytes_sent),
                    int(server_round_before), int(self.current_round),
                    f"{rtt_sec:.3f}", f"{waited_sec:.3f}", now_str()
                ])

            # metrics logs (2 rows)
            log_stage_row(
                SEED, round_id, "LOCAL",
                train_loss, train_acc, val_loss, val_acc,
                local_acc, local_mp, local_mr, local_mf1,
                local_wp, local_wr, local_wf1,
                switch_on, self.gamma_local, self.gamma_global, delta_f1,
                sent_type
            )
            log_stage_row(
                SEED, round_id, "GLOBAL",
                train_loss, train_acc, val_loss, val_acc,
                mixed_acc, mixed_mp, mixed_mr, mixed_mf1,
                mixed_wp, mixed_wr, mixed_wf1,
                switch_on, self.gamma_local, self.gamma_global, delta_f1,
                sent_type
            )

            print(f"[{CLIENT_ID}] ✅ Barrier passed: {server_round_before} -> {self.current_round} (wait={waited_sec:.2f}s)")

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
    ensure_csv_headers()
    PTFLClient4_LocalFirst_StrictBarrier().run()
