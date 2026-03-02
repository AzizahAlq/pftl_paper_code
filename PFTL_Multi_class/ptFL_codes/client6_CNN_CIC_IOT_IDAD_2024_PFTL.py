#!/usr/bin/env python3.10
# ============================================================
# client6_IDAD_PTFL_intelligent_v2_fixed_match_standalone.py
# ============================================================

import os, time, csv, pickle, random, grpc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tensorflow.keras import layers, models, optimizers

import myproto_pb2
import myproto_pb2_grpc

# --- Reproducibility ---
SEED = 45
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# --- Config ---
CLIENT_ID = "client6"
SERVER_ADDRESS = "localhost:50051"
NUM_ROUNDS = 20

CSV_PATH = "/Users/azizahalq/Desktop/PFTL_project3/Datasets_processed/D6_CIC_IOT_IDAD_2024/Client6_Balanced_Final.csv"
LABEL_COL = "Label"

OUT_DIR = f"/Users/azizahalq/Desktop/PFTL_project3/ptfl_logs/{CLIENT_ID}_intelligent_v2"
os.makedirs(OUT_DIR, exist_ok=True)
CURVE_CSV = os.path.join(OUT_DIR, f"{CLIENT_ID}_local_vs_global_curve.csv")

# unified split rule
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)  # 0.17647058823529413

# Intelligent Hyperparams
EPS, GAMMA_GLOBAL_INIT = 0.001, 0.50
ETA, TAU = 0.05, 0.05
GAMMA_MIN, GAMMA_MAX = 0.10, 0.90
BARRIER_POLL_SEC = 1.0

PRIVATE_DIM, SHARED_DIM = 16, 8
LR = 1e-3
BATCH_SIZE = 256

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

def clip(x, lo, hi):
    return float(max(lo, min(hi, x)))

class Client6IntelligentFixed:
    def __init__(self):
        if not os.path.exists(CURVE_CSV):
            with open(CURVE_CSV, "w", newline="") as f:
                csv.writer(f).writerow(["seed","round","local_macro_f1","global_macro_f1","switch_on","gamma_global"])

        self.gamma_glob = float(GAMMA_GLOBAL_INIT)

        self.channel = grpc.insecure_channel(SERVER_ADDRESS)
        self.stub = myproto_pb2_grpc.AggregatorStub(self.channel)

        self._load_data()
        self.model = build_cnn(self.input_shape, self.num_classes)

        self._wait_for_aggregator()

    def _load_data(self):
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

        self.le = LabelEncoder()
        y = self.le.fit_transform(y_raw).astype(int)
        self.le_classes = self.le.classes_
        self.num_classes = int(len(self.le_classes))

        X = X_df.values.astype(np.float32)

        # unified split rule
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=VAL_SIZE_FROM_TRAIN, random_state=SEED, stratify=y_tr)

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_va = imp.transform(X_va)
        X_te = imp.transform(X_te)

        s = StandardScaler()
        self.X_train = s.fit_transform(X_tr)[..., None]
        self.X_val   = s.transform(X_va)[..., None]
        self.X_test  = s.transform(X_te)[..., None]

        self.y_train, self.y_val, self.y_test = y_tr, y_va, y_te
        self.input_shape = (self.X_train.shape[1], 1)

    def _pull_global(self):
        resp = self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])
        w = pickle.loads(resp.weights) if resp.weights else None
        return w, int(resp.round)

    def _wait_for_aggregator(self):
        while True:
            try:
                w, r = self._pull_global()
                self.current_round = int(r)
                if w and isinstance(w, (list, tuple)) and len(w) == 2:
                    self.model.get_layer("shared_dense").set_weights(w)
                print(f"[{CLIENT_ID}] Aggregator reachable (server_round={self.current_round})")
                break
            except Exception:
                time.sleep(2)

    def run(self):
        for r_idx in range(NUM_ROUNDS):
            round_id = r_idx + 1
            print(f"\n[{CLIENT_ID}] ===== Round {round_id}/{NUM_ROUNDS} (server_round={self.current_round}) =====")

            # 1) Local train
            self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=BATCH_SIZE, verbose=1)

            local_w = self.model.get_layer("shared_dense").get_weights()
            y_lp = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
            _, _, l_f1, _ = precision_recall_fscore_support(self.y_val, y_lp, average="macro", zero_division=0)

            # 2) Mix with global
            glob_w, _ = self._pull_global()
            if not (isinstance(glob_w, (list, tuple)) and len(glob_w) == 2):
                glob_w = local_w

            mixed_w = [(1.0 - self.gamma_glob) * l + self.gamma_glob * g for l, g in zip(local_w, glob_w)]
            self.model.get_layer("shared_dense").set_weights(mixed_w)

            y_mp = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
            _, _, m_f1, _ = precision_recall_fscore_support(self.y_val, y_mp, average="macro", zero_division=0)

            # 3) Switch + gamma
            switch = int(m_f1 >= (l_f1 + EPS))
            if not switch:
                self.model.get_layer("shared_dense").set_weights(local_w)

            delta = float(m_f1 - l_f1)
            self.gamma_glob = clip(self.gamma_glob + ETA * np.tanh(delta / max(TAU, 1e-8)), GAMMA_MIN, GAMMA_MAX)

            print(f"[{CLIENT_ID}] MacroF1 Local={l_f1:.6f} | Mixed={m_f1:.6f} | Switch={'ON' if switch else 'OFF'} | Gamma={self.gamma_glob:.4f}")

            with open(CURVE_CSV, "a", newline="") as f:
                csv.writer(f).writerow([SEED, round_id, f"{l_f1:.16f}", f"{m_f1:.16f}", switch, f"{self.gamma_glob:.16f}"])

            # 4) Push update
            payload = pickle.dumps(self.model.get_layer("shared_dense").get_weights())
            server_round_before = int(self.current_round)
            self.stub.SendSharedUpdate(
                myproto_pb2.SharedUpdate(weights=payload, round=int(self.current_round), num_samples=int(len(self.X_train))),
                metadata=[("client_id", CLIENT_ID)]
            )

            # 5) Strict barrier (wait server round increments)
            target = server_round_before + 1
            while True:
                w_new, r_check = self._pull_global()
                if int(r_check) >= target:
                    self.current_round = int(r_check)
                    if w_new and isinstance(w_new, (list, tuple)) and len(w_new) == 2:
                        self.model.get_layer("shared_dense").set_weights(w_new)
                    print(f"[{CLIENT_ID}] Barrier passed -> server_round={self.current_round}")
                    break
                time.sleep(BARRIER_POLL_SEC)

        print(f"\n[{CLIENT_ID}] ===== FINAL TEST EVALUATION =====")
        probs = self.model.predict(self.X_test, verbose=0)
        y_pred = (probs.reshape(-1) >= 0.5).astype(int) if self.num_classes == 2 else np.argmax(probs, axis=1)

        acc = float(accuracy_score(self.y_test, y_pred))
        mp, mr, mf1, _ = precision_recall_fscore_support(self.y_test, y_pred, average="macro", zero_division=0)
        wp, wr, wf1, _ = precision_recall_fscore_support(self.y_test, y_pred, average="weighted", zero_division=0)

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
        print(classification_report(self.y_test, y_pred, target_names=self.le_classes, zero_division=0))

        print("\n==== Confusion Matrix ====")
        print(confusion_matrix(self.y_test, y_pred))

        print("\n===== SAVED (CSV ONLY) =====")
        print("Curve CSV:", CURVE_CSV)

if __name__ == "__main__":
    Client6IntelligentFixed().run()
