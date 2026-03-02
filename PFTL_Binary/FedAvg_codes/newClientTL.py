# ===== Unseen Client: simple γ-blend of shared weights (seeded) =====
import os, random
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------- simple γ-blend ----------
def blend(local, global_, gamma=(0.5, 0.5)):
    return [gamma[0]*l + gamma[1]*g for l, g in zip(local, global_)]

# ---------- model ----------
def build_client_cnn(input_steps):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_steps, 1)),
        tf.keras.layers.Conv1D(4, 5, activation='relu'),   # padding='valid'
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation='relu'),                       # private
        tf.keras.layers.Dense(4, activation='relu', name="shared_dense"),  # shared
        tf.keras.layers.Dropout(0.5, seed=SEED),
        tf.keras.layers.Dense(2, activation='relu'),                       # private
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ---------- data ----------
DATA_PATH = "/Users/azizahalq/Desktop/FL_project3/Datasets_processed/D3_CICIOT2023/balanced_Merged63_sampled_32000.csv"
MODEL_DIR = "/Users/azizahalq/Desktop/FL_project3"  # contains global_model_round{r}.keras

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
X = df.drop(columns=["binary_label"]).apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf],0).fillna(0)
y = df["binary_label"].astype(int)

X_trf, X_te, y_trf, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_trf, y_trf, test_size=0.1, random_state=42, stratify=y_trf)

sc = StandardScaler()
X_tr = sc.fit_transform(X_tr); X_va = sc.transform(X_va); X_te = sc.transform(X_te)
n_feat = X_tr.shape[1]
X_tr = X_tr.reshape((-1, n_feat, 1)).astype("float32")
X_va = X_va.reshape((-1, n_feat, 1)).astype("float32")
X_te = X_te.reshape((-1, n_feat, 1)).astype("float32")

w = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_tr), y=y_tr)
cw = dict(enumerate(w))

# ---------- evaluate rounds ----------
gamma = (0.5, 0.5)   # (local, global)
results = []

for r in range(1, 9):
    path = os.path.join(MODEL_DIR, f"global_model_round{r}.keras")
    if not os.path.exists(path):
        print(f"Round {r}: missing model -> {path}")
        continue

    # get GLOBAL shared weights
    gmodel = tf.keras.models.load_model(path, compile=False)
    global_shared = gmodel.get_layer("shared_dense").get_weights()

    # build local model and get LOCAL shared weights
    model = build_client_cnn(n_feat)
    local_shared = model.get_layer("shared_dense").get_weights()

    # γ-blend then set
    model.get_layer("shared_dense").set_weights(blend(local_shared, global_shared, gamma))

    # train + eval
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=8, batch_size=256, class_weight=cw, verbose=0)
    pr = model.predict(X_te, verbose=0).ravel()
    yb = (pr > 0.5).astype(int)

    acc = accuracy_score(y_te, yb)
    prec = precision_score(y_te, yb, zero_division=0)
    rec = recall_score(y_te, yb, zero_division=0)
    f1 = f1_score(y_te, yb, zero_division=0)

    print(f"Round {r} — Acc {acc:.6f}  Prec {prec:.6f}  Rec {rec:.6f}  F1 {f1:.6f}")
    results.append((r, acc, prec, rec, f1))

# optional: save CSV
pd.DataFrame(results, columns=["Round","Accuracy","Precision","Recall","F1_Score"]).to_csv(
    "/Users/azizahalq/Desktop/FL_project3/unseen_simple_gamma(0.1_0.9)_results.csv", index=False
)
