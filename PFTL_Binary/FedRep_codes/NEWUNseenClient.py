# ===== Standalone CNN for NEW Client (seeded, no 'same' padding) =====
# ---- Reproducible seeding (place BEFORE heavy TF imports) ----
import os, random
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC_OPS"] = "1"

import numpy as np
random.seed(SEED); np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# ---------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score
)
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Load Dataset ==========
df = pd.read_csv("/Users/azizahalq/Desktop/FL_project3/Datasets_processed/D3_CICIOT2023/balanced_Merged63_sampled_32000.csv")
df.columns = df.columns.str.strip()
X = df.drop(columns=["binary_label"])
y = df["binary_label"].astype(int)

# numeric cleanup
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)

# ========== Train / Val / Test ==========
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
)

# ========== Scale + reshape for Conv1D ==========
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

n_features = X_train.shape[1]
X_train = X_train.reshape((-1, n_features, 1)).astype("float32")
X_val   = X_val.reshape((-1, n_features, 1)).astype("float32")
X_test  = X_test.reshape((-1, n_features, 1)).astype("float32")

# ========== Class Weights ==========
weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
cw = dict(enumerate(weights))

# ========== Build CNN (same as PTFL style; no 'same') ==========
model = models.Sequential([
    layers.Input(shape=(n_features, 1)),
    layers.Conv1D(4, 5, activation='relu'),  # default padding='valid'
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(4, activation='relu'),                        # private
    layers.Dense(4, activation='relu', name="shared_dense"),   # shared-compatible
    layers.Dropout(0.5, seed=SEED)  ,                      # private
    layers.Dense(1, activation='sigmoid')                      # output
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ========== Train ==========
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8, batch_size=256,
    class_weight=cw, verbose=1
)

# ========== Predict ==========
pred_train = model.predict(X_train, verbose=0).ravel()
pred_val   = model.predict(X_val,   verbose=0).ravel()
pred_test  = model.predict(X_test,  verbose=0).ravel()

pred_test_bin = (pred_test > 0.5).astype(int)

# ========== Metrics ==========
acc = accuracy_score(y_test, pred_test_bin)
prec = precision_score(y_test, pred_test_bin, zero_division=0)
rec = recall_score(y_test, pred_test_bin, zero_division=0)
f1 = f1_score(y_test, pred_test_bin, zero_division=0)
cm = confusion_matrix(y_test, pred_test_bin)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, pred_test_bin, zero_division=0))
print("Confusion Matrix:\n", cm)

# ========== Confusion Matrix Plot ==========
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomalous"],
            yticklabels=["Normal", "Anomalous"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix - NEW Client (Standalone CNN)")
plt.tight_layout(); plt.show()

# ========== ROC Curves ==========
def plot_roc(y_true, y_scores, label):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.2f})")

plt.figure(figsize=(8, 6))
plot_roc(y_train, pred_train, "Train")
plot_roc(y_val,   pred_val,   "Validation")
plot_roc(y_test,  pred_test,  "Test")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves - NEW Client (Standalone CNN)")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# ========== Training History ==========
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid()

plt.tight_layout(); plt.show()
