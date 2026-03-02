#!/usr/bin/env python3.10
# ============================================================
# FedAvg — ONE Local vs ONE Global curve (Average across clients)
# Using:
#   /Users/azizahalq/Desktop/PFTL_Binary/logs/fedavg_full_model/
#     client1_local_global_macro_f1_by_round.csv
#     ...
#     client6_local_global_macro_f1_by_round.csv
# Output:
#   .../plots_fedavg/FEDAVG_all_clients_local_vs_global.png
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/Users/azizahalq/Desktop/PFTL_Binary/logs/fedper_share_all_except_clf"
OUT_DIR  = os.path.join(BASE_DIR, "plots_fedavg")
os.makedirs(OUT_DIR, exist_ok=True)

CLIENT_IDS = [f"client{i}" for i in range(1, 7)]  # client1..client6

def load_client_file(base_dir: str, client_id: str) -> pd.DataFrame:
    #/Users/azizahalq/Desktop/PFTL_Binary/logs/fedper_share_all_except_clf/client1_local_global_macro_f1_by_round.csv
    path = os.path.join(base_dir, f"{client_id}_local_global_macro_f1_by_round.csv")
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()

    df["client_id"] = client_id
    df["source_file"] = os.path.basename(path)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept both styles just in case
    # required: round, local_macro_f1, global_macro_f1
    rename_map = {}
    if "server_round" in df.columns and "round" not in df.columns:
        rename_map["server_round"] = "round"
    df = df.rename(columns=rename_map)

    return df

def main():
    dfs = []
    for cid in CLIENT_IDS:
        dfi = load_client_file(BASE_DIR, cid)
        if not dfi.empty:
            dfs.append(dfi)

    if not dfs:
        print("No client*_local_global_macro_f1_by_round.csv files found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df = normalize_columns(df)

    print("Loaded rows:", len(df))
    print("Columns found:", df.columns.tolist())

    required = {"round", "local_macro_f1", "global_macro_f1"}
    if not required.issubset(set(df.columns)):
        print("Required columns missing.")
        print("Found columns:", df.columns.tolist())
        return

    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["local_macro_f1"] = pd.to_numeric(df["local_macro_f1"], errors="coerce")
    df["global_macro_f1"] = pd.to_numeric(df["global_macro_f1"], errors="coerce")
    df = df.dropna(subset=["round", "local_macro_f1", "global_macro_f1"])

    # Average across all clients for each round
    local_mean  = df.groupby("round")["local_macro_f1"].mean().sort_index()
    global_mean = df.groupby("round")["global_macro_f1"].mean().sort_index()

    print("\nLocal mean:\n", local_mean)
    print("\nGlobal mean:\n", global_mean)

    # Plot
    plt.figure(figsize=(9, 5))

    plt.plot(
        local_mean.index, local_mean.values,
        marker="s", linestyle="--", linewidth=2,
        label="Local (before sync)"
    )

    plt.plot(
        global_mean.index, global_mean.values,
        marker="o", linestyle="-", linewidth=2,
        label="Global (after sync)"
    )

    plt.title("FEDPER — Macro-F1 vs Rounds (Average across clients)", fontsize=18)
    plt.xlabel("Round", fontsize=16)
    plt.ylabel("Macro-F1", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "FEDPER_all_clients_local_vs_global.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()