#!/usr/bin/env python3.10
# ============================================================
# PFTL — ONE Local vs ONE Global curve
# Using client*_local_global_macro_f1_by_round.csv
# ============================================================

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/Users/azizahalq/Desktop/PFTL_Binary/logs"
OUT_DIR  = os.path.join(BASE_DIR, "plots_pftl")
os.makedirs(OUT_DIR, exist_ok=True)

def load_all_files(base_dir):
    pattern = os.path.join(base_dir, "pftl_gamma_*", "*_local_global_macro_f1_by_round.csv")
    files = glob.glob(pattern)

    if not files:
        print(" No local_global_macro_f1 files found.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

def main():
    df = load_all_files(BASE_DIR)

    if df.empty:
        return

    print("Loaded rows:", len(df))
    print("Columns found:", df.columns.tolist())

    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns:
    # round, local_macro_f1, global_macro_f1
    required = {"round", "local_macro_f1", "global_macro_f1"}
    if not required.issubset(set(df.columns)):
        print("Required columns missing.")
        print("Found columns:", df.columns.tolist())
        return

    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["local_macro_f1"] = pd.to_numeric(df["local_macro_f1"], errors="coerce")
    df["global_macro_f1"] = pd.to_numeric(df["global_macro_f1"], errors="coerce")

    df = df.dropna(subset=["round", "local_macro_f1", "global_macro_f1"])

    # ===== Merge all clients + all gammas =====
    local_mean  = df.groupby("round")["local_macro_f1"].mean().sort_index()
    global_mean = df.groupby("round")["global_macro_f1"].mean().sort_index()

    print("\nLocal mean:\n", local_mean)
    print("\nGlobal mean:\n", global_mean)

    # ===== Plot =====
    plt.figure(figsize=(9, 5))

    plt.plot(local_mean.index, local_mean.values,
         marker="s", linestyle="--", linewidth=2,
         label="Local (before sync)")

    plt.plot(global_mean.index, global_mean.values,
         marker="o", linestyle="-", linewidth=2,
         label="Global (after sync)")

    plt.title("PFTL — Macro-F1 vs Rounds (Avrage across all clients)", fontsize=18)
    plt.xlabel("Round", fontsize=16)
    plt.ylabel("Macro-F1", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "pftl_all_merged_local_vs_global.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("\n Saved:", out_path)

if __name__ == "__main__":
    main()