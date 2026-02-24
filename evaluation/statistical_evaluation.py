import pickle

import numpy as np
from scipy.stats import wilcoxon

from utils.build_result_paths import build_models_result_paths

# =====================================================
# Configuration
# =====================================================

NUM_RUNS = 5
DATASET = "email1"

AP_KEY = "val_aps"
MODE = "mean"
ALPHA = 0.05

models = {
    "DyRep": ("dyrep", "rnn", None),
    "JODIE": ("jodie", "rnn", None),
    "TGN_NoMem": ("tgn", "no-mem", None),
    "TGN_Id": ("tgn", "id", None),
    "TGN_Time": ("tgn", "time", None),
    "TGN_SEAL": ("tgn", "seal", "2h"),
}

results = build_models_result_paths(DATASET, NUM_RUNS, models)


# =====================================================
# Helper Functions
# =====================================================

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_ap(res, key=AP_KEY, mode="best"):
    aps = np.asarray(res[key])

    if mode == "best":
        return aps.max()
    elif mode == "mean":
        return aps.mean()
    else:
        raise ValueError("mode must be 'best' or 'mean'")


def collect_model_aps(file_list, mode):
    return np.array([
        extract_ap(read_pickle(p), mode=mode)
        for p in file_list
    ])


# =====================================================
# Main Evaluation
# =====================================================

def main():
    print("=" * 60)
    print(f"AP evaluation mode: {MODE.upper()}")
    print(f"AP source key: {AP_KEY}")
    print("=" * 60)

    # Collect APs
    model_aps = {
        name: collect_model_aps(paths, MODE)
        for name, paths in results.items()
    }

    # Print mean ± std
    print(f"\nMean ± Std AP over {NUM_RUNS} runs:\n")
    for name, aps in model_aps.items():
        print(f"{name:12s}: {aps.mean():.4f} ± {aps.std():.4f}")

    # Statistical comparison
    print("\n" + "=" * 60)
    print("Paired Wilcoxon Signed-Rank Test")
    print("H1: TGN-Time > Baseline")
    print("=" * 60 + "\n")

    tgn_seal_ap = model_aps["TGN_SEAL"]

    for baseline in ["DyRep", "JODIE", "TGN_NoMem", "TGN_Id", "TGN_Time"]:
        baseline_ap = model_aps[baseline]

        stat, p_value = wilcoxon(
            tgn_seal_ap,
            baseline_ap,
            alternative="greater"
        )

        significant = "✅ YES" if p_value < ALPHA else "❌ NO"

        print(f"TGN-SEAL vs {baseline}")
        print(f"  p-value      : {p_value:.4e}")
        print(f"  significant? : {significant}")
        print("-" * 45)


if __name__ == "__main__":
    main()
