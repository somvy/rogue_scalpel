import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path("results/mmlu_random_vectors")
MODEL_PATHS = {
    "Llama-3.1-8B": BASE_DIR / "meta_llama_Llama_3.1_8B_Instruct" / "mmlu_random_vectors_results.json",
    "Qwen2.5-7B": BASE_DIR / "Qwen_Qwen2.5_7B_Instruct" / "mmlu_random_vectors_results.json",
    "Falcon3-7B": BASE_DIR / "tiiuae_Falcon3_7B_Instruct" / "mmlu_random_vectors_results.json",
}


def load_results(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def coef_to_median_overall(results: dict) -> dict:
    """
    Build mapping: relative_coef (float) -> median overall accuracy across vectors.
    Includes baseline as coef 0.0.
    """
    rel_coefs = results["relative_coefficients"]
    steered = results["steered_accuracies"]
    baseline_overall = results["baseline"]["overall_accuracy"]

    # Initialize accumulator dict
    coef_to_vals = {0.0: []}
    for c in rel_coefs:
        coef_to_vals[float(c)] = []

    # Collect accuracies per vector per coef
    for _, vec_res in steered.items():
        # Baseline (same for all vectors, but we keep per-vector entries)
        coef_to_vals[0.0].append(baseline_overall)
        for c in rel_coefs:
            c_str = str(c)
            if c_str in vec_res and isinstance(vec_res[c_str], dict):
                coef_to_vals[float(c)].append(vec_res[c_str]["overall_accuracy"])

    # Median for each coef
    coef_to_median = {}
    for c, vals in coef_to_vals.items():
        if not vals:
            coef_to_median[c] = np.nan
        else:
            coef_to_median[c] = float(np.median(vals))
    return coef_to_median


def summarize_model(path: Path):
    res = load_results(path)
    coef_med = coef_to_median_overall(res)
    return coef_med, res["model_name"], res["layer_index"], res["sample_size"]


def main():
    summaries = {}
    for short_name, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"Warning: results file not found for {short_name}: {path}")
            continue
        summaries[short_name] = summarize_model(path)

    if not summaries:
        print("No result files found. Make sure you ran mmlu_eval_random.py first.")
        return

    plt.figure(figsize=(8, 5))

    for short_name, (coef_med, model_name, layer_idx, sample_size) in summaries.items():
        xs = sorted(coef_med.keys())
        ys = [coef_med[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=f"{short_name} (layer {layer_idx}, n={sample_size})")

    plt.xlabel("Relative coefficient (0 = baseline)")
    plt.ylabel("Median overall MMLU accuracy")
    plt.title("Median MMLU accuracy vs relative steering coefficient (100 random vectors)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Show plot; you can also save it by uncommenting the next line
    plt.savefig("mmlu_random_vectors_median_accuracy.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()


