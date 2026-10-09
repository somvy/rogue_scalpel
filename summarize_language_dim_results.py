import argparse
import csv
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize language DIM safety results")
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Directory with evaluated JSON files (e.g., .../generations/eval)",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def safe_rate(stats):
    total = stats["safe"] + stats["unsafe"]
    return (stats["safe"] / total) if total else 0.0


def unsafe_rate(stats):
    total = stats["safe"] + stats["unsafe"]
    return (stats["unsafe"] / total) if total else 0.0


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.eval_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)

    baseline_stats = {"safe": 0, "unsafe": 0}
    by_lang_coeff = defaultdict(lambda: defaultdict(lambda: {"safe": 0, "unsafe": 0}))
    token_mode = None
    n_files = 0

    for fname in os.listdir(args.eval_dir):
        if not fname.endswith(".json") or fname == "summary.json":
            continue
        fpath = os.path.join(args.eval_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        n_files += 1
        token_mode = token_mode or data.get("token_mode", "unknown")

        b = data.get("baseline_evaluation", {})
        bj = b.get("judgment")
        if bj in ("safe", "unsafe"):
            baseline_stats[bj] += 1

        for lang, coeff_map in data.get("steered_responses", {}).items():
            for coeff, eval_obj in coeff_map.items():
                if isinstance(eval_obj, dict):
                    j = eval_obj.get("judgment")
                    if j in ("safe", "unsafe"):
                        by_lang_coeff[lang][coeff][j] += 1

    baseline_unsafe = unsafe_rate(baseline_stats)
    rows = []
    for lang, coeff_map in by_lang_coeff.items():
        for coeff, stats in coeff_map.items():
            ur = unsafe_rate(stats)
            rows.append(
                {
                    "language": lang,
                    "token_mode": token_mode,
                    "coeff": coeff,
                    "unsafe_rate": ur,
                    "safe_rate": safe_rate(stats),
                    "delta_vs_baseline": ur - baseline_unsafe,
                    "n": stats["safe"] + stats["unsafe"],
                }
            )

    rows.sort(key=lambda r: r["delta_vs_baseline"], reverse=True)

    csv_path = os.path.join(out_dir, "language_dim_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "language",
                "token_mode",
                "coeff",
                "unsafe_rate",
                "safe_rate",
                "delta_vs_baseline",
                "n",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(out_dir, "language_dim_summary.json")
    payload = {
        "num_files": n_files,
        "token_mode": token_mode,
        "baseline": {
            "safe": baseline_stats["safe"],
            "unsafe": baseline_stats["unsafe"],
            "unsafe_rate": baseline_unsafe,
        },
        "rows": rows,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
