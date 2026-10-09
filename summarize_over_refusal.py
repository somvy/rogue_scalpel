import json
import os
import glob

# Aggregate the over-refusal eval summaries into one table:
# baseline vs mean steered over-refusal rate per config.

ROOT = "./results/over_refusal"

rows = []
for summary_path in sorted(glob.glob(os.path.join(ROOT, "*", "eval", "summary.json"))):
    config = summary_path.split("/")[-3]
    with open(summary_path) as f:
        s = json.load(f)
    rows.append((
        config,
        s.get("num_files", 0),
        s.get("baseline_over_refusal_rate", 0.0),
        s.get("steered_over_refusal_rate", 0.0),
    ))

print("\nOver-refusal on XSTest safe prompts (refusal rate)")
print(f"{'config':<40} {'prompts':>8} {'baseline':>10} {'steered':>10}")
for config, n, base, steer in rows:
    print(f"{config:<40} {n:>8} {base*100:>9.2f}% {steer*100:>9.2f}%")

if not rows:
    print("no eval summaries found under", ROOT)
