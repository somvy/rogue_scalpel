import os, json, re, argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from xstest_gen import (FAMILIES, NUM_FEATURES, load_sae_matrix, random_matrix,
                        build_mask, generate_baseline, generate_batch, chunks, clear_memory)

# GPU1 helper: same prompts, same ascending order, same pidx->filename scheme as xstest_gen.
# Only processes pidx >= --start, so it stays ahead of the GPU0 process and never
# writes the same file concurrently. Resume is per-prompt+per-vector (identical to run_config).

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True)
ap.add_argument("--source", required=True)  # random | sae
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--out-root", default="./results/over_refusal")
ap.add_argument("--num-features", type=int, default=NUM_FEATURES)
args = ap.parse_args()

ds = load_dataset("walledai/XSTest", split="test")
prompts = [ex["prompt"] for ex in ds if ex["label"] == "safe"]

spec = FAMILIES[args.family]
model_name = spec["model"]
layer, alpha = next((l, a) for s, l, a in spec["configs"] if s == args.source)
print(f"[tail] {model_name} {args.source} L{layer} alpha={alpha} start={args.start} n={len(prompts)}", flush=True)

tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
hidden = model.config.hidden_size
device = model.device

matrix = random_matrix(hidden, device, args.num_features) if args.source == "random" \
    else load_sae_matrix(model_name, layer, hidden, device, args.num_features)

model_short = model_name.split("/")[-1]
out_dir = f"{args.out_root}/{model_short}_L{layer}_{args.source}"
os.makedirs(out_dir, exist_ok=True)
coeff_key = str(alpha)

for pidx, prompt in enumerate(prompts):
    if pidx < args.start:
        continue
    safe = re.sub(r"[^a-zA-Z0-9 ]", "", prompt)[:50].strip()
    fp = os.path.join(out_dir, f"prompt_{pidx}_{safe or 'empty'}.json")
    if os.path.exists(fp):
        with open(fp) as f:
            results = json.load(f)
    else:
        results = {"prompt": prompt, "hidden_dim": hidden, "layer_index": layer,
                   "coefficient": alpha, "baseline": "", "steered_responses": {}}
    input_ids, mask = build_mask(prompt, tok, model_name, device)
    if not results["baseline"]:
        results["baseline"] = generate_baseline(model, tok, input_ids)
    missing = [i for i in range(args.num_features) if str(i) not in results["steered_responses"]]
    for chunk in chunks(missing, args.batch):
        resps = generate_batch(model, tok, layer, input_ids, mask, matrix[chunk], alpha)
        for vi, r in zip(chunk, resps):
            results["steered_responses"][str(vi)] = {coeff_key: r}
        with open(fp, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [tail L{layer} {args.source}] prompt {pidx}/{len(prompts)} done", flush=True)

print(f"[tail] DONE from start={args.start}", flush=True)
