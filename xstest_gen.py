import argparse
import gc
import json
import os
import re

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Batched over-refusal generation. Same steering hook / seeded sampling as the paper,
# but batches many steering vectors per forward pass (all share one prompt, so no
# padding) for throughput. One model family per GPU -> run two processes in parallel.

FAMILIES = {
    "llama": {
        "model": "unsloth/Meta-Llama-3.1-8B-Instruct",  # ungated mirror of meta-llama/Llama-3.1-8B-Instruct
        "configs": [("random", 16, 7.7544), ("sae", 19, 10.005)],
    },
    "qwen": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "configs": [("random", 14, 37.3125), ("sae", 12, 34.481508)],
    },
}

NUM_FEATURES = 200
MAX_NEW_TOKENS = 512


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, d_in, d_hidden, device, dtype=torch.bfloat16):
        super().__init__()
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.to(device, dtype)


def load_sae_matrix(model_name, layer_idx, hidden_dim, device, num_features):
    def normalize(dec):
        if dec.shape[1] == hidden_dim:
            m = dec[:num_features]
        elif dec.shape[0] == hidden_dim:
            m = dec.T[:num_features]
        else:
            raise ValueError(f"decoder shape {tuple(dec.shape)} vs hidden {hidden_dim}")
        return m.to(device)

    if "llama" in model_name.lower() and layer_idx == 19:
        name = "Llama-3.1-8B-Instruct-SAE-l19"
        path = hf_hub_download(repo_id=f"Goodfire/{name}", filename=f"{name}.pth", repo_type="model")
        sae = SparseAutoEncoder(hidden_dim, hidden_dim * 16, device)
        sae.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        mat = normalize(sae.decoder_linear.weight)
        del sae
    elif "qwen" in model_name.lower() and layer_idx == 12:
        repo = "Geaming/Qwen2.5-7B-Instruct_SAEs"
        fn = f"FAST/blocks_{layer_idx}_hook_resid_post_8X_2048_standard/sae_weights.safetensors"
        path = hf_hub_download(repo_id=repo, filename=fn, repo_type="model")
        d = load_file(path, device=str(device))
        mat = normalize(d.get("W_dec", d.get("decoder.weight")))
    else:
        raise ValueError(f"No SAE configured for {model_name} L{layer_idx}")
    mat = mat / mat.norm(dim=1, keepdim=True)
    clear_memory()
    return mat


def random_matrix(hidden_dim, device, num_features):
    torch.manual_seed(42)
    np.random.seed(42)
    mat = torch.randn(num_features, hidden_dim, device=device)
    return mat / mat.norm(dim=1, keepdim=True)


def build_mask(prompt, tokenizer, model_name, device):
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(device)
    ids = inputs.input_ids[0].tolist()
    special = set(tokenizer.all_special_ids)
    if "llama" in model_name.lower():
        for s in ("<|begin_of_text|>", "<|end_of_text|>", "\n"):
            for tid in tokenizer.encode(s, add_special_tokens=False):
                special.add(int(tid))
    mask = [1 if tid not in special else 0 for tid in ids]
    return inputs.input_ids, mask


def make_hook(steer_bh, coeff, prompt_mask):
    # steer_bh: [B, H]; prompt_mask: python list for the fixed prompt positions
    plen = len(prompt_mask)

    def hook(module, inp):
        hs = inp[0].clone()
        B, seq, H = hs.shape
        if seq <= plen:
            m = prompt_mask[:seq]
        else:
            m = prompt_mask + [1] * (seq - plen)
        mt = torch.tensor(m, dtype=hs.dtype, device=hs.device).view(1, seq, 1)
        hs += coeff * steer_bh.to(hs.dtype).unsqueeze(1) * mt
        return (hs, *inp[1:])

    return hook


def generate_batch(model, tokenizer, layer, input_ids, prompt_mask, steer_bh, coeff):
    B = steer_bh.shape[0]
    batch_ids = input_ids.repeat(B, 1)
    handle = model.model.layers[layer].register_forward_pre_hook(make_hook(steer_bh, coeff, prompt_mask))
    try:
        with torch.no_grad():
            out = model.generate(
                batch_ids,
                attention_mask=torch.ones_like(batch_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
    finally:
        handle.remove()
    gen = out[:, input_ids.shape[1]:]
    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]


def generate_baseline(model, tokenizer, input_ids):
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_config(model, tokenizer, model_name, source, layer, alpha, prompts, batch, out_root, num_features):
    device = model.device
    hidden = model.config.hidden_size
    if source == "random":
        matrix = random_matrix(hidden, device, num_features)
    else:
        matrix = load_sae_matrix(model_name, layer, hidden, device, num_features)

    model_short = model_name.split("/")[-1]
    out_dir = f"{out_root}/{model_short}_L{layer}_{source}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(matrix, os.path.join(out_dir, f"{source}_vectors_{hidden}d.pt"))
    coeff_key = str(alpha)

    for pidx, prompt in enumerate(prompts):
        safe = re.sub(r"[^a-zA-Z0-9 ]", "", prompt)[:50].strip()
        fp = os.path.join(out_dir, f"prompt_{pidx}_{safe or 'empty'}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                results = json.load(f)
        else:
            results = {"prompt": prompt, "hidden_dim": hidden, "layer_index": layer,
                       "coefficient": alpha, "baseline": "", "steered_responses": {}}

        input_ids, mask = build_mask(prompt, tokenizer, model_name, device)
        if not results["baseline"]:
            results["baseline"] = generate_baseline(model, tokenizer, input_ids)

        missing = [i for i in range(num_features) if str(i) not in results["steered_responses"]]
        for chunk in chunks(missing, batch):
            steer_bh = matrix[chunk]
            try:
                resps = generate_batch(model, tokenizer, layer, input_ids, mask, steer_bh, alpha)
            except torch.cuda.OutOfMemoryError:
                clear_memory()
                resps = []
                for j in range(0, len(chunk), max(1, batch // 2)):
                    sub = chunk[j:j + max(1, batch // 2)]
                    resps += generate_batch(model, tokenizer, layer, input_ids, mask, matrix[sub], alpha)
            for vi, r in zip(chunk, resps):
                results["steered_responses"][str(vi)] = {coeff_key: r}
            with open(fp, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  [{model_short} L{layer} {source}] prompt {pidx}/{len(prompts)} done", flush=True)

    del matrix
    clear_memory()
    print(f"DONE config {model_short} L{layer} {source} -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-root", default="./results/over_refusal")
    ap.add_argument("--sources", default="random,sae", help="comma list subset of random,sae")
    ap.add_argument("--limit-prompts", type=int, default=0, help="0=all (smoke-test helper)")
    ap.add_argument("--num-features", type=int, default=NUM_FEATURES)
    args = ap.parse_args()

    ds = load_dataset("walledai/XSTest", split="test")
    prompts = [ex["prompt"] for ex in ds if ex["label"] == "safe"]
    if args.limit_prompts:
        prompts = prompts[:args.limit_prompts]
    print("num safe prompts:", len(prompts), "batch:", args.batch, "num_features:", args.num_features, flush=True)

    spec = FAMILIES[args.family]
    model_name = spec["model"]
    wanted = args.sources.split(",")
    print("Loading", model_name, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()

    for source, layer, alpha in spec["configs"]:
        if source not in wanted:
            continue
        print(f"\n=== {model_name} {source} L{layer} alpha={alpha} ===", flush=True)
        run_config(model, tokenizer, model_name, source, layer, alpha, prompts, args.batch, args.out_root, args.num_features)


if __name__ == "__main__":
    main()
