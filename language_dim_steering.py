import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LANGUAGES: List[Tuple[str, str]] = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("ru", "Russian"),
    ("zh", "Chinese"),
    ("ar", "Arabic"),
    ("hi", "Hindi"),
    ("pt", "Portuguese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("it", "Italian"),
    ("tr", "Turkish"),
    ("pl", "Polish"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("uk", "Ukrainian"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
    ("id", "Indonesian"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Language DIM steering for Llama-3.1-8B-Instruct")
    # parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer-index", type=int, default=14)
    # parser.add_argument("--layer-index", type=int, default=16)
    parser.add_argument("--num-benign-prompts", type=int, default=100)
    parser.add_argument("--num-harmful-prompts", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--coeffs", type=float, nargs="+", default=[0.5, 1, 2, 4])
    parser.add_argument(
        "--token-mode",
        type=str,
        default="last_user_token",
        choices=["last_user_token", "all_prompt_tokens_mean", "both"],
    )
    parser.add_argument(
        "--run-stage",
        type=str,
        default="both",
        choices=["compute_dim", "steer_eval", "benign_check", "smoke_test", "both"],
    )
    parser.add_argument(
        "--num-benign-check-prompts",
        type=int,
        default=20,
        help="How many benign prompts to run for visible language-shift check JSONs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=str, default="./results/language_dim")
    parser.add_argument(
        "--dim-prompt-mode",
        type=str,
        default="repeated_single",
        choices=["repeated_single", "dataset"],
        help="How to build prompts for DIM vector computation",
    )
    parser.add_argument(
        "--dim-base-prompt",
        type=str,
        default="Explain photosynthesis in simple terms.",
        help="Base prompt used when --dim-prompt-mode repeated_single",
    )
    parser.add_argument(
        "--smoke-prompt",
        type=str,
        default="Explain photosynthesis in simple terms.",
        help="Single prompt for quick smoke test",
    )
    parser.add_argument(
        "--smoke-lang",
        type=str,
        default="all",
        help="Language code to pick DIM vector for smoke test (e.g., es, ru, zh) or 'all'",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="en",
        help="Source language code for steering arithmetic r_target - r_source",
    )
    return parser.parse_args()


def safe_filename(text: str, max_len: int = 70) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", text).strip()
    return (cleaned[:max_len].strip() or "empty").replace(" ", "_")


def build_language_prompt(prompt_text: str, lang_name: str, lang_code: str) -> str:
    if lang_code == "en":
        return prompt_text
    return (
        f"Please rewrite the following request in {lang_name}, keep meaning unchanged, "
        f"and answer only in {lang_name}.\n\nRequest:\n{prompt_text}"
    )


def chat_to_inputs(tokenizer, prompt: str, model_device):
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(chat_prompt, return_tensors="pt").to(model_device)


def non_special_mask(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    special_ids = set(tokenizer.all_special_ids)
    ids = input_ids[0].tolist()
    mask = [0 if t in special_ids else 1 for t in ids]
    return torch.tensor(mask, device=input_ids.device, dtype=torch.bool)


def extract_representation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    layer_index: int,
    token_mode: str,
) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[layer_index + 1][0]  # [seq_len, hidden]
    mask = non_special_mask(tokenizer, input_ids)
    if token_mode == "last_user_token":
        idxs = torch.where(mask)[0]
        if len(idxs) == 0:
            idx = hs.shape[0] - 1
        else:
            idx = int(idxs[-1].item())
        rep = hs[idx]
    elif token_mode == "all_prompt_tokens_mean":
        if mask.any():
            rep = hs[mask].mean(dim=0)
        else:
            rep = hs.mean(dim=0)
    else:
        raise ValueError(f"Unsupported token_mode: {token_mode}")
    return rep.float().detach()


def compute_language_dim_vectors(
    model,
    tokenizer,
    benign_prompts: List[str],
    layer_index: int,
    token_mode: str,
) -> Dict[str, torch.Tensor]:
    per_lang_reps = {code: [] for code, _ in LANGUAGES}
    for prompt_text in tqdm(benign_prompts, desc=f"DIM reps ({token_mode})"):
        for lang_code, lang_name in LANGUAGES:
            lp = build_language_prompt(prompt_text, lang_name, lang_code)
            inputs = chat_to_inputs(tokenizer, lp, model.device)
            rep = extract_representation(model, tokenizer, inputs.input_ids, layer_index, token_mode)
            per_lang_reps[lang_code].append(rep)

    lang_means = {k: torch.stack(v, dim=0).mean(dim=0) for k, v in per_lang_reps.items()}
    # ReCoVeR-style content vector: mean across all languages.
    content_vec = torch.stack([lang_means[code] for code, _ in LANGUAGES], dim=0).mean(dim=0)

    # Language representations: r_lang = v_lang - c
    vectors = {}
    for lang_code, _ in LANGUAGES:
        raw_vec = lang_means[lang_code] - content_vec
        norm = raw_vec.norm().clamp_min(1e-12)
        # vectors[lang_code] = raw_vec / norm
        vectors[lang_code] = raw_vec
    return vectors


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = chat_to_inputs(tokenizer, prompt, model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    vector: torch.Tensor,
    coeff: float,
    layer_index: int,
    max_new_tokens: int,
) -> str:
    inputs = chat_to_inputs(tokenizer, prompt, model.device)
    fixed_ids = inputs.input_ids[0].tolist()
    special_ids = set(tokenizer.all_special_ids)
    mask_fixed = [1 if tid not in special_ids else 0 for tid in fixed_ids]
    fixed_len = len(mask_fixed)

    def steering_pre_hook(module, hook_inputs):
        hs = hook_inputs[0].clone()
        bsz, seq_len, _ = hs.shape
        if seq_len <= fixed_len:
            curr_mask = mask_fixed[:seq_len]
        else:
            curr_mask = mask_fixed + [1] * (seq_len - fixed_len)
        mask_tensor = torch.tensor(curr_mask, device=hs.device, dtype=hs.dtype).view(bsz, seq_len, 1)
        local_vec = vector.to(device=hs.device, dtype=hs.dtype)
        hs = hs + coeff * local_vec * mask_tensor
        return (hs, *hook_inputs[1:])

    handle = model.model.layers[layer_index].register_forward_pre_hook(steering_pre_hook)
    try:
        with torch.no_grad():
            out = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
    finally:
        handle.remove()
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def save_vectors_and_metadata(
    save_dir: str,
    vectors: Dict[str, torch.Tensor],
    model_name: str,
    layer_index: int,
    token_mode: str,
    num_benign_prompts: int,
    dim_prompt_mode: str,
    dim_base_prompt: str,
):
    os.makedirs(save_dir, exist_ok=True)
    vector_path = os.path.join(save_dir, "language_vectors.pt")
    serializable = {k: v.cpu() for k, v in vectors.items()}
    torch.save(serializable, vector_path)

    metadata = {
        "model_name": model_name,
        "layer_index": layer_index,
        "token_mode": token_mode,
        "num_benign_prompts": num_benign_prompts,
        "languages": [c for c, _ in LANGUAGES],
        "vector_path": vector_path,
        "vector_definition": "r_lang = normalize(v_lang - c), where c = mean(v_lang over all languages)",
        "dim_prompt_mode": dim_prompt_mode,
        "dim_base_prompt": dim_base_prompt if dim_prompt_mode == "repeated_single" else "",
    }
    with open(os.path.join(save_dir, "language_vectors_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def run_steering_eval(
    model,
    tokenizer,
    harmful_prompts: List[str],
    vectors: Dict[str, torch.Tensor],
    coeffs: List[float],
    out_dir: str,
    model_name: str,
    layer_index: int,
    token_mode: str,
    max_new_tokens: int,
    source_lang: str,
):
    os.makedirs(out_dir, exist_ok=True)
    for prompt_idx, prompt in enumerate(tqdm(harmful_prompts, desc=f"Steering eval ({token_mode})")):
        filename = f"prompt_{prompt_idx}_{safe_filename(prompt)}.json"
        out_path = os.path.join(out_dir, filename)
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                result = json.load(f)
        else:
            baseline = generate_baseline(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            result = {
                "prompt": prompt,
                "model_name": model_name,
                "layer_index": layer_index,
                "token_mode": token_mode,
                "steering_vectors_type": "language_dim",
                "baseline": baseline,
                "steered_responses": {},
            }

        for lang_code, _ in LANGUAGES:
            if lang_code not in result["steered_responses"]:
                result["steered_responses"][lang_code] = {}
            if source_lang not in vectors:
                raise ValueError(f"source_lang '{source_lang}' not in vectors")
            vec = vectors[lang_code] - vectors[source_lang]
            for coeff in coeffs:
                key = str(coeff)
                if key in result["steered_responses"][lang_code]:
                    continue
                response = generate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    vector=vec,
                    coeff=coeff,
                    layer_index=layer_index,
                    max_new_tokens=max_new_tokens,
                )
                result["steered_responses"][lang_code][key] = response

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def run_benign_language_check(
    model,
    tokenizer,
    benign_prompts: List[str],
    vectors: Dict[str, torch.Tensor],
    coeffs: List[float],
    out_dir: str,
    model_name: str,
    layer_index: int,
    token_mode: str,
    max_new_tokens: int,
    source_lang: str,
):
    """
    Save 'same style' JSONs as harmful eval, but on benign prompts,
    so language shift can be inspected manually.
    """
    os.makedirs(out_dir, exist_ok=True)
    for prompt_idx, prompt in enumerate(tqdm(benign_prompts, desc=f"Benign language check ({token_mode})")):
        filename = f"prompt_{prompt_idx}_{safe_filename(prompt)}.json"
        out_path = os.path.join(out_dir, filename)
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                result = json.load(f)
        else:
            baseline = generate_baseline(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            result = {
                "prompt": prompt,
                "model_name": model_name,
                "layer_index": layer_index,
                "token_mode": token_mode,
                "steering_vectors_type": "language_dim",
                "eval_type": "benign_language_check",
                "baseline": baseline,
                "steered_responses": {},
            }

        for lang_code, _ in LANGUAGES:
            if lang_code not in result["steered_responses"]:
                result["steered_responses"][lang_code] = {}
            if source_lang not in vectors:
                raise ValueError(f"source_lang '{source_lang}' not in vectors")
            vec = vectors[lang_code] - vectors[source_lang]
            for coeff in coeffs:
                key = str(coeff)
                if key in result["steered_responses"][lang_code]:
                    continue
                response = generate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    vector=vec,
                    coeff=coeff,
                    layer_index=layer_index,
                    max_new_tokens=100,
                )
                result["steered_responses"][lang_code][key] = response
                # Incremental save so first generations are visible immediately.
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def run_smoke_test(
    model,
    tokenizer,
    prompt: str,
    vectors: Dict[str, torch.Tensor],
    coeffs: List[float],
    out_dir: str,
    model_name: str,
    layer_index: int,
    token_mode: str,
    max_new_tokens: int,
    smoke_lang: str,
    source_lang: str,
):
    os.makedirs(out_dir, exist_ok=True)
    if source_lang not in vectors:
        raise ValueError(f"source_lang '{source_lang}' not in vectors")

    if smoke_lang == "all":
        target_languages = [code for code, _ in LANGUAGES]
    else:
        if smoke_lang not in vectors:
            raise ValueError(f"Unknown smoke language '{smoke_lang}'. Available: {sorted(list(vectors.keys()))}")
        target_languages = [smoke_lang]

    for target_lang in target_languages:
        out_path = os.path.join(out_dir, f"smoke_{target_lang}_{safe_filename(prompt)}.json")
        result = {
            "prompt": prompt,
            "model_name": model_name,
            "layer_index": layer_index,
            "token_mode": token_mode,
            "steering_vectors_type": "language_dim",
            "eval_type": "smoke_test",
            "source_language": source_lang,
            "smoke_language": target_lang,
            "baseline": generate_baseline(model, tokenizer, prompt, max_new_tokens=max_new_tokens),
            "steered_responses": {target_lang: {}},
        }

        vec = vectors[target_lang] - vectors[source_lang]
        for coeff in coeffs:
            key = str(coeff)
            response = generate_with_steering(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                vector=vec,
                coeff=coeff,
                layer_index=layer_index,
                max_new_tokens=max_new_tokens,
            )
            result["steered_responses"][target_lang][key] = response
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Saved smoke test JSON to {out_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    token_modes = [args.token_mode] if args.token_mode != "both" else ["last_user_token", "all_prompt_tokens_mean"]

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    # Prompts used for DIM vector computation
    if args.dim_prompt_mode == "repeated_single":
        benign_prompts = [args.dim_base_prompt] * args.num_benign_prompts
    else:
        benign_ds = load_dataset("tatsu-lab/alpaca", split="train")
        benign_prompts = [x["instruction"] for x in benign_ds.select(range(args.num_benign_prompts))]

    harmful_ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    harmful_prompts = [x["Goal"] for x in harmful_ds.select(range(args.num_harmful_prompts))]
    benign_check_prompts = benign_prompts[: args.num_benign_check_prompts]

    model_tag = args.model_name.split("/")[-1]
    base_dir = os.path.join(args.results_root, f"{model_tag}_layer{args.layer_index}")
    os.makedirs(base_dir, exist_ok=True)

    for token_mode in token_modes:
        token_dir = os.path.join(base_dir, token_mode)
        vectors_dir = os.path.join(token_dir, "vectors")
        generations_dir = os.path.join(token_dir, "generations")
        benign_check_dir = os.path.join(token_dir, "benign_language_check")
        smoke_dir = os.path.join(token_dir, "smoke_test")
        vector_path = os.path.join(vectors_dir, "language_vectors.pt")

        vectors = None
        if args.run_stage in ("compute_dim", "both"):
            vectors = compute_language_dim_vectors(
                model=model,
                tokenizer=tokenizer,
                benign_prompts=benign_prompts,
                layer_index=args.layer_index,
                token_mode=token_mode,
            )
            save_vectors_and_metadata(
                save_dir=vectors_dir,
                vectors=vectors,
                model_name=args.model_name,
                layer_index=args.layer_index,
                token_mode=token_mode,
                num_benign_prompts=args.num_benign_prompts,
                dim_prompt_mode=args.dim_prompt_mode,
                dim_base_prompt=args.dim_base_prompt,
            )
            print(f"Saved vectors for {token_mode} to {vectors_dir}")

        if args.run_stage in ("steer_eval", "both", "benign_check", "smoke_test"):
            if vectors is None:
                if not os.path.exists(vector_path):
                    print(
                        f"Vectors not found for {token_mode} at {vector_path}. "
                        "Computing them now..."
                    )
                    vectors = compute_language_dim_vectors(
                        model=model,
                        tokenizer=tokenizer,
                        benign_prompts=benign_prompts,
                        layer_index=args.layer_index,
                        token_mode=token_mode,
                    )
                    save_vectors_and_metadata(
                        save_dir=vectors_dir,
                        vectors=vectors,
                        model_name=args.model_name,
                        layer_index=args.layer_index,
                        token_mode=token_mode,
                        num_benign_prompts=args.num_benign_prompts,
                        dim_prompt_mode=args.dim_prompt_mode,
                        dim_base_prompt=args.dim_base_prompt,
                    )
                    print(f"Saved vectors for {token_mode} to {vectors_dir}")
                else:
                    vectors = torch.load(vector_path, map_location="cpu")
            if vectors is None:
                vectors = torch.load(vector_path, map_location="cpu")

        if args.run_stage in ("benign_check", "both"):
            run_benign_language_check(
                model=model,
                tokenizer=tokenizer,
                benign_prompts=benign_check_prompts,
                vectors=vectors,
                coeffs=args.coeffs,
                out_dir=benign_check_dir,
                model_name=args.model_name,
                layer_index=args.layer_index,
                token_mode=token_mode,
                max_new_tokens=args.max_new_tokens,
                source_lang=args.source_lang,
            )
            print(f"Saved benign language-check JSONs for {token_mode} to {benign_check_dir}")

        if args.run_stage in ("steer_eval", "both"):
            run_steering_eval(
                model=model,
                tokenizer=tokenizer,
                harmful_prompts=harmful_prompts,
                vectors=vectors,
                coeffs=args.coeffs,
                out_dir=generations_dir,
                model_name=args.model_name,
                layer_index=args.layer_index,
                token_mode=token_mode,
                max_new_tokens=args.max_new_tokens,
                source_lang=args.source_lang,
            )
            print(f"Saved steering generations for {token_mode} to {generations_dir}")

        if args.run_stage == "smoke_test":
            run_smoke_test(
                model=model,
                tokenizer=tokenizer,
                prompt=args.smoke_prompt,
                vectors=vectors,
                coeffs=args.coeffs,
                out_dir=smoke_dir,
                model_name=args.model_name,
                layer_index=args.layer_index,
                token_mode=token_mode,
                max_new_tokens=args.max_new_tokens,
                smoke_lang=args.smoke_lang,
                source_lang=args.source_lang,
            )


if __name__ == "__main__":
    main()
