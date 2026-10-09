import argparse
import gc
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate random SAE feature steering on MMLU (Llama-3.x), analogous to mmlu_eval_random.py.\n"
            "Samples random SAE features from a Goodfire SAE repo, uses their decoder vectors as steering directions,\n"
            "and sweeps steering coefficients while measuring MMLU accuracy."
        )
    )

    p.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id for the base model.",
    )
    p.add_argument(
        "--layer_index",
        type=int,
        default=19,
        help="Transformer layer index to inject the steering vector into (forward pre-hook).",
    )
    p.add_argument(
        "--expansion_factor",
        type=int,
        default=16,
        help="SAE expansion factor (Goodfire Llama SAEs are typically 16x).",
    )
    p.add_argument(
        "--num_features",
        type=int,
        default=300,
        help="Number of random SAE features to evaluate.",
    )
    p.add_argument(
        "--feature_selection",
        type=str,
        default="dangerous",
        choices=["random", "dangerous"],
        help=(
            "How to choose SAE features: "
            "'random' samples features uniformly at random; "
            "'dangerous' selects top-k SAE features that break the most JailbreakBench prompts "
            "(requires --jailbreak_dir containing eval JSONs with per-feature judgments)."
        ),
    )
    p.add_argument(
        "--jailbreak_dir",
        type=str,
        default="./results/all_prompts/meta-llama/Llama-3.1-8B-Instruct_19_sae/eval",
        help=(
            "Directory containing JailbreakBench eval JSONs with a top-level 'steered_responses' mapping "
            "feature_idx -> coeff -> {judgment: ...}. Required when --feature_selection=dangerous."
        ),
    )
    p.add_argument(
        "--jbb_dataset",
        type=str,
        default="JailbreakBench/JBB-Behaviors",
        help="JailbreakBench dataset name (used for category mapping).",
    )
    p.add_argument(
        "--jbb_config",
        type=str,
        default="behaviors",
        help="JailbreakBench dataset config name.",
    )
    p.add_argument(
        "--jbb_split",
        type=str,
        default="harmful",
        help="JailbreakBench dataset split name.",
    )
    p.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="Number of MMLU questions to sample across all subjects.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Max tokens to generate for each MMLU question (kept small for letter-only answer).",
    )
    p.add_argument(
        "--base_coefficient",
        type=float,
        default=13.34,
        help="Base steering coefficient; actual coefficients are base * relative coefficients.",
    )
    p.add_argument(
        "--relative_coefficients",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
        help="Relative coefficient sweep values.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for feature sampling and dataset sampling.",
    )
    p.add_argument(
        "--output_base_dir",
        type=str,
        default="./results/mmlu_random_sae_features",
        help="Base output directory.",
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional HF token (otherwise use HF_TOKEN env var / cached credentials).",
    )

    return p.parse_args()


class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f


def load_sae_from_goodfire(
    sae_name: str,
    *,
    d_model: int,
    expansion_factor: int,
    device: torch.device,
) -> SparseAutoEncoder:
    file_path = hf_hub_download(
        repo_id=f"Goodfire/{sae_name}",
        filename=f"{sae_name}.pth",
        repo_type="model",
    )

    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    sae_dict = torch.load(file_path, weights_only=True, map_location=device)
    sae.load_state_dict(sae_dict)
    sae.eval()
    return sae


def select_dangerous_features_from_jbb(
    *,
    jailbreak_dir: str,
    num_features: int,
    jbb_dataset: str,
    jbb_config: str,
    jbb_split: str,
) -> list[int]:
    """
    Replicates the core logic from mmlu_eval.py:
    - Load JailbreakBench dataset to map prompt_i -> Category
    - Read eval JSON files in jailbreak_dir (expects 'steered_responses' with 'judgment')
    - Count unsafe judgments per feature (across all files / coefficients)
    - Return top-k feature indices by break count
    """
    if not jailbreak_dir:
        raise ValueError("--jailbreak_dir is required when --feature_selection=dangerous")
    if not os.path.isdir(jailbreak_dir):
        raise ValueError(f"--jailbreak_dir does not exist or is not a directory: {jailbreak_dir}")

    print("Analyzing JailbreakBench results to find top dangerous SAE features...")
    dataset = load_dataset(jbb_dataset, jbb_config, split=jbb_split)
    categories = [example["Category"] for example in dataset]
    category_mapping = {f"prompt_{i}": categories[i] for i in range(len(categories))}

    json_files = [f for f in os.listdir(jailbreak_dir) if f.endswith(".json")]
    if not json_files:
        raise ValueError(f"No .json files found in jailbreak_dir={jailbreak_dir}")

    # feature_idx -> break_count
    vector_break_counts: dict[int, int] = defaultdict(int)

    for json_file in tqdm(json_files, desc="Scanning Jailbreak eval JSONs"):
        # Match mmlu_eval.py mapping convention (prompt_{i}_...)
        parts = json_file.split("_")
        if len(parts) < 2:
            continue
        json_file_short = parts[0] + "_" + parts[1]
        if json_file_short not in category_mapping:
            continue

        with open(os.path.join(jailbreak_dir, json_file), "r") as f:
            data = json.load(f)

        steered = data.get("steered_responses")
        if not isinstance(steered, dict):
            continue

        # IMPORTANT: we only count entries that already have explicit judgments.
        for feature_idx, feature_data in steered.items():
            if not isinstance(feature_data, dict):
                continue
            for _, response_data in feature_data.items():
                if isinstance(response_data, dict) and "judgment" in response_data:
                    judgment = response_data["judgment"]
                else:
                    # Don't guess judgments from raw strings; skip if missing.
                    continue

                if isinstance(judgment, str) and judgment.strip().lower() == "unsafe":
                    try:
                        vector_break_counts[int(feature_idx)] += 1
                    except Exception:
                        continue

    top = sorted(vector_break_counts.items(), key=lambda x: x[1], reverse=True)[:num_features]
    top_indices = [idx for idx, _ in top]
    if not top_indices:
        raise ValueError(
            "Found 0 dangerous features (no explicit 'judgment'=='unsafe' entries). "
            "Check your eval JSON format in --jailbreak_dir."
        )

    print(f"Selected {len(top_indices)} dangerous features (top-{num_features}).")
    return top_indices


def get_mmlu_prompt(example):
    subject = example["subject"]
    choices = example["choices"]
    choice_a, choice_b, choice_c, choice_d = choices[0], choices[1], choices[2], choices[3]
    question = example["question"]
    test_part = (
        f"Question: {question}\n\n"
        f"A. {choice_a}\n"
        f"B. {choice_b}\n"
        f"C. {choice_c}\n"
        f"D. {choice_d}\n\n"
        "Answer:"
    )
    return (
        f"The following is a multiple choice question about {subject}. "
        "Choose the correct answer (A, B, C, or D) and respond with only the letter. "
        "No explanation or reasoning needed.\n\n"
        + test_part
    )


def extract_answer(generated_text):
    generated_text = generated_text.strip().upper()
    matches = re.findall(r"\b([A-D])\b", generated_text)
    return matches[-1] if matches else None


def get_chat_prompt(messages, tokenizer, model_name: str):
    # Match the defensive pattern in mmlu_eval_random.py
    if "falcon" in model_name.lower():
        return f"User:{messages[0]['content']}\nAssistant:"
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"Warning: Could not apply chat template: {e}")
        return messages[0]["content"]


def generate_with_steering(
    prompt_content: str,
    steering_vector: torch.Tensor,
    coefficient: float,
    *,
    model,
    tokenizer,
    layer_index: int,
    model_name: str,
    max_new_tokens: int,
):
    messages = [{"role": "user", "content": prompt_content}]
    chat_prompt = get_chat_prompt(messages, tokenizer, model_name)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    fixed_prompt_ids = inputs.input_ids[0].tolist()
    special_ids = set(tokenizer.all_special_ids)
    mask_fixed = [1 if token_id not in special_ids else 0 for token_id in fixed_prompt_ids]
    fixed_prompt_length = len(fixed_prompt_ids)

    def steering_pre_hook(module, input):
        hidden_states = input[0].clone()
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len <= fixed_prompt_length:
            current_mask = mask_fixed[:seq_len]
        else:
            current_mask = mask_fixed + [1] * (seq_len - fixed_prompt_length)

        mask_tensor = torch.tensor(
            current_mask,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).view(batch_size, seq_len, 1)

        hidden_states += coefficient * steering_vector * mask_tensor
        return (hidden_states, *input[1:])

    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.layers

    handle = layers[layer_index].register_forward_pre_hook(steering_pre_hook)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    handle.remove()
    return tokenizer.decode(outputs[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)


def compute_accuracies(
    mmlu_test,
    *,
    model,
    tokenizer,
    layer_index: int,
    model_name: str,
    max_new_tokens: int,
    is_baseline: bool,
    vec: torch.Tensor | None = None,
    coeff: float | None = None,
):
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for example in tqdm(mmlu_test, desc="Processing MMLU questions"):
        prompt_content = get_mmlu_prompt(example)
        if is_baseline:
            messages = [{"role": "user", "content": prompt_content}]
            chat_prompt = get_chat_prompt(messages, tokenizer, model_name)
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            generated = tokenizer.decode(outputs[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        else:
            assert vec is not None and coeff is not None
            generated = generate_with_steering(
                prompt_content,
                vec,
                coeff,
                model=model,
                tokenizer=tokenizer,
                layer_index=layer_index,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )

        pred = extract_answer(generated)
        correct_letter = chr(65 + example["answer"])
        subject = example["subject"]
        subject_stats[subject]["total"] += 1
        if pred == correct_letter:
            subject_stats[subject]["correct"] += 1

    total_correct = sum(stats["correct"] for stats in subject_stats.values())
    total_questions = sum(stats["total"] for stats in subject_stats.values())
    per_subject = {
        sub: (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0.0
        for sub, stats in subject_stats.items()
    }
    overall = (total_correct / total_questions) if total_questions > 0 else 0.0
    return {"overall_accuracy": overall, "per_subject": per_subject}


def main():
    args = parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_base_dir, exist_ok=True)

    # Load MMLU once
    print("Loading all MMLU categories and sampling questions...")
    configs = get_dataset_config_names("cais/mmlu")
    print(f"Found {len(configs)} subjects in MMLU")

    all_tests = []
    for config_name in tqdm(configs, desc="Loading MMLU subjects"):
        try:
            ds = load_dataset("cais/mmlu", config_name, split="test")
            all_tests.append(ds)
        except Exception as e:
            print(f"Failed to load {config_name}: {e}")

    combined_test = concatenate_datasets(all_tests).shuffle(seed=args.seed)
    mmlu_test = combined_test.select(range(args.sample_size))
    print(f"Sampled {len(mmlu_test)} questions from {len(configs)} subjects")

    model_name = args.model_name
    model_safe_name = model_name.replace("/", "_").replace("-", "_")
    model_output_dir = os.path.join(args.output_base_dir, model_safe_name)
    os.makedirs(model_output_dir, exist_ok=True)

    try:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        total_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        layer_index = args.layer_index
        print(f"Model has {total_layers} layers, using layer {layer_index}")
        print(f"Hidden dimension: {hidden_dim}")

        # Load SAE and sample random features
        sae_name = f"Llama-3.1-8B-Instruct-SAE-l{layer_index}"
        print(f"Loading SAE from Goodfire/{sae_name} ...")
        sae = load_sae_from_goodfire(
            sae_name,
            d_model=hidden_dim,
            expansion_factor=args.expansion_factor,
            device=model.device,
        )

        d_hidden = hidden_dim * args.expansion_factor
        if args.num_features > d_hidden:
            raise ValueError(f"--num_features ({args.num_features}) > d_hidden ({d_hidden})")

        if args.feature_selection == "random":
            print(f"Sampling {args.num_features} random SAE features from d_hidden={d_hidden} ...")
            feature_indices = np.random.choice(d_hidden, size=args.num_features, replace=False).astype(int).tolist()
        else:
            feature_indices = select_dangerous_features_from_jbb(
                jailbreak_dir=args.jailbreak_dir,
                num_features=args.num_features,
                jbb_dataset=args.jbb_dataset,
                jbb_config=args.jbb_config,
                jbb_split=args.jbb_split,
            )
            # Filter indices to SAE hidden dimension (defensive)
            feature_indices = [i for i in feature_indices if 0 <= int(i) < d_hidden][: args.num_features]
            if not feature_indices:
                raise ValueError(
                    f"No valid dangerous feature indices in range [0, {d_hidden}) after filtering. "
                    "Are you using the matching SAE/layer for the jailbreak eval run?"
                )

        # Decoder vectors: [num_features, hidden_dim]
        steering_matrix = sae.decoder_linear.weight.T[feature_indices]
        steering_matrix = steering_matrix / steering_matrix.norm(dim=1, keepdim=True)
        steering_matrix = steering_matrix.to(dtype=torch.bfloat16, device=model.device)

        vector_path = os.path.join(model_output_dir, f"random_sae_features_{args.num_features}_{hidden_dim}d.pt")
        torch.save(steering_matrix, vector_path)
        print(f"Saved SAE feature vectors to {vector_path}")

        steering_coefficients = [args.base_coefficient * rel for rel in args.relative_coefficients]

        results_filepath = os.path.join(model_output_dir, "mmlu_random_sae_features_results.json")
        if os.path.exists(results_filepath):
            print(f"Loading existing results from: {results_filepath}")
            with open(results_filepath, "r") as f:
                results = json.load(f)
        else:
            results = {
                "model_name": model_name,
                "layer_index": layer_index,
                "total_layers": total_layers,
                "hidden_dim": hidden_dim,
                "sae_name": sae_name,
                "expansion_factor": args.expansion_factor,
                "feature_selection": args.feature_selection,
                "jailbreak_dir": args.jailbreak_dir if args.feature_selection == "dangerous" else None,
                "selected_feature_indices": feature_indices,
                "num_features": args.num_features,
                "base_coefficient": args.base_coefficient,
                "relative_coefficients": args.relative_coefficients,
                "steering_coefficients": steering_coefficients,
                "sample_size": args.sample_size,
                "max_new_tokens": args.max_new_tokens,
                "steering_vectors_path": vector_path,
                "seed": args.seed,
                "baseline": None,
                "steered_accuracies": {},
            }

        if results["baseline"] is None:
            print("Computing baseline accuracies...")
            results["baseline"] = compute_accuracies(
                mmlu_test,
                model=model,
                tokenizer=tokenizer,
                layer_index=layer_index,
                model_name=model_name,
                max_new_tokens=args.max_new_tokens,
                is_baseline=True,
            )
            with open(results_filepath, "w") as f:
                json.dump(results, f, indent=2)
            print("Saved baseline results")

        print("Evaluating random SAE features...")
        for local_idx, feature_idx in enumerate(feature_indices):
            vec_key = f"feature_{feature_idx}"
            if vec_key in results["steered_accuracies"]:
                print(f"Skipping feature {feature_idx} ({local_idx}/{args.num_features-1}) (already processed)")
                continue

            vec = steering_matrix[local_idx]
            feature_results = {}
            for rel_coeff, abs_coeff in zip(args.relative_coefficients, steering_coefficients):
                try:
                    print(
                        f"  Feature {feature_idx} ({local_idx}/{args.num_features-1}), "
                        f"Relative {rel_coeff} (abs: {abs_coeff:.4f})..."
                    )
                    acc = compute_accuracies(
                        mmlu_test,
                        model=model,
                        tokenizer=tokenizer,
                        layer_index=layer_index,
                        model_name=model_name,
                        max_new_tokens=args.max_new_tokens,
                        is_baseline=False,
                        vec=vec,
                        coeff=abs_coeff,
                    )
                    feature_results[str(rel_coeff)] = acc
                except Exception as e:
                    feature_results[str(rel_coeff)] = f"ERROR: {str(e)}"
                    print(f"    Error: {str(e)}")
                    clear_memory()

            results["steered_accuracies"][vec_key] = feature_results

            with open(results_filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results after feature {feature_idx}")

        print(f"\nCompleted! Results saved to {results_filepath}")
        print("\nGenerating summary statistics...")
        baseline_acc = results["baseline"]["overall_accuracy"]
        print(f"Baseline Overall Accuracy: {baseline_acc:.4f}")

        all_steered_accuracies = []
        for _, feature_results in results["steered_accuracies"].items():
            for _, acc_data in feature_results.items():
                if isinstance(acc_data, dict) and "overall_accuracy" in acc_data:
                    all_steered_accuracies.append(acc_data["overall_accuracy"])

        if all_steered_accuracies:
            avg_steered_acc = sum(all_steered_accuracies) / len(all_steered_accuracies)
            print(f"Average Steered Accuracy: {avg_steered_acc:.4f}")
            print(f"Performance Change: {avg_steered_acc - baseline_acc:+.4f}")

    finally:
        # best-effort cleanup
        if "model" in locals():
            del model
        if "tokenizer" in locals():
            del tokenizer
        clear_memory()


if __name__ == "__main__":
    main()


