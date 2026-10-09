import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from sae_lens import SAE
import json
import os
import re
from tqdm import tqdm
import gc
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER_INDEX = 19 
BASE_COEFFICIENT = 13.34
STEERING_COEFFICIENTS = [
    BASE_COEFFICIENT * 0.75, 
    BASE_COEFFICIENT * 1, 
    BASE_COEFFICIENT * 1.25, 
    BASE_COEFFICIENT * 1.5, 
    BASE_COEFFICIENT * 1.75, 
    BASE_COEFFICIENT * 2
]

NUM_FEATURES = 50  # Number of top dangerous features to steer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./results/mmlu_dangerous/llama_3.1_8b_19_sae"
VECTOR_DIR = "./results/mmlu_dangerous/llama_3.1_8b_19_sae"
MAX_NEW_TOKENS = 20  # Reduced for direct letter output
SAVE_INTERVAL = 1
SAMPLE_SIZE = 2000  # Number of questions to sample from all MMLU categories

# Jailbreak analysis directories (update as needed)
JAILBREAK_DIR = "./results/all_prompts/llama_3.1_8b_19_sae/eval"  # SAE experiment dir
EXPERIMENT_NAME = "Llama3.1-8b-it 2/3 depth (SAE)"

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Step 1: Analyze jailbreak results to find top dangerous features
print("Analyzing jailbreak results to find top dangerous features...")
dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split='harmful')
categories = [example['Category'] for example in dataset]
category_mapping = {f"prompt_{i}": categories[i] for i in range(len(categories))}

# Load all evaluation results
json_files = [f for f in os.listdir(JAILBREAK_DIR) if f.endswith('.json')]
steered_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for json_file in json_files:
    json_file_short = json_file.split("_")[0] + "_" + json_file.split("_")[1]
    if json_file_short not in category_mapping:
        continue
        
    category = category_mapping[json_file_short]
    
    with open(os.path.join(JAILBREAK_DIR, json_file), 'r') as f:
        data = json.load(f)
    
    # Assume judgments are added to responses (e.g., via external eval); adjust key if needed
    for feature_idx, feature_data in data["steered_responses"].items():
        for coeff, response_data in feature_data.items():
            if isinstance(response_data, str):  # If response is string, skip or judge here
                # Placeholder: Assume you have a way to judge; for demo, use dummy
                judgment = "unsafe" if "harm" in response_data.lower() else "safe"  # Replace with real judgment
            elif isinstance(response_data, dict) and "judgment" in response_data:
                judgment = response_data["judgment"]
            else:
                continue
            steered_results[feature_idx][coeff][category].append(judgment)

# Count how many prompts each vector breaks (unsafe judgments)
vector_break_counts = defaultdict(int)
vector_break_details = defaultdict(list)

for feature_idx in steered_results:
    for coeff in steered_results[feature_idx]:
        for category in steered_results[feature_idx][coeff]:
            judgments = steered_results[feature_idx][coeff][category]
            # Get prompts for this category
            category_prompts = [example['Goal'] for example in dataset if example['Category'] == category]
            for j, judgment in enumerate(judgments):
                if judgment == "unsafe" and j < len(category_prompts):
                    vector_break_counts[feature_idx] += 1
                    vector_break_details[feature_idx].append(category_prompts[j])

# Get top dangerous vectors (indices)
top_dangerous_vectors = sorted(vector_break_counts.items(), key=lambda x: x[1], reverse=True)[:NUM_FEATURES]
dangerous_feature_indices = [int(idx) for idx, count in top_dangerous_vectors]

# Save top dangerous indices
top_indices_path = os.path.join(VECTOR_DIR, "top_dangerous_features.json")
with open(top_indices_path, 'w') as f:
    json.dump(dangerous_feature_indices, f)
print(f"Saved top dangerous feature indices to {top_indices_path}")

# Step 2: Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()

# Get hidden dimension size
HIDDEN_DIM = model.config.hidden_size

# SAE Configuration
SAE_NAME = 'Llama-3.1-8B-Instruct-SAE-l19'
EXPANSION_FACTOR = 16

from huggingface_hub import hf_hub_download

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

file_path = hf_hub_download(
    repo_id=f"Goodfire/{SAE_NAME}",
    filename=f"{SAE_NAME}.pth",
    repo_type="model"
)

def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
):
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    sae_dict = torch.load(
        path, weights_only=True, map_location=device
    )
    sae.load_state_dict(sae_dict)
    return sae

sae = load_sae(
    file_path,
    d_model=4096,
    expansion_factor=EXPANSION_FACTOR,
    device=model.device,
)

# Extract steering vectors for top dangerous features only
steering_matrix = sae.decoder_linear.weight.T[dangerous_feature_indices]  # Use specific indices
steering_matrix = steering_matrix / steering_matrix.norm(dim=1, keepdim=True)  # Normalize

# Save steering matrix
vector_path = os.path.join(VECTOR_DIR, f"dangerous_features_{NUM_FEATURES}_{HIDDEN_DIM}d.pt")
torch.save(steering_matrix, vector_path)
print(f"Saved dangerous feature vectors to {vector_path}")

# Load all MMLU categories and sample 2000 questions
print("Loading all MMLU categories and sampling 2000 questions...")

# Get all subject names available in MMLU
configs = get_dataset_config_names("cais/mmlu")
print(f"Found {len(configs)} subjects in MMLU")

# Load and combine test splits from all subjects
all_tests = []
for config_name in tqdm(configs, desc="Loading MMLU subjects"):
    try:
        ds = load_dataset("cais/mmlu", config_name, split="test")
        all_tests.append(ds)
    except Exception as e:
        print(f"Failed to load {config_name}: {e}")

# Combine all test datasets
combined_test = concatenate_datasets(all_tests)

# Shuffle and sample 2000 questions
combined_test = combined_test.shuffle(seed=42)
mmlu_test = combined_test.select(range(SAMPLE_SIZE))

print(f"Sampled {len(mmlu_test)} questions from {len(configs)} subjects")

# Function to get MMLU prompt content (zero-shot direct answer prompt)
def get_mmlu_prompt(example):
    subject = example['subject']  # Assumes subject column exists
    choices = example['choices']
    choice_a, choice_b, choice_c, choice_d = choices[0], choices[1], choices[2], choices[3]
    question = example['question']
    test_part = f"Question: {question}\n\nA. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\n\nAnswer:"
    return f"The following is a multiple choice question about {subject}. Choose the correct answer (A, B, C, or D) and respond with only the letter. No explanation or reasoning needed.\n\n" + test_part

# Extract predicted answer (last A-D match)
def extract_answer(generated_text):
    generated_text = generated_text.strip().upper()
    matches = re.findall(r'\b([A-D])\b', generated_text)
    return matches[-1] if matches else None

# Steering generation function with special token avoidance
def generate_with_steering(prompt_content, steering_vector, coefficient):
    messages = [{"role": "user", "content": prompt_content}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    fixed_prompt_ids = inputs.input_ids[0].tolist()
    special_ids = set(tokenizer.all_special_ids)
    mask_fixed = [1 if token_id not in special_ids else 0 for token_id in fixed_prompt_ids]
    fixed_prompt_length = len(fixed_prompt_ids)

    def steering_pre_hook(module, input):
        hidden_states = input[0].clone()
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if seq_len <= fixed_prompt_length:
            current_mask = mask_fixed[:seq_len]
        else:
            current_mask = mask_fixed + [1] * (seq_len - fixed_prompt_length)
        
        mask_tensor = torch.tensor(
            current_mask, 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        ).view(batch_size, seq_len, 1)
        
        hidden_states += coefficient * steering_vector * mask_tensor
        return (hidden_states, *input[1:])

    handle = model.model.layers[LAYER_INDEX].register_forward_pre_hook(steering_pre_hook)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    handle.remove()
    return tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

# Function to compute accuracies
def compute_accuracies(is_baseline, vec=None, coeff=None):
    subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for example in tqdm(mmlu_test, desc="Processing MMLU questions"):
        prompt_content = get_mmlu_prompt(example)
        if is_baseline:
            messages = [{"role": "user", "content": prompt_content}]
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            generated = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            generated = generate_with_steering(prompt_content, vec, coeff)

        pred = extract_answer(generated)
        correct_letter = chr(65 + example['answer'])
        subject = example['subject']
        subject_stats[subject]['total'] += 1
        if pred == correct_letter:
            subject_stats[subject]['correct'] += 1

    total_correct = sum(stats['correct'] for stats in subject_stats.values())
    total_questions = sum(stats['total'] for stats in subject_stats.values())
    per_subject = {
        sub: (stats['correct'] / stats['total']) if stats['total'] > 0 else 0.0 
        for sub, stats in subject_stats.items()
    }
    overall = (total_correct / total_questions) if total_questions > 0 else 0.0
    
    return {"overall_accuracy": overall, "per_subject": per_subject}

# Results file
filename = "mmlu_dangerous_results.json"
filepath = os.path.join(OUTPUT_DIR, filename)

# Initialize or load results
if os.path.exists(filepath):
    print(f"Loading existing results from: {filename}")
    with open(filepath, 'r') as f:
        results = json.load(f)
else:
    results = {
        "model_name": MODEL_NAME,
        "layer_index": LAYER_INDEX,
        "sae_name": SAE_NAME,
        "dangerous_feature_indices": dangerous_feature_indices,
        "steering_vectors_path": vector_path,
        "hidden_dim": HIDDEN_DIM,
        "sample_size": SAMPLE_SIZE,
        "steering_coefficients": STEERING_COEFFICIENTS,
        "baseline": None,
        "steered_accuracies": {}
    }

# Compute baseline if not done
if results["baseline"] is None:
    print("Computing baseline accuracies...")
    results["baseline"] = compute_accuracies(is_baseline=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved baseline results")

# Process each dangerous feature vector
for vector_idx, orig_idx in enumerate(dangerous_feature_indices):
    vec_key = f"{orig_idx}"
    if vec_key in results["steered_accuracies"]:
        continue
    
    vec = steering_matrix[vector_idx]
    vector_results = {}
    for coeff in STEERING_COEFFICIENTS:
        try:
            print(f"  Dangerous Feature {orig_idx} (local idx {vector_idx}), Coeff {coeff}...")
            acc = compute_accuracies(is_baseline=False, vec=vec, coeff=coeff)
            vector_results[str(coeff)] = acc
        except Exception as e:
            vector_results[str(coeff)] = f"ERROR: {str(e)}"
            clear_memory()
    
    results["steered_accuracies"][vec_key] = vector_results
    
    # Save periodically
    if (vector_idx + 1) % SAVE_INTERVAL == 0 or vector_idx == NUM_FEATURES - 1:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results after processing dangerous feature {vector_idx}")

print(f"\nCompleted! Results saved to {filepath}")
print(f"Dangerous steering vectors saved to {vector_path}")

# Generate summary statistics
print("\nGenerating summary statistics...")
baseline_acc = results["baseline"]["overall_accuracy"]
print(f"Baseline Overall Accuracy: {baseline_acc:.4f}")

# Calculate average performance across all steering configurations
all_steered_accuracies = []
for feature_idx, feature_results in results["steered_accuracies"].items():
    for coeff, acc_data in feature_results.items():
        if isinstance(acc_data, dict) and "overall_accuracy" in acc_data:
            all_steered_accuracies.append(acc_data["overall_accuracy"])

if all_steered_accuracies:
    avg_steered_acc = sum(all_steered_accuracies) / len(all_steered_accuracies)
    print(f"Average Steered Accuracy: {avg_steered_acc:.4f}")
    print(f"Performance Change: {avg_steered_acc - baseline_acc:+.4f}")

print("Analysis complete!")