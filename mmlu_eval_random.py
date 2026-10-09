import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
import json
import os
import re
from tqdm import tqdm
import gc
import numpy as np
from collections import defaultdict

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Configuration
MODELS_CONFIG = [
    # {
    #     "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    #     "layer_index": 16,  # Will be set to middle layer dynamically
    #     "base_coefficient": 10.3392,  # Can be adjusted per model
    # },
    # {
    #     "model_name": "Qwen/Qwen2.5-7B-Instruct",
    #     "layer_index": 14,
    #     "base_coefficient": 74.625,
    # },
    # {
    #     "model_name": "tiiuae/Falcon3-7B-Instruct",
    #     "layer_index": 14,
    #     "base_coefficient": 213.996,
    # },

    {
        "model_name": "google/gemma-2-2b-it",
        "layer_index": 12,
        "base_coefficient": 205,
    },
]

RELATIVE_COEFFICIENTS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 10, 20]
NUM_VECTORS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 20
SAVE_INTERVAL = 1
SAMPLE_SIZE = 200  # Number of questions to sample from all MMLU categories

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directories
OUTPUT_BASE_DIR = "./results/mmlu_random_vectors"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Load all MMLU categories and sample questions (do this once for all models)
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

# Shuffle and sample questions
combined_test = combined_test.shuffle(seed=42)
mmlu_test = combined_test.select(range(SAMPLE_SIZE))

print(f"Sampled {len(mmlu_test)} questions from {len(configs)} subjects")

# Function to get MMLU prompt content (zero-shot direct answer prompt)
def get_mmlu_prompt(example):
    subject = example['subject']
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

# Helper function to get chat template (handles different model types)
def get_chat_prompt(messages, tokenizer, model_name):
    """Get chat prompt, handling different model architectures"""
    if "falcon" in model_name.lower():
        # Falcon models use a simpler template
        return f"User:{messages[0]['content']}\nAssistant:"
    else:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"Warning: Could not apply chat template: {e}")
            return messages[0]['content']

# Steering generation function with special token avoidance
def generate_with_steering(prompt_content, steering_vector, coefficient, model, tokenizer, layer_index, model_name):
    messages = [{"role": "user", "content": prompt_content}]
    chat_prompt = get_chat_prompt(messages, tokenizer, model_name)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    fixed_prompt_ids = inputs.input_ids[0].tolist()
    special_ids = set(tokenizer.all_special_ids)
    # For Gemma, add "user" and "model" tokens and newlines (part of chat template but not in special_ids)
    if "gemma" in model_name.lower():
        special_ids.add(tokenizer.encode("user", add_special_tokens=False)[0])
        special_ids.add(tokenizer.encode("model", add_special_tokens=False)[0])
        special_ids.add(tokenizer.encode("\n", add_special_tokens=False)[0])
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
        
        if mask_tensor.shape == (1, 1, 1):
            hidden_states += coefficient * steering_vector
        else:
            hidden_states = hidden_states
        # hidden_states += coefficient * steering_vector * mask_tensor
        # print(mask_tensor.shape, mask_tensor)
        return (hidden_states, *input[1:])

    # Get layers - most models use model.model.layers, but some architectures differ
    try:
        layers = model.model.layers
    except AttributeError:
        # Fallback for different architectures
        layers = model.layers
    
    handle = layers[layer_index].register_forward_pre_hook(steering_pre_hook)
    
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
def compute_accuracies(mmlu_test, model, tokenizer, layer_index, model_name, is_baseline, vec=None, coeff=None):
    subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for example in tqdm(mmlu_test, desc="Processing MMLU questions"):
        prompt_content = get_mmlu_prompt(example)
        if is_baseline:
            messages = [{"role": "user", "content": prompt_content}]
            chat_prompt = get_chat_prompt(messages, tokenizer, model_name)
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
            generated = generate_with_steering(prompt_content, vec, coeff, model, tokenizer, layer_index, model_name)

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

# Process each model
for model_config in MODELS_CONFIG:
    model_name = model_config["model_name"]
    base_coeff = model_config["base_coefficient"]
    steering_coefficients = [base_coeff * rel for rel in RELATIVE_COEFFICIENTS]
    
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"{'='*80}")
    
    # Create model-specific output directory
    model_safe_name = model_name.replace("/", "_").replace("-", "_")
    model_output_dir = os.path.join(OUTPUT_BASE_DIR, model_safe_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()
        
        # Get model info
        total_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        
        # Determine middle layer if not set
        layer_index = model_config.get("layer_index")
        if layer_index is None:
            layer_index = total_layers // 2
        print(f"Model has {total_layers} layers, using layer {layer_index} (middle layer)")
        print(f"Hidden dimension: {hidden_dim}")
        
        # Generate random steering vectors
        print(f"Generating {NUM_VECTORS} random steering vectors...")
        np.random.seed(42)  # For reproducibility
        random_vectors = np.random.randn(NUM_VECTORS, hidden_dim).astype(np.float32)
        
        # Normalize vectors
        random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)
        steering_matrix = torch.tensor(random_vectors, dtype=torch.bfloat16, device=model.device)
        
        # Save steering matrix
        vector_path = os.path.join(model_output_dir, f"random_vectors_{NUM_VECTORS}_{hidden_dim}d.pt")
        torch.save(steering_matrix, vector_path)
        print(f"Saved random vectors to {vector_path}")
        
        # Results file
        results_filename = "mmlu_random_vectors_results.json"
        results_filepath = os.path.join(model_output_dir, results_filename)
        
        # Initialize or load results
        if os.path.exists(results_filepath):
            print(f"Loading existing results from: {results_filename}")
            with open(results_filepath, 'r') as f:
                results = json.load(f)
        else:
            results = {
                "model_name": model_name,
                "layer_index": layer_index,
                "total_layers": total_layers,
                "hidden_dim": hidden_dim,
                "num_vectors": NUM_VECTORS,
                "base_coefficient": base_coeff,
                "relative_coefficients": RELATIVE_COEFFICIENTS,
                "steering_coefficients": steering_coefficients,
                "sample_size": SAMPLE_SIZE,
                "steering_vectors_path": vector_path,
                "baseline": None,
                "steered_accuracies": {}
            }
        
        # Compute baseline if not done
        if results["baseline"] is None:
            print("Computing baseline accuracies...")
            results["baseline"] = compute_accuracies(
                mmlu_test, model, tokenizer, layer_index, model_name, is_baseline=True
            )
            with open(results_filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print("Saved baseline results")
        
        # Process each random vector
        for vector_idx in range(NUM_VECTORS):
            vec_key = f"vector_{vector_idx}"
            if vec_key in results["steered_accuracies"]:
                print(f"Skipping vector {vector_idx} (already processed)")
                continue
            
            vec = steering_matrix[vector_idx]
            vector_results = {}
            
            for rel_coeff, abs_coeff in zip(RELATIVE_COEFFICIENTS, steering_coefficients):
                try:
                    print(f"  Vector {vector_idx}/{NUM_VECTORS-1}, Relative Coeff {rel_coeff} (abs: {abs_coeff:.2f})...")
                    acc = compute_accuracies(
                        mmlu_test, model, tokenizer, layer_index, model_name,
                        is_baseline=False, vec=vec, coeff=abs_coeff
                    )
                    vector_results[str(rel_coeff)] = acc
                except Exception as e:
                    vector_results[str(rel_coeff)] = f"ERROR: {str(e)}"
                    print(f"    Error: {str(e)}")
                    clear_memory()
            
            results["steered_accuracies"][vec_key] = vector_results
            
            # Save periodically
            if (vector_idx + 1) % SAVE_INTERVAL == 0 or vector_idx == NUM_VECTORS - 1:
                with open(results_filepath, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved results after processing vector {vector_idx}")
        
        print(f"\nCompleted for {model_name}! Results saved to {results_filepath}")
        
        # Generate summary statistics
        print("\nGenerating summary statistics...")
        baseline_acc = results["baseline"]["overall_accuracy"]
        print(f"Baseline Overall Accuracy: {baseline_acc:.4f}")
        
        # Calculate average performance across all steering configurations
        all_steered_accuracies = []
        for vec_key, vec_results in results["steered_accuracies"].items():
            for rel_coeff, acc_data in vec_results.items():
                if isinstance(acc_data, dict) and "overall_accuracy" in acc_data:
                    all_steered_accuracies.append(acc_data["overall_accuracy"])
        
        if all_steered_accuracies:
            avg_steered_acc = sum(all_steered_accuracies) / len(all_steered_accuracies)
            print(f"Average Steered Accuracy: {avg_steered_acc:.4f}")
            print(f"Performance Change: {avg_steered_acc - baseline_acc:+.4f}")
        
    except Exception as e:
        print(f"Error processing model {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up memory
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        clear_memory()
        print(f"Cleared memory after processing {model_name}\n")

print("\n" + "="*80)
print("All models processed!")
print("="*80)

