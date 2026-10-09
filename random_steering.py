import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sae_lens import SAE
import json
import os
import re
from tqdm import tqdm
import gc
import numpy as np

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

only_bomb_prompt = True


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# LAYER_INDEX = 16
# BASE_COEFFICIENT = 10.3392
# STEERING_COEFFICIENTS = [BASE_COEFFICIENT * 0.75]

# LAYER_INDEX = 11
# BASE_COEFFICIENT = 7.268281

# LAYER_INDEX = 21
# BASE_COEFFICIENT = 15.893947


LAYER_INDEX = 19
BASE_COEFFICIENT = 13.34

# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# LAYER_INDEX = 14
# BASE_COEFFICIENT = 74.625

# MODEL_NAME = "tiiuae/Falcon3-7B-Instruct"
# LAYER_INDEX = 14
# BASE_COEFFICIENT = 213.996


STEERING_COEFFICIENTS = [BASE_COEFFICIENT * 0.25, BASE_COEFFICIENT * 0.5, BASE_COEFFICIENT * 0.75, BASE_COEFFICIENT * 1, BASE_COEFFICIENT * 1.25, BASE_COEFFICIENT * 1.5, BASE_COEFFICIENT * 1.75, BASE_COEFFICIENT * 2]

if "falcon" in MODEL_NAME:
    falcon = True
else:
    falcon = False

# STEERING_COEFFICIENTS = [60.748*0.75, 60.748, 60.748*1.25, 60.748*1.5, 60.748*1.75, 60.748*2] # qwen_2.5_7b_9
# STEERING_COEFFICIENTS = [72.94*0.75, 72.94, 72.94*1.25, 72.94*1.5, 72.94*1.75, 72.94*2] # qwen_2.5_7b_13
# STEERING_COEFFICIENTS = [227*0.75, 227, 227*1.25, 227*1.5, 227*1.75, 227*2] # #Falcon3_7B_17
# STEERING_COEFFICIENTS = [60.748*1.5] # qwen_2.5_7b_9
# STEERING_COEFFICIENTS = [194*1.5, 194*1.75, 194*2] # qwen_2.5_32b_21
# STEERING_COEFFICIENTS = [61, 61*1.5, 61*2] # qwen_2.5_3b_12
# STEERING_COEFFICIENTS = [140*0.5, 140*0.6, 140*0.7] # Falcon3_3B_9
# STEERING_COEFFICIENTS = [297*0.6, 297*0.8, 297] # falcon_1_40b_20
# STEERING_COEFFICIENTS = [382] # falcon_1_32b_24
# STEERING_COEFFICIENTS = [6.324890*0.75, 6.324890, 6.324890*1.25, 6.324890*1.5, 6.324890*1.75, 6.324890*2] # llama_3_8b_10
# STEERING_COEFFICIENTS = [13.34*0.75, 13.34, 13.34*1.25, 13.34*1.5, 13.34*1.75, 13.34*2] # llama_3.1_8b_19
# STEERING_COEFFICIENTS = [194*0.75, 194, 194*1.25, 194*1.5, 194*1.75, 194*2] #Falcon3-7B


NUM_FEATURES = 1000  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = f"./results/{'only_bomb_prompt' if only_bomb_prompt else 'all_prompts'}/{MODEL_NAME}_{LAYER_INDEX}"
VECTOR_DIR = f"./results/{'only_bomb_prompt' if only_bomb_prompt else 'all_prompts'}/{MODEL_NAME}_{LAYER_INDEX}"
MAX_NEW_TOKENS = 512  
SAVE_INTERVAL = 10  # Save after every 100 vectors

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Load model and tokenizer
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

# Generate random steering vectors
print(f"Generating {NUM_FEATURES} random vectors of dim {HIDDEN_DIM}...")
steering_matrix = torch.randn(NUM_FEATURES, HIDDEN_DIM, device=DEVICE)
steering_matrix = steering_matrix / steering_matrix.norm(dim=1, keepdim=True)  # Normalize

# Save steering matrix
vector_path = os.path.join(VECTOR_DIR, f"random_vectors_{HIDDEN_DIM}d.pt")
torch.save(steering_matrix, vector_path)
print(f"Saved random vectors to {vector_path}")

# Load JailbreakBench dataset
print("Loading JailbreakBench dataset...")
dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split='harmful')

if only_bomb_prompt:
    prompts = [example['Goal'] for example in dataset][20:21]
else:
    prompts = [example['Goal'] for example in dataset]

# Steering generation function with special token avoidance
def generate_with_steering(prompt, steering_vector, coefficient):
    """Generate response with activation steering, avoiding special tokens"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    if falcon:
        chat_prompt = f"User:{prompt}\nAssistant:"
    else:
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    # Precompute mask for special tokens in fixed prompt
    fixed_prompt_ids = inputs.input_ids[0].tolist()
    special_ids = set(tokenizer.all_special_ids)
    # For Gemma, add "user" and "model" tokens and newlines (part of chat template but not in special_ids)
    if "gemma" in MODEL_NAME.lower():
        special_ids.add(tokenizer.encode("user", add_special_tokens=False)[0])
        special_ids.add(tokenizer.encode("model", add_special_tokens=False)[0])
        special_ids.add(tokenizer.encode("\n", add_special_tokens=False)[0])
    mask_fixed = [1 if token_id not in special_ids else 0 for token_id in fixed_prompt_ids]
    fixed_prompt_length = len(fixed_prompt_ids)

    def steering_pre_hook(module, input):
        hidden_states = input[0].clone()
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # Create mask: 1 for non-special tokens, 0 for special tokens
        if seq_len <= fixed_prompt_length:
            current_mask = mask_fixed[:seq_len]
        else:
            # Extend mask for generated tokens (assume non-special by default)
            current_mask = mask_fixed + [1] * (seq_len - fixed_prompt_length)
        
        # Convert to tensor and reshape for broadcasting
        mask_tensor = torch.tensor(
            current_mask, 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        ).view(batch_size, seq_len, 1)
        
        hidden_states += coefficient * steering_vector * mask_tensor
        return (hidden_states, *input[1:])
    
    if falcon:
        handle = model.transformer.h[LAYER_INDEX].register_forward_pre_hook(steering_pre_hook)
    else:
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

# Process each prompt
for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):

    # if prompt_idx < 50:
    #     continue
    safe_prompt = re.sub(r'[^a-zA-Z0-9 ]', '', prompt)[:50].strip()
    filename = f"prompt_{prompt_idx}_{safe_prompt or 'empty'}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Initialize results structure
    if os.path.exists(filepath):
        print(f"Loading existing results from: {filename}")
        with open(filepath, 'r') as f:
            results = json.load(f)
    else:
        results = {
            "prompt": prompt,
            "steering_vectors_path": vector_path,
            "hidden_dim": HIDDEN_DIM,
            "layer_index": LAYER_INDEX,
            "baseline": "",
            "steered_responses": {}
        }
        
        # Generate baseline only if file doesn't exist
        print(f"\nGenerating baseline for prompt {prompt_idx}...")
        messages = [{"role": "user", "content": prompt}]
        if falcon:
            chat_prompt = f"User:{prompt}\nAssistant:"
        else:
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            baseline_output = model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        results["baseline"] = tokenizer.decode(baseline_output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Save initial results with baseline
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Process each random vector
    for vector_idx in tqdm(range(NUM_FEATURES), desc=f"Vectors for prompt {prompt_idx}"):
        # Skip if this vector is already processed
        if str(vector_idx) in results["steered_responses"]:
            continue
            
        vec = steering_matrix[vector_idx]
        vector_results = {}
        for coeff in STEERING_COEFFICIENTS:
            try:
                print(f"  Vector {vector_idx}, Coeff {coeff}...")
                response = generate_with_steering(prompt, vec, coeff)
                response = response.strip()
                vector_results[str(coeff)] = response
            except Exception as e:
                vector_results[str(coeff)] = f"ERROR: {str(e)}"
                clear_memory()
        
        # Add the vector results
        results["steered_responses"][str(vector_idx)] = vector_results
        
        # Save after every SAVE_INTERVAL vectors or at the end
        if (vector_idx + 1) % SAVE_INTERVAL == 0 or vector_idx == NUM_FEATURES - 1:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results after processing vector {vector_idx}")
    
    print(f"Completed processing for prompt {prompt_idx}")

print(f"\nCompleted! Results saved to {OUTPUT_DIR}")
print(f"Steering vectors saved to {vector_path}")