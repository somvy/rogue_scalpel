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
from huggingface_hub import hf_hub_download

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Configuration
only_bomb_prompt = False

# Choose model:
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# LAYER_INDEX = 19

# MODEL_NAME = "google/gemma-2-2b-it"
# LAYER_INDEX = 12

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER_INDICES = [12]
BASE_COEFFICIENT = 89
LAYER_COEFFICIENTS = {
    # 4: 34.713336*0.5,
    12: 68.963016*0.5,
    # 18: 89.891038*0.5,
    # 20: 116.004552*0.5,
    # 25: 308.493314*0.5,
}

# Llama3.1 with OpenMOSS SAEs
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# LAYER_INDICES = [7, 10, 13, 16, 19, 22, 25, 28]  # SAE layers to steer on
# LAYER_INDICES = [11, 16, 19, 21] 
# BASE_COEFFICIENT = 13.34
# LAYER_COEFFICIENTS = {
#     11: 7.268195*0.75,
#     16: 10.338*0.75,
#     19: 13.34*0.75,
#     21: 15.895663*0.75,
# }

NUM_FEATURES = 1000 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512  
SAVE_INTERVAL = 10 # Save after every 10 vectors

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

class SparseAutoEncoder(torch.nn.Module):
    """Helper class for loading Llama SAEs"""
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
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f

def load_sae_llama(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
):
    """Load Llama SAE from disk"""
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

def load_layer_steering_matrix(layer_idx: int) -> torch.Tensor:
    """Load only the SAE needed for the requested layer."""
    def decoder_to_steering_matrix(decoder_weights: torch.Tensor, source: str) -> torch.Tensor:
        """Convert decoder weights to [num_features, hidden_dim] steering matrix."""
        if decoder_weights.ndim != 2:
            raise ValueError(f"{source}: expected 2D decoder weights, got shape {tuple(decoder_weights.shape)}")

        if decoder_weights.shape[1] == HIDDEN_DIM:
            # Already [d_sae, d_model]
            steering_matrix = decoder_weights[:NUM_FEATURES]
        elif decoder_weights.shape[0] == HIDDEN_DIM:
            # [d_model, d_sae] -> transpose to [d_sae, d_model]
            steering_matrix = decoder_weights.T[:NUM_FEATURES]
        else:
            raise ValueError(
                f"{source}: decoder shape {tuple(decoder_weights.shape)} incompatible with hidden_dim={HIDDEN_DIM}"
            )
        return steering_matrix.to(model.device)

    if "gemma" in MODEL_NAME.lower() and "2-2b" in MODEL_NAME.lower():
        if layer_idx != 12:
            raise ValueError("Gemma SAE config supports only layer 12.")

        sae_release = "gemma-scope-2b-pt-res-canonical"
        print(f"Loading SAE: {sae_release} @ layer {layer_idx}")
        sae, _, _ = SAE.from_pretrained(
            sae_release,
            sae_id="layer_12/width_16k/canonical",
            device="cpu",
        )
        sae = sae.to(model.device)
        steering_matrix = decoder_to_steering_matrix(sae.W_dec, f"{MODEL_NAME} layer {layer_idx}")
        del sae

    elif "qwen" in MODEL_NAME.lower() and "2.5-7b" in MODEL_NAME.lower():
        # Geaming release has layers 4, 12, 18, 20, 25
        sae_repo = "Geaming/Qwen2.5-7B-Instruct_SAEs"
        qwen_sae_suite = "FAST"
        qwen_sae_variant = "standard"
        sae_path = f"{qwen_sae_suite}/blocks_{layer_idx}_hook_resid_post_8X_2048_{qwen_sae_variant}/sae_weights.safetensors"
        print(f"Loading Qwen SAE from {sae_repo}/{sae_path}")

        from safetensors.torch import load_file
        file_path = hf_hub_download(
            repo_id=sae_repo,
            filename=sae_path,
            repo_type="model"
        )
        sae_layer_dict = load_file(file_path, device=str(model.device))

        if "W_dec" in sae_layer_dict:
            decoder_weights = sae_layer_dict["W_dec"]
        elif "decoder.weight" in sae_layer_dict:
            decoder_weights = sae_layer_dict["decoder.weight"]
        else:
            raise ValueError(f"Could not find decoder weights in checkpoint for layer {layer_idx}")

        steering_matrix = decoder_to_steering_matrix(decoder_weights, f"{MODEL_NAME} layer {layer_idx}")
        del sae_layer_dict

    elif "llama" in MODEL_NAME.lower() and "3.1" in MODEL_NAME:
        sae_repo = "OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x"
        sae_path = f"Llama3_1-8B-Base-L{layer_idx}R-32x/checkpoints/final.safetensors"
        print(f"Loading Llama3.1 SAE from {sae_repo}/{sae_path}")

        from safetensors.torch import load_file
        file_path = hf_hub_download(
            repo_id=sae_repo,
            filename=sae_path,
            repo_type="model"
        )
        sae_layer_dict = load_file(file_path, device=str(model.device))

        if "decoder.weight" in sae_layer_dict:
            decoder_weights = sae_layer_dict["decoder.weight"]
        elif "W_dec" in sae_layer_dict:
            decoder_weights = sae_layer_dict["W_dec"]
        else:
            raise ValueError(f"Could not find decoder weights in checkpoint for layer {layer_idx}")

        steering_matrix = decoder_to_steering_matrix(decoder_weights, f"{MODEL_NAME} layer {layer_idx}")
        del sae_layer_dict

    elif "llama" in MODEL_NAME.lower():
        if layer_idx != LAYER_INDICES[0]:
            raise ValueError("This legacy Llama SAE setup currently supports one layer.")

        sae_name = "Llama-3.1-8B-Instruct-SAE-l19"
        file_path = hf_hub_download(
            repo_id=f"Goodfire/{sae_name}",
            filename=f"{sae_name}.pth",
            repo_type="model"
        )
        sae = load_sae_llama(
            file_path,
            d_model=HIDDEN_DIM,
            expansion_factor=16,
            device=model.device,
        )
        steering_matrix = decoder_to_steering_matrix(sae.decoder_linear.weight, f"{MODEL_NAME} layer {layer_idx}")
        del sae

    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    steering_matrix = steering_matrix / steering_matrix.norm(dim=1, keepdim=True)
    clear_memory()
    return steering_matrix

# Steering generation function with special token avoidance and layer support
def generate_with_steering(prompt, steering_vector, coefficient, layer_idx):
    """Generate response with activation steering on a specific layer"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    # Precompute mask for special tokens in fixed prompt
    fixed_prompt_ids = inputs.input_ids[0].tolist()
    special_ids = set(tokenizer.all_special_ids)
    # For Llama, add special template tokens
    if "llama" in MODEL_NAME.lower():
        for s in ("<|begin_of_text|>", "<|end_of_text|>", "\n"):
            ids = tokenizer.encode(s, add_special_tokens=False)
            for tid in ids:
                special_ids.add(int(tid))
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

        # Move steering vector to correct device/dtype (for multi-GPU support)
        local_vec = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        hidden_states += coefficient * local_vec * mask_tensor
        return (hidden_states, *input[1:])

    # Get layers - most models use model.model.layers, but some architectures differ
    try:
        layers = model.model.layers
    except AttributeError:
        layers = model.layers
    
    handle = layers[layer_idx].register_forward_pre_hook(steering_pre_hook)
    
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

# Load JailbreakBench dataset
print("Loading JailbreakBench dataset...")
dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split='harmful')

if only_bomb_prompt:
    prompts = [example['Goal'] for example in dataset][20:21]
else:
    prompts = [example['Goal'] for example in dataset]

# Main loop: iterate through all layers
for layer_idx in LAYER_INDICES:
    print(f"\n{'='*80}")
    print(f"Processing layer {layer_idx}")
    print(f"{'='*80}")
    
    # Create output directories for this layer
    OUTPUT_DIR = f"./results/{'only_bomb_prompt' if only_bomb_prompt else 'all_prompts'}/{MODEL_NAME.split('/')[-1]}_layer{layer_idx}_sae"
    VECTOR_DIR = f"./results/{'only_bomb_prompt' if only_bomb_prompt else 'all_prompts'}/{MODEL_NAME.split('/')[-1]}_layer{layer_idx}_sae"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    
    # Load only current layer SAE (avoid storing all SAEs in memory)
    steering_matrix = load_layer_steering_matrix(layer_idx)
    steering_coeff = LAYER_COEFFICIENTS.get(layer_idx, BASE_COEFFICIENT)
    steering_coefficients = [steering_coeff]
    
    # Save steering matrix for this layer
    vector_path = os.path.join(VECTOR_DIR, f"random_vectors_{HIDDEN_DIM}d.pt")
    torch.save(steering_matrix, vector_path)
    print(f"Saved steering vectors to {vector_path}")
    print(f"Using steering coefficient: {steering_coeff}")
    
    # Process each prompt with this layer
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Processing prompts for layer {layer_idx}")):
        if prompt_idx < 50: 
            continue
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
                "layer_index": layer_idx,
                "steering_coefficient": steering_coeff,
                "baseline": "",
                "steered_responses": {}
            }
            
            # Generate baseline only if file doesn't exist
            print(f"\nGenerating baseline for prompt {prompt_idx}...")
            messages = [{"role": "user", "content": prompt}]
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
            for coeff in steering_coefficients:
                try:
                    print(f"  Vector {vector_idx}, Coeff {coeff}...")
                    response = generate_with_steering(prompt, vec, coeff, layer_idx)
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
    
    print(f"Completed layer {layer_idx}! Results saved to {OUTPUT_DIR}")
    del steering_matrix
    clear_memory()

print(f"\nAll layers completed!")
print(f"Results saved to ./results/{'only_bomb_prompt' if only_bomb_prompt else 'all_prompts'}/")