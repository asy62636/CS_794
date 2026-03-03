"""
Batch generation with prefix caching using radix tree
"""
import os
import torch
import tiktoken
from model_radix import GPT
from prefix_cache import PrefixCache
import time

# Configuration
device = 'cpu' if torch.cuda.is_available() else 'cpu'
model_type = 'gpt2-xl'  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
max_new_tokens = 50
temperature = 0.8
top_k = 200
seed = 1337

# Batch of prompts (can be different lengths!)
prompts = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Spain?",
    "What is the weather like today?",
    "What is the weather like tomorrow?",
    "Tell me a joke about",
]

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print("="*80)
print("BATCH GENERATION WITH PREFIX CACHING")
print("="*80)

# Load model
print(f"\nLoading {model_type}...")
model = GPT.from_pretrained(model_type, dict(dropout=0.0))
model.eval()
model.to(device)
print(f"✓ Model loaded on {device}")

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

# Encode all prompts
print(f"\n{'='*80}")
print(f"ENCODING {len(prompts)} PROMPTS")
print(f"{'='*80}")

batch_tokens = []
for i, prompt in enumerate(prompts):
    tokens = enc.encode(prompt)
    batch_tokens.append(tokens)
    print(f"Prompt {i+1}: '{prompt}' ({len(tokens)} tokens)")

# ============================================================================
# PHASE 1: BUILD RADIX TREE
# ============================================================================
print(f"\n{'='*80}")
print("PHASE 1: BUILDING RADIX TREE")
print(f"{'='*80}")

cache = PrefixCache()

for i, tokens in enumerate(batch_tokens):
    print(f"\nInserting prompt {i+1}...")
    cache.insert_structure_only(tokens)

print("\nTree structure built:")
cache.print_tree()

# ============================================================================
# PHASE 2: COMPUTE ALL KV CACHES
# ============================================================================
print(f"\n{'='*80}")
print("PHASE 2: COMPUTING KV CACHES FOR TREE")
print(f"{'='*80}")

start_compute = time.time()
cache.compute_all_kv_caches(model)
compute_time = time.time() - start_compute

print(f"\n✓ KV cache computation complete in {compute_time:.2f}s")
print(f"  Tokens computed: {cache.total_tokens_computed}")

# ============================================================================
# PHASE 3: GENERATE FOR EACH PROMPT
# ============================================================================
print(f"\n{'='*80}")
print("PHASE 3: GENERATING OUTPUTS")
print(f"{'='*80}")

outputs = []
total_gen_time = 0
total_tokens_generated = 0

for i, (prompt, tokens) in enumerate(zip(prompts, batch_tokens)):
    print(f"\n{'-'*80}")
    print(f"Prompt {i+1}/{len(prompts)}: '{prompt}'")
    print(f"{'-'*80}")
    
    # Get cached KV for this prompt
    cached_kv, position = cache.get_kv_for_prompt(tokens)
    
    if cached_kv is not None:
        print(f"✓ Using cached KV up to position {position}")
    else:
        print(f"⚠️  No cache found, will compute from scratch")
    
    # Prepare input tensor
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Generate with cached KV
    start_gen = time.time()
    
    # Use the cached KV as starting point
    past_kvs = cached_kv
    curr_idx = idx
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Only pass last token (we already have KV for the rest)
        if past_kvs is not None:
            idx_input = curr_idx[:, [-1]]
        else:
            idx_input = curr_idx
        
        # Forward pass
        logits, _, past_kvs = model(idx_input, past_kvs)
        
        # Sample next token
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append
        curr_idx = torch.cat((curr_idx, idx_next), dim=1)
    
    gen_time = time.time() - start_gen
    total_gen_time += gen_time
    
    # Decode
    output_text = enc.decode(curr_idx[0].tolist())
    tokens_gen = max_new_tokens
    total_tokens_generated += tokens_gen
    
    print(f"\nGenerated text:")
    print(f"  {output_text}")
    print(f"\nStats:")
    print(f"  Generation time: {gen_time:.3f}s")
    print(f"  Tokens/second: {tokens_gen/gen_time:.2f}")
    
    outputs.append(output_text)

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

cache.stats()

print(f"Generation statistics:")
print(f"  Total prompts: {len(prompts)}")
print(f"  Total tokens generated: {total_tokens_generated}")
print(f"  Total generation time: {total_gen_time:.2f}s")
print(f"  Average time per prompt: {total_gen_time/len(prompts):.2f}s")
print(f"  Average tokens/second: {total_tokens_generated/total_gen_time:.2f}")

print(f"\nPhase breakdown:")
print(f"  Tree building: negligible")
print(f"  KV computation: {compute_time:.2f}s")
print(f"  Generation: {total_gen_time:.2f}s")
print(f"  Total: {compute_time + total_gen_time:.2f}s")

print(f"\n{'='*80}")
print("COMPLETE!")
print(f"{'='*80}")