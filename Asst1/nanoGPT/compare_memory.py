"""
Compare memory usage: Original GPT vs KV-Cached GPT
"""
import torch
import psutil
import tiktoken
from model import GPT as GPT_Cached
from model_og import GPT as GPT_Original

def measure_memory(model, idx, num_tokens, model_name):
    """Measure memory usage during generation"""
    print(f"\n{model_name}:")
    
    process = psutil.Process()
    
    # Get baseline memory
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Generate
    torch.manual_seed(42)
    output = model.generate(idx, max_new_tokens=num_tokens, temperature=1.0)
    
    # Get memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before
    
    print(f"  Memory before: {mem_before:.2f} MB")
    print(f"  Memory after:  {mem_after:.2f} MB")
    print(f"  Memory used:   {mem_used:.2f} MB")
    
    return mem_before, mem_after, mem_used, output

def main():
    device = 'cpu'
    
    print("="*80)
    print("KV Cache Memory Comparison")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    model_cached = GPT_Cached.from_pretrained('gpt2', dict(dropout=0.0))
    model_original = GPT_Original.from_pretrained('gpt2', dict(dropout=0.0))
    
    model_cached.eval()
    model_original.eval()
    
    model_cached.to(device)
    model_original.to(device)
    
    print("✅ Models loaded")
    
    # Prepare input
    enc = tiktoken.get_encoding("gpt2")
    prompt = "The quick brown fox jumps over the lazy dog"
    tokens = enc.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Input tokens: {len(tokens)}")
    
    # Test different output lengths
    test_lengths = [10, 50, 100, 200]
    
    results = []
    
    for num_tokens in test_lengths:
        print("\n" + "="*80)
        print(f"Generating {num_tokens} tokens...")
        print("="*80)
        
        # Measure original
        mem_before_orig, mem_after_orig, mem_used_orig, out_orig = measure_memory(
            model_original, idx, num_tokens, "Original GPT (No Cache)"
        )
        
        # Clear cache between runs (garbage collection)
        import gc
        gc.collect()
        
        # Measure cached
        mem_before_cached, mem_after_cached, mem_used_cached, out_cached = measure_memory(
            model_cached, idx, num_tokens, "KV-Cached GPT"
        )
        
        # Compare
        mem_diff = mem_used_cached - mem_used_orig
        
        print(f"\n{'Memory comparison:':<25}")
        print(f"  {'Original used:':<20} {mem_used_orig:.2f} MB")
        print(f"  {'Cached used:':<20} {mem_used_cached:.2f} MB")
        print(f"  {'Difference:':<20} {mem_diff:+.2f} MB ({mem_diff/mem_used_orig*100:+.1f}%)")
        
        if mem_diff > 0:
            print(f"  ⚠️  KV cache uses {mem_diff:.2f} MB MORE memory")
        else:
            print(f"  ✅ KV cache uses {abs(mem_diff):.2f} MB LESS memory")
        
        results.append({
            'tokens': num_tokens,
            'mem_orig': mem_used_orig,
            'mem_cached': mem_used_cached,
            'diff': mem_diff
        })
        
        # Clear for next iteration
        gc.collect()
    
    # Summary table
    print("\n" + "="*80)
    print("MEMORY SUMMARY")
    print("="*80)
    print(f"{'Tokens':<10} {'Original (MB)':<15} {'Cached (MB)':<15} {'Difference (MB)':<15}")
    print("-"*80)
    for r in results:
        diff_str = f"{r['diff']:+.2f}"
        print(f"{r['tokens']:<10} {r['mem_orig']:<15.2f} {r['mem_cached']:<15.2f} {diff_str:<15}")
    
    # Calculate theoretical cache size for GPT-2
    print("\n" + "="*80)
    print("THEORETICAL KV CACHE SIZE (GPT-2)")
    print("="*80)
    
    n_layers = 12
    n_heads = 12
    head_dim = 64
    
    for num_tokens in test_lengths:
        # 2 (K and V) × n_layers × n_heads × num_tokens × head_dim × 4 bytes (float32)
        cache_size_bytes = 2 * n_layers * n_heads * num_tokens * head_dim * 4
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        
        print(f"  {num_tokens} tokens: ~{cache_size_mb:.2f} MB")
    
    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)

if __name__ == "__main__":
    main()