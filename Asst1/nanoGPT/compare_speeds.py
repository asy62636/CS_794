"""
Compare inference speed: Original GPT vs KV-Cached GPT
"""
import torch
import time
import tiktoken
from model import GPT as GPT_Cached
from model_og import GPT as GPT_Original

def benchmark_model(model, idx, num_tokens, model_name, seed=42):
    """Benchmark a single model"""
    print(f"\n{model_name}:")
    
    # Warm up (important!) - uses whatever random state
    _ = model.generate(idx, max_new_tokens=5, temperature=1.0)
    
    # NOW set the seed (after warm-up)
    torch.manual_seed(seed)
    
    # Actual benchmark with fresh seed
    start_time = time.time()
    output = model.generate(idx, max_new_tokens=num_tokens, temperature=1.0)
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_token = total_time / num_tokens
    tokens_per_sec = num_tokens / total_time
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Time per token: {time_per_token:.4f}s")
    print(f"  Tokens/second: {tokens_per_sec:.2f}")
    
    return total_time, time_per_token, output

def main():
    device = 'cuda'  # Change to 'cuda' if you have GPU
    
    print("="*80)
    print("KV Cache Speed Comparison")
    print("="*80)
    
    # Load both models
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
    test_lengths = [10, 25, 50, 100]
    
    results = []
    
    for num_tokens in test_lengths:
        print("\n" + "="*80)
        print(f"Generating {num_tokens} tokens...")
        print("="*80)
        
        # Benchmark original
        time_orig, tpt_orig, out_orig = benchmark_model(
            model_original, idx, num_tokens, "Original GPT (No Cache)"
        )
        
        # Benchmark cached
        time_cached, tpt_cached, out_cached = benchmark_model(
            model_cached, idx, num_tokens, "KV-Cached GPT"
        )
        
        # Calculate speedup
        speedup = time_orig / time_cached
        
        print(f"\n{'SPEEDUP:':<20} {speedup:.2f}x faster")
        print(f"{'Time saved:':<20} {time_orig - time_cached:.3f}s ({(1 - time_cached/time_orig)*100:.1f}%)")
        
        # Verify outputs match (with same seed, they should!)
        if torch.equal(out_orig, out_cached):
            print(f"{'Output match:':<20} ✅ Identical outputs")
        else:
            print(f"{'Output match:':<20} ⚠️  Different outputs (check random seed)")
        
        results.append({
            'tokens': num_tokens,
            'time_orig': time_orig,
            'time_cached': time_cached,
            'speedup': speedup
        })
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Tokens':<10} {'Original':<12} {'Cached':<12} {'Speedup':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['tokens']:<10} {r['time_orig']:<12.3f} {r['time_cached']:<12.3f} {r['speedup']:<12.2f}x")
    
    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)

if __name__ == "__main__":
    main()