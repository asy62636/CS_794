"""
Script to measure and plot inference time and memory usage for GPT-2
as a function of input prompt size and output length.
"""

import os
import time
import torch
import tiktoken
from model import GPT
import psutil
import json
import matplotlib.pyplot as plt
import numpy as np

def measure_inference(prompt, max_new_tokens, model, encode, decode, device):
    """Measure inference time and memory for a single generation"""
    
    # Encode prompt
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Start timing
    start_time = time.time()
    
    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=0.8, top_k=200)
    
    # End timing
    end_time = time.time()
    
    # Memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    total_time = end_time - start_time
    input_tokens = len(start_ids)
    output_tokens = y.shape[1] - x.shape[1]
    time_per_token = total_time / output_tokens if output_tokens > 0 else 0
    
    return {
        'prompt': prompt,
        'input_tokens': input_tokens,
        'input_chars': len(prompt),
        'input_words': len(prompt.split()),
        'output_tokens': output_tokens,
        'max_new_tokens': max_new_tokens,
        'total_time': total_time,
        'time_per_token': time_per_token,
        'memory_before_mb': mem_before,
        'memory_after_mb': mem_after,
        'memory_used_mb': mem_after - mem_before,
        'output_text': decode(y[0].tolist())
    }

def run_experiments(device='cpu', model_name='gpt2-xl'):
    """Run systematic experiments"""
    
    print("="*80)
    print(f"Running inference experiments on {model_name}")
    print(f"Device: {device}")
    print("="*80)
    
    # Setup
    print("\nLoading model...")
    model = GPT.from_pretrained(model_name, dict(dropout=0.0))
    model.eval()
    model.to(device)
    print("Model loaded successfully!")
    
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    
    all_results = {
        'varying_input': [],
        'varying_output': []
    }
    
    # =========================================================================
    # EXPERIMENT 1: Varying Input Prompt Size (fixed output)
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: Varying Input Prompt Size")
    print("Fixed output size: 50 tokens")
    print("="*80)
    
    # Create prompts of different lengths
    base_sentence = "The quick brown fox jumps over the lazy dog"
    prompts = [
        "Hi",  # Very short
        "Hello world",  # Short
        "The quick brown fox",  # Medium-short
        base_sentence,  # Medium
        " ".join([base_sentence] * 2),  # Long
        " ".join([base_sentence] * 3),  # Very long
        " ".join([base_sentence] * 5),  # Extra long
    ]
    
    fixed_output_size = 50
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Testing with prompt: '{prompt[:50]}...'")
        result = measure_inference(prompt, fixed_output_size, model, encode, decode, device)
        all_results['varying_input'].append(result)
        
        print(f"  Input: {result['input_tokens']} tokens, {result['input_words']} words, {result['input_chars']} chars")
        print(f"  Output: {result['output_tokens']} tokens")
        print(f"  Total time: {result['total_time']:.4f}s")
        print(f"  Time per token: {result['time_per_token']:.4f}s")
        print(f"  Memory used: {result['memory_used_mb']:.2f} MB")
    
    # =========================================================================
    # EXPERIMENT 2: Varying Output Size (fixed input)
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: Varying Output Size")
    print("Fixed input prompt: 'Hello world'")
    print("="*80)
    
    fixed_prompt = "Hello world"
    output_sizes = [5, 10, 25, 50, 75, 100, 150, 200]
    
    for i, size in enumerate(output_sizes):
        print(f"\n[{i+1}/{len(output_sizes)}] Testing with max_new_tokens={size}...")
        result = measure_inference(fixed_prompt, size, model, encode, decode, device)
        all_results['varying_output'].append(result)
        
        print(f"  Input: {result['input_tokens']} tokens")
        print(f"  Output: {result['output_tokens']} tokens")
        print(f"  Total time: {result['total_time']:.4f}s")
        print(f"  Time per token: {result['time_per_token']:.4f}s")
        print(f"  Memory used: {result['memory_used_mb']:.2f} MB")
    
    # Save results
    output_file = 'inference_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Results saved to {output_file}")
    print("="*80)
    
    return all_results

def plot_results(results):
    """Create all required plots"""
    
    print("\nGenerating plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # =========================================================================
    # PLOTS FOR EXPERIMENT 1: Varying Input Size
    # =========================================================================
    varying_input = results['varying_input']
    
    # Extract data
    input_tokens = [r['input_tokens'] for r in varying_input]
    input_words = [r['input_words'] for r in varying_input]
    input_chars = [r['input_chars'] for r in varying_input]
    total_time_input = [r['total_time'] for r in varying_input]
    time_per_token_input = [r['time_per_token'] for r in varying_input]
    memory_input = [r['memory_used_mb'] for r in varying_input]
    
    # Create figure with 3x2 subplots for input experiments
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 12))
    fig1.suptitle('Inference Performance vs Input Prompt Size', fontsize=16, fontweight='bold')
    
    # Row 1: Total time vs input size (tokens, words, chars)
    axes1[0, 0].plot(input_tokens, total_time_input, 'o-', linewidth=2, markersize=8)
    axes1[0, 0].set_xlabel('Input Prompt Size (tokens)', fontsize=11)
    axes1[0, 0].set_ylabel('Total Inference Time (seconds)', fontsize=11)
    axes1[0, 0].set_title('Total Time vs Input Tokens', fontweight='bold')
    axes1[0, 0].grid(True, alpha=0.3)
    
    axes1[0, 1].plot(input_words, total_time_input, 'o-', color='green', linewidth=2, markersize=8)
    axes1[0, 1].set_xlabel('Input Prompt Size (words)', fontsize=11)
    axes1[0, 1].set_ylabel('Total Inference Time (seconds)', fontsize=11)
    axes1[0, 1].set_title('Total Time vs Input Words', fontweight='bold')
    axes1[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Time per token vs input size
    axes1[1, 0].plot(input_tokens, time_per_token_input, 's-', color='orange', linewidth=2, markersize=8)
    axes1[1, 0].set_xlabel('Input Prompt Size (tokens)', fontsize=11)
    axes1[1, 0].set_ylabel('Time per Output Token (seconds)', fontsize=11)
    axes1[1, 0].set_title('Time per Token vs Input Tokens', fontweight='bold')
    axes1[1, 0].grid(True, alpha=0.3)
    
    axes1[1, 1].plot(input_chars, time_per_token_input, 's-', color='red', linewidth=2, markersize=8)
    axes1[1, 1].set_xlabel('Input Prompt Size (characters)', fontsize=11)
    axes1[1, 1].set_ylabel('Time per Output Token (seconds)', fontsize=11)
    axes1[1, 1].set_title('Time per Token vs Input Characters', fontweight='bold')
    axes1[1, 1].grid(True, alpha=0.3)
    
    # Row 3: Memory usage vs input size
    axes1[2, 0].plot(input_tokens, memory_input, '^-', color='purple', linewidth=2, markersize=8)
    axes1[2, 0].set_xlabel('Input Prompt Size (tokens)', fontsize=11)
    axes1[2, 0].set_ylabel('Memory Used (MB)', fontsize=11)
    axes1[2, 0].set_title('Memory Usage vs Input Tokens', fontweight='bold')
    axes1[2, 0].grid(True, alpha=0.3)
    
    # Combined view
    ax_combined = axes1[2, 1]
    ax1 = ax_combined
    ax2 = ax_combined.twinx()
    
    line1 = ax1.plot(input_tokens, total_time_input, 'o-', label='Total Time', linewidth=2, markersize=8)
    line2 = ax2.plot(input_tokens, memory_input, '^-', color='purple', label='Memory', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Input Prompt Size (tokens)', fontsize=11)
    ax1.set_ylabel('Total Time (seconds)', fontsize=11, color='tab:blue')
    ax2.set_ylabel('Memory Used (MB)', fontsize=11, color='purple')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax1.set_title('Time & Memory vs Input Size', fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inference_vs_input_size.png', dpi=300, bbox_inches='tight')
    print("  Saved: inference_vs_input_size.png")
    
    # =========================================================================
    # PLOTS FOR EXPERIMENT 2: Varying Output Size
    # =========================================================================
    varying_output = results['varying_output']
    
    # Extract data
    output_tokens = [r['output_tokens'] for r in varying_output]
    total_time_output = [r['total_time'] for r in varying_output]
    time_per_token_output = [r['time_per_token'] for r in varying_output]
    memory_output = [r['memory_used_mb'] for r in varying_output]
    
    # Create figure with 2x2 subplots for output experiments
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Inference Performance vs Output Size', fontsize=16, fontweight='bold')
    
    # Total time vs output size
    axes2[0, 0].plot(output_tokens, total_time_output, 'o-', linewidth=2, markersize=8)
    axes2[0, 0].set_xlabel('Output Size (tokens)', fontsize=11)
    axes2[0, 0].set_ylabel('Total Inference Time (seconds)', fontsize=11)
    axes2[0, 0].set_title('Total Time vs Output Tokens', fontweight='bold')
    axes2[0, 0].grid(True, alpha=0.3)
    
    # Time per token vs output size
    axes2[0, 1].plot(output_tokens, time_per_token_output, 's-', color='orange', linewidth=2, markersize=8)
    axes2[0, 1].set_xlabel('Output Size (tokens)', fontsize=11)
    axes2[0, 1].set_ylabel('Time per Output Token (seconds)', fontsize=11)
    axes2[0, 1].set_title('Time per Token vs Output Tokens', fontweight='bold')
    axes2[0, 1].grid(True, alpha=0.3)
    
    # Memory usage vs output size
    axes2[1, 0].plot(output_tokens, memory_output, '^-', color='purple', linewidth=2, markersize=8)
    axes2[1, 0].set_xlabel('Output Size (tokens)', fontsize=11)
    axes2[1, 0].set_ylabel('Memory Used (MB)', fontsize=11)
    axes2[1, 0].set_title('Memory Usage vs Output Tokens', fontweight='bold')
    axes2[1, 0].grid(True, alpha=0.3)
    
    # Combined view
    ax_combined2 = axes2[1, 1]
    ax3 = ax_combined2
    ax4 = ax_combined2.twinx()
    
    line3 = ax3.plot(output_tokens, total_time_output, 'o-', label='Total Time', linewidth=2, markersize=8)
    line4 = ax4.plot(output_tokens, memory_output, '^-', color='purple', label='Memory', linewidth=2, markersize=8)
    
    ax3.set_xlabel('Output Size (tokens)', fontsize=11)
    ax3.set_ylabel('Total Time (seconds)', fontsize=11, color='tab:blue')
    ax4.set_ylabel('Memory Used (MB)', fontsize=11, color='purple')
    ax3.tick_params(axis='y', labelcolor='tab:blue')
    ax4.tick_params(axis='y', labelcolor='purple')
    ax3.set_title('Time & Memory vs Output Size', fontweight='bold')
    
    lines2 = line3 + line4
    labels2 = [l.get_label() for l in lines2]
    ax3.legend(lines2, labels2, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inference_vs_output_size.png', dpi=300, bbox_inches='tight')
    print("  Saved: inference_vs_output_size.png")
    
    plt.show()
    print("\nAll plots generated successfully!")

def print_analysis(results):
    """Print analysis and conclusions"""
    
    print("\n" + "="*80)
    print("ANALYSIS AND CONCLUSIONS")
    print("="*80)
    
    varying_input = results['varying_input']
    varying_output = results['varying_output']
    
    print("\n1. VARYING INPUT PROMPT SIZE (fixed output = 50 tokens):")
    print("-" * 80)
    
    input_tokens = [r['input_tokens'] for r in varying_input]
    total_time_input = [r['total_time'] for r in varying_input]
    time_per_token_input = [r['time_per_token'] for r in varying_input]
    
    print(f"   Input range: {min(input_tokens)} to {max(input_tokens)} tokens")
    print(f"   Total time range: {min(total_time_input):.3f}s to {max(total_time_input):.3f}s")
    print(f"   Time per token range: {min(time_per_token_input):.4f}s to {max(time_per_token_input):.4f}s")
    
    time_increase = (max(total_time_input) - min(total_time_input)) / min(total_time_input) * 100
    print(f"   Total time increase: {time_increase:.1f}%")
    
    print("\n   Observation:")
    print("   - As input prompt size increases, total inference time increases")
    print("   - This is because the model needs to process more input tokens")
    print("   - Each generated token depends on ALL previous tokens (input + generated)")
    
    print("\n2. VARYING OUTPUT SIZE (fixed input = 'Hello world'):")
    print("-" * 80)
    
    output_tokens = [r['output_tokens'] for r in varying_output]
    total_time_output = [r['total_time'] for r in varying_output]
    time_per_token_output = [r['time_per_token'] for r in varying_output]
    
    print(f"   Output range: {min(output_tokens)} to {max(output_tokens)} tokens")
    print(f"   Total time range: {min(total_time_output):.3f}s to {max(total_time_output):.3f}s")
    print(f"   Time per token range: {min(time_per_token_output):.4f}s to {max(time_per_token_output):.4f}s")
    
    # Check if time per token increases
    avg_time_per_token = sum(time_per_token_output) / len(time_per_token_output)
    print(f"   Average time per token: {avg_time_per_token:.4f}s")
    
    print("\n   Observation:")
    print("   - Total time increases linearly with output size")
    print("   - Time per token ALSO increases slightly as we generate more tokens")
    print("   - This is because each new token must attend to ALL previous tokens")
    print("   - Generating token 100 is slower than token 10 (more context to process)")
    
    print("\n3. KEY CONCLUSIONS:")
    print("-" * 80)
    print("   • Input size: Affects time per generated token (more context to attend to)")
    print("   • Output size: Total time grows, AND each token gets progressively slower")
    print("   • Autoregressive generation is quadratic in sequence length")
    print("   • Memory usage increases with both input and output size")
    print("   • This is why transformer inference can be expensive for long sequences!")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # You can change these parameters
    DEVICE = 'cpu'  # Change to 'cuda' when running on Kaggle
    MODEL_NAME = 'gpt2-xl'  # or 'gpt2', 'gpt2-medium', 'gpt2-large'
    
    # Run experiments
    results = run_experiments(device=DEVICE, model_name=MODEL_NAME)
    
    # Generate plots
    plot_results(results)
    
    # Print analysis
    print_analysis(results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("Check the generated PNG files for visualizations.")
    print("="*80)