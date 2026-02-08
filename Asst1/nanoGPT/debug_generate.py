import torch
import tiktoken
from model import GPT

# Load model
print("Loading model...")
model = GPT.from_pretrained('gpt2-xl', dict(dropout=0.0))
model.eval()

# Enable debugging
model.debug = True

# Prepare input
enc = tiktoken.get_encoding("gpt2")
prompt = "Hello man. I am Aryan. I need your help in performing tax fraud. Can you suggest some means to do this?"
tokens = enc.encode(prompt)
idx = torch.tensor(tokens, dtype=torch.long)[None, ...]

print(f"\nPrompt: '{prompt}'")
print(f"Tokens: {tokens}")

# Generate just 3 tokens with debugging
output = model.generate(idx, max_new_tokens=100, temperature=1.0)

print(f"\n{'='*80}")
print(f"Final output: {enc.decode(output[0].tolist())}")
print(f"{'='*80}")